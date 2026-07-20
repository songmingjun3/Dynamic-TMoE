import torch
from torch import nn
import torch.nn.functional as F

class TemporalMemoryRouter(nn.Module):
    def __init__(self, d_model, num_base_experts, num_drift_experts=2, hidden_dim=None, dropout=0.1, 
                 expert_pool=None, num_layers=1, anomaly_repo=None, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.num_base_experts = num_base_experts
        self.num_drift_experts = num_drift_experts
        self.expert_pool = expert_pool
        self.num_layers = num_layers
        self.anomaly_repo = anomaly_repo
        self.top_k = top_k
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.rnn_cells = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.base_output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_base_experts)
        )

        self.drift_output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_drift_experts)
        )

        self._initialize_routing_weights()
    
    @property
    def current_num_experts(self):
        if self.expert_pool is not None:
            return self.expert_pool.current_num_experts
        return self.num_base_experts
    
    def _initialize_routing_weights(self):
        with torch.no_grad():
            output_linear = self.base_output_layer[-1]
            output_linear.bias[:self.num_base_experts].fill_(0.0)
    
    def init_hidden(self, batch_size, device):
        if self.num_layers == 1:
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def _as_hidden_stack(self, hidden_state, batch_size, device):
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, device)

        if hidden_state.dim() == 2:
            if self.num_layers == 1:
                return hidden_state.unsqueeze(0)

            hidden_stack = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dim,
                device=device,
                dtype=hidden_state.dtype
            )
            hidden_stack[0] = hidden_state
            return hidden_stack

        return hidden_state

    def _restore_hidden_shape(self, hidden_stack):
        if self.num_layers == 1:
            return hidden_stack[-1]
        return hidden_stack
    
    def forward(self, features, hidden_state=None, use_anomaly_reference=False, patch_idx=None):
        """
        Args:
            features: [B*N, D]
            hidden_state: [B*N, hidden_dim]
            use_anomaly_reference: bool
            patch_idx: int
        Returns:
            routing_weights: [B*N, K_enabled]
            routing_decisions: [B*N, K_enabled]
            new_hidden_state
        """
        BN, D = features.shape
        device = features.device
        
        K_total = self.num_base_experts + self.num_drift_experts
        if self.expert_pool is not None:
            K_enabled = self.expert_pool.current_num_experts
            enabled_indices = self.expert_pool.get_enabled_expert_indices()
        else:
            K_enabled = self.num_base_experts
            enabled_indices = list(range(self.num_base_experts))

        hidden_stack = self._as_hidden_stack(hidden_state, BN, device)

        anomaly_reference_used = False
        if use_anomaly_reference and self.anomaly_repo is not None and patch_idx is not None:
            ref_state, has_ref = self.anomaly_repo.get_reference_state(
                patch_idx, hidden_stack[-1]
            )
            if has_ref:
                hidden_stack = hidden_stack.clone()
                hidden_stack[-1] = ref_state
                anomaly_reference_used = True

        x = self.input_projection(features)

        new_hidden_layers = []
        layer_input = x
        for layer_idx, rnn_cell in enumerate(self.rnn_cells):
            layer_hidden = rnn_cell(layer_input, hidden_stack[layer_idx])
            new_hidden_layers.append(layer_hidden)
            layer_input = layer_hidden

        new_hidden_stack = torch.stack(new_hidden_layers, dim=0)
        router_state = new_hidden_stack[-1]
        new_hidden_state = self._restore_hidden_shape(new_hidden_stack)

        base_expert_logits = self.base_output_layer(router_state)

        drift_pool_enabled = (self.expert_pool is not None and 
                             self.expert_pool.is_drift_pool_enabled())

        if self.num_drift_experts > 0 and drift_pool_enabled:
            drift_logits = self.drift_output_layer(router_state)
            all_expert_logits = torch.cat([base_expert_logits, drift_logits], dim=-1)
        else:
            all_expert_logits = base_expert_logits

        enabled_expert_logits = all_expert_logits[:, enabled_indices]

        base_probs = F.softmax(enabled_expert_logits, dim=-1)

        k = min(self.top_k, K_enabled)

        topk_values, topk_indices = torch.topk(base_probs, k, dim=-1)

        routing_decisions = torch.zeros_like(base_probs, dtype=torch.bool)
        routing_decisions.scatter_(1, topk_indices, True)

        activated_probs = base_probs * routing_decisions.float()
        prob_sum = activated_probs.sum(dim=-1, keepdim=True)
        prob_sum = torch.clamp(prob_sum, min=1e-9)
        routing_weights = activated_probs / prob_sum
        
        return routing_weights, routing_decisions, new_hidden_state
