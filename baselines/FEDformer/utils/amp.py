from __future__ import annotations

import torch


def supports_grad_scaling(model: torch.nn.Module) -> bool:
    return not any(parameter.is_complex() for parameter in model.parameters())


def build_grad_scaler(model: torch.nn.Module):
    enabled = supports_grad_scaling(model)
    if not enabled:
        print(
            "AMP autocast enabled; GradScaler disabled because FEDformer "
            "contains complex-valued parameters."
        )
    return torch.cuda.amp.GradScaler(enabled=enabled)
