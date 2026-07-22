import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
FEDFORMER_ROOT = REPO_ROOT / "baselines" / "FEDformer"
sys.path.insert(0, str(FEDFORMER_ROOT))

from utils.amp import supports_grad_scaling  # noqa: E402


class ComplexModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(4, dtype=torch.complex64)
        )


def test_complex_parameters_disable_grad_scaling_but_support_optimizer_step():
    model = ComplexModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    assert not supports_grad_scaling(model)

    loss = model.weight.abs().square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()


def test_real_parameters_allow_grad_scaling():
    assert supports_grad_scaling(torch.nn.Linear(4, 2))
