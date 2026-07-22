import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FEDFORMER_ROOT = REPO_ROOT / "baselines" / "FEDformer"


def test_fourier_block_handles_non_power_of_two_fp16_input():
    script = f"""
import sys
import torch
sys.path.insert(0, {str(FEDFORMER_ROOT)!r})
from layers.FourierCorrelation import FourierBlock
block = FourierBlock(8, 8, 96, modes=4, mode_select_method='else')
inputs = torch.randn(2, 96, 8, 1, dtype=torch.float16)
outputs, attention = block(inputs, inputs, inputs, None)
assert outputs.shape == (2, 8, 1, 96)
assert outputs.dtype == inputs.dtype
assert attention is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
