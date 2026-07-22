import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TIMESNET_ROOT = REPO_ROOT / "baselines" / "TimesNet"


def test_fft_for_period_handles_non_power_of_two_fp16_input():
    script = f"""
import sys
import torch
sys.path.insert(0, {str(TIMESNET_ROOT)!r})
from models.TimesNet import FFT_for_Period
inputs = torch.randn(2, 816, 4, dtype=torch.float16)
periods, weights = FFT_for_Period(inputs, k=2)
assert len(periods) == 2
assert weights.shape == (2, 2)
assert weights.dtype == inputs.dtype
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
