import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TIMESNET_ROOT = REPO_ROOT / "baselines" / "TimesNet"
sys.path.insert(0, str(TIMESNET_ROOT))

from models.TimesNet import FFT_for_Period  # noqa: E402


def test_fft_for_period_handles_non_power_of_two_fp16_input():
    inputs = torch.randn(2, 816, 4, dtype=torch.float16)

    periods, weights = FFT_for_Period(inputs, k=2)

    assert len(periods) == 2
    assert weights.shape == (2, 2)
    assert weights.dtype == inputs.dtype
