import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TFPS_ROOT = REPO_ROOT / "baselines" / "TFPS"


def test_tfps_fft_handles_non_power_of_two_fp16_patch_axis():
    script = f"""
import sys
import torch
sys.path.insert(0, {str(TFPS_ROOT)!r})
from layers.PatchTST_MoE_backbone_cluster_frequency import (
    fft2_real_fp32,
    ifft2_real_fp32,
)
inputs = torch.randn(2, 7, 128, 12, dtype=torch.float16)
frequency = fft2_real_fp32(inputs)
restored = ifft2_real_fp32(frequency)
assert frequency.shape == inputs.shape
assert restored.shape == inputs.shape
assert frequency.dtype == inputs.dtype
assert restored.dtype == inputs.dtype
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
