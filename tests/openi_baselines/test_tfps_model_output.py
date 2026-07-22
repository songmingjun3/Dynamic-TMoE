import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TFPS_ROOT = REPO_ROOT / "baselines" / "TFPS"


def test_tfps_tuple_output_preserves_affinities_and_forecast():
    script = f"""
import sys
import torch
sys.path.insert(0, {str(TFPS_ROOT)!r})
from utils.model_output import unpack_model_output
time_affinity = torch.randn(2, 3)
frequency_affinity = torch.randn(2, 3)
forecast = torch.randn(2, 96, 7)
actual = unpack_model_output((time_affinity, frequency_affinity, forecast))
assert actual[0] is time_affinity
assert actual[1] is frequency_affinity
assert actual[2] is forecast
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
