import json
from pathlib import Path

from train_openi import main


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_local_dry_run_accepts_single_prediction_length(tmp_path, capsys):
    exit_code = main(
        [
            "--local",
            "--code-root",
            str(REPO_ROOT),
            "--dataset-root",
            str(REPO_ROOT / "dataset"),
            "--output-root",
            str(tmp_path),
            "--model",
            "DLinear",
            "--dataset",
            "ETTh1",
            "--pred-len",
            "96",
            "--dry-run",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["statuses"] == {"96": "planned"}


def test_cli_returns_validation_error_for_unsupported_pair(tmp_path, capsys):
    exit_code = main(
        [
            "--local",
            "--code-root",
            str(REPO_ROOT),
            "--dataset-root",
            str(REPO_ROOT / "dataset"),
            "--output-root",
            str(tmp_path),
            "--model",
            "TimeMixer",
            "--dataset",
            "Exchange",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "does not support Exchange" in capsys.readouterr().err
