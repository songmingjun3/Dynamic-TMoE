import json
from pathlib import Path

from train_openi import main


REPO_ROOT = Path(__file__).resolve().parents[2]


def native_option(command, name):
    argv = command["argv"]
    return argv[argv.index(name) + 1]


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


def test_cli_accepts_per_horizon_batch_sizes(tmp_path, capsys):
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
            "96,192",
            "--batch-size",
            "192:64,96:128",
            "--dry-run",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    argv_96 = output["commands"]["96"][0]["argv"]
    argv_192 = output["commands"]["192"][0]["argv"]
    assert exit_code == 0
    assert argv_96[argv_96.index("--batch_size") + 1] == "128"
    assert argv_192[argv_192.index("--batch_size") + 1] == "64"


def test_cli_rejects_incomplete_batch_size_mapping_before_training(
    tmp_path, capsys
):
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
            "96,192",
            "--batch-size",
            "96:128",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "Missing batch size for prediction lengths: 192" in capsys.readouterr().err
    assert not (tmp_path / "DLinear" / "ETTh1" / "96" / "status.json").exists()


def test_cli_dry_run_executes_model_dataset_cartesian_product(
    tmp_path, capsys
):
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
            "DLinear,PatchTST",
            "--dataset",
            "ETTh1,Weather",
            "--pred-len",
            "96",
            "--batch-size",
            "96:32",
            "--dry-run",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert [
        (item["model"], item["dataset"])
        for item in output["tasks"]
    ] == [
        ("DLinear", "ETTh1"),
        ("DLinear", "Weather"),
        ("PatchTST", "ETTh1"),
        ("PatchTST", "Weather"),
    ]
    assert all(
        native_option(
            item["result"]["commands"]["96"][0], "--batch_size"
        )
        == "32"
        for item in output["tasks"]
    )


def test_cli_preflight_rejects_unsupported_pair_without_running_any_task(
    tmp_path, capsys
):
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
            "DLinear,TimeMixer",
            "--dataset",
            "ETTh1,ILI",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "TimeMixer does not support ILI" in capsys.readouterr().err
    assert not (tmp_path / "DLinear").exists()
