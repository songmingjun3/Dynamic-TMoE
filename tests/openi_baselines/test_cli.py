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


def test_cli_accepts_timemixer_exchange(tmp_path, capsys):
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

    output = json.loads(capsys.readouterr().out)
    command = output["commands"]["96"][0]
    assert exit_code == 0
    assert native_option(command, "--data") == "custom"
    assert native_option(command, "--enc_in") == "8"
    assert native_option(command, "--dec_in") == "8"
    assert native_option(command, "--c_out") == "8"


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


def test_cli_accepts_num_workers_and_applies_it_to_every_process(
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
            "ST-MTM",
            "--dataset",
            "ETTh1",
            "--pred-len",
            "96",
            "--num-workers",
            "2",
            "--dry-run",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    commands = output["commands"]["96"]
    assert exit_code == 0
    assert len(commands) == 2
    assert all(
        native_option(command, "--num_workers") == "2"
        for command in commands
    )


def test_cli_rejects_non_positive_num_workers(tmp_path, capsys):
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
            "--num-workers",
            "0",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "num-workers must be a positive integer" in capsys.readouterr().err


def test_cli_accepts_acceleration_options(tmp_path, capsys):
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
            "PatchTST",
            "--dataset",
            "ETTh1",
            "--pred-len",
            "96",
            "--use-amp",
            "true",
            "--patience",
            "4",
            "--pin-memory",
            "true",
            "--persistent-workers",
            "true",
            "--prefetch-factor",
            "2",
            "--cudnn-benchmark",
            "true",
            "--cpu-threads",
            "1",
            "--dry-run",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    command = output["commands"]["96"][0]
    assert exit_code == 0
    assert "--use_amp" in command["argv"]
    assert native_option(command, "--patience") == "4"
    assert command["env"]["OPENI_PIN_MEMORY"] == "1"
    assert command["env"]["OPENI_PERSISTENT_WORKERS"] == "1"
    assert command["env"]["OPENI_PREFETCH_FACTOR"] == "2"
    assert command["env"]["OPENI_CUDNN_BENCHMARK"] == "1"
    assert command["env"]["OMP_NUM_THREADS"] == "1"


def test_cli_accepts_false_amp_value_without_forwarding_native_flag(
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
            "96",
            "--use-amp",
            "false",
            "--pin-memory",
            "false",
            "--persistent-workers",
            "false",
            "--cudnn-benchmark",
            "false",
            "--dry-run",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    command = output["commands"]["96"][0]
    assert exit_code == 0
    assert "--use_amp" not in command["argv"]
    assert command["env"]["OPENI_PIN_MEMORY"] == "0"
    assert command["env"]["OPENI_PERSISTENT_WORKERS"] == "0"
    assert command["env"]["OPENI_CUDNN_BENCHMARK"] == "0"


def test_cli_rejects_non_positive_acceleration_values(tmp_path, capsys):
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
            "--prefetch-factor",
            "0",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert (
        "prefetch-factor must be a positive integer"
        in capsys.readouterr().err
    )


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
