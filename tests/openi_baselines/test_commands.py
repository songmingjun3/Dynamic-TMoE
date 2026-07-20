import sys
from pathlib import Path

import pytest

from openi_baselines.commands import CommandLayout, build_processes
from openi_baselines.paths import resolve_dataset_file
from openi_baselines.registry import get_task, iter_tasks


REPO_ROOT = Path(__file__).resolve().parents[2]


def option(argv, name):
    index = argv.index(name)
    return argv[index + 1]


def make_layout(tmp_path, dataset):
    return CommandLayout(
        repo_root=REPO_ROOT,
        data=resolve_dataset_file(REPO_ROOT / "dataset", dataset),
        task_output=tmp_path / "task",
    )


def test_dlinear_command_uses_mounted_data_gpu_zero_and_requested_horizon(tmp_path):
    layout = make_layout(tmp_path, "ETTh1")

    process = build_processes(get_task("DLinear", "ETTh1"), 336, layout)[0]

    assert process.argv[0] == sys.executable
    assert Path(process.argv[2]) == REPO_ROOT / "baselines" / "Dlinear" / "run_longExp.py"
    assert option(process.argv, "--root_path") == str(layout.data.root_path)
    assert option(process.argv, "--data_path") == "ETTh1.csv"
    assert option(process.argv, "--pred_len") == "336"
    assert option(process.argv, "--gpu") == "0"
    # These baseline implementations save via args.checkpoints but reload from
    # a hard-coded ./checkpoints path during evaluation. Keep both paths aligned.
    assert option(process.argv, "--checkpoints") == "./checkpoints"


@pytest.mark.parametrize(
    ("model", "expected_native_model"),
    [
        ("FEDformer", "FEDformer"),
        ("FITS", "FITS"),
        ("PatchTST", "PatchTST"),
        ("RAFT", "RAFT"),
        ("TFPS", "PatchTST_MoE_cluster"),
        ("TimeMixer", "TimeMixer"),
        ("TimesNet", "TimesNet"),
    ],
)
def test_standard_models_emit_native_model_and_prediction_length(
    tmp_path, model, expected_native_model
):
    layout = make_layout(tmp_path, "Weather")

    process = build_processes(get_task(model, "Weather"), 720, layout)[0]

    assert option(process.argv, "--model") == expected_native_model
    assert option(process.argv, "--pred_len") == "720"
    assert option(process.argv, "--gpu") == "0"


def test_ili_command_uses_short_sequence_and_label_lengths(tmp_path):
    layout = make_layout(tmp_path, "ILI")

    process = build_processes(get_task("TimesNet", "ILI"), 24, layout)[0]

    assert option(process.argv, "--seq_len") == "36"
    assert option(process.argv, "--label_len") == "18"


def test_stmtm_builds_pretrain_then_finetune_with_shared_pretrain_directory(tmp_path):
    layout = make_layout(tmp_path, "Weather")

    processes = build_processes(get_task("ST-MTM", "Weather"), 720, layout)

    assert [process.stage for process in processes] == ["pretrain", "finetune"]
    assert option(processes[0].argv, "--task_name") == "pretrain"
    assert option(processes[1].argv, "--task_name") == "finetune"
    assert option(processes[0].argv, "--pretrain_checkpoints") == option(
        processes[1].argv, "--pretrain_checkpoints"
    )
    assert option(processes[1].argv, "--pred_len") == "720"


def test_all_registered_commands_reference_existing_entrypoints(tmp_path):
    for task in iter_tasks():
        layout = make_layout(tmp_path / task.model / task.dataset, task.dataset)
        for pred_len in task.horizons:
            for process in build_processes(task, pred_len, layout):
                assert Path(process.argv[2]).is_file(), process.argv[2]
                assert process.cwd.is_relative_to(layout.task_output)


@pytest.mark.parametrize("model", ["DLinear", "FEDformer", "TimesNet"])
def test_batch_size_override_replaces_or_adds_native_option(tmp_path, model):
    layout = make_layout(tmp_path, "ETTh1")

    process = build_processes(
        get_task(model, "ETTh1"), 96, layout, batch_size=48
    )[0]

    assert option(process.argv, "--batch_size") == "48"
    assert process.argv.count("--batch_size") == 1


def test_stmtm_batch_size_override_applies_to_both_stages(tmp_path):
    layout = make_layout(tmp_path, "ETTh1")

    processes = build_processes(
        get_task("ST-MTM", "ETTh1"), 96, layout, batch_size=24
    )

    assert [
        option(process.argv, "--batch_size") for process in processes
    ] == ["24", "24"]
