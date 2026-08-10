from pathlib import Path

from openi_baselines.commands import CommandLayout, build_processes
from openi_baselines.paths import resolve_dataset_file
from openi_baselines.registry import parse_task_matrix
from openi_baselines.strict_config import apply_strict_config


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "openi_paper_strict_v1" / "matrix.json"


def option(argv, name):
    index = argv.index(name)
    return argv[index + 1]


def strict_task(dataset):
    return apply_strict_config(
        parse_task_matrix("Dynamic_TMoE", dataset), CONFIG
    )[0]


def make_layout(tmp_path, dataset):
    return CommandLayout(
        repo_root=REPO_ROOT,
        data=resolve_dataset_file(REPO_ROOT / "dataset", dataset),
        task_output=tmp_path / "task",
    )


def test_strict_etth1_96_emits_table7_overrides(tmp_path):
    task = strict_task("ETTh1")
    process = build_processes(
        task, 96, make_layout(tmp_path, "ETTh1")
    )[0]

    assert task.strict_config == "paper_strict_v1"
    assert option(process.argv, "--patch_len") == "48"
    assert option(process.argv, "--stride") == "12"
    assert option(process.argv, "--learning_rate") == "0.0012"
    assert option(process.argv, "--dropout") == "0.4"
    assert option(process.argv, "--num_temporal_moe_layers") == "2"
    assert option(process.argv, "--num_rnn_layers") == "1"
    assert option(process.argv, "--batch_size") == "256"
    assert option(process.argv, "--model_id").endswith(
        "_96_s2021_paper_strict_v1"
    )
    assert process.argv.count("--learning_rate") == 1
    assert "--use_amp" not in process.argv


def test_strict_ili_36_emits_short_context_and_ili_overrides(tmp_path):
    task = strict_task("ILI")
    process = build_processes(
        task, 36, make_layout(tmp_path, "ILI")
    )[0]

    assert option(process.argv, "--seq_len") == "36"
    assert option(process.argv, "--label_len") == "18"
    assert option(process.argv, "--patch_len") == "24"
    assert option(process.argv, "--stride") == "2"
    assert option(process.argv, "--learning_rate") == "0.0004"
    assert option(process.argv, "--dropout") == "0.1"
    assert option(process.argv, "--d_model") == "1024"
    assert option(process.argv, "--factor") == "3"
    assert option(process.argv, "--e_layers") == "4"
    assert option(process.argv, "--batch_size") == "64"


def test_strict_config_covers_all_table7_tasks():
    tasks = apply_strict_config(
        parse_task_matrix(
            "Dynamic_TMoE",
            "ETTh1,ETTh2,ETTm1,ETTm2,Traffic,Electricity,Weather,ILI,Exchange",
        ),
        CONFIG,
    )
    assert sum(len(task.horizon_overrides) for task in tasks) == 36
