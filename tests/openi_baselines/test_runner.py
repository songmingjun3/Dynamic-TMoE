import json
import sys
from pathlib import Path

from openi_baselines.paths import resolve_dataset_file
from openi_baselines.registry import get_task
from openi_baselines.runner import run_task
from openi_baselines.types import ProcessSpec


REPO_ROOT = Path(__file__).resolve().parents[2]
FAKE_TRAIN = Path(__file__).parent / "fixtures" / "fake_train.py"


def fake_builder(fail=(), calls=None):
    failed = set(fail)

    def build(task, pred_len, layout, batch_size=None, num_workers=None):
        if calls is not None:
            calls.append((pred_len, batch_size, num_workers))
        cwd = layout.task_output / "native_work" / "train"
        cwd.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, str(FAKE_TRAIN), "--pred-len", str(pred_len)]
        if pred_len in failed:
            argv.append("--fail")
        return (
            ProcessSpec(
                stage="train",
                argv=tuple(argv),
                cwd=cwd,
                env={"PYTHONUNBUFFERED": "1"},
            ),
        )

    return build


def run_fake(tmp_path, pred_lengths, **kwargs):
    return run_task(
        task=get_task("DLinear", "ETTh1"),
        pred_lengths=pred_lengths,
        repo_root=REPO_ROOT,
        data=resolve_dataset_file(REPO_ROOT / "dataset", "ETTh1"),
        output_root=tmp_path,
        process_builder=kwargs.pop("process_builder", fake_builder()),
        **kwargs,
    )


def test_runner_continues_after_one_horizon_fails(tmp_path):
    result = run_fake(
        tmp_path,
        (96, 192, 336),
        process_builder=fake_builder(fail={192}),
    )

    assert result.exit_code != 0
    assert result.statuses == {96: "succeeded", 192: "failed", 336: "succeeded"}
    assert (tmp_path / "DLinear" / "ETTh1" / "96" / "metrics.json").is_file()
    assert (tmp_path / "DLinear" / "ETTh1" / "192" / "checkpoints").is_dir()


def test_runner_skips_success_unless_force(tmp_path):
    calls = []
    builder = fake_builder(calls=calls)

    first = run_fake(tmp_path, (96,), process_builder=builder)
    resumed = run_fake(tmp_path, (96,), process_builder=builder)
    forced = run_fake(tmp_path, (96,), process_builder=builder, force=True)

    assert first.statuses[96] == "succeeded"
    assert resumed.statuses[96] == "skipped"
    assert forced.statuses[96] == "succeeded"
    assert calls == [(96, None, None), (96, None, None)]


def test_runner_writes_dataset_level_json_and_csv_summaries(tmp_path):
    run_fake(tmp_path, (96, 192))
    dataset_output = tmp_path / "DLinear" / "ETTh1"

    status = json.loads((dataset_output / "status_summary.json").read_text(encoding="utf-8"))

    assert status["model"] == "DLinear"
    assert status["statuses"] == {"96": "succeeded", "192": "succeeded"}
    assert (dataset_output / "metrics_summary.json").is_file()
    assert (dataset_output / "metrics_summary.csv").is_file()


def test_dry_run_builds_commands_without_starting_processes(tmp_path):
    result = run_fake(tmp_path, (96,), dry_run=True)

    assert result.exit_code == 0
    assert result.statuses[96] == "planned"
    assert result.commands[96][0]["stage"] == "train"
    assert not (tmp_path / "DLinear" / "ETTh1" / "96" / "status.json").exists()


def test_runner_routes_each_horizon_batch_size(tmp_path):
    calls = []

    run_fake(
        tmp_path,
        (96, 192),
        batch_sizes={96: 128, 192: 64},
        process_builder=fake_builder(calls=calls),
        dry_run=True,
    )

    assert calls == [(96, 128, None), (192, 64, None)]


def test_runner_routes_num_workers_to_every_horizon(tmp_path):
    calls = []

    run_fake(
        tmp_path,
        (96, 192),
        num_workers=4,
        process_builder=fake_builder(calls=calls),
        dry_run=True,
    )

    assert calls == [(96, None, 4), (192, None, 4)]
