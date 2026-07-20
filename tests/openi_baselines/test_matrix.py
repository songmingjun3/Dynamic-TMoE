import json
from pathlib import Path

import pytest

from openi_baselines.matrix import (
    execute_matrix,
    plan_matrix,
    write_matrix_summaries,
)
from openi_baselines.paths import resolve_dataset_file
from openi_baselines.registry import parse_task_matrix
from openi_baselines.runner import RunResult


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_plan_matrix_uses_explicit_horizons_and_global_batches():
    plans = plan_matrix(
        parse_task_matrix("DLinear,PatchTST", "ETTh1"),
        pred_len_value="96,192",
        batch_size_value="192:64,96:128",
    )

    assert [
        (plan.task.model, plan.pred_lengths, plan.batch_sizes)
        for plan in plans
    ] == [
        ("DLinear", (96, 192), {96: 128, 192: 64}),
        ("PatchTST", (96, 192), {96: 128, 192: 64}),
    ]


def test_plan_matrix_all_uses_horizon_union_for_global_batch_validation():
    plans = plan_matrix(
        parse_task_matrix("DLinear", "ETTh1,ILI"),
        pred_len_value="all",
        batch_size_value=(
            "24:8,36:8,48:8,60:8,96:32,192:16,336:8,720:4"
        ),
    )

    assert plans[0].pred_lengths == (96, 192, 336, 720)
    assert plans[0].batch_sizes == {
        96: 32,
        192: 16,
        336: 8,
        720: 4,
    }
    assert plans[1].pred_lengths == (24, 36, 48, 60)
    assert plans[1].batch_sizes == {24: 8, 36: 8, 48: 8, 60: 8}


def test_plan_matrix_rejects_explicit_horizon_invalid_for_any_task():
    with pytest.raises(ValueError, match="DLinear/ILI"):
        plan_matrix(
            parse_task_matrix("DLinear", "ETTh1,ILI"),
            pred_len_value="96",
            batch_size_value=None,
        )


def test_execute_matrix_is_serial_and_continues_after_failed_task(tmp_path):
    plans = plan_matrix(
        parse_task_matrix("DLinear,PatchTST", "ETTh1"),
        pred_len_value="96",
        batch_size_value="96:32",
    )
    calls = []

    def recording_runner(**kwargs):
        calls.append(
            (kwargs["task"].model, kwargs["batch_sizes"][96])
        )
        failed = kwargs["task"].model == "DLinear"
        return RunResult(
            exit_code=1 if failed else 0,
            statuses={96: "failed" if failed else "succeeded"},
        )

    result = execute_matrix(
        plans,
        repo_root=tmp_path,
        data_by_dataset={
            "ETTh1": resolve_dataset_file(
                REPO_ROOT / "dataset", "ETTh1"
            )
        },
        output_root=tmp_path / "output",
        task_runner=recording_runner,
    )

    assert calls == [("DLinear", 32), ("PatchTST", 32)]
    assert result.exit_code == 1
    assert [item.result.exit_code for item in result.tasks] == [1, 0]


def test_execute_matrix_isolates_unexpected_task_exceptions(tmp_path):
    plans = plan_matrix(
        parse_task_matrix("DLinear,PatchTST", "ETTh1"),
        pred_len_value="96",
        batch_size_value=None,
    )
    calls = []

    def raising_runner(**kwargs):
        calls.append(kwargs["task"].model)
        if kwargs["task"].model == "DLinear":
            raise RuntimeError("broken task")
        return RunResult(exit_code=0, statuses={96: "succeeded"})

    result = execute_matrix(
        plans,
        repo_root=tmp_path,
        data_by_dataset={
            "ETTh1": resolve_dataset_file(
                REPO_ROOT / "dataset", "ETTh1"
            )
        },
        output_root=tmp_path / "output",
        task_runner=raising_runner,
    )

    assert calls == ["DLinear", "PatchTST"]
    assert result.tasks[0].error == "RuntimeError: broken task"
    assert result.tasks[0].result.statuses == {96: "failed"}
    assert (
        tmp_path
        / "output"
        / "DLinear"
        / "ETTh1"
        / "matrix_error.log"
    ).is_file()


def test_write_matrix_summaries_records_every_horizon(tmp_path):
    plans = plan_matrix(
        parse_task_matrix("DLinear", "ETTh1"),
        pred_len_value="96,192",
        batch_size_value=None,
    )

    def successful_runner(**kwargs):
        return RunResult(
            exit_code=0,
            statuses={96: "succeeded", 192: "skipped"},
        )

    result = execute_matrix(
        plans,
        repo_root=tmp_path,
        data_by_dataset={
            "ETTh1": resolve_dataset_file(
                REPO_ROOT / "dataset", "ETTh1"
            )
        },
        output_root=tmp_path,
        task_runner=successful_runner,
    )
    write_matrix_summaries(tmp_path, result)

    payload = json.loads(
        (tmp_path / "multi_task_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["exit_code"] == 0
    assert payload["rows"] == [
        {
            "model": "DLinear",
            "dataset": "ETTh1",
            "pred_len": 96,
            "status": "succeeded",
            "task_exit_code": 0,
        },
        {
            "model": "DLinear",
            "dataset": "ETTh1",
            "pred_len": 192,
            "status": "skipped",
            "task_exit_code": 0,
        },
    ]
    assert (tmp_path / "multi_task_summary.csv").is_file()
