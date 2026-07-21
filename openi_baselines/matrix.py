from __future__ import annotations

import csv
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from .paths import DatasetLocation
from .registry import parse_batch_sizes, parse_pred_lengths
from .runner import RunResult, run_task
from .types import AccelerationOptions, TaskSpec


@dataclass(frozen=True)
class MatrixTaskPlan:
    task: TaskSpec
    pred_lengths: tuple[int, ...]
    batch_sizes: Mapping[int, int]


@dataclass
class MatrixTaskResult:
    plan: MatrixTaskPlan
    result: RunResult
    error: str | None = None


@dataclass
class MatrixResult:
    exit_code: int
    tasks: list[MatrixTaskResult] = field(default_factory=list)

    def rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for item in self.tasks:
            for pred_len, status in item.result.statuses.items():
                rows.append(
                    {
                        "model": item.plan.task.model,
                        "dataset": item.plan.task.dataset,
                        "pred_len": pred_len,
                        "status": status,
                        "task_exit_code": item.result.exit_code,
                    }
                )
        return rows

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "exit_code": self.exit_code,
            "rows": self.rows(),
            "tasks": [
                {
                    "model": item.plan.task.model,
                    "dataset": item.plan.task.dataset,
                    "pred_lengths": list(item.plan.pred_lengths),
                    "result": item.result.as_dict(),
                    "error": item.error,
                }
                for item in self.tasks
            ],
        }
        if len(self.tasks) == 1:
            payload.update(self.tasks[0].result.as_dict())
            payload["exit_code"] = self.exit_code
        return payload


def plan_matrix(
    tasks: tuple[TaskSpec, ...],
    *,
    pred_len_value: str | None,
    batch_size_value: str | None,
) -> tuple[MatrixTaskPlan, ...]:
    selections = tuple(
        (task, parse_pred_lengths(task, pred_len_value))
        for task in tasks
    )

    horizon_union: list[int] = []
    for _, pred_lengths in selections:
        for pred_len in pred_lengths:
            if pred_len not in horizon_union:
                horizon_union.append(pred_len)

    global_batches = parse_batch_sizes(
        tuple(horizon_union), batch_size_value
    )
    return tuple(
        MatrixTaskPlan(
            task=task,
            pred_lengths=pred_lengths,
            batch_sizes={
                pred_len: global_batches[pred_len]
                for pred_len in pred_lengths
                if pred_len in global_batches
            },
        )
        for task, pred_lengths in selections
    )


TaskRunner = Callable[..., RunResult]


def execute_matrix(
    plans: tuple[MatrixTaskPlan, ...],
    *,
    repo_root: Path,
    data_by_dataset: Mapping[str, DatasetLocation],
    output_root: Path,
    force: bool = False,
    dry_run: bool = False,
    num_workers: int | None = None,
    acceleration: AccelerationOptions | None = None,
    task_runner: TaskRunner = run_task,
) -> MatrixResult:
    output_root = Path(output_root)
    task_results: list[MatrixTaskResult] = []
    for plan in plans:
        try:
            result = task_runner(
                task=plan.task,
                pred_lengths=plan.pred_lengths,
                repo_root=repo_root,
                data=data_by_dataset[plan.task.dataset],
                output_root=output_root,
                force=force,
                dry_run=dry_run,
                batch_sizes=dict(plan.batch_sizes),
                num_workers=num_workers,
                acceleration=acceleration,
            )
            item = MatrixTaskResult(plan=plan, result=result)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = RunResult(
                exit_code=1,
                statuses={
                    pred_len: "failed"
                    for pred_len in plan.pred_lengths
                },
            )
            item = MatrixTaskResult(
                plan=plan,
                result=result,
                error=error,
            )
            if not dry_run:
                error_path = (
                    output_root
                    / plan.task.model
                    / plan.task.dataset
                    / "matrix_error.log"
                )
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
        task_results.append(item)

    return MatrixResult(
        exit_code=(
            1
            if any(
                item.result.exit_code != 0 for item in task_results
            )
            else 0
        ),
        tasks=task_results,
    )


def write_matrix_summaries(
    output_root: Path, result: MatrixResult
) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result_payload = result.as_dict()
    payload = {
        "exit_code": result.exit_code,
        "rows": result.rows(),
        "tasks": result_payload["tasks"],
    }
    (output_root / "multi_task_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fields = [
        "model",
        "dataset",
        "pred_len",
        "status",
        "task_exit_code",
    ]
    with (output_root / "multi_task_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result.rows())
