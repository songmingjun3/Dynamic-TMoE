from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .registry import parse_batch_sizes, parse_pred_lengths
from .types import TaskSpec


@dataclass(frozen=True)
class MatrixTaskPlan:
    task: TaskSpec
    pred_lengths: tuple[int, ...]
    batch_sizes: Mapping[int, int]


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
