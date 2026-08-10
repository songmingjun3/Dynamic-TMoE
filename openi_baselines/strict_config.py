from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .types import TaskSpec


REQUIRED_COMMON_FIELDS = {
    "task_name",
    "is_training",
    "features",
    "des",
    "itr",
    "seed",
    "train_sample_limit",
    "channel_independence",
    "use_relation_layer",
    "enable_drift_detection",
    "n_heads",
    "e_layers",
    "d_layers",
    "d_ff",
    "factor",
    "expand",
    "d_conv",
    "num_drift_experts",
    "train_epochs",
    "patience",
    "cycle_length",
    "drift_window_size",
    "drift_k_sigma",
    "finetune_epochs",
    "num_workers",
    "use_amp",
    "gpu",
    "checkpoints",
}
REQUIRED_HORIZON_FIELDS = {
    "patch_len",
    "stride",
    "learning_rate",
    "dropout",
    "num_temporal_moe_layers",
    "num_rnn_layers",
    "d_model",
    "batch_size",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid strict configuration JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Strict configuration root must be an object")
    return payload


def _horizon_overrides(raw: object, *, dataset: str, expected: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError(f"Strict configuration horizons for {dataset} must be an object")
    result: dict[int, dict[str, Any]] = {}
    for raw_horizon, values in raw.items():
        try:
            horizon = int(raw_horizon)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid horizon {raw_horizon!r} for {dataset}") from exc
        if not isinstance(values, dict):
            raise ValueError(f"Overrides for {dataset}/{horizon} must be an object")
        result[horizon] = dict(values)

    expected_set = set(expected)
    actual_set = set(result)
    if actual_set != expected_set:
        missing = ", ".join(str(item) for item in sorted(expected_set - actual_set))
        extra = ", ".join(str(item) for item in sorted(actual_set - expected_set))
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(f"Horizon mismatch for {dataset}: {'; '.join(details)}")
    return result


def apply_strict_config(
    tasks: tuple[TaskSpec, ...],
    config_path: str | Path,
) -> tuple[TaskSpec, ...]:
    """Attach a versioned strict matrix's runtime overrides to task specs."""

    path = Path(config_path).expanduser().resolve()
    payload = _read_json(path)
    if payload.get("schema") != 1:
        raise ValueError("Unsupported strict configuration schema")
    matrix_id = payload.get("matrix_id")
    if not isinstance(matrix_id, str) or not matrix_id:
        raise ValueError("Strict configuration requires a non-empty matrix_id")
    if payload.get("model") != "Dynamic_TMoE":
        raise ValueError("Strict configuration model must be Dynamic_TMoE")

    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Strict configuration requires a datasets object")

    configured: list[TaskSpec] = []
    for task in tasks:
        if task.model != "Dynamic_TMoE":
            raise ValueError("Strict configuration can only be applied to Dynamic_TMoE")
        raw_dataset = datasets.get(task.dataset)
        if not isinstance(raw_dataset, dict):
            raise ValueError(f"Strict configuration has no dataset {task.dataset}")
        common = raw_dataset.get("common", {})
        if not isinstance(common, dict):
            raise ValueError(f"Common overrides for {task.dataset} must be an object")
        missing_common = sorted(REQUIRED_COMMON_FIELDS - set(common))
        if missing_common:
            raise ValueError(
                f"Strict configuration missing common fields for {task.dataset}: "
                + ", ".join(missing_common)
            )
        horizons = _horizon_overrides(
            raw_dataset.get("horizons"),
            dataset=task.dataset,
            expected=task.horizons,
        )
        for horizon, values in horizons.items():
            missing_horizon = sorted(REQUIRED_HORIZON_FIELDS - set(values))
            if missing_horizon:
                raise ValueError(
                    f"Strict configuration missing fields for "
                    f"{task.dataset}/{horizon}: "
                    + ", ".join(missing_horizon)
                )
        configured.append(
            replace(
                task,
                parameters=dict(common),
                horizon_overrides=horizons,
                strict_config=matrix_id,
            )
        )
    return tuple(configured)
