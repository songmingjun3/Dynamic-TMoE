from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from .types import TaskSpec


LONG_HORIZONS = (96, 192, 336, 720)
ILI_HORIZONS = (24, 36, 48, 60)

DATASETS = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Electricity",
    "Exchange",
    "ILI",
    "Traffic",
    "Weather",
)

MODEL_DEFINITIONS = {
    "DLinear": ("baselines/Dlinear/run_longExp.py", "DLinear", DATASETS),
    "Dynamic_TMoE": ("run.py", "Dynamic_TMoE", DATASETS),
    "FEDformer": ("baselines/FEDformer/run.py", "FEDformer", DATASETS),
    "FITS": ("baselines/FITS/run_longExp_F.py", "FITS", DATASETS),
    "PatchTST": ("baselines/PatchTST/run_longExp.py", "PatchTST", DATASETS),
    "RAFT": ("baselines/RAFT/run.py", "RAFT", DATASETS),
    "ST-MTM": ("baselines/st-mtm/run.py", "STMTM", DATASETS),
    "TFPS": (
        "baselines/TFPS/run_longExp.py",
        "PatchTST_MoE_cluster",
        DATASETS,
    ),
    "TimeMixer": (
        "baselines/TimeMixer/run.py",
        "TimeMixer",
        (
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "Electricity",
            "Exchange",
            "ILI",
            "Traffic",
            "Weather",
        ),
    ),
    "TimesNet": ("baselines/TimesNet/run.py", "TimesNet", DATASETS),
}


class UnsupportedTaskError(ValueError):
    """Raised when a model/dataset pair is absent from repository scripts."""


def _normalize(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


MODEL_ALIASES = {_normalize(name): name for name in MODEL_DEFINITIONS}
MODEL_ALIASES.update({"stmtm": "ST-MTM", "dlinear": "DLinear"})

DATASET_ALIASES = {_normalize(name): name for name in DATASETS}
DATASET_ALIASES.update(
    {
        "ecl": "Electricity",
        "electric": "Electricity",
        "exchangerate": "Exchange",
        "illness": "ILI",
        "influenza": "ILI",
    }
)


def _build_tasks() -> tuple[TaskSpec, ...]:
    tasks: list[TaskSpec] = []
    for model, (entrypoint, native_model, datasets) in MODEL_DEFINITIONS.items():
        for dataset in datasets:
            tasks.append(
                TaskSpec(
                    model=model,
                    dataset=dataset,
                    entrypoint=Path(entrypoint),
                    data_kind=dataset,
                    native_model=native_model,
                    horizons=ILI_HORIZONS if dataset == "ILI" else LONG_HORIZONS,
                    stages=("pretrain", "finetune") if model == "ST-MTM" else ("train",),
                )
            )
    return tuple(tasks)


TASKS = _build_tasks()
TASK_INDEX = {(task.model, task.dataset): task for task in TASKS}


def iter_tasks() -> Iterable[TaskSpec]:
    return iter(TASKS)


def _canonical_model(value: str) -> str:
    try:
        return MODEL_ALIASES[_normalize(value)]
    except KeyError as exc:
        supported = ", ".join(MODEL_DEFINITIONS)
        raise UnsupportedTaskError(
            f"Unsupported model {value!r}. Supported models: {supported}"
        ) from exc


def _canonical_dataset(value: str) -> str:
    try:
        return DATASET_ALIASES[_normalize(value)]
    except KeyError as exc:
        supported = ", ".join(DATASETS)
        raise UnsupportedTaskError(
            f"Unsupported dataset {value!r}. Supported datasets: {supported}"
        ) from exc


def get_task(model: str, dataset: str) -> TaskSpec:
    canonical_model = _canonical_model(model)
    canonical_dataset = _canonical_dataset(dataset)
    try:
        return TASK_INDEX[(canonical_model, canonical_dataset)]
    except KeyError as exc:
        supported = ", ".join(
            task.dataset for task in TASKS if task.model == canonical_model
        )
        raise UnsupportedTaskError(
            f"{canonical_model} does not support {canonical_dataset}. "
            f"Supported datasets: {supported}"
        ) from exc


def _parse_canonical_list(
    value: str,
    *,
    label: str,
    canonicalize: Callable[[str], str],
) -> tuple[str, ...]:
    selected: list[str] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            raise ValueError(
                f"{label} values must be comma-separated names"
            )
        canonical = canonicalize(item)
        if canonical not in selected:
            selected.append(canonical)
    return tuple(selected)


def parse_models(value: str) -> tuple[str, ...]:
    return _parse_canonical_list(
        value, label="model", canonicalize=_canonical_model
    )


def parse_datasets(value: str) -> tuple[str, ...]:
    return _parse_canonical_list(
        value, label="dataset", canonicalize=_canonical_dataset
    )


def parse_task_matrix(
    model_value: str, dataset_value: str
) -> tuple[TaskSpec, ...]:
    models = parse_models(model_value)
    datasets = parse_datasets(dataset_value)
    return tuple(
        get_task(model, dataset)
        for model in models
        for dataset in datasets
    )


def parse_pred_lengths(task: TaskSpec, value: str | None) -> tuple[int, ...]:
    if value is None or value.strip().casefold() == "all":
        return task.horizons

    selected: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("Prediction lengths must be comma-separated integers")
        try:
            pred_len = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid prediction length {item!r}") from exc
        if pred_len not in task.horizons:
            allowed = ", ".join(str(length) for length in task.horizons)
            raise ValueError(
                f"Invalid prediction length {pred_len} for {task.model}/{task.dataset}. "
                f"Allowed values: {allowed}"
            )
        if pred_len not in selected:
            selected.append(pred_len)
    return tuple(selected)


def parse_batch_sizes(
    pred_lengths: tuple[int, ...], value: str | None
) -> dict[int, int]:
    if value is None:
        return {}

    parsed: dict[int, int] = {}
    for raw_item in value.split(","):
        item = raw_item.strip()
        if item.count(":") != 1:
            raise ValueError(
                f"Invalid batch size mapping {item!r}. "
                "Expected PRED_LEN:BATCH_SIZE"
            )

        raw_pred_len, raw_batch_size = (
            part.strip() for part in item.split(":")
        )
        try:
            pred_len = int(raw_pred_len)
        except ValueError as exc:
            raise ValueError(
                f"Invalid prediction length {raw_pred_len!r} "
                "in batch size mapping"
            ) from exc
        if pred_len in parsed:
            raise ValueError(
                f"Duplicate batch size mapping for prediction length {pred_len}"
            )

        try:
            batch_size = int(raw_batch_size)
        except ValueError as exc:
            raise ValueError(f"Invalid batch size {raw_batch_size!r}") from exc
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")
        parsed[pred_len] = batch_size

    selected = set(pred_lengths)
    provided = set(parsed)
    missing = selected - provided
    extra = provided - selected
    if missing:
        values = ", ".join(str(item) for item in sorted(missing))
        raise ValueError(f"Missing batch size for prediction lengths: {values}")
    if extra:
        values = ", ".join(str(item) for item in sorted(extra))
        raise ValueError(
            "Batch sizes provided for unselected prediction lengths: "
            f"{values}"
        )
    return parsed
