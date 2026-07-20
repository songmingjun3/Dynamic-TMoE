from __future__ import annotations

from collections.abc import Iterable
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
