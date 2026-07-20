from __future__ import annotations

import shutil
from pathlib import Path


CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt"}
PREDICTION_SUFFIXES = {".npy", ".npz", ".csv"}
RESULT_SUFFIXES = {".txt", ".json", ".csv", ".npy", ".npz", ".pdf"}


def _category(path: Path) -> str | None:
    name = path.name.casefold()
    suffix = path.suffix.casefold()
    if suffix in CHECKPOINT_SUFFIXES or "checkpoint" in name or name.startswith("ckpt"):
        return "checkpoints"
    if suffix in PREDICTION_SUFFIXES and (
        name.startswith("pred") or "prediction" in name
    ):
        return "predictions"
    if suffix in RESULT_SUFFIXES and (
        name.startswith("result")
        or name.startswith("metric")
        or suffix == ".pdf"
        or name.startswith("true")
    ):
        return "native_results"
    return None


def collect_artifacts(source_root: str | Path, destination_root: str | Path) -> dict[str, int]:
    source = Path(source_root).resolve()
    destination = Path(destination_root).resolve()
    counts = {"checkpoints": 0, "predictions": 0, "native_results": 0}

    for path in sorted(source.rglob("*")):
        resolved = path.resolve()
        if not path.is_file() or resolved == destination or destination in resolved.parents:
            continue
        category = _category(path)
        if category is None:
            continue
        relative = path.relative_to(source)
        target = destination / category / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if resolved != target.resolve():
            shutil.copy2(path, target)
        counts[category] += 1
    return counts
