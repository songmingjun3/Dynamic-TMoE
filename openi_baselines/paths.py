from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetLocation:
    root_path: Path
    file: Path


DATASET_FILES = {
    "ETTh1": Path("ETT-small/ETTh1.csv"),
    "ETTh2": Path("ETT-small/ETTh2.csv"),
    "ETTm1": Path("ETT-small/ETTm1.csv"),
    "ETTm2": Path("ETT-small/ETTm2.csv"),
    "Electricity": Path("electricity/electricity.csv"),
    "Exchange": Path("exchange_rate/exchange_rate.csv"),
    "ILI": Path("illness/national_illness.csv"),
    "Traffic": Path("traffic/traffic.csv"),
    "Weather": Path("weather/weather.csv"),
}


def _is_repo(path: Path) -> bool:
    return (path / "baselines").is_dir()


def resolve_repo(code_root: str | Path) -> Path:
    root = Path(code_root).expanduser().resolve()
    if _is_repo(root):
        return root

    preferred = root / "Dynamic-TMoE"
    if _is_repo(preferred):
        return preferred

    matches = sorted(path for path in root.iterdir() if path.is_dir() and _is_repo(path))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not locate a repository containing baselines under {root}"
    )


def resolve_dataset_file(dataset_root: str | Path, dataset: str) -> DatasetLocation:
    root = Path(dataset_root).expanduser().resolve()
    try:
        relative_file = DATASET_FILES[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {dataset!r}") from exc

    candidates = [root / relative_file]
    if root.is_dir():
        candidates.extend(
            child / relative_file for child in root.iterdir() if child.is_dir()
        )

    for candidate in candidates:
        if candidate.is_file():
            return DatasetLocation(root_path=candidate.parent, file=candidate)

    if root.is_dir():
        fallback = sorted(root.rglob(relative_file.name))
        if len(fallback) == 1:
            return DatasetLocation(root_path=fallback[0].parent, file=fallback[0])

    raise FileNotFoundError(
        f"Could not find {relative_file.name} for {dataset} under {root}"
    )
