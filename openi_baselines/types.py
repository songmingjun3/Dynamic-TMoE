from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class TaskSpec:
    """Immutable description of one supported model/dataset reproduction task."""

    model: str
    dataset: str
    entrypoint: Path
    data_kind: str
    native_model: str
    horizons: tuple[int, ...]
    parameters: Mapping[str, object] = field(default_factory=dict)
    horizon_overrides: Mapping[int, Mapping[str, object]] = field(default_factory=dict)
    stages: tuple[str, ...] = ("train",)
