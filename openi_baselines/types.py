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


@dataclass(frozen=True)
class ProcessSpec:
    stage: str
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AccelerationOptions:
    use_amp: bool = False
    patience: int | None = None
    pin_memory: bool | None = None
    persistent_workers: bool | None = None
    prefetch_factor: int | None = None
    cudnn_benchmark: bool | None = None
    cpu_threads: int | None = None
