from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value == "1"


def loader_kwargs(
    args,
    *,
    default_pin_memory: bool = False,
    default_persistent_workers: bool = False,
    default_prefetch_factor: int | None = None,
) -> dict[str, object]:
    benchmark = os.environ.get("OPENI_CUDNN_BENCHMARK")
    if benchmark is not None:
        import torch

        torch.backends.cudnn.benchmark = benchmark == "1"

    workers = int(args.num_workers)
    kwargs: dict[str, object] = {
        "pin_memory": _env_bool(
            "OPENI_PIN_MEMORY", default_pin_memory
        ),
        "persistent_workers": workers > 0
        and _env_bool(
            "OPENI_PERSISTENT_WORKERS",
            default_persistent_workers,
        ),
    }
    if workers > 0:
        raw_prefetch = os.environ.get("OPENI_PREFETCH_FACTOR")
        prefetch_factor = (
            int(raw_prefetch)
            if raw_prefetch is not None
            else default_prefetch_factor
        )
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs
