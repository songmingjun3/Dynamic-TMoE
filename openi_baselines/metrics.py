from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping


class MetricsNotFoundError(ValueError):
    """Raised when evaluation output has no complete MSE/MAE pair."""


NUMBER = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf))"
METRIC_PATTERN = re.compile(
    rf"\b(mse|mae|rmse|mape|mspe|rse|corr)\s*:\s*({NUMBER})",
    re.IGNORECASE,
)


def parse_text_metrics(text: str) -> dict[str, float]:
    complete: list[dict[str, float]] = []
    for line in text.splitlines():
        metrics = {
            match.group(1).casefold(): float(match.group(2))
            for match in METRIC_PATTERN.finditer(line)
        }
        if "mse" in metrics and "mae" in metrics:
            complete.append(metrics)
    if not complete:
        raise MetricsNotFoundError("Evaluation output must contain both MSE and MAE")
    return complete[-1]


def find_metrics(search_root: str | Path) -> tuple[dict[str, float], Path]:
    root = Path(search_root)
    candidates: list[Path] = []
    for pattern in ("result*.txt", "metrics*.txt", "*.log"):
        candidates.extend(sorted(root.rglob(pattern)))

    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
            return parse_text_metrics(text), candidate
        except MetricsNotFoundError:
            continue
    raise MetricsNotFoundError(f"No complete MSE and MAE metrics found under {root}")


def write_metrics(
    target: str | Path,
    *,
    model: str,
    dataset: str,
    pred_len: int,
    metrics: Mapping[str, float],
    source: str | Path,
) -> None:
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "dataset": dataset,
        "pred_len": pred_len,
        "metrics": dict(metrics),
        "source": str(source),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
