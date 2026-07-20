from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .artifacts import collect_artifacts
from .commands import CommandLayout, build_processes
from .metrics import find_metrics, write_metrics
from .paths import DatasetLocation
from .types import ProcessSpec, TaskSpec


ProcessBuilder = Callable[
    [TaskSpec, int, CommandLayout], tuple[ProcessSpec, ...]
]


@dataclass
class RunResult:
    exit_code: int
    statuses: dict[int, str]
    commands: dict[int, list[dict[str, object]]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "exit_code": self.exit_code,
            "statuses": {str(key): value for key, value in self.statuses.items()},
            "commands": {str(key): value for key, value in self.commands.items()},
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["status"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _environment_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "python": platform.python_version(),
        "python_executable": os.path.realpath(os.sys.executable),
        "platform": platform.platform(),
    }
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "torch_cuda": torch.version.cuda,
                "gpus": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except Exception as exc:
        payload["torch_error"] = str(exc)
    return payload


def _command_payload(process: ProcessSpec) -> dict[str, object]:
    return {
        "stage": process.stage,
        "argv": list(process.argv),
        "cwd": str(process.cwd),
        "env": dict(process.env),
    }


def _execute_process(process: ProcessSpec, log_path: Path) -> int:
    process.cwd.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(process.env)

    with log_path.open("a", encoding="utf-8") as log:
        marker = f"\n===== stage={process.stage} started={_now()} =====\n"
        print(marker, end="", flush=True)
        log.write(marker)
        child = subprocess.Popen(
            process.argv,
            cwd=process.cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert child.stdout is not None
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return child.wait()


def _write_summaries(
    dataset_output: Path,
    task: TaskSpec,
    statuses: dict[int, str],
) -> None:
    _write_json(
        dataset_output / "status_summary.json",
        {
            "model": task.model,
            "dataset": task.dataset,
            "updated_at": _now(),
            "statuses": {str(key): value for key, value in statuses.items()},
        },
    )

    rows: list[dict[str, object]] = []
    for pred_len in statuses:
        metrics_path = dataset_output / str(pred_len) / "metrics.json"
        if metrics_path.is_file():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": payload["model"],
                    "dataset": payload["dataset"],
                    "pred_len": payload["pred_len"],
                    **payload["metrics"],
                }
            )
    _write_json(dataset_output / "metrics_summary.json", rows)

    required = ["model", "dataset", "pred_len", "mse", "mae"]
    extras = sorted({key for row in rows for key in row} - set(required))
    fields = required + extras
    csv_path = dataset_output / "metrics_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_task(
    *,
    task: TaskSpec,
    pred_lengths: Iterable[int],
    repo_root: Path,
    data: DatasetLocation,
    output_root: Path,
    force: bool = False,
    dry_run: bool = False,
    process_builder: ProcessBuilder = build_processes,
) -> RunResult:
    dataset_output = Path(output_root) / task.model / task.dataset
    statuses: dict[int, str] = {}
    commands: dict[int, list[dict[str, object]]] = {}

    for pred_len in pred_lengths:
        task_output = dataset_output / str(pred_len)
        status_path = task_output / "status.json"
        if not force and _read_status(status_path) == "succeeded":
            statuses[pred_len] = "skipped"
            continue

        layout = CommandLayout(
            repo_root=Path(repo_root),
            data=data,
            task_output=task_output,
        )
        processes = process_builder(task, pred_len, layout)
        commands[pred_len] = [_command_payload(process) for process in processes]

        if dry_run:
            statuses[pred_len] = "planned"
            continue

        task_output.mkdir(parents=True, exist_ok=True)
        _write_json(task_output / "command.json", commands[pred_len])
        _write_json(task_output / "environment.json", _environment_payload())
        started_at = _now()
        _write_json(
            status_path,
            {
                "status": "running",
                "model": task.model,
                "dataset": task.dataset,
                "pred_len": pred_len,
                "started_at": started_at,
            },
        )

        exit_codes: dict[str, int] = {}
        error: str | None = None
        try:
            for process in processes:
                exit_code = _execute_process(
                    process, task_output / "logs" / "train.log"
                )
                exit_codes[process.stage] = exit_code
                if exit_code != 0:
                    raise RuntimeError(
                        f"Stage {process.stage} exited with code {exit_code}"
                    )

            artifact_counts = collect_artifacts(
                task_output / "native_work", task_output
            )
            metrics, source = find_metrics(task_output)
            write_metrics(
                task_output / "metrics.json",
                model=task.model,
                dataset=task.dataset,
                pred_len=pred_len,
                metrics=metrics,
                source=source,
            )
            status = "succeeded"
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            artifact_counts = collect_artifacts(
                task_output / "native_work", task_output
            )
            (task_output / "logs").mkdir(parents=True, exist_ok=True)
            (task_output / "logs" / "error.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )

        statuses[pred_len] = status
        _write_json(
            status_path,
            {
                "status": status,
                "model": task.model,
                "dataset": task.dataset,
                "pred_len": pred_len,
                "started_at": started_at,
                "finished_at": _now(),
                "exit_codes": exit_codes,
                "artifact_counts": artifact_counts,
                "error": error,
            },
        )

    if not dry_run:
        dataset_output.mkdir(parents=True, exist_ok=True)
        _write_summaries(dataset_output, task, statuses)

    exit_code = 1 if any(status == "failed" for status in statuses.values()) else 0
    return RunResult(exit_code=exit_code, statuses=statuses, commands=commands)
