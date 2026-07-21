from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openi_baselines.matrix import (
    execute_matrix,
    plan_matrix,
    write_matrix_summaries,
)
from openi_baselines.paths import resolve_dataset_file, resolve_repo
from openi_baselines.platform import PlatformContext, prepare_platform
from openi_baselines.registry import (
    UnsupportedTaskError,
    parse_task_matrix,
)
from openi_baselines.types import AccelerationOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenI baseline/dataset reproduction tasks"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pred-len", default=None)
    parser.add_argument(
        "--batch-size",
        default=None,
        help="Comma-separated PRED_LEN:BATCH_SIZE mapping",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override the DataLoader worker count for every selected baseline",
    )
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--patience", type=int)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        help="Set OMP_NUM_THREADS and MKL_NUM_THREADS for native processes",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context: PlatformContext | None = None
    try:
        tasks = parse_task_matrix(args.model, args.dataset)
        plans = plan_matrix(
            tasks,
            pred_len_value=args.pred_len,
            batch_size_value=args.batch_size,
        )
        if args.num_workers is not None and args.num_workers <= 0:
            raise ValueError("num-workers must be a positive integer")
        for name in ("patience", "prefetch_factor", "cpu_threads"):
            value = getattr(args, name)
            if value is not None and value <= 0:
                raise ValueError(
                    f"{name.replace('_', '-')} must be a positive integer"
                )
        acceleration = AccelerationOptions(
            use_amp=args.use_amp,
            patience=args.patience,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            cudnn_benchmark=args.cudnn_benchmark,
            cpu_threads=args.cpu_threads,
        )
        context = prepare_platform(
            local=args.local,
            code_root=args.code_root,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
        )
        repo_root = resolve_repo(context.code_path)
        datasets = dict.fromkeys(
            plan.task.dataset for plan in plans
        )
        data_by_dataset = {
            dataset: resolve_dataset_file(context.dataset_path, dataset)
            for dataset in datasets
        }
        result = execute_matrix(
            plans,
            repo_root=repo_root,
            data_by_dataset=data_by_dataset,
            output_root=context.output_path,
            force=args.force,
            dry_run=args.dry_run,
            num_workers=args.num_workers,
            acceleration=acceleration,
        )
        if not args.dry_run:
            write_matrix_summaries(context.output_path, result)
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        return result.exit_code
    except (UnsupportedTaskError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if context is not None:
            context.upload_output()


if __name__ == "__main__":
    raise SystemExit(main())
