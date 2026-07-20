import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_shell_script(path: Path):
    variables = {}
    commands = []
    current = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                commands.append(" ".join(current))
                current = []
            continue
        if line.startswith("#") or line.startswith("export "):
            continue

        assign = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.+)", line)
        if assign and not current:
            variables[assign.group(1)] = assign.group(2).strip().strip('"').strip("'")
            continue

        if line.endswith("\\"):
            current.append(line[:-1].strip())
            continue

        if current:
            current.append(line)
            commands.append(" ".join(current))
            current = []
        elif line.startswith("python "):
            commands.append(line)

    if current:
        commands.append(" ".join(current))

    parsed = []
    for command in commands:
        for name, value in variables.items():
            command = command.replace(f"${name}", value)
        tokens = shlex.split(command, posix=False)
        if len(tokens) >= 3 and tokens[0] == "python" and tokens[1] == "-u" and tokens[2] == "run.py":
            parsed.append(tokens[1:])
    return parsed


def get_arg(tokens, name, default=None):
    if name not in tokens:
        return default
    idx = tokens.index(name)
    if idx + 1 >= len(tokens):
        return default
    return tokens[idx + 1].strip("'\"")


def set_arg(tokens, name, value):
    if name in tokens:
        tokens[tokens.index(name) + 1] = str(value)
    else:
        tokens.extend([name, str(value)])


def parse_metrics(log_path: Path):
    mse = mae = None
    if not log_path.exists():
        return mse, mae
    metric_re = re.compile(r"mse:([0-9.eE+-]+),\s*mae:([0-9.eE+-]+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = metric_re.search(line)
        if match:
            mse = float(match.group(1))
            mae = float(match.group(2))
    return mse, mae


def completed_keys(summary_path: Path):
    if not summary_path.exists():
        return set()
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        return {
            (row["dataset"], row["seq_len"], row["pred_len"])
            for row in csv.DictReader(handle)
            if row.get("status") == "complete"
        }


def append_summary(summary_path: Path, row):
    exists = summary_path.exists()
    fieldnames = [
        "dataset",
        "seq_len",
        "pred_len",
        "effective_batch_size",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "train_epochs",
        "patience",
        "mse",
        "mae",
        "status",
        "elapsed_seconds",
        "log",
        "command",
    ]
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=32)
    parser.add_argument("--only-pred-len", default=None)
    parser.add_argument("--summary", default="output/traffic_accum_summary.csv")
    parser.add_argument("--state", default="output/traffic_accum_state.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    python_exe = Path(args.python).resolve()
    output_dir = root / "output"
    output_dir.mkdir(exist_ok=True)
    summary_path = root / args.summary
    state_path = root / args.state

    queue = []
    for tokens in parse_shell_script(root / "scripts" / "traffic.sh"):
        if args.only_pred_len is not None and get_arg(tokens, "--pred_len", "") != str(args.only_pred_len):
            continue
        set_arg(tokens, "--batch_size", args.micro_batch_size)
        set_arg(tokens, "--gradient_accumulation_steps", args.accumulation_steps)
        set_arg(tokens, "--num_workers", 0)
        set_arg(tokens, "--finetune_epochs", 3)
        # Intentionally do not set train_sample_limit: strict-ish traffic uses the full training split.
        queue.append(tokens)

    done = completed_keys(summary_path)
    with state_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "total", "dataset", "seq_len", "pred_len", "status", "detail"])
        writer.writeheader()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    for index, tokens in enumerate(queue, start=1):
        dataset = "traffic"
        seq_len = get_arg(tokens, "--seq_len", "")
        pred_len = get_arg(tokens, "--pred_len", "")
        key = (dataset, seq_len, pred_len)
        if key in done:
            with state_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=["index", "total", "dataset", "seq_len", "pred_len", "status", "detail"]).writerow(
                    {"index": index, "total": len(queue), "dataset": dataset, "seq_len": seq_len, "pred_len": pred_len, "status": "skipped", "detail": "already complete"}
                )
            continue

        suffix = f"traffic_accum{args.accumulation_steps}_{seq_len}_{pred_len}"
        log_path = output_dir / f"repro_{suffix}.log"
        err_path = output_dir / f"repro_{suffix}.err.log"
        command = [str(python_exe)] + tokens

        with state_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=["index", "total", "dataset", "seq_len", "pred_len", "status", "detail"]).writerow(
                {"index": index, "total": len(queue), "dataset": dataset, "seq_len": seq_len, "pred_len": pred_len, "status": "running", "detail": log_path.relative_to(root).as_posix()}
            )

        start = time.time()
        with log_path.open("w", encoding="utf-8") as stdout, err_path.open("w", encoding="utf-8") as stderr:
            return_code = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr).wait()
        elapsed = time.time() - start
        mse, mae = parse_metrics(log_path)
        status = "complete" if return_code == 0 and mse is not None else "failed"

        append_summary(
            summary_path,
            {
                "dataset": dataset,
                "seq_len": seq_len,
                "pred_len": pred_len,
                "effective_batch_size": args.micro_batch_size * args.accumulation_steps,
                "micro_batch_size": args.micro_batch_size,
                "gradient_accumulation_steps": args.accumulation_steps,
                "train_epochs": get_arg(tokens, "--train_epochs", ""),
                "patience": get_arg(tokens, "--patience", ""),
                "mse": mse if mse is not None else "",
                "mae": mae if mae is not None else "",
                "status": status,
                "elapsed_seconds": f"{elapsed:.3f}",
                "log": log_path.relative_to(root).as_posix(),
                "command": " ".join(command),
            },
        )

        with state_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=["index", "total", "dataset", "seq_len", "pred_len", "status", "detail"]).writerow(
                {"index": index, "total": len(queue), "dataset": dataset, "seq_len": seq_len, "pred_len": pred_len, "status": status, "detail": log_path.relative_to(root).as_posix()}
            )


if __name__ == "__main__":
    sys.exit(main())
