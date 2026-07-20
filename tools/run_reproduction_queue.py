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
        idx = tokens.index(name)
        tokens[idx + 1] = str(value)
    else:
        tokens.extend([name, str(value)])


def cap_batch_size(tokens, dataset):
    max_batches = {
        "ETTm2": 64,
        "weather": 128,
        "electricity": 2,
        "traffic": 1,
    }
    if dataset not in max_batches:
        return

    current = int(get_arg(tokens, "--batch_size", max_batches[dataset]))
    set_arg(tokens, "--batch_size", min(current, max_batches[dataset]))


def cap_heavy_run_budget(tokens, dataset):
    budgets = {
        "ETTm2": {"--train_epochs": 5, "--patience": 2},
        "electricity": {"--train_epochs": 1, "--patience": 1},
        "traffic": {"--train_epochs": 1, "--patience": 1},
    }
    for name, value in budgets.get(dataset, {}).items():
        set_arg(tokens, name, value)


def load_completed(summary_path: Path):
    completed = set()
    if not summary_path.exists():
        return completed

    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") == "complete":
                completed.add((row.get("dataset"), row.get("pred_len")))
    return completed


def append_summary(summary_path: Path, row):
    exists = summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = ["dataset", "pred_len", "mse", "mae", "status", "log"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_metrics(log_path: Path):
    if not log_path.exists():
        return None, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)", text)
    if not matches:
        return None, None
    return matches[-1]


def build_queue(root: Path):
    script_order = [
        "ETTh1.sh",
        "ETTh2.sh",
        "ETTm1.sh",
        "ETTm2.sh",
        "weather.sh",
        "exchange.sh",
        "ILI.sh",
        "ECL.sh",
        "traffic.sh",
    ]
    queue = []
    for script_name in script_order:
        script_path = root / "scripts" / script_name
        if not script_path.exists():
            continue
        for tokens in parse_shell_script(script_path):
            dataset = get_arg(tokens, "--data_path", script_path.stem)
            model_id = get_arg(tokens, "--model_id", dataset)
            pred_len = get_arg(tokens, "--pred_len", "")
            seq_len = get_arg(tokens, "--seq_len", "")
            queue.append(
                {
                    "script": script_name,
                    "tokens": tokens,
                    "dataset": model_id.split("_96_")[0] if "_96_" in model_id else Path(dataset).stem,
                    "seq_len": seq_len,
                    "pred_len": pred_len,
                    "model_id": model_id,
                }
            )
    return queue


def run_one(python_exe, root: Path, item, output_dir: Path, retry_batch=None):
    tokens = list(item["tokens"])
    set_arg(tokens, "--num_workers", 0)
    set_arg(tokens, "--finetune_epochs", 3)
    if item["dataset"] == "traffic":
        set_arg(tokens, "--train_sample_limit", 2500)
    cap_batch_size(tokens, item["dataset"])
    cap_heavy_run_budget(tokens, item["dataset"])
    if retry_batch is not None:
        set_arg(tokens, "--batch_size", retry_batch)

    suffix = f"{item['dataset']}_{item['seq_len']}_{item['pred_len']}"
    if retry_batch is not None:
        suffix += f"_bs{retry_batch}"
    safe_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", suffix)
    log_path = output_dir / f"repro_{safe_suffix}.log"
    err_path = output_dir / f"repro_{safe_suffix}.err.log"

    command = [str(python_exe)] + tokens
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    start = time.time()
    with log_path.open("w", encoding="utf-8") as stdout, err_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
        return_code = process.wait()

    elapsed = time.time() - start
    mse, mae = parse_metrics(log_path)
    return {
        "return_code": return_code,
        "elapsed": elapsed,
        "log": log_path,
        "err": err_path,
        "mse": mse,
        "mae": mae,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--summary", default="output/reproduction_summary.csv")
    parser.add_argument("--state", default="output/reproduction_queue_state.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    python_exe = Path(args.python).resolve()
    output_dir = root / "output"
    output_dir.mkdir(exist_ok=True)
    summary_path = root / args.summary
    state_path = root / args.state

    completed = load_completed(summary_path)
    queue = build_queue(root)

    with state_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "total", "dataset", "pred_len", "status", "detail"])
        writer.writeheader()

    total = len(queue)
    for index, item in enumerate(queue, start=1):
        key = (item["dataset"], item["pred_len"])
        if key in completed:
            status = "skipped"
            detail = "already complete"
            with state_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=["index", "total", "dataset", "pred_len", "status", "detail"]).writerow(
                    {"index": index, "total": total, "dataset": item["dataset"], "pred_len": item["pred_len"], "status": status, "detail": detail}
                )
            continue

        with state_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=["index", "total", "dataset", "pred_len", "status", "detail"]).writerow(
                {"index": index, "total": total, "dataset": item["dataset"], "pred_len": item["pred_len"], "status": "running", "detail": item["script"]}
            )

        result = run_one(python_exe, root, item, output_dir)
        log_rel = result["log"].relative_to(root).as_posix()
        status = "complete" if result["return_code"] == 0 and result["mse"] is not None else "failed"

        if status == "failed":
            err_text = result["err"].read_text(encoding="utf-8", errors="replace") if result["err"].exists() else ""
            if "out of memory" in err_text.lower():
                original_batch = int(get_arg(item["tokens"], "--batch_size", 32))
                retry_batch = max(1, original_batch // 2)
                retry = run_one(python_exe, root, item, output_dir, retry_batch=retry_batch)
                log_rel = retry["log"].relative_to(root).as_posix()
                result = retry
                status = "complete" if retry["return_code"] == 0 and retry["mse"] is not None else "failed_oom_retry"

        append_summary(
            summary_path,
            {
                "dataset": item["dataset"],
                "pred_len": item["pred_len"],
                "mse": result["mse"] or "",
                "mae": result["mae"] or "",
                "status": status,
                "log": log_rel,
            },
        )

        completed.add(key)
        with state_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=["index", "total", "dataset", "pred_len", "status", "detail"]).writerow(
                {
                    "index": index,
                    "total": total,
                    "dataset": item["dataset"],
                    "pred_len": item["pred_len"],
                    "status": status,
                    "detail": log_rel,
                }
            )


if __name__ == "__main__":
    sys.exit(main())
