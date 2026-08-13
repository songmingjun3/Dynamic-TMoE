from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
DOCS = ROOT / "docs"
INVENTORY_PATH = OUT / "baseline_tmoe_results_inventory.json"
REPORT_PATH = DOCS / "baseline_tmoe_results_summary.md"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def number(value: object) -> float | None:
    return float(value) if finite(value) else None


def metric_matches(text: str) -> list[tuple[float, float, float | None]]:
    pattern = re.compile(
        r"mse\s*:\s*([0-9.eE+-]+|nan).*?mae\s*:\s*([0-9.eE+-]+|nan)"
        r"(?:.*?rse\s*:\s*([0-9.eE+-]+|nan))?",
        re.IGNORECASE | re.DOTALL,
    )
    matches = []
    for mse, mae, rse in pattern.findall(text):
        if finite(mse) and finite(mae):
            matches.append((float(mse), float(mae), number(rse)))
    return matches


def last_metrics(path: Path) -> tuple[float, float, float | None] | None:
    if not path.exists():
        return None
    matches = metric_matches(path.read_text(encoding="utf-8", errors="replace"))
    return matches[-1] if matches else None


def first_setting(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("mse:"):
            return line
    return ""


def infer_seq_len(path: Path) -> int | None:
    for match in re.finditer(r"(?:^|_)sl(\d+)(?:_|$)", path.read_text(encoding="utf-8", errors="replace")):
        return int(match.group(1))
    return None


def metric_evidence_for_task(dataset_dir: Path, pred_len: str) -> tuple[Path | None, tuple[float, float, float | None] | None, int | None]:
    pred_dir = dataset_dir / str(pred_len)
    candidates = sorted(pred_dir.rglob("result*.txt"))
    for candidate in candidates:
        metrics = last_metrics(candidate)
        if metrics:
            setting = first_setting(candidate)
            seq_len = infer_seq_len(candidate)
            return candidate, metrics, seq_len
        seq_len = infer_seq_len(candidate)
    return None, None, seq_len if candidates else None


def task_key(model: str, dataset: str, pred_len: str | int) -> tuple[str, str, str]:
    return model, dataset, str(pred_len)


def load_task_statuses() -> dict[tuple[str, str, str], dict]:
    statuses: dict[tuple[str, str, str], dict] = {}
    for path in ROOT.glob("baselines/*/test_results/**/multi_task_summary.csv"):
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            model = row.get("model", "")
            dataset = row.get("dataset", "")
            pred_len = row.get("pred_len", "")
            if model and dataset and pred_len:
                statuses[task_key(model, dataset, pred_len)] = {
                    "status": row.get("status", ""),
                    "exit_code": row.get("task_exit_code", ""),
                    "source": rel(path),
                }
    return statuses


def add_record(records: list[dict], **kwargs) -> None:
    record = {
        "model": None,
        "dataset": None,
        "seq_len": None,
        "pred_len": None,
        "mse": None,
        "mae": None,
        "rse": None,
        "status": None,
        "environment": None,
        "protocol": None,
        "source": None,
    }
    record.update(kwargs)
    records.append(record)


def collect_structured_baselines(records: list[dict], no_numeric: list[dict], failures: list[dict]) -> None:
    statuses = load_task_statuses()
    for path in sorted(ROOT.glob("baselines/*/test_results/**/metrics_summary.csv")):
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        if not rows:
            continue
        model_from_path = path.parts[path.parts.index("baselines") + 1]
        for row in rows:
            model = row.get("model") or model_from_path
            dataset = row.get("dataset", "")
            pred_len = row.get("pred_len", "")
            if not dataset or not pred_len:
                continue
            status_info = statuses.get(task_key(model, dataset, pred_len), {})
            evidence_path, evidence_metrics, seq_len = metric_evidence_for_task(path.parent, pred_len)
            mse = number(row.get("mse"))
            mae = number(row.get("mae"))
            rse = number(row.get("rse"))
            if mse is not None and mae is not None:
                add_record(
                    records,
                    model=model,
                    dataset=dataset,
                    seq_len=seq_len,
                    pred_len=int(pred_len),
                    mse=mse,
                    mae=mae,
                    rse=rse,
                    status="complete_numeric",
                    environment="baseline_openi_v1",
                    protocol="baseline_test_results",
                    source=rel(path),
                    task_status=status_info.get("status"),
                )
            else:
                no_numeric.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "pred_len": int(pred_len),
                        "status": status_info.get("status", "unknown"),
                        "exit_code": status_info.get("exit_code"),
                        "source": rel(path),
                        "metric_evidence": rel(evidence_path) if evidence_path else None,
                    }
                )

    fits_path = ROOT / "baselines/FITS/test_results/multi_task_summary.csv"
    if fits_path.exists():
        for row in csv.DictReader(fits_path.open(encoding="utf-8-sig")):
            dataset = row.get("dataset", "")
            pred_len = row.get("pred_len", "")
            log_path = ROOT / f"baselines/FITS/test_results/FITS/{dataset}/{pred_len}/logs/train.log"
            error_path = ROOT / f"baselines/FITS/test_results/FITS/{dataset}/{pred_len}/logs/error.log"
            observed = last_metrics(log_path)
            add_failure(
                failures,
                model="FITS",
                dataset=dataset,
                pred_len=int(pred_len),
                failure_class="final_artifact_failure_after_metric",
                status="failed",
                exit_code=int(row.get("task_exit_code") or 1),
                observed_mse=observed[0] if observed else None,
                observed_mae=observed[1] if observed else None,
                observed_rse=observed[2] if observed else None,
                error_summary="metric was printed, then test failed at np.save with ValueError for an inhomogeneous array",
                source=rel(fits_path),
                log=rel(log_path),
                error_log=rel(error_path),
                environment="baseline_openi_v1",
            )

    tfps_path = ROOT / "baselines/TFPS/test_results/multi_task_summary.csv"
    if tfps_path.exists():
        for row in csv.DictReader(tfps_path.open(encoding="utf-8-sig")):
            if row.get("status") == "failed":
                dataset = row.get("dataset", "")
                pred_len = row.get("pred_len", "")
                log_path = ROOT / f"baselines/TFPS/test_results/TFPS/{dataset}/{pred_len}/logs/train.log"
                error_path = ROOT / f"baselines/TFPS/test_results/TFPS/{dataset}/{pred_len}/logs/error.log"
                add_failure(
                    failures,
                    model="TFPS",
                    dataset=dataset,
                    pred_len=int(pred_len),
                    failure_class="task_failed",
                    status="failed",
                    exit_code=int(row.get("task_exit_code") or 1),
                    error_summary="task status is failed; no finite metric row was produced",
                    source=rel(tfps_path),
                    log=rel(log_path),
                    error_log=rel(error_path),
                    environment="baseline_openi_v1",
                )


PATCH_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "Exchange", "ILI", "Electricity", "traffic"]


def patch_dataset(setting: str) -> str:
    for dataset in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "national_illness", "Electricity", "traffic"):
        if setting.startswith(dataset + "_"):
            return {"exchange_rate": "Exchange", "national_illness": "ILI"}.get(dataset, dataset)
    if setting.startswith("96_"):
        return "weather"
    if setting.startswith("336_"):
        return "Electricity"
    if setting.startswith("36_"):
        return "ILI"
    return "unknown"


def collect_patchtst(records: list[dict], failures: list[dict]) -> None:
    result_path = ROOT / "baselines/PatchTST/result.txt"
    text = result_path.read_text(encoding="utf-8", errors="replace") if result_path.exists() else ""
    pattern = re.compile(
        r"(?m)^([^\s]+)\s*\n\s*mse:([0-9.eE+-]+),\s*mae:([0-9.eE+-]+),\s*rse:([0-9.eE+-]+)"
    )
    state_rows = list(csv.DictReader((OUT / "patchtst_paper_matrix_state.csv").open(encoding="utf-8-sig")))
    last_state = {row.get("name"): row for row in state_rows}
    for setting, mse, mae, rse in pattern.findall(text):
        dataset = patch_dataset(setting)
        seq_match = re.search(r"sl(\d+)", setting)
        pred_match = re.search(r"pl(\d+)", setting)
        if dataset == "unknown" or not pred_match:
            continue
        pred_len = int(pred_match.group(1))
        state_dataset = {"traffic": "Traffic", "weather": "Weather"}.get(dataset, dataset)
        state = last_state.get(f"PatchTST_{state_dataset}_{pred_len}", {})
        status = "checkpoint_recovered_test" if dataset == "traffic" and pred_len == 720 else "complete_numeric"
        add_record(
            records,
            model="PatchTST",
            dataset=dataset,
            seq_len=int(seq_match.group(1)) if seq_match else None,
            pred_len=pred_len,
            mse=float(mse),
            mae=float(mae),
            rse=float(rse),
            status=status,
            environment="patchtst_local_cuda",
            protocol="patchtst_paper_matrix",
            source=rel(result_path),
            queue_status=state.get("status"),
        )

    failures.append(
        {
            "model": "PatchTST",
            "dataset": "traffic",
            "pred_len": 720,
            "failure_class": "full_training_failed_checkpoint_recovered_test",
            "status": "failed",
            "exit_code": -1073741819,
            "error_summary": "full training stopped at epoch 89/iter 300; stderr contains no OOM message; later test-only used the saved best checkpoint",
            "source": rel(OUT / "patchtst_paper_matrix_state.csv"),
            "log": rel(ROOT / "baselines/PatchTST/logs/LongForecasting/PatchTST_traffic_96_720.log"),
            "stderr": rel(OUT / "patchtst_paper_matrix_Traffic_720.stderr.log"),
            "environment": "patchtst_local_cuda",
        }
    )


def parse_tmoe_dir(name: str) -> tuple[str, int, int] | None:
    if "smoke" in name.lower():
        return None
    prefix = "long_term_forecast_"
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    datasets = ["national_illness", "ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "traffic", "weather", "Exchange"]
    for dataset in datasets:
        marker = dataset + "_"
        if rest.startswith(marker):
            tail = rest[len(marker):]
            match = re.match(r"(\d+)_(\d+)_Dynamic_TMoE", tail)
            if match:
                return dataset, int(match.group(1)), int(match.group(2))
    return None


def collect_tmoe(records: list[dict], smoke_records: list[dict], failures: list[dict]) -> None:
    metric_lookup: dict[tuple[str, int, int], float] = {}
    for metric_path in sorted((ROOT / "results").glob("*/metrics.npy")):
        parsed = parse_tmoe_dir(metric_path.parent.name)
        if parsed is None:
            continue
        dataset, seq_len, pred_len = parsed
        try:
            import numpy as np

            values = np.load(metric_path, allow_pickle=True).tolist()
        except Exception:
            continue
        if len(values) >= 3 and finite(values[2]):
            metric_lookup[(dataset.lower(), seq_len, pred_len)] = float(values[2])

    summary_path = OUT / "reproduction_summary_latest.csv"
    for row in csv.DictReader(summary_path.open(encoding="utf-8-sig")):
        dataset = row.get("dataset", "")
        pred_len = int(row.get("pred_len", "0"))
        log = ROOT / row.get("log", "")
        match = re.search(r"_(\d+)_(\d+)\.log$", log.name)
        seq_len = int(match.group(1)) if match else None
        rse = metric_lookup.get((dataset.lower(), seq_len, pred_len)) if seq_len is not None else None
        add_record(
            records,
            model="Dynamic_TMoE",
            dataset=dataset,
            seq_len=seq_len,
            pred_len=pred_len,
            mse=float(row["mse"]),
            mae=float(row["mae"]),
            rse=rse,
            status="complete_reproduction_queue_modified_budget",
            environment="dynamic_tmoe_local_cuda",
            protocol="reproduction_queue_modified_budget",
            source=rel(summary_path),
            log=rel(log),
        )

    for metric_path in sorted((ROOT / "results").glob("*/metrics.npy")):
        if "smoke" not in metric_path.parent.name.lower():
            continue
        parsed = parse_tmoe_dir(metric_path.parent.name.replace("smoke_cuda_", "").replace("smoke_", ""))
        try:
            import numpy as np

            values = np.load(metric_path, allow_pickle=True).tolist()
        except Exception:
            continue
        if parsed is None or len(values) < 3:
            continue
        dataset, seq_len, pred_len = parsed
        smoke_records.append(
            {
                "model": "Dynamic_TMoE",
                "dataset": dataset,
                "seq_len": seq_len,
                "pred_len": pred_len,
                "mse": float(values[1]),
                "mae": float(values[0]),
                "rse": float(values[2]),
                "status": "smoke_not_for_paper",
                "environment": "dynamic_tmoe_local_cuda",
                "source": rel(metric_path),
            }
        )

    traffic_path = OUT / "traffic_accum_summary.csv"
    for row in csv.DictReader(traffic_path.open(encoding="utf-8-sig")):
        status = row.get("status", "")
        base = {
            "model": "Dynamic_TMoE",
            "dataset": row.get("dataset", "traffic"),
            "seq_len": int(row.get("seq_len", "96")),
            "pred_len": int(row.get("pred_len", "0")),
            "environment": "dynamic_tmoe_local_cuda",
            "protocol": "traffic_gradient_accumulation",
            "source": rel(traffic_path),
            "log": row.get("log"),
        }
        if status == "complete":
            add_record(
                records,
                **base,
                mse=float(row["mse"]),
                mae=float(row["mae"]),
                status="complete_traffic_gradient_accumulation",
                effective_batch_size=int(row["effective_batch_size"]),
                gradient_accumulation_steps=int(row["gradient_accumulation_steps"]),
            )
        elif status == "failed":
            add_failure(
                failures,
                **base,
                failure_class="superseded_failed_attempt",
                status="failed",
                exit_code=None,
                error_summary="early traffic accumulated-gradient attempt failed; a later complete row exists; no retry is requested",
                command=row.get("command"),
                elapsed_seconds=float(row.get("elapsed_seconds") or 0),
            )


def add_failure(failures: list[dict], **kwargs) -> None:
    failures.append(kwargs)


def environments() -> list[dict]:
    return [
        {
            "id": "baseline_openi_v1",
            "scope": "DLinear/FEDformer/FITS/TFPS/TimeMixer/TimesNet structured baseline artifacts",
            "python": "3.11.11",
            "python_executable": "/usr/local/bin/python3.11",
            "platform": "Linux-4.15.0-156-generic-x86_64-with-glibc2.35 or Linux-5.15.0-112-generic-x86_64-with-glibc2.35",
            "torch": "2.6.0+cu124",
            "cuda": "12.4",
            "gpu": "Tesla V100S-PCIE-32GB",
            "evidence": "baselines/*/test_results/**/environment.json",
            "note": "requirements.txt files are repository declarations; use environment.json as the recorded runtime evidence.",
        },
        {
            "id": "patchtst_local_cuda",
            "scope": "PatchTST paper matrix",
            "python": "3.8.20",
            "python_executable": "D:/software/miniconda/envs/Time-Series-Library/python.exe",
            "platform": "Windows local audit",
            "torch": "2.4.1",
            "cuda": "11.8",
            "gpu": "NVIDIA GeForce RTX 3060",
            "evidence": "tools/run_patchtst_paper_matrix.ps1; baselines/PatchTST/requirements.txt; live environment observation on 2026-08-04",
        },
        {
            "id": "dynamic_tmoe_local_cuda",
            "scope": "Dynamic_TMoE reproduction queue and traffic accumulation runs",
            "python": "3.10.20",
            "python_executable": "F:/Dynamic-TMoE/.conda/dynamic_tmoe_cuda/python.exe",
            "platform": "Windows local audit",
            "torch": "2.4.1+cu124",
            "cuda": "12.4",
            "gpu": "NVIDIA GeForce RTX 3060",
            "evidence": "tools/run_reproduction_queue.ps1; output/traffic_accum_summary.csv; live environment observation on 2026-08-04",
        },
    ]


def protocols() -> dict:
    return {
        "baseline_openi_v1": {
            "scope": "Structured baseline artifacts under baselines/*/test_results",
            "datasets": "Model-specific coverage shown in the coverage table",
            "note": "Use each result directory's command.json/environment.json and metrics_summary.csv together; do not infer missing metrics from task success alone.",
        },
        "patchtst_paper_matrix": {
            "model": "PatchTST",
            "random_seed": 2021,
            "seq_len_by_dataset": {"ETTh1": 336, "ETTh2": 96, "ETTm1": 336, "ETTm2": 336, "Weather": 96, "Exchange": 96, "ILI": 36, "Electricity": 336, "Traffic": 96},
            "pred_lens": "ETT/Weather/Exchange/Electricity/Traffic: 96,192,336,720; ILI: 24,36,48,60",
            "model_args": "e_layers=3, n_heads=16, d_model=128, d_ff=256, dropout=0.2, fc_dropout=0.2, patch_len=16, stride=8, train_epochs=100, lradj=TST, pct_start=0.2, num_workers=0, itr=1",
            "batch_and_lr": "ETT/Weather=128, Exchange=32, ILI=16, Electricity=32, Traffic=24; lr=0.0001 except ILI=0.0025",
            "source": "tools/run_patchtst_paper_matrix.ps1",
        },
        "reproduction_queue_modified_budget": {
            "model": "Dynamic_TMoE",
            "device": "CUDA_VISIBLE_DEVICES=0",
            "forced_args": "num_workers=0, finetune_epochs=3",
            "batch_caps": "ETTm2=64, Weather=128, Electricity=2, Traffic=1",
            "heavy_budget_caps": "ETTm2 train_epochs=5/patience=2; Electricity=1/1; Traffic=1/1",
            "source": "tools/run_reproduction_queue.py and tools/run_reproduction_queue.ps1",
        },
        "traffic_gradient_accumulation": {
            "model": "Dynamic_TMoE",
            "dataset": "Traffic",
            "seq_len": 96,
            "channels": 862,
            "effective_batch_size": 32,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 32,
            "train_epochs": 60,
            "patience": 10,
            "finetune_epochs": 3,
            "source": "output/traffic_accum_summary.csv",
        },
    }


def coverage(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["model"]].append(record)
    output = []
    for model, rows in sorted(grouped.items()):
        output.append(
            {
                "model": model,
                "numeric_records": len(rows),
                "datasets": sorted({str(row["dataset"]) for row in rows}),
                "environments": sorted({row["environment"] for row in rows}),
                "protocols": sorted({row.get("protocol") for row in rows if row.get("protocol")}),
            }
        )
    return output


def build_inventory() -> dict:
    records: list[dict] = []
    failures: list[dict] = []
    no_numeric: list[dict] = []
    smoke_records: list[dict] = []
    collect_structured_baselines(records, no_numeric, failures)
    collect_patchtst(records, failures)
    collect_tmoe(records, smoke_records, failures)
    records.sort(key=lambda row: (row["model"], str(row["dataset"]), row["pred_len"] or 0, row["status"]))
    failures.sort(key=lambda row: (row.get("model", ""), str(row.get("dataset", "")), row.get("pred_len", 0), row.get("failure_class", "")))
    return {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "origin_date": "2026-08-04",
            "verification_status": "ANALYZED",
            "version_label": "baseline_tmoe_results_v1",
        },
        "audit": {
            "no_retry_policy": True,
            "failed_results_are_record_only": True,
            "active_training_process_observed": False,
            "gpu_observation_at_audit": "NVIDIA GeForce RTX 3060; 14% utilization; 2472 MiB / 12288 MiB observed after result scan; not attributed to a training task",
            "scope_note": "Only existing files were read. Numeric rows from different protocols/environments must not be pooled without the protocol and environment fields.",
        },
        "environments": environments(),
        "protocols": protocols(),
        "coverage": coverage(records),
        "records": records,
        "no_numeric_metric": no_numeric,
        "failures": failures,
        "smoke_records": smoke_records,
        "sources": [
            "output/reproduction_summary_latest.csv",
            "output/traffic_accum_summary.csv",
            "output/patchtst_paper_matrix_state.csv",
            "output/baseline_queue_state.csv",
            "baselines/PatchTST/result.txt",
            "results/*/metrics.npy",
            "baselines/*/test_results/**/metrics_summary.csv",
            "baselines/*/test_results/**/multi_task_summary.csv",
            "baselines/*/test_results/**/environment.json",
            "tools/run_patchtst_paper_matrix.ps1",
            "tools/run_reproduction_queue.py",
            "tools/run_reproduction_queue.ps1",
            "requirements.txt",
        ],
    }


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def render_report(data: dict) -> str:
    lines = [
        "# Dynamic-TMoE Baseline / TMoE Results Summary",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: `experiment-agent`",
        "- Origin Mode: `validate`",
        "- Origin Date: `2026-08-04`",
        "- Verification Status: `ANALYZED`",
        "- Version Label: `baseline_tmoe_results_v1`",
        "",
        "## 使用规则",
        "",
        "- 本清单只整理已有文件，不启动、不重试失败任务；失败结果只存档。",
        "- `complete_numeric` 可在注明环境、协议和来源后引用。",
        "- `checkpoint_recovered_test` 不是完整独立训练，只能作为检查点恢复测试引用。",
        "- `no_numeric_metric`、`failed`、`smoke_not_for_paper` 不进入论文主结果表。",
        "- 不同环境、数据划分、训练预算和协议的数值不得直接混合比较；优先使用同一环境和同一协议的行。",
        "",
        "## 环境注册表",
        "",
        "| ID | 适用结果 | Python / Torch / CUDA | GPU / 平台 | 证据 |",
        "|---|---|---|---|---|",
    ]
    for env in data["environments"]:
        lines.append(
            f"| `{env['id']}` | {env['scope']} | {env['python']} / {env['torch']} / {env['cuda']} | {env['gpu']} / {env['platform']} | `{env['evidence']}` |"
        )
    lines += [
        "",
        "baseline_openi_v1 的 `requirements.txt` 仅是各仓库声明依赖，实际运行环境以结果目录中的 `environment.json` 为准。两个 local CUDA 环境的版本来自 2026-08-04 现场复核；结果文件的时间和运行配置仍以各自来源文件为准。",
        "",
        "## 协议登记",
        "",
        "- **PatchTST paper matrix**：`random_seed=2021`；数据集输入长度为 ETTh1=336、ETTh2=96、ETTm1/ETTm2=336、Weather/Exchange=96、ILI=36、Electricity=336、Traffic=96；全局 `e_layers=3, n_heads=16, d_model=128, d_ff=256, patch_len=16, stride=8, train_epochs=100, num_workers=0`；批量与学习率按构建器脚本中的数据集设置。",
        "- **Dynamic_TMoE reproduction queue**：`num_workers=0`、`finetune_epochs=3`；ETTm2/Weather/Electricity/Traffic 有批量上限，ETTm2/Electricity/Traffic 还有缩短的 epoch/patience 预算，因此必须标记为 `reproduction_queue_modified_budget`。",
        "- **Dynamic_TMoE Traffic gradient accumulation**：Traffic 862 通道，`seq_len=96`，micro batch=1，梯度累积 32，有效 batch=32，60 epochs，patience=10，finetune=3；这是独立于 reproduction queue 的协议。",
        "",
        "## 覆盖概览",
        "",
        "| 模型 | 数值记录数 | 数据集 | 环境 | 协议 |",
        "|---|---:|---|---|---|",
    ]
    for item in data["coverage"]:
        lines.append(
            f"| {item['model']} | {item['numeric_records']} | {', '.join(item['datasets'])} | {', '.join(item['environments'])} | {', '.join(item['protocols'])} |"
        )
    lines += [
        "",
        "## 数值结果",
        "",
        "字段为 `dataset / seq_len / pred_len / MSE / MAE / RSE / status / source`。完整逐条 JSON 记录见 `output/baseline_tmoe_results_inventory.json`。",
        "",
    ]
    by_model: dict[str, list[dict]] = defaultdict(list)
    for record in data["records"]:
        by_model[record["model"]].append(record)
    for model in sorted(by_model):
        lines += [f"### {model}", "", "| Dataset | Seq | Pred | MSE | MAE | RSE | Status | Protocol |", "|---|---:|---:|---:|---:|---:|---|---|"]
        for row in by_model[model]:
            lines.append(
                f"| {row['dataset']} | {fmt(row['seq_len'])} | {fmt(row['pred_len'])} | {fmt(row['mse'])} | {fmt(row['mae'])} | {fmt(row['rse'])} | `{row['status']}` | `{row.get('protocol', '')}` |"
            )
        lines.append("")

    lines += [
        "## 失败与不可用记录",
        "",
        "以下项目只记录，不重试，也不作为 complete 结果使用。每条明细在 JSON 的 `failures` 中保留。",
        "",
        "| 模型 / 数据集 | 数量或状态 | 分类 | 退出码 | 说明 | 来源 |",
        "|---|---:|---|---:|---|---|",
    ]
    grouped_failures: dict[tuple, list[dict]] = defaultdict(list)
    for failure in data["failures"]:
        grouped_failures[(failure.get("model"), failure.get("dataset"), failure.get("failure_class"), failure.get("exit_code"), failure.get("error_summary"), failure.get("source"))].append(failure)
    for key, rows in sorted(grouped_failures.items(), key=lambda item: str(item[0])):
        model, dataset, failure_class, exit_code, summary, source = key
        lines.append(f"| {model} / {dataset} | {len(rows)} | `{failure_class}` | {fmt(exit_code)} | {summary} | `{source}` |")
    lines += [
        "",
        "### 无数值证据",
        "",
        "| 模型 | 条目数 | 说明 | 来源 |",
        "|---|---:|---|---|",
    ]
    no_numeric_by_model = Counter(row["model"] for row in data["no_numeric_metric"])
    no_numeric_sources = defaultdict(set)
    for row in data["no_numeric_metric"]:
        no_numeric_sources[row["model"]].add(row["source"])
    for model, count in sorted(no_numeric_by_model.items()):
        lines.append(f"| {model} | {count} | 状态文件可能为 succeeded，但没有有限 MSE/MAE；不可引用为数值结果 | {', '.join('`'+x+'`' for x in sorted(no_numeric_sources[model]))} |")
    lines += [
        "| RAFT | - | 当前仓库内未发现可追溯结构化数值证据 | `baselines/RAFT` |",
        "| st-mtm | - | 当前仓库内未发现可追溯结构化数值证据 | `baselines/st-mtm` |",
        "",
        "FITS 的 24 个任务均为 `exit_code=1`；训练日志曾输出 MSE/MAE，但测试阶段随后在 `np.save` 因不规则数组触发 `ValueError`，因此只能列为失败记录。PatchTST Traffic-720 的完整训练以 `-1073741819` 失败，后来使用已有 best checkpoint 完成 test-only，主表明确标记为 `checkpoint_recovered_test`。",
        "",
        "## Smoke 结果（不用于论文主表）",
        "",
        "| 模型 | 数据集 | Pred | MSE | MAE | RSE | 来源 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in data["smoke_records"]:
        lines.append(f"| {row['model']} | {row['dataset']} | {row['pred_len']} | {fmt(row['mse'])} | {fmt(row['mae'])} | {fmt(row['rse'])} | `{row['source']}` |")
    lines += [
        "",
        "## 来源清单",
        "",
    ]
    for source in data["sources"]:
        lines.append(f"- `{source}`")
    lines += [
        "",
        "## 推荐引用写法",
        "",
        "引用某一行时同时写明：模型、数据集、输入长度、预测长度、环境 ID、协议、MSE/MAE/RSE 和来源文件。不要把 `reproduction_queue_modified_budget`、`traffic_gradient_accumulation`、PatchTST paper matrix 和 baseline_openi_v1 的行合并计算平均值或宣称同协议 SOTA。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    data = build_inventory()
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    INVENTORY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_report(data), encoding="utf-8")
    print(json.dumps({"inventory": rel(INVENTORY_PATH), "report": rel(REPORT_PATH), "records": len(data["records"]), "failures": len(data["failures"]), "no_numeric": len(data["no_numeric_metric"]), "smoke": len(data["smoke_records"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
