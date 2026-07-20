# OpenI Multi-Task Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow comma-separated model and dataset arrays in `train_openi.py`, preflight their Cartesian product, execute every task sequentially, and emit root-level multi-task summaries.

**Architecture:** Canonical list and Cartesian-product parsing lives in the registry. A focused `openi_baselines/matrix.py` module plans per-task horizons and global batch mappings, executes existing `run_task()` calls sequentially, isolates task failures, and writes matrix summaries; the CLI prepares the platform once and supplies fully resolved inputs.

**Tech Stack:** Python 3.10+, argparse, dataclasses, csv/json, pytest

---

## File Structure

- Modify `openi_baselines/registry.py`: canonicalize comma-separated model/dataset lists and build the ordered Cartesian product.
- Modify `openi_baselines/__init__.py`: export the new registry interfaces.
- Create `openi_baselines/matrix.py`: plan horizons/batches, execute tasks serially, aggregate results, and write JSON/CSV summaries.
- Modify `train_openi.py`: preflight the full matrix, resolve all data before training, execute once, and upload once.
- Create `tests/openi_baselines/test_matrix.py`: cover planning, batch unions, serial execution, failure continuation, and summaries.
- Modify `tests/openi_baselines/test_registry.py`: cover list parsing and Cartesian-product validation.
- Modify `tests/openi_baselines/test_cli.py`: cover single-task compatibility, multi-task dry-run, and preflight rejection.
- Modify `docs/openi-baseline-reproduction.md`: document array syntax, strict validation, global batches, and summary files.

### Task 1: Parse Canonical Lists and the Cartesian Product

**Files:**
- Modify: `openi_baselines/registry.py`
- Modify: `openi_baselines/__init__.py`
- Test: `tests/openi_baselines/test_registry.py`

- [ ] **Step 1: Write failing registry tests**

Add imports and tests:

```python
from openi_baselines.registry import parse_datasets, parse_models, parse_task_matrix


def test_model_and_dataset_lists_are_canonical_deduplicated_and_ordered():
    assert parse_models("patchtst,DLinear,patchtst") == ("PatchTST", "DLinear")
    assert parse_datasets("weather,ecl,Weather") == ("Weather", "Electricity")


def test_task_matrix_uses_model_major_cartesian_order():
    tasks = parse_task_matrix("DLinear,PatchTST", "ETTh1,Weather")
    assert [(task.model, task.dataset) for task in tasks] == [
        ("DLinear", "ETTh1"),
        ("DLinear", "Weather"),
        ("PatchTST", "ETTh1"),
        ("PatchTST", "Weather"),
    ]


def test_task_matrix_rejects_any_unsupported_pair():
    with pytest.raises(UnsupportedTaskError, match="TimeMixer.*ILI"):
        parse_task_matrix("DLinear,TimeMixer", "ETTh1,ILI")


@pytest.mark.parametrize(("parser", "label"), [(parse_models, "model"), (parse_datasets, "dataset")])
def test_name_lists_reject_empty_items(parser, label):
    with pytest.raises(ValueError, match=f"{label} values must be comma-separated"):
        parser("DLinear,,PatchTST" if label == "model" else "ETTh1,,Weather")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/openi_baselines/test_registry.py -v`

Expected: collection fails because the three parsing functions do not exist.

- [ ] **Step 3: Implement canonical parsing and matrix construction**

Add to `registry.py`:

```python
def _parse_canonical_list(value: str, *, label: str, canonicalize) -> tuple[str, ...]:
    selected: list[str] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            raise ValueError(f"{label} values must be comma-separated names")
        canonical = canonicalize(item)
        if canonical not in selected:
            selected.append(canonical)
    return tuple(selected)


def parse_models(value: str) -> tuple[str, ...]:
    return _parse_canonical_list(value, label="model", canonicalize=_canonical_model)


def parse_datasets(value: str) -> tuple[str, ...]:
    return _parse_canonical_list(value, label="dataset", canonicalize=_canonical_dataset)


def parse_task_matrix(model_value: str, dataset_value: str) -> tuple[TaskSpec, ...]:
    models = parse_models(model_value)
    datasets = parse_datasets(dataset_value)
    return tuple(get_task(model, dataset) for model in models for dataset in datasets)
```

Export `parse_models`, `parse_datasets`, and `parse_task_matrix` from `openi_baselines/__init__.py`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/openi_baselines/test_registry.py -v`

Expected: all registry tests pass.

- [ ] **Step 5: Commit**

```bash
git add openi_baselines/registry.py openi_baselines/__init__.py tests/openi_baselines/test_registry.py
git commit -m "feat: parse OpenI task matrices"
```

### Task 2: Plan Horizons and Global Batch Mappings

**Files:**
- Create: `openi_baselines/matrix.py`
- Create: `tests/openi_baselines/test_matrix.py`

- [ ] **Step 1: Write failing planning tests**

Create `test_matrix.py` with:

```python
import pytest

from openi_baselines.matrix import plan_matrix
from openi_baselines.registry import parse_task_matrix


def test_plan_matrix_uses_explicit_horizons_and_global_batches():
    plans = plan_matrix(
        parse_task_matrix("DLinear,PatchTST", "ETTh1"),
        pred_len_value="96,192",
        batch_size_value="192:64,96:128",
    )
    assert [(plan.task.model, plan.pred_lengths, plan.batch_sizes) for plan in plans] == [
        ("DLinear", (96, 192), {96: 128, 192: 64}),
        ("PatchTST", (96, 192), {96: 128, 192: 64}),
    ]


def test_plan_matrix_all_uses_horizon_union_for_global_batch_validation():
    plans = plan_matrix(
        parse_task_matrix("DLinear", "ETTh1,ILI"),
        pred_len_value="all",
        batch_size_value=(
            "24:8,36:8,48:8,60:8,96:32,192:16,336:8,720:4"
        ),
    )
    assert plans[0].pred_lengths == (96, 192, 336, 720)
    assert plans[0].batch_sizes == {96: 32, 192: 16, 336: 8, 720: 4}
    assert plans[1].pred_lengths == (24, 36, 48, 60)
    assert plans[1].batch_sizes == {24: 8, 36: 8, 48: 8, 60: 8}


def test_plan_matrix_rejects_explicit_horizon_invalid_for_any_task():
    with pytest.raises(ValueError, match="DLinear/ILI"):
        plan_matrix(
            parse_task_matrix("DLinear", "ETTh1,ILI"),
            pred_len_value="96",
            batch_size_value=None,
        )
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/openi_baselines/test_matrix.py -v`

Expected: collection fails because `openi_baselines.matrix` does not exist.

- [ ] **Step 3: Implement immutable matrix plans**

Create `matrix.py` with these planning interfaces:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .registry import parse_batch_sizes, parse_pred_lengths
from .types import TaskSpec


@dataclass(frozen=True)
class MatrixTaskPlan:
    task: TaskSpec
    pred_lengths: tuple[int, ...]
    batch_sizes: Mapping[int, int]


def plan_matrix(
    tasks: tuple[TaskSpec, ...],
    *,
    pred_len_value: str | None,
    batch_size_value: str | None,
) -> tuple[MatrixTaskPlan, ...]:
    selections = tuple(
        (task, parse_pred_lengths(task, pred_len_value)) for task in tasks
    )
    horizon_union: list[int] = []
    for _, pred_lengths in selections:
        for pred_len in pred_lengths:
            if pred_len not in horizon_union:
                horizon_union.append(pred_len)
    global_batches = parse_batch_sizes(tuple(horizon_union), batch_size_value)
    return tuple(
        MatrixTaskPlan(
            task=task,
            pred_lengths=pred_lengths,
            batch_sizes={
                pred_len: global_batches[pred_len]
                for pred_len in pred_lengths
                if pred_len in global_batches
            },
        )
        for task, pred_lengths in selections
    )
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/openi_baselines/test_matrix.py -v`

Expected: three planning tests pass.

- [ ] **Step 5: Commit**

```bash
git add openi_baselines/matrix.py tests/openi_baselines/test_matrix.py
git commit -m "feat: plan OpenI multi-task horizons"
```

### Task 3: Execute Serially and Write Root Summaries

**Files:**
- Modify: `openi_baselines/matrix.py`
- Modify: `tests/openi_baselines/test_matrix.py`

- [ ] **Step 1: Write failing execution and summary tests**

Append tests using real `RunResult` values and a recording runner:

```python
import json

from openi_baselines.matrix import execute_matrix, write_matrix_summaries
from openi_baselines.paths import resolve_dataset_file
from openi_baselines.runner import RunResult


def test_execute_matrix_is_serial_and_continues_after_failed_task(tmp_path):
    plans = plan_matrix(
        parse_task_matrix("DLinear,PatchTST", "ETTh1"),
        pred_len_value="96",
        batch_size_value="96:32",
    )
    calls = []

    def recording_runner(**kwargs):
        calls.append((kwargs["task"].model, kwargs["batch_sizes"][96]))
        failed = kwargs["task"].model == "DLinear"
        return RunResult(
            exit_code=1 if failed else 0,
            statuses={96: "failed" if failed else "succeeded"},
        )

    result = execute_matrix(
        plans,
        repo_root=tmp_path,
        data_by_dataset={"ETTh1": resolve_dataset_file(REPO_ROOT / "dataset", "ETTh1")},
        output_root=tmp_path / "output",
        task_runner=recording_runner,
    )
    assert calls == [("DLinear", 32), ("PatchTST", 32)]
    assert result.exit_code == 1
    assert [item.result.exit_code for item in result.tasks] == [1, 0]


def test_write_matrix_summaries_records_every_horizon(tmp_path):
    plans = plan_matrix(
        parse_task_matrix("DLinear", "ETTh1"),
        pred_len_value="96,192",
        batch_size_value=None,
    )

    def successful_runner(**kwargs):
        return RunResult(exit_code=0, statuses={96: "succeeded", 192: "skipped"})

    result = execute_matrix(
        plans,
        repo_root=tmp_path,
        data_by_dataset={"ETTh1": resolve_dataset_file(REPO_ROOT / "dataset", "ETTh1")},
        output_root=tmp_path,
        task_runner=successful_runner,
    )
    write_matrix_summaries(tmp_path, result)
    payload = json.loads((tmp_path / "multi_task_summary.json").read_text(encoding="utf-8"))
    assert payload["exit_code"] == 0
    assert payload["rows"] == [
        {"model": "DLinear", "dataset": "ETTh1", "pred_len": 96, "status": "succeeded", "task_exit_code": 0},
        {"model": "DLinear", "dataset": "ETTh1", "pred_len": 192, "status": "skipped", "task_exit_code": 0},
    ]
    assert (tmp_path / "multi_task_summary.csv").is_file()
```

Define `REPO_ROOT = Path(__file__).resolve().parents[2]` and import `Path` at the top of the test module.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/openi_baselines/test_matrix.py -v`

Expected: imports fail because execution and summary interfaces do not exist.

- [ ] **Step 3: Implement matrix results and serial execution**

Add to `matrix.py`:

```python
import csv
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .paths import DatasetLocation
from .runner import RunResult, run_task

TaskRunner = Callable[..., RunResult]


@dataclass
class MatrixTaskResult:
    plan: MatrixTaskPlan
    result: RunResult
    error: str | None = None


@dataclass
class MatrixResult:
    exit_code: int
    tasks: list[MatrixTaskResult] = field(default_factory=list)

    def rows(self) -> list[dict[str, object]]:
        rows = []
        for item in self.tasks:
            for pred_len, status in item.result.statuses.items():
                rows.append({
                    "model": item.plan.task.model,
                    "dataset": item.plan.task.dataset,
                    "pred_len": pred_len,
                    "status": status,
                    "task_exit_code": item.result.exit_code,
                })
        return rows

    def as_dict(self) -> dict[str, object]:
        payload = {
            "exit_code": self.exit_code,
            "rows": self.rows(),
            "tasks": [
                {
                    "model": item.plan.task.model,
                    "dataset": item.plan.task.dataset,
                    "pred_lengths": list(item.plan.pred_lengths),
                    "result": item.result.as_dict(),
                    "error": item.error,
                }
                for item in self.tasks
            ],
        }
        if len(self.tasks) == 1:
            payload.update(self.tasks[0].result.as_dict())
            payload["exit_code"] = self.exit_code
        return payload


def execute_matrix(
    plans: tuple[MatrixTaskPlan, ...],
    *, repo_root: Path,
    data_by_dataset: Mapping[str, DatasetLocation],
    output_root: Path,
    force: bool = False,
    dry_run: bool = False,
    task_runner: TaskRunner = run_task,
) -> MatrixResult:
    task_results: list[MatrixTaskResult] = []
    for plan in plans:
        try:
            result = task_runner(
                task=plan.task,
                pred_lengths=plan.pred_lengths,
                repo_root=repo_root,
                data=data_by_dataset[plan.task.dataset],
                output_root=output_root,
                force=force,
                dry_run=dry_run,
                batch_sizes=dict(plan.batch_sizes),
            )
            item = MatrixTaskResult(plan=plan, result=result)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = RunResult(
                exit_code=1,
                statuses={pred_len: "failed" for pred_len in plan.pred_lengths},
            )
            item = MatrixTaskResult(plan=plan, result=result, error=error)
            if not dry_run:
                error_path = output_root / plan.task.model / plan.task.dataset / "matrix_error.log"
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
        task_results.append(item)
    return MatrixResult(
        exit_code=1 if any(item.result.exit_code != 0 for item in task_results) else 0,
        tasks=task_results,
    )
```

- [ ] **Step 4: Implement JSON/CSV summary writing**

Add:

```python
def write_matrix_summaries(output_root: Path, result: MatrixResult) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {"exit_code": result.exit_code, "rows": result.rows(), "tasks": result.as_dict()["tasks"]}
    (output_root / "multi_task_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = ["model", "dataset", "pred_len", "status", "task_exit_code"]
    with (output_root / "multi_task_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result.rows())
```

- [ ] **Step 5: Run tests and verify GREEN**

Run: `python -m pytest tests/openi_baselines/test_matrix.py -v`

Expected: all planning, execution, continuation, and summary tests pass.

- [ ] **Step 6: Commit**

```bash
git add openi_baselines/matrix.py tests/openi_baselines/test_matrix.py
git commit -m "feat: execute and summarize OpenI task matrices"
```

### Task 4: Integrate Full Preflight into the CLI

**Files:**
- Modify: `train_openi.py`
- Modify: `tests/openi_baselines/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Add helpers to read a native option and tests:

```python
def native_option(command, name):
    argv = command["argv"]
    return argv[argv.index(name) + 1]


def test_cli_dry_run_executes_model_dataset_cartesian_product(tmp_path, capsys):
    exit_code = main([
        "--local", "--code-root", str(REPO_ROOT),
        "--dataset-root", str(REPO_ROOT / "dataset"),
        "--output-root", str(tmp_path),
        "--model", "DLinear,PatchTST",
        "--dataset", "ETTh1,Weather",
        "--pred-len", "96",
        "--batch-size", "96:32",
        "--dry-run",
    ])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert [(item["model"], item["dataset"]) for item in output["tasks"]] == [
        ("DLinear", "ETTh1"),
        ("DLinear", "Weather"),
        ("PatchTST", "ETTh1"),
        ("PatchTST", "Weather"),
    ]
    assert all(native_option(item["result"]["commands"]["96"][0], "--batch_size") == "32" for item in output["tasks"])


def test_cli_preflight_rejects_unsupported_pair_without_running_any_task(tmp_path, capsys):
    exit_code = main([
        "--local", "--code-root", str(REPO_ROOT),
        "--dataset-root", str(REPO_ROOT / "dataset"),
        "--output-root", str(tmp_path),
        "--model", "DLinear,TimeMixer",
        "--dataset", "ETTh1,ILI",
        "--dry-run",
    ])
    assert exit_code == 2
    assert "TimeMixer does not support ILI" in capsys.readouterr().err
    assert not (tmp_path / "DLinear").exists()
```

Keep the existing single-task tests unchanged to prove backward compatibility.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/openi_baselines/test_cli.py -v`

Expected: multi-task dry-run fails because comma-separated model/dataset values are unsupported.

- [ ] **Step 3: Replace single-task setup with matrix preflight**

In `train_openi.py`, replace `get_task`, `parse_pred_lengths`, `parse_batch_sizes`, and direct `run_task` usage with:

```python
from openi_baselines.matrix import execute_matrix, plan_matrix, write_matrix_summaries
from openi_baselines.registry import UnsupportedTaskError, parse_task_matrix
```

Use this order inside `main()`:

```python
tasks = parse_task_matrix(args.model, args.dataset)
plans = plan_matrix(
    tasks,
    pred_len_value=args.pred_len,
    batch_size_value=args.batch_size,
)
context = prepare_platform(...)
repo_root = resolve_repo(context.code_path)
data_by_dataset = {
    dataset: resolve_dataset_file(context.dataset_path, dataset)
    for dataset in dict.fromkeys(plan.task.dataset for plan in plans)
}
result = execute_matrix(
    plans,
    repo_root=repo_root,
    data_by_dataset=data_by_dataset,
    output_root=context.output_path,
    force=args.force,
    dry_run=args.dry_run,
)
if not args.dry_run:
    write_matrix_summaries(context.output_path, result)
print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
return result.exit_code
```

Update the parser description to `Run OpenI baseline/dataset reproduction tasks` and retain the existing `finally` upload so it executes exactly once.

- [ ] **Step 4: Run CLI and full tests**

Run:

```bash
python -m pytest tests/openi_baselines/test_cli.py -v
python -m pytest tests/openi_baselines -q
```

Expected: CLI tests and the full suite pass; single-task output is now a one-item matrix result.

- [ ] **Step 5: Commit**

```bash
git add train_openi.py tests/openi_baselines/test_cli.py
git commit -m "feat: orchestrate OpenI task matrices"
```

### Task 5: Document and Verify Multi-Task Submission

**Files:**
- Modify: `docs/openi-baseline-reproduction.md`

- [ ] **Step 1: Update the guide with exact submission examples**

Add this usage content:

```markdown
多个模型和数据集使用逗号分隔，并按笛卡尔积顺序执行：

`--model DLinear,PatchTST --dataset ETTh1,Weather --pred-len 96,192 --batch-size 96:128,192:64`

任何组合或显式预测长度不受支持时，矩阵在训练前整体拒绝。一个任务训练失败后，其余任务继续执行，最终通过 `multi_task_summary.json` 和 `multi_task_summary.csv` 汇总。
```

Document the `all` plus ILI union example from the design and state that tasks execute sequentially on one GPU.

- [ ] **Step 2: Run final verification**

Run:

```powershell
python -m pytest tests/openi_baselines -v
python -X pycache_prefix="$env:TEMP\dynamic-tmoe-openi-pycache" -m compileall -q train_openi.py openi_baselines
python tools/generate_openi_scripts.py --check
python train_openi.py --local --code-root . --dataset-root dataset --output-root "$env:TEMP\dynamic-tmoe-openi-matrix-dry-run" --model DLinear,PatchTST --dataset ETTh1,Weather --pred-len 96,192 --batch-size 96:128,192:64 --dry-run
git diff --check
```

Expected: all tests pass; compilation and generator checks exit 0; dry-run returns four tasks in model-major order and each horizon command contains its global batch size.

- [ ] **Step 3: Commit**

```bash
git add docs/openi-baseline-reproduction.md
git commit -m "docs: explain OpenI multi-task matrices"
```
