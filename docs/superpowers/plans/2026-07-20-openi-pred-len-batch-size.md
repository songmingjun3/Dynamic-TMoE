# OpenI Per-Horizon Batch Size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict `pred_len:batch_size` CLI mapping so each selected prediction horizon can override the native baseline batch size independently.

**Architecture:** Parse and validate the mapping next to prediction-length parsing, pass the resulting dictionary through the CLI and runner, and apply the per-horizon value centrally to every generated `ProcessSpec`. A central argv override keeps all nine baseline builders compatible and applies the same value to both ST-MTM stages.

**Tech Stack:** Python 3.10+, argparse, dataclasses, pytest

---

## File Structure

- Modify `openi_baselines/registry.py`: parse and strictly validate `pred_len:batch_size` mappings.
- Modify `openi_baselines/__init__.py`: export the new parser.
- Modify `openi_baselines/commands.py`: optionally replace or append `--batch_size` on every native process.
- Modify `openi_baselines/runner.py`: select the override for each horizon and pass it to the process builder.
- Modify `train_openi.py`: expose `--batch-size` and validate it before resolving paths or starting processes.
- Modify `tests/openi_baselines/test_registry.py`: cover valid and invalid mapping syntax and set equality.
- Modify `tests/openi_baselines/test_commands.py`: cover single-stage, implicit-default and ST-MTM overrides.
- Modify `tests/openi_baselines/test_runner.py`: verify each horizon receives its own override.
- Modify `tests/openi_baselines/test_cli.py`: verify CLI forwarding and pre-execution validation.
- Modify `docs/openi-baseline-reproduction.md`: document the OpenI startup file and mapping examples.

### Task 1: Parse and Validate Batch Size Mappings

**Files:**
- Modify: `openi_baselines/registry.py`
- Modify: `openi_baselines/__init__.py`
- Test: `tests/openi_baselines/test_registry.py`

- [ ] **Step 1: Write failing parser tests**

Add imports and tests equivalent to:

```python
from openi_baselines.registry import parse_batch_sizes


def test_batch_size_mapping_matches_selected_horizons():
    assert parse_batch_sizes((96, 192), "192:64,96:128") == {192: 64, 96: 128}
    assert parse_batch_sizes((96, 192), None) == {}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("96:128", "Missing batch size for prediction lengths: 192"),
        ("96:128,192:64,336:32", "Batch sizes provided for unselected prediction lengths: 336"),
        ("96:128,96:64,192:32", "Duplicate batch size mapping for prediction length 96"),
        ("96=128,192:64", "Expected PRED_LEN:BATCH_SIZE"),
        ("96:0,192:64", "Batch size must be a positive integer"),
        ("96:large,192:64", "Invalid batch size"),
    ],
)
def test_batch_size_mapping_rejects_invalid_values(value, message):
    with pytest.raises(ValueError, match=message):
        parse_batch_sizes((96, 192), value)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/openi_baselines/test_registry.py -v`

Expected: collection fails because `parse_batch_sizes` does not exist.

- [ ] **Step 3: Implement the strict parser**

Add this public interface in `registry.py`:

```python
def parse_batch_sizes(
    pred_lengths: tuple[int, ...], value: str | None
) -> dict[int, int]:
    if value is None:
        return {}

    parsed: dict[int, int] = {}
    for raw_item in value.split(","):
        item = raw_item.strip()
        if item.count(":") != 1:
            raise ValueError(
                f"Invalid batch size mapping {item!r}. "
                "Expected PRED_LEN:BATCH_SIZE"
            )
        raw_pred_len, raw_batch_size = (part.strip() for part in item.split(":"))
        try:
            pred_len = int(raw_pred_len)
        except ValueError as exc:
            raise ValueError(f"Invalid prediction length {raw_pred_len!r} in batch size mapping") from exc
        if pred_len in parsed:
            raise ValueError(
                f"Duplicate batch size mapping for prediction length {pred_len}"
            )
        try:
            batch_size = int(raw_batch_size)
        except ValueError as exc:
            raise ValueError(f"Invalid batch size {raw_batch_size!r}") from exc
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")
        parsed[pred_len] = batch_size

    selected = set(pred_lengths)
    provided = set(parsed)
    missing = selected - provided
    extra = provided - selected
    if missing:
        values = ", ".join(str(item) for item in sorted(missing))
        raise ValueError(f"Missing batch size for prediction lengths: {values}")
    if extra:
        values = ", ".join(str(item) for item in sorted(extra))
        raise ValueError(
            f"Batch sizes provided for unselected prediction lengths: {values}"
        )
    return parsed
```

Export `parse_batch_sizes` from `openi_baselines/__init__.py`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/openi_baselines/test_registry.py -v`

Expected: all registry tests pass.

- [ ] **Step 5: Commit**

```bash
git add openi_baselines/registry.py openi_baselines/__init__.py tests/openi_baselines/test_registry.py
git commit -m "feat: parse per-horizon batch sizes"
```

### Task 2: Apply Overrides to Native Baseline Commands

**Files:**
- Modify: `openi_baselines/commands.py`
- Test: `tests/openi_baselines/test_commands.py`

- [ ] **Step 1: Write failing command tests**

Add tests equivalent to:

```python
@pytest.mark.parametrize("model", ["DLinear", "FEDformer", "TimesNet"])
def test_batch_size_override_replaces_or_adds_native_option(tmp_path, model):
    layout = make_layout(tmp_path, "ETTh1")
    process = build_processes(
        get_task(model, "ETTh1"), 96, layout, batch_size=48
    )[0]
    assert option(process.argv, "--batch_size") == "48"
    assert process.argv.count("--batch_size") == 1


def test_stmtm_batch_size_override_applies_to_both_stages(tmp_path):
    layout = make_layout(tmp_path, "ETTh1")
    processes = build_processes(
        get_task("ST-MTM", "ETTh1"), 96, layout, batch_size=24
    )
    assert [option(process.argv, "--batch_size") for process in processes] == ["24", "24"]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/openi_baselines/test_commands.py -v`

Expected: FAIL because `build_processes` does not accept `batch_size`.

- [ ] **Step 3: Implement a central immutable argv override**

Import `replace` from `dataclasses`, then add:

```python
def _with_batch_size(process: ProcessSpec, batch_size: int) -> ProcessSpec:
    argv = list(process.argv)
    option = "--batch_size"
    if option in argv:
        argv[argv.index(option) + 1] = str(batch_size)
    else:
        argv.extend((option, str(batch_size)))
    return replace(process, argv=tuple(argv))
```

Change the public builder signature and final normalization:

```python
def build_processes(
    task: TaskSpec,
    pred_len: int,
    layout: CommandLayout,
    batch_size: int | None = None,
) -> tuple[ProcessSpec, ...]:
    # existing horizon validation and builder dispatch
    processes = (result,) if isinstance(result, ProcessSpec) else result
    if batch_size is not None:
        processes = tuple(_with_batch_size(process, batch_size) for process in processes)
    return processes
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/openi_baselines/test_commands.py -v`

Expected: all command tests pass.

- [ ] **Step 5: Commit**

```bash
git add openi_baselines/commands.py tests/openi_baselines/test_commands.py
git commit -m "feat: override native baseline batch sizes"
```

### Task 3: Route Per-Horizon Values Through Runner and CLI

**Files:**
- Modify: `openi_baselines/runner.py`
- Modify: `train_openi.py`
- Modify: `tests/openi_baselines/test_runner.py`
- Modify: `tests/openi_baselines/test_cli.py`

- [ ] **Step 1: Write failing runner and CLI tests**

Update the test builder to accept `batch_size=None`, record `(pred_len, batch_size)`, and add:

```python
def test_runner_routes_each_horizon_batch_size(tmp_path):
    calls = []
    run_fake(
        tmp_path,
        (96, 192),
        batch_sizes={96: 128, 192: 64},
        process_builder=fake_builder(calls=calls),
        dry_run=True,
    )
    assert calls == [(96, 128), (192, 64)]
```

Add a CLI dry-run assertion:

```python
def test_cli_accepts_per_horizon_batch_sizes(tmp_path, capsys):
    exit_code = main([
        "--local", "--code-root", str(REPO_ROOT),
        "--dataset-root", str(REPO_ROOT / "dataset"),
        "--output-root", str(tmp_path),
        "--model", "DLinear", "--dataset", "ETTh1",
        "--pred-len", "96,192",
        "--batch-size", "192:64,96:128",
        "--dry-run",
    ])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "128" in output["commands"]["96"][0]["argv"]
    assert "64" in output["commands"]["192"][0]["argv"]
```

Also test that a missing horizon returns exit code 2 and creates no task status file.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/openi_baselines/test_runner.py tests/openi_baselines/test_cli.py -v`

Expected: FAIL because runner and CLI do not accept batch size mappings.

- [ ] **Step 3: Route the mapping**

Change `ProcessBuilder` to accept a fourth `int | None` argument. Add `batch_sizes: dict[int, int] | None = None` to `run_task`, normalize it with `batch_sizes = batch_sizes or {}`, and call:

```python
processes = process_builder(task, pred_len, layout, batch_sizes.get(pred_len))
```

In `train_openi.py`, add:

```python
parser.add_argument(
    "--batch-size",
    default=None,
    help="Comma-separated PRED_LEN:BATCH_SIZE mapping",
)
```

After `pred_lengths = parse_pred_lengths(...)`, call:

```python
batch_sizes = parse_batch_sizes(pred_lengths, args.batch_size)
```

Pass `batch_sizes=batch_sizes` to `run_task` and include `parse_batch_sizes` in the import list. Keep `ValueError` in the existing exit-code-2 error path.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/openi_baselines/test_runner.py tests/openi_baselines/test_cli.py -v`

Expected: all runner and CLI tests pass.

- [ ] **Step 5: Commit**

```bash
git add openi_baselines/runner.py train_openi.py tests/openi_baselines/test_runner.py tests/openi_baselines/test_cli.py
git commit -m "feat: route per-horizon batch size overrides"
```

### Task 4: Document and Verify the OpenI Interface

**Files:**
- Modify: `docs/openi-baseline-reproduction.md`

- [ ] **Step 1: Update the usage guide**

Add a section containing the concrete OpenI fields and examples:

```markdown
启动文件：`train_openi.py`

单个预测长度参数：

`--model DLinear --dataset ETTh1 --pred-len 96 --batch-size 96:128`

多个预测长度参数：

`--model DLinear --dataset ETTh1 --pred-len 96,192 --batch-size 96:128,192:64`

传入 `--batch-size` 时，每个选中的预测长度都必须有且仅有一个映射。
不传时保留 baseline 默认 batch size。ST-MTM 的映射值同时应用于预训练和微调。
```

- [ ] **Step 2: Run focused and full verification**

Run:

```bash
python -m pytest tests/openi_baselines -v
python -X pycache_prefix="$env:TEMP\dynamic-tmoe-openi-pycache" -m compileall -q train_openi.py openi_baselines
python train_openi.py --local --code-root . --dataset-root dataset --output-root output/openi-dry-run --model DLinear --dataset ETTh1 --pred-len 96,192 --batch-size 96:128,192:64 --dry-run
python tools/generate_openi_scripts.py --check
git diff --check
```

Expected: all tests pass; compilation, dry-run, generator check and whitespace check exit 0; dry-run commands contain `--batch_size 128` for horizon 96 and `--batch_size 64` for horizon 192.

- [ ] **Step 3: Commit**

```bash
git add docs/openi-baseline-reproduction.md
git commit -m "docs: explain per-horizon batch size parameters"
```
