# OpenI Baseline Reproduction Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a manifest-driven OpenI runner plus generated model/dataset/prediction-length launchers for every repository-supported baseline reproduction task.

**Architecture:** A small `openi_baselines` package separates immutable experiment registration, path/platform integration, native command construction, process execution, artifact collection, and metric normalization. `train_openi.py` is the only public Python entry point; deterministic shell launchers delegate to it and contain no duplicated hyperparameters.

**Tech Stack:** Python 3.10+, standard library, PyTorch environment introspection, `c2net` on OpenI, pytest, POSIX shell.

---

## File map

- Create `train_openi.py`: public CLI and final exit-code handling.
- Create `openi_baselines/__init__.py`: public package exports.
- Create `openi_baselines/types.py`: immutable task/process dataclasses.
- Create `openi_baselines/registry.py`: supported matrices, aliases, horizons, and native experiment parameters.
- Create `openi_baselines/paths.py`: code, dataset, and output resolution.
- Create `openi_baselines/platform.py`: OpenI/local platform contexts and upload finalization.
- Create `openi_baselines/commands.py`: baseline-specific native argv generation.
- Create `openi_baselines/metrics.py`: native result parsing and normalized summaries.
- Create `openi_baselines/artifacts.py`: checkpoint, prediction, and native-result collection.
- Create `openi_baselines/runner.py`: resume-aware multi-horizon orchestration.
- Create `tools/generate_openi_scripts.py`: deterministic launcher generation.
- Create `tests/openi_baselines/`: focused unit and subprocess integration tests.
- Generate `scripts/openi/<model>/<dataset>/{all,pred_*}.sh`: OpenI boot scripts.
- Modify `README.md`: submission and artifact instructions.

### Task 1: Registry and request resolution

**Files:**
- Create: `openi_baselines/__init__.py`
- Create: `openi_baselines/types.py`
- Create: `openi_baselines/registry.py`
- Test: `tests/openi_baselines/test_registry.py`

- [ ] **Step 1: Write failing registry tests**

```python
from openi_baselines.registry import get_task, iter_tasks, parse_pred_lengths


def test_registry_contains_expected_supported_matrix():
    tasks = {(task.model, task.dataset) for task in iter_tasks()}
    assert len(tasks) == 79
    assert ("TimeMixer", "Exchange") not in tasks
    assert ("TimeMixer", "ILI") not in tasks
    assert ("DLinear", "Exchange") in tasks
    assert ("ST-MTM", "Weather") in tasks


def test_names_are_case_insensitive_and_alias_aware():
    assert get_task("dlinear", "electricity").model == "DLinear"
    assert get_task("st-mtm", "ecl").dataset == "Electricity"


def test_prediction_length_selection_supports_all_single_and_subset():
    task = get_task("PatchTST", "ETTh1")
    assert parse_pred_lengths(task, None) == (96, 192, 336, 720)
    assert parse_pred_lengths(task, "336") == (336,)
    assert parse_pred_lengths(task, "96,720") == (96, 720)
```

- [ ] **Step 2: Run the tests and verify the missing-package failure**

Run: `python -m pytest tests/openi_baselines/test_registry.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'openi_baselines'`.

- [ ] **Step 3: Implement immutable task specifications and complete matrix registration**

```python
@dataclass(frozen=True)
class TaskSpec:
    model: str
    dataset: str
    entrypoint: str
    data_kind: str
    native_model: str
    horizons: tuple[int, ...]
    parameters: Mapping[str, object]
    horizon_overrides: Mapping[int, Mapping[str, object]] = field(default_factory=dict)
    stages: tuple[str, ...] = ("train",)
```

Register all nine datasets for DLinear, FEDformer, FITS, PatchTST, RAFT, ST-MTM, TFPS, and TimesNet. Register seven repository-supported datasets for TimeMixer. Dataset aliases include `ECL -> Electricity`, `ExchangeRate -> Exchange`, and `Illness -> ILI`.

- [ ] **Step 4: Run registry tests**

Run: `python -m pytest tests/openi_baselines/test_registry.py -v`

Expected: all registry tests pass.

- [ ] **Step 5: Commit registry changes**

```bash
git add openi_baselines/__init__.py openi_baselines/types.py openi_baselines/registry.py tests/openi_baselines/test_registry.py
git commit -m "feat: register OpenI baseline task matrix"
```

### Task 2: Platform and mounted path resolution

**Files:**
- Create: `openi_baselines/paths.py`
- Create: `openi_baselines/platform.py`
- Test: `tests/openi_baselines/test_paths.py`
- Test: `tests/openi_baselines/test_platform.py`

- [ ] **Step 1: Write failing path and local-platform tests**

```python
def test_resolve_repo_accepts_nested_openi_checkout(tmp_path):
    repo = tmp_path / "code" / "Dynamic-TMoE"
    (repo / "baselines" / "Dlinear").mkdir(parents=True)
    assert resolve_repo(tmp_path / "code") == repo


def test_resolve_dataset_finds_selected_ts_mount(tmp_path):
    csv = tmp_path / "dataset" / "TS" / "ETT-small" / "ETTh1.csv"
    csv.parent.mkdir(parents=True)
    csv.touch()
    assert resolve_dataset_file(tmp_path / "dataset", "ETTh1").file == csv


def test_local_context_uses_explicit_roots_without_c2net(tmp_path):
    context = prepare_platform(
        local=True,
        code_root=tmp_path / "code",
        dataset_root=tmp_path / "dataset",
        output_root=tmp_path / "output",
    )
    assert context.output_path == tmp_path / "output"
```

- [ ] **Step 2: Verify expected import failures**

Run: `python -m pytest tests/openi_baselines/test_paths.py tests/openi_baselines/test_platform.py -v`

Expected: tests fail because `paths` and `platform` modules do not exist.

- [ ] **Step 3: Implement deterministic path discovery and lazy c2net integration**

`prepare_platform()` imports `c2net.context` only in OpenI mode. Local mode requires explicit roots and exposes a no-op uploader. `resolve_dataset_file()` maps registry dataset names to expected mounted relative paths and uses a bounded recursive fallback when the mount has an extra dataset-name directory.

- [ ] **Step 4: Run path/platform tests**

Run: `python -m pytest tests/openi_baselines/test_paths.py tests/openi_baselines/test_platform.py -v`

Expected: all tests pass without `c2net` installed locally.

- [ ] **Step 5: Commit platform changes**

```bash
git add openi_baselines/paths.py openi_baselines/platform.py tests/openi_baselines/test_paths.py tests/openi_baselines/test_platform.py
git commit -m "feat: resolve OpenI platform paths"
```

### Task 3: Native command construction

**Files:**
- Create: `openi_baselines/commands.py`
- Test: `tests/openi_baselines/test_commands.py`

- [ ] **Step 1: Write failing representative command tests**

```python
def test_dlinear_command_uses_mounted_data_gpu_zero_and_requested_horizon(layout):
    process = build_processes(get_task("DLinear", "ETTh1"), 336, layout)[0]
    assert process.argv[0] == sys.executable
    assert process.argv[1].endswith("baselines/Dlinear/run_longExp.py")
    assert option(process.argv, "--root_path") == str(layout.data_file.parent)
    assert option(process.argv, "--pred_len") == "336"
    assert option(process.argv, "--gpu") == "0"


def test_stmtm_builds_pretrain_then_finetune(layout):
    processes = build_processes(get_task("ST-MTM", "Weather"), 720, layout)
    assert [process.stage for process in processes] == ["pretrain", "finetune"]
    assert "--pretrained_model" in processes[1].argv


def test_all_registered_commands_reference_existing_entrypoints(repo_layout):
    for task in iter_tasks():
        for pred_len in task.horizons:
            for process in build_processes(task, pred_len, repo_layout.for_task(task)):
                assert Path(process.argv[1]).is_file()
```

- [ ] **Step 2: Verify command tests fail because the builder is missing**

Run: `python -m pytest tests/openi_baselines/test_commands.py -v`

Expected: import fails for `openi_baselines.commands`.

- [ ] **Step 3: Implement builders using registry parameters and per-horizon overrides**

Every process uses an argv tuple rather than shell interpolation. Each builder sets GPU 0, the mounted dataset path, a task-unique checkpoint path, and native output-compatible working directory. ST-MTM fine-tuning receives the pretraining checkpoint path produced by its first stage.

- [ ] **Step 4: Run complete command-generation tests**

Run: `python -m pytest tests/openi_baselines/test_commands.py -v`

Expected: every registered process references an existing Python entry point and all assertions pass.

- [ ] **Step 5: Commit command changes**

```bash
git add openi_baselines/commands.py tests/openi_baselines/test_commands.py
git commit -m "feat: build native baseline commands"
```

### Task 4: Metrics and artifact normalization

**Files:**
- Create: `openi_baselines/metrics.py`
- Create: `openi_baselines/artifacts.py`
- Test: `tests/openi_baselines/test_metrics.py`
- Test: `tests/openi_baselines/test_artifacts.py`

- [ ] **Step 1: Write failing normalization tests**

```python
def test_parse_text_metrics_uses_last_complete_evaluation():
    text = "epoch mse:0.8, mae:0.7\nfinal mse:0.42, mae:0.31, rse:0.5"
    assert parse_text_metrics(text)["mse"] == 0.42
    assert parse_text_metrics(text)["mae"] == 0.31


def test_collect_artifacts_classifies_checkpoint_and_prediction(tmp_path):
    (tmp_path / "checkpoint.pth").write_bytes(b"weights")
    (tmp_path / "pred.npy").write_bytes(b"prediction")
    counts = collect_artifacts(tmp_path, tmp_path / "normalized")
    assert counts == {"checkpoints": 1, "predictions": 1, "native_results": 0}
```

- [ ] **Step 2: Verify missing-module failures**

Run: `python -m pytest tests/openi_baselines/test_metrics.py tests/openi_baselines/test_artifacts.py -v`

Expected: imports fail for `metrics` and `artifacts`.

- [ ] **Step 3: Implement parsers, JSON/CSV writers, and non-recursive-safe collection**

Text parsing accepts scientific notation and `nan`, chooses the last complete MSE/MAE pair, and preserves optional RMSE, MAPE, MSPE, RSE, and correlation. Artifact collection excludes the normalized destination itself, retains relative source paths, and never removes native files.

- [ ] **Step 4: Run normalization tests**

Run: `python -m pytest tests/openi_baselines/test_metrics.py tests/openi_baselines/test_artifacts.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit normalization changes**

```bash
git add openi_baselines/metrics.py openi_baselines/artifacts.py tests/openi_baselines/test_metrics.py tests/openi_baselines/test_artifacts.py
git commit -m "feat: normalize baseline artifacts and metrics"
```

### Task 5: Resume-aware orchestration and CLI

**Files:**
- Create: `openi_baselines/runner.py`
- Create: `train_openi.py`
- Test: `tests/openi_baselines/fixtures/fake_train.py`
- Test: `tests/openi_baselines/test_runner.py`
- Test: `tests/openi_baselines/test_cli.py`

- [ ] **Step 1: Write failing process-orchestration tests**

```python
def test_runner_continues_after_one_horizon_fails(fake_plan, tmp_path):
    result = run_task(fake_plan(fail={192}), pred_lengths=(96, 192, 336), output=tmp_path)
    assert result.exit_code != 0
    assert result.statuses[96] == "succeeded"
    assert result.statuses[192] == "failed"
    assert result.statuses[336] == "succeeded"


def test_runner_skips_success_unless_force(fake_plan, tmp_path):
    run_task(fake_plan(), pred_lengths=(96,), output=tmp_path)
    resumed = run_task(fake_plan(), pred_lengths=(96,), output=tmp_path)
    forced = run_task(fake_plan(), pred_lengths=(96,), output=tmp_path, force=True)
    assert resumed.statuses[96] == "skipped"
    assert forced.statuses[96] == "succeeded"
```

- [ ] **Step 2: Verify orchestration tests fail for missing runner**

Run: `python -m pytest tests/openi_baselines/test_runner.py tests/openi_baselines/test_cli.py -v`

Expected: import fails for `openi_baselines.runner` or the CLI is absent.

- [ ] **Step 3: Implement status lifecycle, streaming, continuation, final upload, and CLI options**

The CLI supports `--model`, `--dataset`, `--pred-len`, `--force`, `--dry-run`, `--local`, `--code-root`, `--dataset-root`, and `--output-root`. OpenI mode always calls upload from finalization. `--dry-run` prints JSON process specifications without launching native processes.

- [ ] **Step 4: Run orchestration and CLI tests**

Run: `python -m pytest tests/openi_baselines/test_runner.py tests/openi_baselines/test_cli.py -v`

Expected: continuation, skip, force, partial artifacts, summaries, CLI validation, and exit codes pass.

- [ ] **Step 5: Commit orchestration changes**

```bash
git add openi_baselines/runner.py train_openi.py tests/openi_baselines/fixtures/fake_train.py tests/openi_baselines/test_runner.py tests/openi_baselines/test_cli.py
git commit -m "feat: orchestrate OpenI baseline reproductions"
```

### Task 6: Deterministic launch-script generation

**Files:**
- Create: `tools/generate_openi_scripts.py`
- Create: `tests/openi_baselines/test_script_generator.py`
- Generate: `scripts/openi/`

- [ ] **Step 1: Write failing generator tests**

```python
def test_generator_creates_all_and_per_horizon_scripts(tmp_path):
    generate_scripts(tmp_path)
    base = tmp_path / "PatchTST" / "ETTh1"
    assert (base / "all.sh").is_file()
    assert (base / "pred_96.sh").is_file()
    assert "--pred-len 96" in (base / "pred_96.sh").read_text()


def test_generator_emits_expected_script_count(tmp_path):
    generate_scripts(tmp_path)
    assert len(list(tmp_path.rglob("*.sh"))) == 395
```

The count is 79 model/dataset `all.sh` scripts plus 316 per-horizon scripts.

- [ ] **Step 2: Verify generator tests fail for the missing module**

Run: `python -m pytest tests/openi_baselines/test_script_generator.py -v`

Expected: import fails for `tools.generate_openi_scripts`.

- [ ] **Step 3: Implement deterministic generation and regenerate the full tree**

Each script uses `set -euo pipefail`, resolves the repository root relative to the script path, and calls `python -u "$REPO_ROOT/train_openi.py"` with canonical model/dataset arguments. Generation removes only previously generated `.sh` files below `scripts/openi` and leaves unrelated files untouched.

- [ ] **Step 4: Run generator and validate all scripts**

Run: `python tools/generate_openi_scripts.py`

Run: `python -m pytest tests/openi_baselines/test_script_generator.py -v`

Run on Linux/OpenI: `find scripts/openi -name '*.sh' -print0 | xargs -0 -n1 bash -n`

Expected: 395 scripts are generated; tests and shell syntax checks pass.

- [ ] **Step 5: Commit generated launchers**

```bash
git add tools/generate_openi_scripts.py tests/openi_baselines/test_script_generator.py scripts/openi
git commit -m "feat: generate fine-grained OpenI launchers"
```

### Task 7: Documentation and full verification

**Files:**
- Modify: `README.md`
- Create: `docs/openi-baseline-reproduction.md`

- [ ] **Step 1: Document image requirements, task submission, CLI examples, resume behavior, and output layout**

The guide includes commands for all-length, single-length, subset, dry-run, and local validation modes. It explains that `c2net` must exist in the built image and that numerical reproduction is validated by OpenI GPU jobs.

- [ ] **Step 2: Run the complete automated suite**

Run: `python -m pytest tests/openi_baselines -v`

Expected: zero failures.

- [ ] **Step 3: Compile all new Python modules**

Run: `python -m compileall -q train_openi.py openi_baselines tools/generate_openi_scripts.py`

Expected: exit code 0 and no output.

- [ ] **Step 4: Run repository-wide registered-command dry-run**

Run: `python train_openi.py --local --code-root . --dataset-root dataset --output-root output/openi-dry-run --model DLinear --dataset ETTh1 --pred-len 96 --dry-run`

Run: `python tools/generate_openi_scripts.py --check`

Expected: valid JSON command output and no generated-file drift.

- [ ] **Step 5: Check the final diff and commit documentation**

```bash
git diff --check
git add README.md docs/openi-baseline-reproduction.md
git commit -m "docs: explain OpenI baseline task submission"
```

- [ ] **Step 6: Record the deployment smoke gate**

Submit `scripts/openi/DLinear/ETTh1/pred_96.sh` as one OpenI V100 training task. Accept deployment only when the task output contains a checkpoint, prediction file, `metrics.json` with MSE/MAE, `status.json` marked succeeded, and the downloadable output is present on the task page.
