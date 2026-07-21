# OpenI All-Baseline Parameter Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish and verify a copy-ready OpenI parameter guide that covers all 79 supported baseline/dataset combinations in two V100-oriented matrix jobs.

**Architecture:** A standalone Markdown guide contains the two authoritative parameter strings, coverage arithmetic, operational warnings, timeout split fallbacks, and artifact checks. Existing documentation links to it; validation uses the real matrix planner and CLI dry-run without changing training code.

**Tech Stack:** Markdown, Python CLI, PowerShell, pytest, Git

---

## File Structure

- Create `docs/openi-all-baseline-run-parameters.md`: authoritative two-job submission and completion guide.
- Modify `docs/openi-baseline-reproduction.md`: link to the full all-baseline parameter guide.
- No Python production or test files change.

### Task 1: Create the Complete Two-Job Parameter Guide

**Files:**
- Create: `docs/openi-all-baseline-run-parameters.md`

- [ ] **Step 1: Write the platform and coverage sections**

Create the document with this header and platform table:

```markdown
# 启智平台全部 Baseline 实验运行参数

## 适用环境

| 配置项 | 值 |
| --- | --- |
| 代码分支 | `codex/openi-baseline-runner` |
| 启动文件 | `train_openi.py` |
| 数据集 | `TS` |
| GPU | 单卡 Tesla V100 32GB |
| 执行方式 | 单 GPU 串行笛卡尔积 |

本方案用两个训练任务覆盖 9 个 baseline、79 个受支持的模型/数据集组合和 316 次 horizon 训练。仓库不支持的 `TimeMixer × Exchange` 与 `TimeMixer × ILI` 不包含在实验矩阵中。
```

- [ ] **Step 2: Add the exact Task A fields and parameter string**

Document task name `all-baselines-common-datasets`, its `63` combinations and `252` horizon rows, then include this single-line value exactly:

```text
--model DLinear,FEDformer,FITS,PatchTST,RAFT,ST-MTM,TFPS,TimeMixer,TimesNet --dataset ETTh1,ETTh2,ETTm1,ETTm2,Electricity,Traffic,Weather --pred-len all --batch-size 96:32,192:16,336:8,720:8
```

List the model-major execution order and state that every combination uses horizons `96,192,336,720`.

- [ ] **Step 3: Add the exact Task B fields and parameter string**

Document task name `all-baselines-exchange-ili`, its `16` combinations and `64` horizon rows, then include this value exactly:

```text
--model DLinear,FEDformer,FITS,PatchTST,RAFT,ST-MTM,TFPS,TimesNet --dataset Exchange,ILI --pred-len all --batch-size 24:16,36:16,48:8,60:8,96:32,192:16,336:8,720:8
```

State that Exchange uses `96,192,336,720`, ILI uses `24,36,48,60`, and the batch mapping covers their union.

- [ ] **Step 4: Document batch semantics and exact-reproduction alternative**

Add this warning:

```markdown
## Batch Size 说明

上述 `--batch-size` 是针对 V100 32GB 的全局保守配置，会覆盖所有 baseline 的原始 batch size。它用于降低大型模型和高维数据集的 OOM 风险，但不等同于每篇论文的原始训练超参数。

如果目标是严格保留仓库中的原始 batch size，请从两个任务参数中完整删除 `--batch-size ...`，不要只删除部分 horizon 映射。
```

- [ ] **Step 5: Add failure, resume, and timeout split instructions**

Document that tasks are serial, failures continue, overall failure returns nonzero, and successful horizons skip only when their status files are present in the same output context. Explain `--force` as a full-matrix override.

Provide these three fallback model groups:

```text
DLinear,FEDformer,FITS
PatchTST,RAFT,ST-MTM
TFPS,TimeMixer,TimesNet
```

For Task B, use `TFPS,TimesNet` for the third group because TimeMixer does not support Exchange or ILI. State that splitting changes only the `--model` value; dataset, horizon and batch parameters remain unchanged.

- [ ] **Step 6: Add artifact and completion checklists**

Document these required artifacts:

```text
<output>/<model>/<dataset>/<pred_len>/status.json
<output>/<model>/<dataset>/<pred_len>/metrics.json
<output>/multi_task_summary.json
<output>/multi_task_summary.csv
```

Add completion checks: Task A summary has 252 rows, Task B has 64 rows, combined coverage is 316 rows, every successful row has MSE/MAE metrics, and failed rows have corresponding logs/status.

### Task 2: Link and Validate the Authoritative Parameters

**Files:**
- Modify: `docs/openi-baseline-reproduction.md`
- Verify: `docs/openi-all-baseline-run-parameters.md`

- [ ] **Step 1: Add the existing-guide link**

After the support-range paragraph in `docs/openi-baseline-reproduction.md`, add:

```markdown
需要一次提交全部受支持实验时，请使用[全部 Baseline 实验运行参数](openi-all-baseline-run-parameters.md)。
```

- [ ] **Step 2: Validate Task A with the real planner**

Run:

```powershell
python -c "from openi_baselines.matrix import plan_matrix; from openi_baselines.registry import parse_task_matrix; tasks=parse_task_matrix('DLinear,FEDformer,FITS,PatchTST,RAFT,ST-MTM,TFPS,TimeMixer,TimesNet','ETTh1,ETTh2,ETTm1,ETTm2,Electricity,Traffic,Weather'); plans=plan_matrix(tasks,pred_len_value='all',batch_size_value='96:32,192:16,336:8,720:8'); assert len(plans)==63; assert sum(len(p.pred_lengths) for p in plans)==252; assert all(dict(p.batch_sizes)=={96:32,192:16,336:8,720:8} for p in plans); print('Task A: 63 tasks, 252 horizons')"
```

Expected: `Task A: 63 tasks, 252 horizons`.

- [ ] **Step 3: Validate Task B with the real planner**

Run:

```powershell
python -c "from openi_baselines.matrix import plan_matrix; from openi_baselines.registry import parse_task_matrix; tasks=parse_task_matrix('DLinear,FEDformer,FITS,PatchTST,RAFT,ST-MTM,TFPS,TimesNet','Exchange,ILI'); plans=plan_matrix(tasks,pred_len_value='all',batch_size_value='24:16,36:16,48:8,60:8,96:32,192:16,336:8,720:8'); assert len(plans)==16; assert sum(len(p.pred_lengths) for p in plans)==64; assert {n for p in plans for n in p.pred_lengths}=={24,36,48,60,96,192,336,720}; print('Task B: 16 tasks, 64 horizons')"
```

Expected: `Task B: 16 tasks, 64 horizons`.

- [ ] **Step 4: Run both full CLI dry-runs and inspect counts**

Run Task A, parse its JSON, and validate every native command:

```powershell
$taskA = python train_openi.py --local --code-root . --dataset-root dataset --output-root "$env:TEMP\openi-all-baselines-task-a" --model DLinear,FEDformer,FITS,PatchTST,RAFT,ST-MTM,TFPS,TimeMixer,TimesNet --dataset ETTh1,ETTh2,ETTm1,ETTm2,Electricity,Traffic,Weather --pred-len all --batch-size 96:32,192:16,336:8,720:8 --dry-run | ConvertFrom-Json
if ($taskA.exit_code -ne 0 -or $taskA.tasks.Count -ne 63 -or $taskA.rows.Count -ne 252) { throw "Task A dry-run count mismatch" }
$taskABatches = @{96='32'; 192='16'; 336='8'; 720='8'}
foreach ($task in $taskA.tasks) { foreach ($horizon in $task.result.commands.PSObject.Properties) { foreach ($command in $horizon.Value) { $argv = @($command.argv); $index = [array]::IndexOf($argv, '--batch_size'); if ($index -lt 0 -or $argv[$index + 1] -ne $taskABatches[[int]$horizon.Name]) { throw "Task A batch mismatch" } } } }
"Task A CLI: $($taskA.tasks.Count) tasks, $($taskA.rows.Count) horizons"
```

Expected: `Task A CLI: 63 tasks, 252 horizons`.

Run Task B with its horizon union:

```powershell
$taskB = python train_openi.py --local --code-root . --dataset-root dataset --output-root "$env:TEMP\openi-all-baselines-task-b" --model DLinear,FEDformer,FITS,PatchTST,RAFT,ST-MTM,TFPS,TimesNet --dataset Exchange,ILI --pred-len all --batch-size 24:16,36:16,48:8,60:8,96:32,192:16,336:8,720:8 --dry-run | ConvertFrom-Json
if ($taskB.exit_code -ne 0 -or $taskB.tasks.Count -ne 16 -or $taskB.rows.Count -ne 64) { throw "Task B dry-run count mismatch" }
$taskBBatches = @{24='16'; 36='16'; 48='8'; 60='8'; 96='32'; 192='16'; 336='8'; 720='8'}
foreach ($task in $taskB.tasks) { foreach ($horizon in $task.result.commands.PSObject.Properties) { foreach ($command in $horizon.Value) { $argv = @($command.argv); $index = [array]::IndexOf($argv, '--batch_size'); if ($index -lt 0 -or $argv[$index + 1] -ne $taskBBatches[[int]$horizon.Name]) { throw "Task B batch mismatch" } } } }
"Task B CLI: $($taskB.tasks.Count) tasks, $($taskB.rows.Count) horizons"
```

Expected: `Task B CLI: 16 tasks, 64 horizons`.

The combined assertions prove:

```text
Task A: tasks.Count = 63, rows.Count = 252, exit_code = 0
Task B: tasks.Count = 16, rows.Count = 64, exit_code = 0
```

For every command, locate `--batch_size` in `argv` and verify its value matches the documented mapping for that `pred_len`.

- [ ] **Step 5: Run repository verification**

Run:

```powershell
python -m pytest tests/openi_baselines -q
python tools/generate_openi_scripts.py --check
git diff --check
```

Expected: 77 tests pass, 395 scripts are current, and the whitespace check exits 0.

### Task 3: Commit and Synchronize the Documentation

**Files:**
- Create: `docs/openi-all-baseline-run-parameters.md`
- Modify: `docs/openi-baseline-reproduction.md`

- [ ] **Step 1: Commit both documentation files**

```bash
git add docs/openi-all-baseline-run-parameters.md docs/openi-baseline-reproduction.md
git commit -m "docs: add complete OpenI baseline parameters"
```

- [ ] **Step 2: Verify final committed state**

Run:

```powershell
python -m pytest tests/openi_baselines -q
git status --short
git log -1 --oneline
```

Expected: 77 tests pass; only the user's pre-existing untracked paths remain; the latest commit is the documentation commit.

- [ ] **Step 3: Push GitHub**

Run:

```bash
git push origin codex/openi-baseline-runner
```

Expected: GitHub branch updates to the documentation commit.

- [ ] **Step 4: Push OpenI**

Run:

```bash
git push openi codex/openi-baseline-runner
```

Expected: OpenI branch updates to the same documentation commit.
