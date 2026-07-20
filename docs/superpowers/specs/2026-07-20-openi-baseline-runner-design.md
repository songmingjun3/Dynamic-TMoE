# OpenI Baseline Reproduction Runner Design

## Goal

Provide a single OpenI-compatible training entry point for every supported long-term forecasting baseline in this repository, plus generated task scripts at model, dataset, and prediction-length granularity. Every task must persist checkpoints, predictions, native results, logs, normalized metrics, environment metadata, and status information through the OpenI `c2net` output channel.

## Scope

The runner covers these baseline directories:

- DLinear
- FEDformer
- FITS
- PatchTST
- RAFT
- ST-MTM
- TFPS
- TimeMixer
- TimesNet

The target datasets are ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Exchange, ILI, Traffic, and Weather. A model/dataset combination is registered only when the repository already contains an experiment script for that combination. TimeMixer therefore covers ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, and Weather; the other eight baselines cover all nine target datasets.

Each model/dataset task runs the standard prediction lengths `96,192,336,720`. ILI uses `24,36,48,60`. Users may omit `--pred-len` to run all standard lengths, provide one value, or provide a comma-separated subset.

ST-MTM tasks run the repository's complete two-stage workflow: self-supervised pretraining followed by fine-tuning and evaluation. Both stages persist their checkpoints and logs.

## Selected Approach

Use a manifest-driven Python runner with generated thin shell scripts.

The manifest is the single source of truth for supported combinations, native entry points, dataset-specific parameters, prediction lengths, output discovery rules, and multi-stage workflows. The Python runner owns platform initialization, validation, task orchestration, status handling, artifact collection, metric normalization, and upload. Generated shell scripts contain no duplicated experiment configuration; they only invoke the shared runner with a model, dataset, and optional prediction length.

This approach preserves each baseline's native Python implementation while avoiding dependence on its hard-coded shell paths, GPU indices, working directories, and output locations.

## Architecture

### Public entry point

`train_openi.py` is the command submitted as the OpenI boot file. Its public interface is:

```text
python train_openi.py \
  --model <baseline> \
  --dataset <dataset> \
  [--pred-len 96|96,192|all] \
  [--force] \
  [--dry-run]
```

Model and dataset names are matched case-insensitively against canonical registry names. Omitting `--pred-len` and passing `all` have the same meaning. Invalid or unsupported combinations fail before any GPU process starts and print the supported values.

### Internal package

`openi_baselines/registry.py` defines immutable task specifications for the nine baselines. It contains no subprocess or platform behavior.

`openi_baselines/paths.py` resolves the repository, selected mounted dataset, and task output directories. It supports the observed OpenI layout (`/tmp/code/Dynamic-TMoE`, `/tmp/dataset/TS`) without hard-coding those paths.

`openi_baselines/platform.py` wraps `c2net.context.prepare` and `upload_output`. A local mode permits unit tests and `--dry-run` without requiring a live OpenI job.

`openi_baselines/commands.py` converts a registered task and prediction length into one or more native process specifications. ST-MTM returns an ordered pretrain/fine-tune sequence; other baselines return one training/evaluation process.

`openi_baselines/runner.py` executes process specifications, streams output to both the OpenI console and log files, continues after an individual prediction-length failure, implements successful-task skipping and `--force`, and returns a final nonzero exit code when any requested item fails.

`openi_baselines/artifacts.py` discovers and copies native checkpoints, predictions, result files, and additional model-specific outputs into the stable task artifact layout.

`openi_baselines/metrics.py` parses the native metric formats and writes normalized JSON and CSV summaries. It preserves every metric exposed by the baseline and requires MSE and MAE for a successful evaluation.

`tools/generate_openi_scripts.py` deterministically regenerates all thin shell launchers from the registry.

## Generated task scripts

For every registered model/dataset combination, generation creates:

```text
scripts/openi/<model>/<dataset>/all.sh
scripts/openi/<model>/<dataset>/pred_<length>.sh
```

For example:

```text
scripts/openi/PatchTST/ETTh1/all.sh
scripts/openi/PatchTST/ETTh1/pred_96.sh
scripts/openi/PatchTST/ETTh1/pred_192.sh
scripts/openi/PatchTST/ETTh1/pred_336.sh
scripts/openi/PatchTST/ETTh1/pred_720.sh
```

Each launcher resolves the repository root relative to itself and executes `train_openi.py`. It is therefore valid both as an OpenI boot command and from a checked-out Linux repository.

## Data and execution flow

1. Initialize the platform context with `c2net.context.prepare()`.
2. Resolve the checked-out repository and selected dataset mount from the returned context.
3. Validate the requested model, dataset, and prediction lengths against the registry.
4. Create an isolated working directory under `c2net_context.output_path/<model>/<dataset>/<pred_len>`.
5. Record the resolved command and environment before starting the native process.
6. Execute the registered native process or ordered ST-MTM stages with GPU 0.
7. Stream stdout and stderr to the OpenI console and `logs/train.log`.
8. Collect checkpoints, predictions, and native result files into the stable artifact layout.
9. Parse and normalize metrics, then update the per-length status.
10. Continue with the remaining requested lengths after a failure.
11. Write dataset-level summaries and call `upload_output()` even when the task partially fails.
12. Exit nonzero after upload if any requested prediction length failed.

## Output contract

Each prediction length produces:

```text
<output_path>/<model>/<dataset>/<pred_len>/
├── checkpoints/
├── predictions/
├── native_results/
├── logs/
│   └── train.log
├── command.json
├── environment.json
├── metrics.json
└── status.json
```

An all-length task additionally produces:

```text
<output_path>/<model>/<dataset>/
├── metrics_summary.csv
├── metrics_summary.json
└── status_summary.json
```

`command.json` records argv, native working directory, stage name, model, dataset, and prediction length. `environment.json` records Python, PyTorch, CUDA availability, CUDA runtime reported by PyTorch, visible GPU names, and relevant package versions.

`status.json` uses `running`, `succeeded`, `failed`, or `skipped` and records timestamps, process exit codes, artifact counts, and an error message when applicable. A pre-existing `succeeded` status skips the item unless `--force` is supplied.

## Metrics contract

Normalized metrics use this shape:

```json
{
  "model": "PatchTST",
  "dataset": "ETTh1",
  "pred_len": 96,
  "metrics": {
    "mse": 0.0,
    "mae": 0.0
  },
  "source": "native result path"
}
```

Additional native values such as RMSE, MAPE, MSPE, RSE, correlation, or training loss remain in `metrics`. Missing MSE or MAE after a nominally successful process is treated as an evaluation failure, because the task has not produced comparable reproduction results.

## Failure handling

- Validation failures stop before platform workload execution.
- A native process failure is recorded for that prediction length; later requested lengths still run.
- Existing logs and partial artifacts are collected on both success and failure.
- ST-MTM fine-tuning does not start if its required pretraining stage fails.
- Upload is attempted in a finalization path regardless of task outcome.
- Upload failure is reported separately and produces a nonzero final exit.
- A dataset-level summary distinguishes succeeded, failed, and skipped prediction lengths.

## Testing strategy

Unit tests cover registry resolution, case-insensitive aliases, prediction-length validation, path discovery, command generation, artifact classification, metrics parsing, status transitions, resume behavior, `--force`, partial failure continuation, and ST-MTM stage dependencies.

Runner tests use small real subprocess fixtures rather than GPU mocks. These fixtures emit representative metrics and artifacts, allowing tests to exercise console streaming, log creation, continuation, collection, and summary generation.

Generator tests compare the expected registry matrix with generated launchers and validate every generated script with `bash -n` when Bash is available.

Repository verification consists of:

1. `pytest` for the runner package and generator.
2. Python bytecode compilation for all new Python files.
3. Shell syntax validation for every generated launcher.
4. A full registry `--dry-run` matrix that resolves every supported model/dataset/prediction-length command without starting GPU work.
5. One OpenI ETTh1-96 smoke task as the deployment gate before submitting the full matrix.

## Non-goals

- Do not rewrite baseline model implementations.
- Do not invent experiments for combinations absent from the repository scripts.
- Do not merge multiple model/dataset combinations into one OpenI task.
- Do not install or downgrade large ML dependencies at task runtime.
- Do not claim full numerical reproduction from local static tests; numerical validation occurs in OpenI GPU jobs.
