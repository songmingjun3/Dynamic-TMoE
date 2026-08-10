# OpenI paper_strict_v1 execution plan

Status: `WAITING_FOR_USER_CONFIRMATION`

This plan is separate from `output/openi_p0`. It does not overwrite the old
P0 state, results, or permanent failure records.

## Scope

- Model: `Dynamic_TMoE`
- Tasks: 36 rows from the paper Table 7
- Project: `songmingjun/Dynamic-TMoE`
- Branch: `codex/openi-baseline-runner`
- Boot file: `train_openi.py`
- Dataset repository: `songmingjun/TS`
- Image: `TMoE`
- Resource: one V100 32GB, 8 CPU, 50GB memory
- Protocol label: `paper_config_on_V100`
- Matrix: `configs/openi_paper_strict_v1/matrix.json`
- Output namespace: `output/openi_paper_strict_v1`

The matrix contains the Table 7 patch length, stride, learning rate, dropout,
MoE/RNN depth, and batch size for every dataset and horizon. It also freezes
seed 2021, AdamW, MSE loss, MSE/MAE reporting, patience 10, top-k 3,
`drift_k_sigma=3.0`, and the repository-specific values not specified by the
paper.

## Execution gates

1. Confirm that the pushed branch contains this matrix and the strict argv
   mapping.
2. Validate all dataset files and the mounted C2Net paths.
3. Generate exactly one task spec for the next planned row.
4. Run `openi_task.py --dry-run` and inspect the exact task name and metadata.
5. Check the global OpenI task list for any `queued`, `running`, or `unknown`
   task before submitting.
6. Submit exactly one task, record `task_id`, `task_url`, and status, then
   wait for its terminal state.
7. On success, download all artifacts and require `c2net/output`,
   `status.json`, `metrics.json`, finite MSE/MAE, command, and environment
   evidence before marking the row `succeeded`.
8. Record failed, OOM, cancelled, or stopped states without retrying. If the
   balance is below 7 points, stop submission and write `next_resume_after`.

## Ordering and naming

Run datasets in the order `ETTh1, ETTh2, ETTm1, ETTm2, Traffic, Electricity,
Weather, ILI, Exchange`, with horizons ascending inside each dataset. New
task names use `psv1-dtmoe-{dataset}-{pred_len}-s2021` and are distinct from
the old `p0v1-*` tasks. The old ETTh1 96/192/336 failures are included as
independent `paper_strict_v1` experiments, never retried under their old keys.

## Existing data comparison

Existing local results remain immutable and are joined by
`(model,dataset,pred_len,seed,protocol)`. The final report will label the 144
complete numeric baseline rows, 36 modified-budget Dynamic_TMoE rows, four
Traffic gradient-accumulation rows, and one recovered PatchTST checkpoint row
separately from `paper_strict_v1`.

## Confirmation gate

No OpenI task is submitted while this plan has status
`WAITING_FOR_USER_CONFIRMATION`. After confirmation, the runner may proceed
only through the gates above and remains globally serial.
