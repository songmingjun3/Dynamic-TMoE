# Traffic OOM fix for `paper_strict_v1`

## Evidence

The original strict matrix uses `batch_size=32` for all four Traffic
horizons. On the OpenI V100 32GB runs, `psv1-dtmoe-traffic-96-s2021` and
`psv1-dtmoe-traffic-192-s2021` both failed after about 1 minute 10 seconds.
`output/openi_paper_strict_v1/state.json` records
`no OOM detail or finite metrics exposed`. This is failure evidence
consistent with the OOM hypothesis, not a claim that a CUDA OOM traceback was
captured.

## Revision and audit rules

`matrix_traffic_oom_fix.json` keeps the original matrix and changes only its
`matrix_id` and Traffic `96/192/336/720` `batch_size` values from 32 to 8.
It is a modified-budget recovery variant and must be reported separately from
`paper_strict_v1`.

- Keep all 9 datasets and 36 horizon tasks covered.
- Compare the complete JSON structures; the only permitted differences are the
  new matrix ID and those four Traffic batch-size paths.
- Do not edit OpenI state/manifest files or submit tasks as part of this fix.
- Accept a rerun only with `status.json`, `metrics.json`, C2Net output, and
  finite MSE/MAE evidence.
