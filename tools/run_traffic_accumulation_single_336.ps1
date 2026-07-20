$root = 'F:\Dynamic-TMoE'
$py = Join-Path $root '.conda\dynamic_tmoe_cuda\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location -LiteralPath $root

$launcher = Join-Path $root 'output\traffic_accum_single_336_launcher.log'
$stdoutLog = Join-Path $root 'output\traffic_accum_single_336_stdout.log'
$stderrLog = Join-Path $root 'output\traffic_accum_single_336_stderr.log'
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] launching single pred_len=336 with $py" | Out-File -LiteralPath $launcher -Encoding utf8

& $py -u tools\run_traffic_accumulation_queue.py --python $py --root $root --micro-batch-size 1 --accumulation-steps 32 --only-pred-len 336 `
  1> $stdoutLog `
  2> $stderrLog

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] queue exited with $LASTEXITCODE" | Add-Content -LiteralPath $launcher -Encoding utf8
