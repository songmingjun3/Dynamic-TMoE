$root = 'F:\Dynamic-TMoE'
$py = Join-Path $root '.conda\dynamic_tmoe_cuda\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location -LiteralPath $root

$launcher = Join-Path $root 'output\traffic_accum_single_192_launcher.log'
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] launching single pred_len=192 with $py" | Out-File -LiteralPath $launcher -Encoding utf8

& $py -u tools\run_traffic_accumulation_queue.py --python $py --root $root --micro-batch-size 1 --accumulation-steps 32 --only-pred-len 192 `
  1> (Join-Path $root 'output\traffic_accum_single_192_stdout.log') `
  2> (Join-Path $root 'output\traffic_accum_single_192_stderr.log')

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] queue exited with $LASTEXITCODE" | Add-Content -LiteralPath $launcher -Encoding utf8
