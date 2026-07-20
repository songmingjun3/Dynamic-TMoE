$ErrorActionPreference = 'Stop'

$root = 'F:\Dynamic-TMoE'
$python = Join-Path $root '.conda\dynamic_tmoe_cuda\python.exe'

Set-Location $root

$env:PYTHONIOENCODING = 'utf-8'
$env:CUDA_VISIBLE_DEVICES = '0'

$stdout = Join-Path $root 'output\queue_stdout.log'
$stderr = Join-Path $root 'output\queue_stderr.log'

& $python -u tools\run_reproduction_queue.py `
  --python $python `
  --root $root `
  1> $stdout `
  2> $stderr
