@echo off
setlocal

set ROOT=F:\Dynamic-TMoE
set PY=%ROOT%\.conda\dynamic_tmoe_cuda\python.exe
set PYTHONIOENCODING=utf-8
set CUDA_VISIBLE_DEVICES=0

cd /d "%ROOT%"
"%PY%" -u tools\run_reproduction_queue.py --python "%PY%" --root "%ROOT%" 1> "%ROOT%\output\queue_stdout.log" 2> "%ROOT%\output\queue_stderr.log"

endlocal
