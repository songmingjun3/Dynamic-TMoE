@echo off
setlocal

set ROOT=F:\Dynamic-TMoE
set PY=%ROOT%\.conda\dynamic_tmoe_cuda\python.exe
set PYTHONIOENCODING=utf-8
set CUDA_VISIBLE_DEVICES=0

cd /d "%ROOT%"
"%PY%" -u tools\run_traffic_strict_queue.py --python "%PY%" --root "%ROOT%" 1> "%ROOT%\output\traffic_strict_queue_stdout.log" 2> "%ROOT%\output\traffic_strict_queue_stderr.log"

endlocal
