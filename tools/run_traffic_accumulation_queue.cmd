@echo off
setlocal

set ROOT=F:\Dynamic-TMoE
set PY=%ROOT%\.conda\dynamic_tmoe_cuda\python.exe
set PYTHONIOENCODING=utf-8
set CUDA_VISIBLE_DEVICES=0

cd /d "%ROOT%"
"%PY%" -u tools\run_traffic_accumulation_queue.py --python "%PY%" --root "%ROOT%" --micro-batch-size 1 --accumulation-steps 32 1> "%ROOT%\output\traffic_accum_queue_stdout.log" 2> "%ROOT%\output\traffic_accum_queue_stderr.log"

endlocal
