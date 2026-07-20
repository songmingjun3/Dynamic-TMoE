@echo off
setlocal

set ROOT=F:\Dynamic-TMoE
set PY=%ROOT%\.conda\dynamic_tmoe_cuda\python.exe
set PYTHONIOENCODING=utf-8
set CUDA_VISIBLE_DEVICES=0

cd /d "%ROOT%"
echo [%DATE% %TIME%] launching single pred_len=192 with "%PY%" > "%ROOT%\output\traffic_accum_single_192_launcher.log"
if not exist "%PY%" (
  echo [%DATE% %TIME%] missing python "%PY%" >> "%ROOT%\output\traffic_accum_single_192_launcher.log"
  exit /b 1
)
"%PY%" -u tools\run_traffic_accumulation_queue.py --python "%PY%" --root "%ROOT%" --micro-batch-size 1 --accumulation-steps 32 --only-pred-len 192 1> "%ROOT%\output\traffic_accum_single_192_stdout.log" 2> "%ROOT%\output\traffic_accum_single_192_stderr.log"
echo [%DATE% %TIME%] queue exited with %ERRORLEVEL% >> "%ROOT%\output\traffic_accum_single_192_launcher.log"

endlocal
