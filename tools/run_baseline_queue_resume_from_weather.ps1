$ErrorActionPreference = 'Stop'

$Root = 'F:\Dynamic-TMoE'
$Bash = 'C:\Program Files\Git\bin\bash.exe'
$BaselinePath = '/d/software/miniconda/envs/Time-Series-Library:/d/software/miniconda/envs/Time-Series-Library/Scripts'
$OutputDir = Join-Path $Root 'output'
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$State = Join-Path $OutputDir 'baseline_queue_state.csv'

if (-not (Test-Path $State)) {
  'time,name,status,exit_code,script,stdout,stderr' | Set-Content -LiteralPath $State -Encoding UTF8
}

function Add-StateRow {
  param(
    [string]$Name,
    [string]$Status,
    [string]$ExitCode,
    [string]$Script,
    [string]$Stdout,
    [string]$Stderr
  )

  $time = (Get-Date).ToString('s')
  Add-Content -LiteralPath $State -Value "$time,$Name,$Status,$ExitCode,$Script,$Stdout,$Stderr"
}

function Invoke-BaselineJob {
  param(
    [string]$Name,
    [string]$Cwd,
    [string]$Script
  )

  $stdout = Join-Path $OutputDir ("baseline_${Name}.stdout.log")
  $stderr = Join-Path $OutputDir ("baseline_${Name}.stderr.log")
  Add-StateRow -Name $Name -Status 'running' -ExitCode '' -Script $Script -Stdout $stdout -Stderr $stderr

  Push-Location (Join-Path $Root $Cwd)
  try {
    $cmd = "export PATH=${BaselinePath}:`$PATH; bash '$Script'"
    & $Bash -lc $cmd 1> $stdout 2> $stderr
    $code = $LASTEXITCODE
  } finally {
    Pop-Location
  }

  if ($code -eq 0) { $status = 'complete' } else { $status = 'failed' }
  Add-StateRow -Name $Name -Status $status -ExitCode $code -Script $Script -Stdout $stdout -Stderr $stderr
  if ($code -ne 0) {
    exit $code
  }
}

function Invoke-PatchTstWeatherResume {
  $name = 'PatchTST_Weather'
  $script = 'manual_resume_weather_192_336_720_num_workers0'
  $stdout = Join-Path $OutputDir 'baseline_PatchTST_Weather.resume.stdout.log'
  $stderr = Join-Path $OutputDir 'baseline_PatchTST_Weather.resume.stderr.log'
  Add-StateRow -Name $name -Status 'running' -ExitCode '' -Script $script -Stdout $stdout -Stderr $stderr

  Push-Location (Join-Path $Root 'baselines\PatchTST')
  try {
    $cmd = @'
export PATH=/d/software/miniconda/envs/Time-Series-Library:/d/software/miniconda/envs/Time-Series-Library/Scripts:$PATH
mkdir -p logs/LongForecasting
seq_len=96
model_name=PatchTST
root_path_name=../../dataset/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
random_seed=2021
for pred_len in 192 336 720
do
  python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 8 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 20 \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --num_workers 0 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
  code=$?
  if [ $code -ne 0 ]; then
    exit $code
  fi
done
'@
    & $Bash -lc $cmd 1> $stdout 2> $stderr
    $code = $LASTEXITCODE
  } finally {
    Pop-Location
  }

  if ($code -eq 0) { $status = 'complete' } else { $status = 'failed' }
  Add-StateRow -Name $name -Status $status -ExitCode $code -Script $script -Stdout $stdout -Stderr $stderr
  if ($code -ne 0) {
    exit $code
  }
}

Invoke-PatchTstWeatherResume

$RemainingJobs = @(
  @{Name='PatchTST_Exchange';Cwd='baselines\PatchTST';  Script='scripts/PatchTST/exchange.sh'},
  @{Name='PatchTST_ILI';     Cwd='baselines\PatchTST';  Script='scripts/PatchTST/illness.sh'},
  @{Name='PatchTST_ECL';     Cwd='baselines\PatchTST';  Script='scripts/PatchTST/electricity.sh'},
  @{Name='PatchTST_Traffic'; Cwd='baselines\PatchTST';  Script='scripts/PatchTST/traffic.sh'},

  @{Name='TimesNet_ETTh1';   Cwd='baselines\TimesNet';  Script='scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh'},
  @{Name='TimesNet_ETTh2';   Cwd='baselines\TimesNet';  Script='scripts/long_term_forecast/ETT_script/TimesNet_ETTh2.sh'},
  @{Name='TimesNet_ETTm1';   Cwd='baselines\TimesNet';  Script='scripts/long_term_forecast/ETT_script/TimesNet_ETTm1.sh'},
  @{Name='TimesNet_ETTm2';   Cwd='baselines\TimesNet';  Script='scripts/long_term_forecast/ETT_script/TimesNet_ETTm2.sh'},
  @{Name='TimesNet_Weather'; Cwd='baselines\TimesNet';  Script='scripts/long_term_forecast/Weather_script/TimesNet.sh'},
  @{Name='TimesNet_Exchange';Cwd='baselines\TimesNet';  Script='scripts/long_term_forecast/Exchange_script/TimesNet.sh'},
  @{Name='TimesNet_ILI';     Cwd='baselines\TimesNet';  Script='scripts/long_term_forecast/ILI_script/TimesNet.sh'},
  @{Name='TimesNet_ECL';     Cwd='baselines\TimesNet';  Script='scripts/long_term_forecast/ECL_script/TimesNet.sh'},
  @{Name='TimesNet_Traffic'; Cwd='baselines\TimesNet';  Script='scripts/long_term_forecast/Traffic_script/TimesNet.sh'},

  @{Name='TimeMixer_ETTh1';  Cwd='baselines\TimeMixer'; Script='scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh'},
  @{Name='TimeMixer_ETTh2';  Cwd='baselines\TimeMixer'; Script='scripts/long_term_forecast/ETT_script/TimeMixer_ETTh2_unify.sh'},
  @{Name='TimeMixer_ETTm1';  Cwd='baselines\TimeMixer'; Script='scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1_unify.sh'},
  @{Name='TimeMixer_ETTm2';  Cwd='baselines\TimeMixer'; Script='scripts/long_term_forecast/ETT_script/TimeMixer_ETTm2_unify.sh'},
  @{Name='TimeMixer_Weather';Cwd='baselines\TimeMixer'; Script='scripts/long_term_forecast/Weather_script/TimeMixer_unify.sh'},
  @{Name='TimeMixer_ECL';    Cwd='baselines\TimeMixer'; Script='scripts/long_term_forecast/ECL_script/TimeMixer_unify.sh'},
  @{Name='TimeMixer_Traffic';Cwd='baselines\TimeMixer'; Script='scripts/long_term_forecast/Traffic_script/TimeMixer_unify.sh'},

  @{Name='FEDformer_M';      Cwd='baselines\FEDformer'; Script='scripts/run_M.sh'},
  @{Name='TFPS_ETTh1';       Cwd='baselines\TFPS';      Script='scripts/etth1.sh'},
  @{Name='TFPS_ETTh2';       Cwd='baselines\TFPS';      Script='scripts/etth2.sh'},
  @{Name='TFPS_ETTm1';       Cwd='baselines\TFPS';      Script='scripts/ettm1.sh'},
  @{Name='TFPS_ETTm2';       Cwd='baselines\TFPS';      Script='scripts/ettm2.sh'},
  @{Name='TFPS_Weather';     Cwd='baselines\TFPS';      Script='scripts/weather.sh'},
  @{Name='TFPS_Exchange';    Cwd='baselines\TFPS';      Script='scripts/exchange.sh'},
  @{Name='TFPS_ILI';         Cwd='baselines\TFPS';      Script='scripts/ILI.sh'},
  @{Name='TFPS_ECL';         Cwd='baselines\TFPS';      Script='scripts/electricity.sh'},
  @{Name='TFPS_Traffic';     Cwd='baselines\TFPS';      Script='scripts/traffic.sh'}
)

foreach ($job in $RemainingJobs) {
  Invoke-BaselineJob -Name $job.Name -Cwd $job.Cwd -Script $job.Script
}
