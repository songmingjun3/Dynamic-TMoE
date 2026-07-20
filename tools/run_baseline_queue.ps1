$ErrorActionPreference = 'Stop'

$Root = 'F:\Dynamic-TMoE'
$Bash = 'C:\Program Files\Git\bin\bash.exe'
$BaselinePath = '/d/software/miniconda/envs/Time-Series-Library:/d/software/miniconda/envs/Time-Series-Library/Scripts'
$OutputDir = Join-Path $Root 'output'
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$State = Join-Path $OutputDir 'baseline_queue_state.csv'

$Jobs = @(
  @{Name='PatchTST_ETTh1';   Cwd='baselines\PatchTST';  Script='scripts/PatchTST/etth1.sh'},
  @{Name='PatchTST_ETTh2';   Cwd='baselines\PatchTST';  Script='scripts/PatchTST/etth2.sh'},
  @{Name='PatchTST_ETTm1';   Cwd='baselines\PatchTST';  Script='scripts/PatchTST/ettm1.sh'},
  @{Name='PatchTST_ETTm2';   Cwd='baselines\PatchTST';  Script='scripts/PatchTST/ettm2.sh'},
  @{Name='PatchTST_Weather'; Cwd='baselines\PatchTST';  Script='scripts/PatchTST/weather.sh'},
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

if (-not (Test-Path $State)) {
  'time,name,status,exit_code,script,stdout,stderr' | Set-Content -LiteralPath $State -Encoding UTF8
}

foreach ($job in $Jobs) {
  $name = $job.Name
  $cwd = Join-Path $Root $job.Cwd
  $script = $job.Script
  $stdout = Join-Path $OutputDir ("baseline_${name}.stdout.log")
  $stderr = Join-Path $OutputDir ("baseline_${name}.stderr.log")
  $start = (Get-Date).ToString('s')
  Add-Content -LiteralPath $State -Value "$start,$name,running,,$script,$stdout,$stderr"
  Push-Location $cwd
  try {
    $cmd = "export PATH=${BaselinePath}:`$PATH; bash '$script'"
    & $Bash -lc $cmd 1> $stdout 2> $stderr
    $code = $LASTEXITCODE
  } finally {
    Pop-Location
  }
  $end = (Get-Date).ToString('s')
  if ($code -eq 0) { $status = 'complete' } else { $status = 'failed' }
  Add-Content -LiteralPath $State -Value "$end,$name,$status,$code,$script,$stdout,$stderr"
  if ($code -ne 0) {
    exit $code
  }
}

