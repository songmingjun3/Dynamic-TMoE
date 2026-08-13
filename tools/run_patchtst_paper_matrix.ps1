$ErrorActionPreference = 'Stop'

$Root = 'F:\Dynamic-TMoE'
$PatchRoot = Join-Path $Root 'baselines\PatchTST'
$Python = 'D:\software\miniconda\envs\Time-Series-Library\python.exe'
$OutputDir = Join-Path $Root 'output'
$LogDir = Join-Path $PatchRoot 'logs\LongForecasting'
$State = Join-Path $OutputDir 'patchtst_paper_matrix_state.csv'
$BaselineState = Join-Path $OutputDir 'baseline_queue_state.csv'
$PauseAfterName = 'PatchTST_Traffic_192'
$PauseFlag = Join-Path $OutputDir 'pause_after_traffic_192.flag'

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not (Test-Path $State)) {
  'time,name,status,exit_code,log,stderr' | Set-Content -LiteralPath $State -Encoding UTF8
}

function Add-StateRow {
  param(
    [string]$Name,
    [string]$Status,
    [string]$ExitCode,
    [string]$Log,
    [string]$Stderr
  )
  $time = (Get-Date).ToString('s')
  Add-Content -LiteralPath $State -Value "$time,$Name,$Status,$ExitCode,$Log,$Stderr"
  Add-Content -LiteralPath $BaselineState -Value "$time,$Name,$Status,$ExitCode,patchtst_paper_matrix,$Log,$Stderr"
}

function Test-CompleteLog {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    return $false
  }
  $tail = Get-Content -LiteralPath $Path -Tail 80 -ErrorAction SilentlyContinue
  return [bool]($tail | Select-String -Pattern 'mse\s*:' -SimpleMatch:$false)
}

function Wait-IfPauseRequested {
  param([string]$CompletedName)

  if ($CompletedName -ne $PauseAfterName -or -not (Test-Path $PauseFlag)) {
    return
  }

  Add-StateRow -Name $CompletedName -Status 'paused_after_complete' -ExitCode '0' -Log '' -Stderr ''
  while (Test-Path $PauseFlag) {
    Start-Sleep -Seconds 60
  }
}

$Jobs = @(
  @{Dataset='ETTh1';       Root='../../dataset/ETT-small/';     Data='ETTh1.csv';            ModelId='ETTh1';            Seq=336; Preds=@(96,192,336,720); Enc=7;   Batch=128; Lr='0.0001'; Patience=$null},
  @{Dataset='ETTh2';       Root='../../dataset/ETT-small/';     Data='ETTh2.csv';            ModelId='ETTh2';            Seq=96;  Preds=@(96,192,336,720); Enc=7;   Batch=128; Lr='0.0001'; Patience=$null},
  @{Dataset='ETTm1';       Root='../../dataset/ETT-small/';     Data='ETTm1.csv';            ModelId='ETTm1';            Seq=336; Preds=@(96,192,336,720); Enc=7;   Batch=128; Lr='0.0001'; Patience=20},
  @{Dataset='ETTm2';       Root='../../dataset/ETT-small/';     Data='ETTm2.csv';            ModelId='ETTm2';            Seq=336; Preds=@(96,192,336,720); Enc=7;   Batch=128; Lr='0.0001'; Patience=20},
  @{Dataset='Weather';     Root='../../dataset/weather/';       Data='weather.csv';          ModelId='weather';          Seq=96;  Preds=@(96,192,336,720); Enc=21;  Batch=128; Lr='0.0001'; Patience=20},
  @{Dataset='Exchange';    Root='../../dataset/exchange_rate/'; Data='exchange_rate.csv';    ModelId='exchange_rate';    Seq=96;  Preds=@(96,192,336,720); Enc=8;   Batch=32;  Lr='0.0001'; Patience=10},
  @{Dataset='ILI';         Root='../../dataset/illness/';       Data='national_illness.csv'; ModelId='national_illness'; Seq=36;  Preds=@(24,36,48,60);    Enc=7;   Batch=16;  Lr='0.0025'; Patience=$null},
  @{Dataset='Electricity'; Root='../../dataset/electricity/';   Data='electricity.csv';      ModelId='Electricity';      Seq=336; Preds=@(96,192,336,720); Enc=321; Batch=32;  Lr='0.0001'; Patience=10},
  @{Dataset='Traffic';     Root='../../dataset/traffic/';       Data='traffic.csv';          ModelId='traffic';          Seq=96;  Preds=@(96,192,336,720); Enc=862; Batch=24;  Lr='0.0001'; Patience=10}
)

$env:PYTHONPATH = "$PatchRoot;$($env:PYTHONPATH)"
$env:PATH = "D:\software\miniconda\envs\Time-Series-Library;D:\software\miniconda\envs\Time-Series-Library\Scripts;$($env:PATH)"

Push-Location $PatchRoot
try {
  foreach ($job in $Jobs) {
    foreach ($pred in $job.Preds) {
      $name = "PatchTST_$($job.Dataset)_$pred"
      $log = Join-Path $LogDir "PatchTST_$($job.ModelId)_$($job.Seq)_$pred.log"
      $stderr = Join-Path $OutputDir "patchtst_paper_matrix_$($job.Dataset)_$pred.stderr.log"

      if (Test-CompleteLog -Path $log) {
        Add-StateRow -Name $name -Status 'skipped_complete' -ExitCode '0' -Log $log -Stderr $stderr
        continue
      }

      if (Test-Path $log) {
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        Copy-Item -LiteralPath $log -Destination "$log.interrupted_$stamp" -Force
      }

      Add-StateRow -Name $name -Status 'running' -ExitCode '' -Log $log -Stderr $stderr

      $args = @(
        '-u', 'run_longExp.py',
        '--random_seed', '2021',
        '--is_training', '1',
        '--root_path', $job.Root,
        '--data_path', $job.Data,
        '--model_id', "$($job.ModelId)_$($job.Seq)_$pred",
        '--model', 'PatchTST',
        '--data', 'custom',
        '--features', 'M',
        '--seq_len', "$($job.Seq)",
        '--pred_len', "$pred",
        '--enc_in', "$($job.Enc)",
        '--e_layers', '3',
        '--n_heads', '16',
        '--d_model', '128',
        '--d_ff', '256',
        '--dropout', '0.2',
        '--fc_dropout', '0.2',
        '--head_dropout', '0',
        '--patch_len', '16',
        '--stride', '8',
        '--des', 'Exp',
        '--train_epochs', '100',
        '--lradj', 'TST',
        '--pct_start', '0.2',
        '--itr', '1',
        '--batch_size', "$($job.Batch)",
        '--num_workers', '0',
        '--learning_rate', $job.Lr
      )
      if ($null -ne $job.Patience) {
        $args += @('--patience', "$($job.Patience)")
      }

      $oldErrorActionPreference = $ErrorActionPreference
      $ErrorActionPreference = 'Continue'
      & $Python @args 1> $log 2> $stderr
      $code = $LASTEXITCODE
      $ErrorActionPreference = $oldErrorActionPreference

      if ($code -eq 0) {
        Add-StateRow -Name $name -Status 'complete' -ExitCode '0' -Log $log -Stderr $stderr
        Wait-IfPauseRequested -CompletedName $name
      } else {
        Add-StateRow -Name $name -Status 'failed' -ExitCode "$code" -Log $log -Stderr $stderr
        exit $code
      }
    }
  }
} finally {
  Pop-Location
}
