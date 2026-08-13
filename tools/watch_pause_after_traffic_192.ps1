$ErrorActionPreference = 'Stop'

$Root = 'F:\Dynamic-TMoE'
$OutputDir = Join-Path $Root 'output'
$State = Join-Path $OutputDir 'baseline_queue_state.csv'
$Flag = Join-Path $OutputDir 'pause_after_traffic_192.flag'
$WatchLog = Join-Path $OutputDir 'pause_after_traffic_192_watchdog.log'
$QueuePid = 15728

function Add-WatchLog {
  param([string]$Message)
  Add-Content -LiteralPath $WatchLog -Value ("{0} {1}" -f (Get-Date).ToString('s'), $Message)
}

function Get-ChildProcessIds {
  param([int]$ParentPid)
  try {
    @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $ParentPid" | ForEach-Object { [int]$_.ProcessId })
  } catch {
    @()
  }
}

Add-WatchLog "watching queue pid=$QueuePid"

while (Test-Path $Flag) {
  if (Test-Path $State) {
    $tail = Get-Content -LiteralPath $State -Tail 80
    $traffic192Complete = [bool]($tail | Where-Object { $_ -match ',PatchTST_Traffic_192,complete,0,' })
    if ($traffic192Complete) {
      Add-WatchLog 'Traffic_192 complete detected; stopping queue scheduler before remaining tasks.'
      $children = Get-ChildProcessIds -ParentPid $QueuePid
      foreach ($childPid in $children) {
        try {
          $child = Get-Process -Id $childPid -ErrorAction Stop
          Add-WatchLog "stopping child pid=$childPid name=$($child.ProcessName)"
          Stop-Process -Id $childPid -Force -ErrorAction Stop
        } catch {
          Add-WatchLog "child pid=$childPid already exited or could not be stopped: $($_.Exception.Message)"
        }
      }
      try {
        $queue = Get-Process -Id $QueuePid -ErrorAction Stop
        Add-WatchLog "stopping queue pid=$QueuePid name=$($queue.ProcessName)"
        Stop-Process -Id $QueuePid -Force -ErrorAction Stop
      } catch {
        Add-WatchLog "queue pid=$QueuePid already exited or could not be stopped: $($_.Exception.Message)"
      }
      break
    }
  }
  Start-Sleep -Seconds 2
}

Add-WatchLog 'watchdog done'
