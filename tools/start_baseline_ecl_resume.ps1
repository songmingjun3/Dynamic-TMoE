$ErrorActionPreference = 'Stop'

$root = 'F:\Dynamic-TMoE'
$script = Join-Path $root 'tools\run_baseline_queue_resume_from_ecl.ps1'
$launcherLog = Join-Path $root 'output\baseline_queue_resume_from_ecl.launcher.log'

New-Item -ItemType Directory -Force -Path (Join-Path $root 'output') | Out-Null

$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
$psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$script`""
$psi.WorkingDirectory = $root
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $false
$psi.RedirectStandardError = $false
$psi.CreateNoWindow = $true

$pathValue = [Environment]::GetEnvironmentVariable('Path', 'Process')
if (-not $pathValue) {
  $pathValue = [Environment]::GetEnvironmentVariable('PATH', 'Process')
}
$removeKeys = @()
foreach ($key in $psi.Environment.Keys) {
  if ($key -ieq 'Path') {
    $removeKeys += $key
  }
}
foreach ($key in $removeKeys) {
  $psi.Environment.Remove($key) | Out-Null
}
if ($pathValue) {
  $psi.Environment['Path'] = $pathValue
}

Add-Content -LiteralPath $launcherLog -Value ("launching {0}" -f (Get-Date).ToString('s'))
$process = [System.Diagnostics.Process]::Start($psi)
if ($process) {
  Add-Content -LiteralPath $launcherLog -Value ("pid={0}" -f $process.Id)
  "pid=$($process.Id)"
} else {
  Add-Content -LiteralPath $launcherLog -Value 'started'
  'started'
}
