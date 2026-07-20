param(
    [Parameter(Mandatory = $true)]
    [string]$FileId,

    [Parameter(Mandatory = $true)]
    [string]$Uuid,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [Parameter(Mandatory = $true)]
    [string]$ProgressPath,

    [long]$ExpectedBytes = 0
)

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Net.Http

$uri = "https://drive.usercontent.google.com/download?id=$FileId&export=download&confirm=t&uuid=$Uuid"
$tempPath = "$OutputPath.part"
$buffer = New-Object byte[] (1024 * 1024)

if (Test-Path -LiteralPath $tempPath) {
    Remove-Item -LiteralPath $tempPath -Force
}

$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromHours(4)
$response = $client.GetAsync($uri, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
$response.EnsureSuccessStatusCode() | Out-Null

$contentLength = $response.Content.Headers.ContentLength
if (-not $contentLength -and $ExpectedBytes -gt 0) {
    $contentLength = $ExpectedBytes
}

$stream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
$file = [System.IO.File]::Open($tempPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)

$downloaded = 0L
$started = Get-Date
$lastReport = Get-Date

try {
    while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
        $file.Write($buffer, 0, $read)
        $downloaded += $read

        $now = Get-Date
        if (($now - $lastReport).TotalSeconds -ge 5) {
            $elapsed = [Math]::Max(($now - $started).TotalSeconds, 0.001)
            $percent = if ($contentLength) { [Math]::Round(($downloaded / $contentLength) * 100, 2) } else { $null }
            $status = [ordered]@{
                status = 'downloading'
                downloaded_bytes = $downloaded
                total_bytes = $contentLength
                percent = $percent
                mb_downloaded = [Math]::Round($downloaded / 1MB, 2)
                mb_total = if ($contentLength) { [Math]::Round($contentLength / 1MB, 2) } else { $null }
                mbps = [Math]::Round(($downloaded / 1MB) / $elapsed, 2)
                updated_at = $now.ToString('s')
            }
            $status | ConvertTo-Json | Set-Content -LiteralPath $ProgressPath -Encoding UTF8
            $lastReport = $now
        }
    }
}
finally {
    $file.Dispose()
    $stream.Dispose()
    $response.Dispose()
    $client.Dispose()
}

Move-Item -LiteralPath $tempPath -Destination $OutputPath -Force

$completed = [ordered]@{
    status = 'complete'
    downloaded_bytes = (Get-Item -LiteralPath $OutputPath).Length
    total_bytes = $contentLength
    percent = 100
    mb_downloaded = [Math]::Round((Get-Item -LiteralPath $OutputPath).Length / 1MB, 2)
    mb_total = if ($contentLength) { [Math]::Round($contentLength / 1MB, 2) } else { $null }
    updated_at = (Get-Date).ToString('s')
}
$completed | ConvertTo-Json | Set-Content -LiteralPath $ProgressPath -Encoding UTF8
