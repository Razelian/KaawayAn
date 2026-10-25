param(
    [string]$Config = "config/site.toml",
    [switch]$NoClassifier
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$dashboard = Start-Process -FilePath "python" -ArgumentList @(
    "-m", "streamlit", "run", "dashboard/dashboard.py"
) -PassThru -NoNewWindow
$ingest = $null
$ingestArgs = @("main.py", "--config", $Config)
if ($NoClassifier) { $ingestArgs += "--no-classifier" }

try {
    while ($true) {
        $ingest = Start-Process -FilePath "python" -ArgumentList $ingestArgs -PassThru -NoNewWindow
        while (-not $ingest.HasExited) {
            if ($dashboard.HasExited) {
                throw "Streamlit dashboard exited with code $($dashboard.ExitCode)"
            }
            Start-Sleep -Seconds 1
            $ingest.Refresh()
        }
        Write-Warning "Ingest exited with code $($ingest.ExitCode); restarting in 2 seconds."
        Start-Sleep -Seconds 2
    }
}
finally {
    if ($ingest -and -not $ingest.HasExited) { Stop-Process -Id $ingest.Id }
    if ($dashboard -and -not $dashboard.HasExited) { Stop-Process -Id $dashboard.Id }
}
