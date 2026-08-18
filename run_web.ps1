param(
    [string]$HostAddress = "127.0.0.1",
    [int]$ApiPort = 8765,
    [int]$WebPort = 5173,
    [string]$PublicHost = "",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $repoRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python is missing: $python. Run install.ps1 first."
}

$wildcardBind = $HostAddress -in @("0.0.0.0", "::")
if ($wildcardBind -and [string]::IsNullOrWhiteSpace($PublicHost)) {
    throw "-PublicHost is required when -HostAddress is a wildcard bind address."
}
$clientHost = if ([string]::IsNullOrWhiteSpace($PublicHost)) { $HostAddress } else { $PublicHost.Trim() }
$frontendOrigin = "http://${clientHost}:$WebPort"
$originalAllowedOrigins = $env:MUSIC_TO_MIDI_ALLOWED_ORIGINS
$configuredOrigins = @($originalAllowedOrigins -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$env:MUSIC_TO_MIDI_ALLOWED_ORIGINS = (@($configuredOrigins + $frontendOrigin) | Select-Object -Unique) -join ","

$logDir = Join-Path $repoRoot "MidiOutput\WebAPI\server-logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$backendOut = Join-Path $logDir "backend.stdout.log"
$backendErr = Join-Path $logDir "backend.stderr.log"
$frontendOut = Join-Path $logDir "frontend.stdout.log"
$frontendErr = Join-Path $logDir "frontend.stderr.log"

$backend = $null
$frontend = $null
try {
    $backend = Start-Process -FilePath $python -WorkingDirectory $repoRoot -ArgumentList @(
        "-m", "src.web_api", "--host", $HostAddress, "--port", "$ApiPort"
    ) -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr -WindowStyle Hidden -PassThru

    $frontend = Start-Process -FilePath $python -WorkingDirectory $repoRoot -ArgumentList @(
        "-m", "http.server", "$WebPort", "--bind", $HostAddress, "--directory", "web"
    ) -RedirectStandardOutput $frontendOut -RedirectStandardError $frontendErr -WindowStyle Hidden -PassThru

    $healthUrl = "http://${HostAddress}:$ApiPort/api/v1/health"
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($backend.HasExited) {
            throw "Inference backend failed to start (exit=$($backend.ExitCode)). Log: $backendErr"
        }
        if ($frontend.HasExited) {
            throw "Web frontend failed to start (exit=$($frontend.ExitCode)). Log: $frontendErr"
        }
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                $ready = $true
                break
            }
        } catch {
            # Readiness polling is bounded; the final failure below is explicit.
        }
        Start-Sleep -Seconds 1
    }
    if (-not $ready) {
        throw "Inference backend did not pass its 60-second health gate: $healthUrl. Log: $backendErr"
    }

    $apiBase = "http://${clientHost}:$ApiPort"
    $webUrl = "http://${clientHost}:$WebPort/?api=$([System.Uri]::EscapeDataString($apiBase))"
    Write-Host "Frontend ready: $webUrl" -ForegroundColor Cyan
    Write-Host "Backend ready: http://${HostAddress}:$ApiPort/docs" -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop both processes." -ForegroundColor DarkGray
    if (-not $NoBrowser) {
        Start-Process $webUrl
    }
    while (-not $backend.HasExited -and -not $frontend.HasExited) {
        Start-Sleep -Milliseconds 500
    }
    if ($backend.HasExited) {
        throw "Inference backend exited unexpectedly (exit=$($backend.ExitCode)). Log: $backendErr"
    }
    throw "Web frontend exited unexpectedly (exit=$($frontend.ExitCode)). Log: $frontendErr"
} finally {
    foreach ($process in @($frontend, $backend)) {
        if ($null -ne $process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -Force
        }
    }
    $env:MUSIC_TO_MIDI_ALLOWED_ORIGINS = $originalAllowedOrigins
}
