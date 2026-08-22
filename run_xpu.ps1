# Music to MIDI - isolated Intel XPU launcher
#Requires -Version 5.1

$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "run.ps1"
& powershell -NoProfile -ExecutionPolicy Bypass -File $launcher `
    -Accelerator xpu -VenvName venv-xpu @args
exit $LASTEXITCODE
