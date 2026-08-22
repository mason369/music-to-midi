# Music to MIDI - isolated Intel XPU installer
#Requires -Version 5.1

$ErrorActionPreference = "Stop"
$installer = Join-Path $PSScriptRoot "install.ps1"
& powershell -NoProfile -ExecutionPolicy Bypass -File $installer `
    -Accelerator xpu -VenvName venv-xpu @args
exit $LASTEXITCODE
