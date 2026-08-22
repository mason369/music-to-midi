# Music to MIDI - isolated Intel XPU portable build
#Requires -Version 5.1

$ErrorActionPreference = "Stop"
$builder = Join-Path $PSScriptRoot "build_web_executables.ps1"
& powershell -NoProfile -ExecutionPolicy Bypass -File $builder `
    -Accelerator xpu -HardlinkAssetStaging @args
exit $LASTEXITCODE
