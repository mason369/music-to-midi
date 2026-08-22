param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$repoRoot = Split-Path -Parent $PSScriptRoot
$sourcePath = Join-Path $repoRoot "tools\universal_launcher\UniversalLauncher.cs"
$iconPath = Join-Path $repoRoot "resources\icons\app.ico"
$outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)

if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
    throw "Universal launcher source is missing: $sourcePath"
}
if (-not (Test-Path -LiteralPath $iconPath -PathType Leaf)) {
    throw "Universal launcher icon is missing: $iconPath"
}

$cscCandidates = @(
    (Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"),
    (Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe")
)
$cscPath = $cscCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1
if (-not $cscPath) {
    throw "Windows C# compiler was not found in the .NET Framework runtime."
}

[System.IO.Directory]::CreateDirectory($outputRoot) | Out-Null
$appLauncher = Join-Path $outputRoot "MusicToMidi.exe"
$backendLauncher = Join-Path $outputRoot "MusicToMidiBackend.exe"

foreach ($path in @($appLauncher, $backendLauncher)) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Force
    }
}

& $cscPath `
    /nologo `
    /optimize+ `
    /codepage:65001 `
    /platform:x64 `
    /target:winexe `
    /define:APP_LAUNCHER `
    "/win32icon:$iconPath" `
    /reference:System.Management.dll `
    /reference:System.Windows.Forms.dll `
    "/out:$appLauncher" `
    $sourcePath
if ($LASTEXITCODE -ne 0) {
    throw "Universal App launcher compilation failed with exit code $LASTEXITCODE."
}

& $cscPath `
    /nologo `
    /optimize+ `
    /codepage:65001 `
    /platform:x64 `
    /target:exe `
    /define:BACKEND_LAUNCHER `
    "/win32icon:$iconPath" `
    /reference:System.Management.dll `
    /reference:System.Windows.Forms.dll `
    "/out:$backendLauncher" `
    $sourcePath
if ($LASTEXITCODE -ne 0) {
    throw "Universal Web backend launcher compilation failed with exit code $LASTEXITCODE."
}

foreach ($path in @($appLauncher, $backendLauncher)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Universal launcher was not produced: $path"
    }
    if ((Get-Item -LiteralPath $path).Length -le 0) {
        throw "Universal launcher is empty: $path"
    }
}

Write-Host "[ok] Universal App launcher: $appLauncher"
Write-Host "[ok] Universal Web backend launcher: $backendLauncher"
