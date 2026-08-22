param(
    [switch]$Clean,
    [string]$PythonExe = "",
    [string]$FfmpegDir = "",
    [ValidateSet("cuda", "xpu")]
    [string]$Accelerator = "cuda",
    [string]$BuildRoot = "",
    [string]$DistRoot = "",
    [switch]$HardlinkAssetStaging
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Accelerator = $Accelerator.ToLowerInvariant()
$ResolvedBuildRoot = [System.IO.Path]::GetFullPath(
    $(if ([string]::IsNullOrWhiteSpace($BuildRoot)) { Join-Path $Root "build" } else { $BuildRoot })
)
$ResolvedDistRoot = [System.IO.Path]::GetFullPath(
    $(if ([string]::IsNullOrWhiteSpace($DistRoot)) { Join-Path $Root "dist" } else { $DistRoot })
)
$DefaultVenv = if ($Accelerator -eq "xpu") { "venv-xpu" } else { "venv" }
$Python = if ($PythonExe) { $PythonExe } else { Join-Path $Root "$DefaultVenv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python is missing: $Python"
}

$PortableArgs = @{
    Accelerator = $Accelerator
    BuildRoot = $ResolvedBuildRoot
    DistRoot = $ResolvedDistRoot
}
if ($Clean) {
    $PortableArgs["Clean"] = $true
}
if ($PythonExe) {
    $PortableArgs["PythonExe"] = $PythonExe
}
if ($FfmpegDir) {
    $PortableArgs["FfmpegDir"] = $FfmpegDir
}
if ($HardlinkAssetStaging) {
    $PortableArgs["HardlinkAssetStaging"] = $true
}

& (Join-Path $Root "build_portable.ps1") @PortableArgs

$FrontendWorkPath = Join-Path $ResolvedBuildRoot "pyinstaller-web-frontend"
$PyInstallerArgs = @(
    "-m", "PyInstaller", "--noconfirm",
    "--workpath", $FrontendWorkPath,
    "--distpath", $ResolvedDistRoot
)
if ($Clean) {
    $PyInstallerArgs += "--clean"
}
& $Python @PyInstallerArgs "MusicToMidiFrontend.spec"
if ($LASTEXITCODE -ne 0) {
    throw "Frontend build failed with exit code $LASTEXITCODE"
}

$AppName = if ($Accelerator -eq "xpu") { "MusicToMidi-XPU-App" } else { "MusicToMidi-App" }
$BackendName = if ($Accelerator -eq "xpu") { "MusicToMidi-XPU-WebBackend" } else { "MusicToMidi-WebBackend" }
$GuiExecutableName = if ($Accelerator -eq "xpu") { "MusicToMidiXpu.exe" } else { "MusicToMidi.exe" }
$BackendExecutableName = if ($Accelerator -eq "xpu") { "MusicToMidiBackendXpu.exe" } else { "MusicToMidiBackend.exe" }
$AppRoot = Join-Path $ResolvedDistRoot $AppName
$BackendRoot = Join-Path $ResolvedDistRoot $BackendName
$FrontendSource = Join-Path $ResolvedDistRoot "MusicToMidiFrontend"
$FrontendRoot = Join-Path $ResolvedDistRoot "MusicToMidi-WebFrontend"

$expectedDistPrefix = $ResolvedDistRoot.TrimEnd("\") + "\"
foreach ($path in @($AppRoot, $BackendRoot, $FrontendSource, $FrontendRoot)) {
    $fullPath = [System.IO.Path]::GetFullPath($path)
    if (-not $fullPath.StartsWith($expectedDistPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Portable output escaped DistRoot: $fullPath"
    }
}
if (Test-Path -LiteralPath $FrontendRoot) {
    Remove-Item -LiteralPath $FrontendRoot -Recurse -Force
}
Move-Item -LiteralPath $FrontendSource -Destination $FrontendRoot

$AppExe = Join-Path $AppRoot $GuiExecutableName
$BackendExe = Join-Path $BackendRoot $BackendExecutableName
$FrontendExe = Join-Path $FrontendRoot "MusicToMidiFrontend.exe"
foreach ($Executable in @($AppExe, $BackendExe, $FrontendExe)) {
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        throw "Expected executable was not produced: $Executable"
    }
    if ((Get-Item -LiteralPath $Executable).Length -le 0) {
        throw "Built executable is empty: $Executable"
    }
}
if (Test-Path -LiteralPath (Join-Path $AppRoot $BackendExecutableName)) {
    throw "Desktop App package must not contain the Web backend executable."
}
if (Test-Path -LiteralPath (Join-Path $BackendRoot $GuiExecutableName)) {
    throw "Web backend package must not contain the desktop App executable."
}

Write-Host "Desktop App: $AppExe" -ForegroundColor Green
Write-Host "Web backend: $BackendExe" -ForegroundColor Green
Write-Host "Web frontend: $FrontendExe" -ForegroundColor Green
