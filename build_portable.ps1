param(
    [switch]$Clean,
    [string]$PythonExe = "",
    [string]$FfmpegDir = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Resolve-Python {
    param([string]$Requested)

    if ($Requested -and (Test-Path $Requested)) {
        return (Resolve-Path $Requested).Path
    }

    $candidates = @(
        (Join-Path $Root "venv\Scripts\python.exe"),
        (Join-Path $Root "scripts\python.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    throw "未找到 Python，可通过 -PythonExe 指定。"
}

function Resolve-ExistingDir {
    param([string[]]$Candidates)

    foreach ($candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    return $null
}

function Copy-Tree {
    param(
        [string]$Source,
        [string]$Destination,
        [string]$Label
    )

    if (-not $Source) {
        Write-Host "[skip] $Label 未找到"
        return $false
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
    Write-Host "[ok] 已收集 $Label -> $Destination"
    return $true
}

$Python = Resolve-Python -Requested $PythonExe
Write-Host "使用 Python: $Python"

$TorchRuntimeRepair = Join-Path $Root "tools\repair_torch_openmp.py"
if (Test-Path $TorchRuntimeRepair) {
    & $Python -m pip install zstandard | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "安装 zstandard 失败，无法修复 torch OpenMP 运行时"
    }
    & $Python $TorchRuntimeRepair
    if ($LASTEXITCODE -ne 0) {
        throw "修复 torch OpenMP 运行时失败"
    }
}

$BuildAssetRoot = Join-Path $Root "build\portable_assets"

if ($Clean) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $Root "build")
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $Root "dist")
}

New-Item -ItemType Directory -Force -Path $BuildAssetRoot | Out-Null

$AudioSeparatorSource = Resolve-ExistingDir @(
    $env:MUSIC_TO_MIDI_BUNDLE_AUDIO_SEPARATOR_DIR,
    (Join-Path $env:USERPROFILE ".music-to-midi\models\audio-separator"),
    (Join-Path $Root "checkpoints\audio-separator")
)
$YourMt3Source = Resolve-ExistingDir @(
    $env:MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR,
    (Join-Path $env:USERPROFILE ".cache\music_ai_models\yourmt3_all"),
    (Join-Path $Root "checkpoints\yourmt3_all")
)
$AriaSource = Resolve-ExistingDir @(
    $env:MUSIC_TO_MIDI_BUNDLE_ARIA_DIR,
    (Join-Path $env:USERPROFILE ".cache\music_ai_models\aria_amt"),
    (Join-Path $Root "checkpoints\aria_amt")
)

$ResolvedFfmpegDir = Resolve-ExistingDir @(
    $FfmpegDir,
    $env:MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR,
    (Join-Path $Root "tools\ffmpeg"),
    (Join-Path $Root "ffmpeg")
)
if (-not $ResolvedFfmpegDir) {
    $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpegCmd) {
        $ResolvedFfmpegDir = Split-Path -Parent $ffmpegCmd.Source
    }
}

$AudioSeparatorBundle = Join-Path $BuildAssetRoot "audio-separator"
$YourMt3Bundle = Join-Path $BuildAssetRoot "yourmt3_all"
$AriaBundle = Join-Path $BuildAssetRoot "aria_amt"
$FfmpegBundle = Join-Path $BuildAssetRoot "ffmpeg"

Copy-Tree -Source $AudioSeparatorSource -Destination $AudioSeparatorBundle -Label "audio-separator 模型" | Out-Null
Copy-Tree -Source $YourMt3Source -Destination $YourMt3Bundle -Label "YourMT3 模型" | Out-Null
Copy-Tree -Source $AriaSource -Destination $AriaBundle -Label "Aria-AMT 模型" | Out-Null

if ($ResolvedFfmpegDir) {
    New-Item -ItemType Directory -Force -Path $FfmpegBundle | Out-Null
    foreach ($name in @("ffmpeg.exe", "ffprobe.exe")) {
        $sourceFile = Join-Path $ResolvedFfmpegDir $name
        if (Test-Path $sourceFile) {
            Copy-Item -Path $sourceFile -Destination (Join-Path $FfmpegBundle $name) -Force
        }
    }
    Write-Host "[ok] 已收集 ffmpeg -> $FfmpegBundle"
} else {
    Write-Host "[warn] 未找到 ffmpeg，将继续构建，但非 WAV 输入将依赖 librosa fallback。"
}

$env:MUSIC_TO_MIDI_BUNDLE_AUDIO_SEPARATOR_DIR = $AudioSeparatorBundle
$env:MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR = $YourMt3Bundle
$env:MUSIC_TO_MIDI_BUNDLE_ARIA_DIR = $AriaBundle
$env:MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR = $FfmpegBundle

try {
    & $Python -m PyInstaller --noconfirm MusicToMidi.spec
} finally {
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_AUDIO_SEPARATOR_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_ARIA_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR -ErrorAction SilentlyContinue
}

$DistDir = Join-Path $Root "dist\MusicToMidi"
if (Test-Path $DistDir) {
    Write-Host ""
    Write-Host "便携版已生成: $DistDir"
    Write-Host "建议分发整个目录，不要只拿单个 exe。"
}
