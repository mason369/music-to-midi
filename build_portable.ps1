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

    throw "Python executable not found. Use -PythonExe to specify it."
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

function Test-WorkingExecutable {
    param([string]$ExecutablePath)

    if ([string]::IsNullOrWhiteSpace($ExecutablePath) -or -not (Test-Path $ExecutablePath)) {
        return $false
    }

    try {
        $output = & $ExecutablePath -version 2>&1
        if ($LASTEXITCODE -ne 0) {
            return $false
        }

        $firstLine = ""
        if ($output -is [System.Array]) {
            if ($output.Count -gt 0) {
                $firstLine = [string]$output[0]
            }
        } else {
            $firstLine = [string]$output
        }

        return $firstLine -like "ffmpeg version*"
    } catch {
        return $false
    }
}

function Resolve-FFmpegBinDir {
    param([string[]]$Candidates)

    foreach ($candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate) -or -not (Test-Path $candidate)) {
            continue
        }

        $resolved = (Resolve-Path $candidate).Path
        $item = Get-Item $resolved

        if (-not $item.PSIsContainer) {
            $fileDescription = [string]$item.VersionInfo.FileDescription
            if ($fileDescription -like "*Chocolatey Shim*") {
                $shimParentDir = Split-Path -Parent $resolved
                $shimRootDir = Split-Path -Parent $shimParentDir
                $shimTargetDir = Join-Path $shimRootDir "lib\ffmpeg\tools\ffmpeg\bin"
                if (Test-Path (Join-Path $shimTargetDir "ffmpeg.exe")) {
                    return $shimTargetDir
                }
            }
        }

        $dirsToCheck = @()

        if ($item.PSIsContainer) {
            $dirsToCheck += $resolved
            $dirsToCheck += (Join-Path $resolved "bin")
        } else {
            $dirsToCheck += (Split-Path -Parent $resolved)
        }

        foreach ($dir in ($dirsToCheck | Select-Object -Unique)) {
            if (Test-WorkingExecutable (Join-Path $dir "ffmpeg.exe")) {
                return $dir
            }
        }

        $parentDir = Split-Path -Parent $resolved
        if ($parentDir) {
            $rootDir = Split-Path -Parent $parentDir
            if ($rootDir) {
                $chocoDir = Join-Path $rootDir "lib\ffmpeg\tools\ffmpeg\bin"
                if (Test-WorkingExecutable (Join-Path $chocoDir "ffmpeg.exe")) {
                    return $chocoDir
                }
            }
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
        Write-Host "[skip] $Label not found"
        return $false
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
    Write-Host "[ok] Collected $Label -> $Destination"
    return $true
}

$Python = Resolve-Python -Requested $PythonExe
Write-Host "Using Python: $Python"

$TorchRuntimeRepair = Join-Path $Root "tools\repair_torch_openmp.py"
if (Test-Path $TorchRuntimeRepair) {
    & $Python -m pip install zstandard | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install zstandard; cannot repair torch OpenMP runtime."
    }
    & $Python $TorchRuntimeRepair
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to repair torch OpenMP runtime."
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
$MirosSource = Resolve-ExistingDir @(
    $env:MUSIC_TO_MIDI_BUNDLE_MIROS_DIR,
    (Join-Path $Root ".tmp\ai4m-miros")
)

$ResolvedFfmpegDir = Resolve-FFmpegBinDir @(
    $FfmpegDir,
    $env:MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR,
    (Join-Path $Root "tools\ffmpeg\bin"),
    (Join-Path $Root "tools\ffmpeg"),
    (Join-Path $Root "ffmpeg\bin"),
    (Join-Path $Root "ffmpeg")
)
if (-not $ResolvedFfmpegDir) {
    $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpegCmd) {
        $ResolvedFfmpegDir = Resolve-FFmpegBinDir @($ffmpegCmd.Source)
    }
}

$AudioSeparatorBundle = Join-Path $BuildAssetRoot "audio-separator"
$YourMt3Bundle = Join-Path $BuildAssetRoot "yourmt3_all"
$AriaBundle = Join-Path $BuildAssetRoot "aria_amt"
$MirosBundle = Join-Path $BuildAssetRoot "ai4m-miros"
$FfmpegBundle = Join-Path $BuildAssetRoot "ffmpeg"

Copy-Tree -Source $AudioSeparatorSource -Destination $AudioSeparatorBundle -Label "audio-separator models" | Out-Null
Copy-Tree -Source $YourMt3Source -Destination $YourMt3Bundle -Label "YourMT3 models" | Out-Null
Copy-Tree -Source $AriaSource -Destination $AriaBundle -Label "Aria-AMT models" | Out-Null
Copy-Tree -Source $MirosSource -Destination $MirosBundle -Label "ai4m-miros source" | Out-Null

if ($ResolvedFfmpegDir) {
    $FfmpegBundleBin = Join-Path $FfmpegBundle "bin"
    New-Item -ItemType Directory -Force -Path $FfmpegBundleBin | Out-Null
    foreach ($name in @("ffmpeg.exe", "ffprobe.exe")) {
        $sourceFile = Join-Path $ResolvedFfmpegDir $name
        if (Test-Path $sourceFile) {
            Copy-Item -Path $sourceFile -Destination (Join-Path $FfmpegBundleBin $name) -Force
        }
    }
    Write-Host "[ok] Collected ffmpeg -> $FfmpegBundleBin"
} else {
    Write-Host "[warn] ffmpeg not found; build will continue, but non-WAV inputs will rely on librosa fallback."
}

$env:MUSIC_TO_MIDI_BUNDLE_AUDIO_SEPARATOR_DIR = $AudioSeparatorBundle
$env:MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR = $YourMt3Bundle
$env:MUSIC_TO_MIDI_BUNDLE_ARIA_DIR = $AriaBundle
$env:MUSIC_TO_MIDI_BUNDLE_MIROS_DIR = $MirosBundle
$env:MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR = $FfmpegBundle

try {
    & $Python -m PyInstaller --noconfirm MusicToMidi.spec
} finally {
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_AUDIO_SEPARATOR_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_ARIA_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_MIROS_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR -ErrorAction SilentlyContinue
}

$DistDir = Join-Path $Root "dist\MusicToMidi"
if (Test-Path $DistDir) {
    Write-Host ""
    Write-Host "Portable build created: $DistDir"
    Write-Host "Distribute the entire directory instead of a single exe."
}
