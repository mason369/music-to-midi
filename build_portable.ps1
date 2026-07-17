param(
    [switch]$Clean,
    [string]$PythonExe = "",
    [string]$FfmpegDir = ""
)

$ErrorActionPreference = "Stop"

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING = "utf-8"

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
        [string]$Label,
        [switch]$Required
    )

    if (-not $Source) {
        if ($Required) {
            throw "Required asset missing: $Label"
        }
        Write-Host "[warn] $Label not found"
        return $false
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
    Write-Host "[ok] Collected $Label -> $Destination"
    return $true
}

function Assert-SixStemAssets {
    param(
        [string]$ModelDir,
        [string]$PythonPath,
        [string]$Label
    )

    if ([string]::IsNullOrWhiteSpace($ModelDir)) {
        throw "Required BS-RoFormer SW Fixed six-stem assets missing: $Label directory was not resolved."
    }

    & $PythonPath (Join-Path $Root "download_multistem_model.py") --cache-dir $ModelDir --check-only
    if ($LASTEXITCODE -ne 0) {
        throw "Invalid BS-RoFormer SW Fixed six-stem assets in ${Label}: $ModelDir"
    }
}

function Assert-PortableModelIdentities {
    param(
        [string]$AudioSeparatorDir,
        [string]$YourMt3Dir,
        [string]$YourMt3SourceDir,
        [string]$AriaAmtDir,
        [string]$ByteDancePianoDir,
        [string]$MirosDir,
        [string]$PythonPath,
        [string]$Label
    )

    $requiredDirectories = [ordered]@{
        "audio-separator" = $AudioSeparatorDir
        "YourMT3" = $YourMt3Dir
        "patched YourMT3 source" = $YourMt3SourceDir
        "Aria-AMT" = $AriaAmtDir
        "ByteDance Piano" = $ByteDancePianoDir
        "MIROS" = $MirosDir
    }
    foreach ($entry in $requiredDirectories.GetEnumerator()) {
        if ([string]::IsNullOrWhiteSpace([string]$entry.Value)) {
            throw "Required $($entry.Key) directory was not resolved for ${Label}."
        }
    }

    $validator = Join-Path $Root "tools\validate_portable_model_assets.py"
    if (-not (Test-Path -LiteralPath $validator -PathType Leaf)) {
        throw "Portable model identity validator is missing: $validator"
    }

    & $PythonPath $validator `
        --audio-separator-dir $AudioSeparatorDir `
        --yourmt3-dir $YourMt3Dir `
        --yourmt3-source-dir $YourMt3SourceDir `
        --aria-amt-dir $AriaAmtDir `
        --bytedance-piano-dir $ByteDancePianoDir `
        --miros-dir $MirosDir `
        --label $Label
    if ($LASTEXITCODE -ne 0) {
        throw "Pinned portable model identity validation failed for ${Label}."
    }
}

function Remove-PathIfExists {
    param(
        [string]$Path,
        [string]$Label
    )

    if (-not (Test-Path $Path)) {
        return
    }

    Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop

    if (Test-Path $Path) {
        throw "Failed to remove $Label at $Path."
    }
}

function Assert-CudaEnabledTorchRuntime {
    param([string]$PythonPath)

    $checkScript = @'
import sys

try:
    import torch
except Exception as exc:
    print(f"Failed to import torch: {exc}", file=sys.stderr)
    sys.exit(2)

cuda_version = torch.version.cuda
torch_version = getattr(torch, "__version__", "unknown")
if not cuda_version:
    print(f"CPU-only PyTorch runtime detected: torch={torch_version}", file=sys.stderr)
    sys.exit(3)

def parse_version(value):
    parts = []
    for part in str(value).split("+", 1)[0].split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts)

torch_tuple = parse_version(torch_version)
cuda_tuple = parse_version(cuda_version)
if torch_tuple < (2, 7, 0) or cuda_tuple < (12, 8):
    print(
        "Unsupported PyTorch/CUDA runtime for GPU portable build: "
        f"torch={torch_version}, cuda={cuda_version}. "
        "Use torch 2.7.0+ built with CUDA 12.8+ for RTX 50-series (sm_120).",
        file=sys.stderr,
    )
    sys.exit(4)

print(f"CUDA-enabled PyTorch runtime detected: torch={torch_version}, cuda={cuda_version}")
'@

    $output = $checkScript | & $PythonPath - 2>&1
    $exitCode = $LASTEXITCODE
    if ($output) {
        $output | ForEach-Object { Write-Host $_ }
    }
    if ($exitCode -ne 0) {
        throw "GPU portable build requires PyTorch 2.7.0+ with CUDA 12.8+. Install torch/torchaudio/torchvision from https://download.pytorch.org/whl/cu128. CPU-only or older CUDA runtimes are not allowed."
    }
}

$Python = Resolve-Python -Requested $PythonExe
Write-Host "Using Python: $Python"
Assert-CudaEnabledTorchRuntime -PythonPath $Python

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
$YourMt3CodeSource = Join-Path $Root "YourMT3\amt\src"

if ($Clean) {
    Remove-PathIfExists -Path (Join-Path $Root "build") -Label "build directory"
    Remove-PathIfExists -Path (Join-Path $Root "dist") -Label "dist directory"
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
$AriaAmtSource = Resolve-ExistingDir @(
    $env:MUSIC_TO_MIDI_BUNDLE_ARIA_AMT_DIR,
    $env:MUSIC_TO_MIDI_BUNDLE_ARIA_DIR,
    (Join-Path $env:USERPROFILE ".cache\music_ai_models\aria_amt"),
    (Join-Path $Root "checkpoints\aria_amt")
)
$ByteDancePianoSource = Resolve-ExistingDir @(
    $env:MUSIC_TO_MIDI_BUNDLE_BYTEDANCE_PIANO_DIR,
    (Join-Path $env:USERPROFILE ".cache\music_ai_models\bytedance_piano"),
    (Join-Path $Root "checkpoints\bytedance_piano")
)
$TransKunV2AugSource = Resolve-ExistingDir @(
    $env:MUSIC_TO_MIDI_BUNDLE_TRANSKUN_V2_AUG_DIR,
    (Join-Path $env:USERPROFILE ".cache\music_ai_models\transkun_v2_aug"),
    (Join-Path $Root "checkpoints\transkun_v2_aug")
)
$MirosSource = Resolve-ExistingDir @(
    $env:MUSIC_TO_MIDI_BUNDLE_MIROS_DIR,
    (Join-Path $Root "external\ai4m-miros"),
    (Join-Path $Root "ai4m-miros"),
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
$AriaAmtBundle = Join-Path $BuildAssetRoot "aria_amt"
$ByteDancePianoBundle = Join-Path $BuildAssetRoot "bytedance_piano"
$TransKunV2AugBundle = Join-Path $BuildAssetRoot "transkun_v2_aug"
$MirosBundle = Join-Path $BuildAssetRoot "ai4m-miros"
$FfmpegBundle = Join-Path $BuildAssetRoot "ffmpeg"

Assert-PortableModelIdentities `
    -AudioSeparatorDir $AudioSeparatorSource `
    -YourMt3Dir $YourMt3Source `
    -YourMt3SourceDir $YourMt3CodeSource `
    -AriaAmtDir $AriaAmtSource `
    -ByteDancePianoDir $ByteDancePianoSource `
    -MirosDir $MirosSource `
    -PythonPath $Python `
    -Label "portable source assets"
Assert-SixStemAssets -ModelDir $AudioSeparatorSource -PythonPath $Python -Label "audio-separator source"
Copy-Tree -Source $AudioSeparatorSource -Destination $AudioSeparatorBundle -Label "audio-separator models" -Required | Out-Null
Assert-SixStemAssets -ModelDir $AudioSeparatorBundle -PythonPath $Python -Label "audio-separator bundle"
Copy-Tree -Source $YourMt3Source -Destination $YourMt3Bundle -Label "YourMT3 models" -Required | Out-Null
Copy-Tree -Source $AriaAmtSource -Destination $AriaAmtBundle -Label "Aria-AMT models" -Required | Out-Null
Copy-Tree -Source $ByteDancePianoSource -Destination $ByteDancePianoBundle -Label "ByteDance Piano models" -Required | Out-Null
Copy-Tree -Source $TransKunV2AugSource -Destination $TransKunV2AugBundle -Label "TransKun V2 Aug models" -Required | Out-Null
$transkunV2AugCheck = @"
from pathlib import Path
from src.core.transkun_v2_aug_transcriber import (
    TRANSKUN_V2_AUG_MODEL_DIR_NAME,
    validate_transkun_v2_aug_model_files,
)
model_dir = Path(r'$TransKunV2AugBundle') / TRANSKUN_V2_AUG_MODEL_DIR_NAME
reason = validate_transkun_v2_aug_model_files(model_dir)
if reason:
    raise RuntimeError(reason)
print(f'TransKun V2 Aug assets verified: {model_dir}')
"@
& $Python -c $transkunV2AugCheck
if ($LASTEXITCODE -ne 0) {
    throw "Invalid TransKun V2 Aug assets in portable bundle: $TransKunV2AugBundle"
}
Copy-Tree -Source $MirosSource -Destination $MirosBundle -Label "ai4m-miros source" -Required | Out-Null
Assert-PortableModelIdentities `
    -AudioSeparatorDir $AudioSeparatorBundle `
    -YourMt3Dir $YourMt3Bundle `
    -YourMt3SourceDir $YourMt3CodeSource `
    -AriaAmtDir $AriaAmtBundle `
    -ByteDancePianoDir $ByteDancePianoBundle `
    -MirosDir $MirosBundle `
    -PythonPath $Python `
    -Label "staged portable model assets"

if (-not $ResolvedFfmpegDir) {
    throw "FFmpeg bundle source not found. Portable builds require ffmpeg.exe and ffprobe.exe for MP3/FLAC/M4A input."
}

$FfmpegBundleBin = Join-Path $FfmpegBundle "bin"
New-Item -ItemType Directory -Force -Path $FfmpegBundleBin | Out-Null
foreach ($name in @("ffmpeg.exe", "ffprobe.exe")) {
    $sourceFile = Join-Path $ResolvedFfmpegDir $name
    if (-not (Test-Path -LiteralPath $sourceFile -PathType Leaf)) {
        throw "Required FFmpeg executable is missing: $sourceFile"
    }
    $destinationFile = Join-Path $FfmpegBundleBin $name
    Copy-Item -LiteralPath $sourceFile -Destination $destinationFile -Force
    if (-not (Test-Path -LiteralPath $destinationFile -PathType Leaf)) {
        throw "Failed to stage required FFmpeg executable: $destinationFile"
    }
}
Write-Host "[ok] Collected ffmpeg and ffprobe -> $FfmpegBundleBin"

$env:MUSIC_TO_MIDI_BUNDLE_AUDIO_SEPARATOR_DIR = $AudioSeparatorBundle
$env:MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR = $YourMt3Bundle
$env:MUSIC_TO_MIDI_BUNDLE_ARIA_AMT_DIR = $AriaAmtBundle
$env:MUSIC_TO_MIDI_BUNDLE_BYTEDANCE_PIANO_DIR = $ByteDancePianoBundle
$env:MUSIC_TO_MIDI_BUNDLE_TRANSKUN_V2_AUG_DIR = $TransKunV2AugBundle
$env:MUSIC_TO_MIDI_BUNDLE_MIROS_DIR = $MirosBundle
$env:MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR = $FfmpegBundle

$PyInstallerExitCode = 0
try {
    & $Python -m PyInstaller --noconfirm MusicToMidi.spec
    $PyInstallerExitCode = $LASTEXITCODE
} finally {
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_AUDIO_SEPARATOR_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_ARIA_AMT_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_BYTEDANCE_PIANO_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_TRANSKUN_V2_AUG_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_MIROS_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR -ErrorAction SilentlyContinue
}

if ($PyInstallerExitCode -ne 0) {
    throw "PyInstaller build failed with exit code $PyInstallerExitCode."
}

$DistDir = Join-Path $Root "dist\MusicToMidi"
if (Test-Path $DistDir) {
    foreach ($noticeName in @("LICENSE", "THIRD_PARTY_NOTICES.md")) {
        $noticeSource = Join-Path $Root $noticeName
        if (-not (Test-Path -LiteralPath $noticeSource -PathType Leaf)) {
            throw "Required distribution notice is missing: $noticeSource"
        }
        Copy-Item -LiteralPath $noticeSource -Destination (Join-Path $DistDir $noticeName) -Force
    }
    Write-Host ""
    Write-Host "Portable build created: $DistDir"
    Write-Host "Distribute the entire directory instead of a single exe."
}
