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
. (Join-Path $Root "scripts\powershell_helpers.ps1")

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

    $sourcePath = [IO.Path]::GetFullPath($Source)
    $destinationPath = [IO.Path]::GetFullPath($Destination)
    if ($sourcePath -eq $destinationPath) {
        throw "Refusing to replace $Label because source and destination are identical: $sourcePath"
    }
    if (Test-Path -LiteralPath $destinationPath) {
        Remove-Item -LiteralPath $destinationPath -Recurse -Force -ErrorAction Stop
    }
    New-Item -ItemType Directory -Force -Path $destinationPath | Out-Null
    Copy-Item -Path (Join-Path $sourcePath "*") -Destination $destinationPath -Recurse -Force
    Write-Host "[ok] Collected $Label -> $destinationPath"
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
        [string]$BeatThisDir,
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
        "Beat This final0" = $BeatThisDir
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
        --beat-this-dir $BeatThisDir `
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
from importlib import metadata

try:
    import torch
    import torchaudio
    import torchvision
except Exception as exc:
    print(f"Failed to import the pinned PyTorch runtime trio: {exc}", file=sys.stderr)
    sys.exit(2)

cuda_version = torch.version.cuda
torch_version = getattr(torch, "__version__", "unknown")
if not cuda_version:
    print(f"CPU-only PyTorch runtime detected: torch={torch_version}", file=sys.stderr)
    sys.exit(3)

expected_versions = {
    "torch": "2.7.0",
    "torchaudio": "2.7.0",
    "torchvision": "0.22.0",
}
installed_versions = {
    package_name: metadata.version(package_name)
    for package_name in expected_versions
}
version_mismatches = {
    package_name: installed_version
    for package_name, installed_version in installed_versions.items()
    if installed_version.split("+", 1)[0] != expected_versions[package_name]
}
if version_mismatches or cuda_version != "12.8":
    print(
        "Unsupported PyTorch/CUDA runtime for GPU portable build: "
        f"packages={installed_versions}, cuda={cuda_version}. "
        "Use exactly torch 2.7.0, torchaudio 2.7.0, and torchvision 0.22.0 "
        "built with CUDA 12.8.",
        file=sys.stderr,
    )
    sys.exit(4)

print(f"Pinned CUDA PyTorch runtime detected: packages={installed_versions}, cuda={cuda_version}")
'@

    $output = $checkScript | & $PythonPath - 2>&1
    $exitCode = $LASTEXITCODE
    if ($output) {
        $output | ForEach-Object { Write-Host $_ }
    }
    if ($exitCode -ne 0) {
        throw "GPU portable build requires exactly torch 2.7.0, torchaudio 2.7.0, and torchvision 0.22.0 built with CUDA 12.8. Install the pinned trio from https://download.pytorch.org/whl/cu128. CPU-only, mismatched, or differently versioned runtimes are not allowed."
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
$beatThisSourceText = & $Python -c "from src.core.beat_this_tracker import validate_beat_this_checkpoint; print(validate_beat_this_checkpoint().parent)"
if ($LASTEXITCODE -ne 0) {
    throw "Beat This final0 checkpoint is unavailable or failed exact identity validation. Run download_beat_this_model.py."
}
$BeatThisSource = Resolve-ExistingDir @(
    $env:MUSIC_TO_MIDI_BUNDLE_BEAT_THIS_DIR,
    ($beatThisSourceText | Select-Object -Last 1)
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
$muscriptorModelSourceText = & $Python -c "from src.utils.muscriptor_downloader import get_cached_muscriptor_paths; [print(get_cached_muscriptor_paths(size, validate_hashes=True)[0].parent) for size in ('small', 'medium', 'large')]"
if ($LASTEXITCODE -ne 0) {
    throw "One or more pinned MuScriptor Small/Medium/Large models are unavailable or failed identity validation. Run download_sota_models.py after accepting all three Hugging Face model terms."
}
$MuscriptorSmallSource = Resolve-ExistingDir @(
    $env:MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_SMALL_DIR,
    ($muscriptorModelSourceText | Select-Object -Index 0)
)
$MuscriptorMediumSource = Resolve-ExistingDir @(
    $env:MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_MEDIUM_DIR,
    ($muscriptorModelSourceText | Select-Object -Index 1)
)
$MuscriptorLargeSource = Resolve-ExistingDir @(
    $env:MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_LARGE_DIR,
    $env:MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_DIR,
    ($muscriptorModelSourceText | Select-Object -Index 2)
)
$muscriptorAssetsSourceText = & $Python -c "from src.utils.muscriptor_soundfont_downloader import download_muscriptor_soundfont; print(download_muscriptor_soundfont(printer=lambda _message: None).parent)"
if ($LASTEXITCODE -ne 0) {
    throw "MuScriptor official SoundFont is not available or failed identity validation. Run download_sota_models.py."
}
$MuscriptorAssetsSource = Resolve-ExistingDir @(
    $env:MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_ASSETS_DIR,
    ($muscriptorAssetsSourceText | Select-Object -Last 1)
)
$fluidsynthSourceText = & $Python -c "from src.utils.fluidsynth_runtime import get_fluidsynth_executable; print(get_fluidsynth_executable().parent.parent)"
if ($LASTEXITCODE -ne 0) {
    throw "Pinned FluidSynth runtime is unavailable. Run download_fluidsynth_runtime.py."
}
$FluidSynthSource = Resolve-ExistingDir @(
    $env:MUSIC_TO_MIDI_BUNDLE_FLUIDSYNTH_DIR,
    ($fluidsynthSourceText | Select-Object -Last 1)
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
$BeatThisBundle = Join-Path $BuildAssetRoot "beat_this"
$TransKunV2AugBundle = Join-Path $BuildAssetRoot "transkun_v2_aug"
$MirosBundle = Join-Path $BuildAssetRoot "ai4m-miros"
$MuscriptorSmallBundle = Join-Path $BuildAssetRoot "muscriptor_small"
$MuscriptorMediumBundle = Join-Path $BuildAssetRoot "muscriptor_medium"
$MuscriptorLargeBundle = Join-Path $BuildAssetRoot "muscriptor_large"
$MuscriptorAssetsBundle = Join-Path $BuildAssetRoot "muscriptor_assets"
$FluidSynthBundle = Join-Path $BuildAssetRoot "fluidsynth"
$FfmpegBundle = Join-Path $BuildAssetRoot "ffmpeg"

Assert-PortableModelIdentities `
    -AudioSeparatorDir $AudioSeparatorSource `
    -YourMt3Dir $YourMt3Source `
    -YourMt3SourceDir $YourMt3CodeSource `
    -AriaAmtDir $AriaAmtSource `
    -ByteDancePianoDir $ByteDancePianoSource `
    -BeatThisDir $BeatThisSource `
    -MirosDir $MirosSource `
    -PythonPath $Python `
    -Label "portable source assets"
Assert-SixStemAssets -ModelDir $AudioSeparatorSource -PythonPath $Python -Label "audio-separator source"
Copy-Tree -Source $AudioSeparatorSource -Destination $AudioSeparatorBundle -Label "audio-separator models" -Required | Out-Null
Assert-SixStemAssets -ModelDir $AudioSeparatorBundle -PythonPath $Python -Label "audio-separator bundle"
Copy-Tree -Source $YourMt3Source -Destination $YourMt3Bundle -Label "YourMT3 models" -Required | Out-Null
Copy-Tree -Source $AriaAmtSource -Destination $AriaAmtBundle -Label "Aria-AMT models" -Required | Out-Null
Copy-Tree -Source $ByteDancePianoSource -Destination $ByteDancePianoBundle -Label "ByteDance Piano models" -Required | Out-Null
Copy-Tree -Source $BeatThisSource -Destination $BeatThisBundle -Label "Beat This final0 model" -Required | Out-Null
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
$pythonExitCode = Invoke-PythonScript -PythonExecutable $Python -Script $transkunV2AugCheck
if ($pythonExitCode -ne 0) {
    throw "Invalid TransKun V2 Aug assets in portable bundle: $TransKunV2AugBundle"
}
Copy-Tree -Source $MirosSource -Destination $MirosBundle -Label "ai4m-miros source" -Required | Out-Null
Copy-Tree -Source $MuscriptorSmallSource -Destination $MuscriptorSmallBundle -Label "MuScriptor-small model" -Required | Out-Null
Copy-Tree -Source $MuscriptorMediumSource -Destination $MuscriptorMediumBundle -Label "MuScriptor-medium model" -Required | Out-Null
Copy-Tree -Source $MuscriptorLargeSource -Destination $MuscriptorLargeBundle -Label "MuScriptor-large model" -Required | Out-Null
Copy-Tree -Source $MuscriptorAssetsSource -Destination $MuscriptorAssetsBundle -Label "MuScriptor playback assets" -Required | Out-Null
Copy-Tree -Source $FluidSynthSource -Destination $FluidSynthBundle -Label "FluidSynth runtime" -Required | Out-Null
$muscriptorPortableCheck = @"
from pathlib import Path
from src.utils.artifact_identity import validate_file_identity
from src.utils.muscriptor_downloader import (
    MUSCRIPTOR_CONFIG_FILENAME,
    MUSCRIPTOR_ARTIFACTS,
    MUSCRIPTOR_MODEL_FILENAME,
)
from src.utils.muscriptor_soundfont_downloader import (
    MUSCRIPTOR_SF2_EXACT_BYTES,
    MUSCRIPTOR_SF2_FILENAME,
    MUSCRIPTOR_SF2_SHA256,
)

model_dirs = {
    'small': Path(r'$MuscriptorSmallBundle'),
    'medium': Path(r'$MuscriptorMediumBundle'),
    'large': Path(r'$MuscriptorLargeBundle'),
}
assets_dir = Path(r'$MuscriptorAssetsBundle')
for model_size, model_dir in model_dirs.items():
    artifact = MUSCRIPTOR_ARTIFACTS[model_size]
    validate_file_identity(model_dir / MUSCRIPTOR_MODEL_FILENAME, expected_size=artifact.model_bytes, expected_sha256=artifact.model_sha256, label=f'staged MuScriptor-{model_size} model')
    validate_file_identity(model_dir / MUSCRIPTOR_CONFIG_FILENAME, expected_size=artifact.config_bytes, expected_sha256=artifact.config_sha256, label=f'staged MuScriptor-{model_size} config')
validate_file_identity(assets_dir / MUSCRIPTOR_SF2_FILENAME, expected_size=MUSCRIPTOR_SF2_EXACT_BYTES, expected_sha256=MUSCRIPTOR_SF2_SHA256, label='staged MuScriptor SoundFont')
print('MuScriptor Small/Medium/Large portable assets verified')
"@
$pythonExitCode = Invoke-PythonScript -PythonExecutable $Python -Script $muscriptorPortableCheck
if ($pythonExitCode -ne 0) {
    throw "MuScriptor portable assets failed exact identity validation."
}
& (Join-Path $FluidSynthBundle "bin\fluidsynth.exe") --version
if ($LASTEXITCODE -ne 0) {
    throw "Staged FluidSynth runtime failed to execute."
}
Assert-PortableModelIdentities `
    -AudioSeparatorDir $AudioSeparatorBundle `
    -YourMt3Dir $YourMt3Bundle `
    -YourMt3SourceDir $YourMt3CodeSource `
    -AriaAmtDir $AriaAmtBundle `
    -ByteDancePianoDir $ByteDancePianoBundle `
    -BeatThisDir $BeatThisBundle `
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
$env:MUSIC_TO_MIDI_BUNDLE_BEAT_THIS_DIR = $BeatThisBundle
$env:MUSIC_TO_MIDI_BUNDLE_TRANSKUN_V2_AUG_DIR = $TransKunV2AugBundle
$env:MUSIC_TO_MIDI_BUNDLE_MIROS_DIR = $MirosBundle
$env:MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_SMALL_DIR = $MuscriptorSmallBundle
$env:MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_MEDIUM_DIR = $MuscriptorMediumBundle
$env:MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_LARGE_DIR = $MuscriptorLargeBundle
$env:MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_DIR = $MuscriptorLargeBundle
$env:MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_ASSETS_DIR = $MuscriptorAssetsBundle
$env:MUSIC_TO_MIDI_BUNDLE_FLUIDSYNTH_DIR = $FluidSynthBundle
$env:MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR = $FfmpegBundle
$PyInstallerBootstrapDir = Join-Path $Root "tools\pyinstaller_bootstrap"
$PyInstallerBootstrap = Join-Path $PyInstallerBootstrapDir "sitecustomize.py"
if (-not (Test-Path -LiteralPath $PyInstallerBootstrap -PathType Leaf)) {
    throw "PyInstaller VC runtime bootstrap is missing: $PyInstallerBootstrap"
}
$BuildVcRuntimeDir = Join-Path $env:WINDIR "System32"
foreach ($name in @("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")) {
    $runtimePath = Join-Path $BuildVcRuntimeDir $name
    if (-not (Test-Path -LiteralPath $runtimePath -PathType Leaf)) {
        throw "Required PyInstaller build runtime is missing: $runtimePath"
    }
}
$OriginalPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($OriginalPythonPath)) {
    $PyInstallerBootstrapDir
} else {
    $PyInstallerBootstrapDir + [IO.Path]::PathSeparator + $OriginalPythonPath
}
$env:MUSIC_TO_MIDI_BUILD_VC_RUNTIME_DIR = $BuildVcRuntimeDir

$PyInstallerExitCode = 0
try {
    & $Python -m PyInstaller --noconfirm MusicToMidi.spec
    $PyInstallerExitCode = $LASTEXITCODE
} finally {
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_AUDIO_SEPARATOR_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_ARIA_AMT_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_BYTEDANCE_PIANO_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_BEAT_THIS_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_TRANSKUN_V2_AUG_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_MIROS_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_SMALL_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_MEDIUM_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_LARGE_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_ASSETS_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_FLUIDSYNTH_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\MUSIC_TO_MIDI_BUILD_VC_RUNTIME_DIR -ErrorAction SilentlyContinue
    if ([string]::IsNullOrWhiteSpace($OriginalPythonPath)) {
        Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $OriginalPythonPath
    }
}

if ($PyInstallerExitCode -ne 0) {
    throw "PyInstaller build failed with exit code $PyInstallerExitCode."
}

$DistDir = Join-Path $Root "dist\MusicToMidi"
if (-not (Test-Path -LiteralPath $DistDir -PathType Container)) {
    throw "PyInstaller reported success but the portable directory is missing: $DistDir"
}

$PortableExe = Join-Path $DistDir "MusicToMidi.exe"
if (-not (Test-Path -LiteralPath $PortableExe -PathType Leaf)) {
    throw "Portable executable is missing: $PortableExe"
}
$GuiRuntimeSelfTest = Start-Process `
    -FilePath $PortableExe `
    -ArgumentList "--self-test-gui-runtime" `
    -PassThru `
    -WindowStyle Hidden
if (-not $GuiRuntimeSelfTest.WaitForExit(120000)) {
    Stop-Process -Id $GuiRuntimeSelfTest.Id -Force -ErrorAction SilentlyContinue
    throw "Portable GUI/Qt/ONNX Runtime self-test timed out after 120 seconds."
}
$GuiRuntimeSelfTest.Refresh()
if ($GuiRuntimeSelfTest.ExitCode -ne 0) {
    throw "Portable GUI/Qt/ONNX Runtime self-test failed with exit code $($GuiRuntimeSelfTest.ExitCode)."
}
Write-Host "[ok] Portable GUI + Qt + ONNX Runtime CUDA load order verified"

$muscriptorDistCheck = @"
from pathlib import Path
from src.utils.artifact_identity import validate_file_identity
from src.core.beat_this_tracker import (
    BEAT_THIS_CHECKPOINT_NAME,
    validate_beat_this_checkpoint,
)
from src.utils.muscriptor_downloader import (
    MUSCRIPTOR_ARTIFACTS,
    MUSCRIPTOR_CONFIG_FILENAME,
    MUSCRIPTOR_MODEL_FILENAME,
)

models_root = Path(r'$DistDir') / '_internal' / 'models'
for model_size, artifact in MUSCRIPTOR_ARTIFACTS.items():
    model_dir = models_root / f'muscriptor_{model_size}'
    validate_file_identity(model_dir / MUSCRIPTOR_MODEL_FILENAME, expected_size=artifact.model_bytes, expected_sha256=artifact.model_sha256, label=f'packaged MuScriptor-{model_size} model')
    validate_file_identity(model_dir / MUSCRIPTOR_CONFIG_FILENAME, expected_size=artifact.config_bytes, expected_sha256=artifact.config_sha256, label=f'packaged MuScriptor-{model_size} config')
print('Packaged MuScriptor Small/Medium/Large assets verified')
validate_beat_this_checkpoint(
    Path(r'$DistDir') / '_internal' / 'models' / 'beat_this' / BEAT_THIS_CHECKPOINT_NAME
)
print('Packaged Beat This final0 asset verified')
"@
$pythonExitCode = Invoke-PythonScript -PythonExecutable $Python -Script $muscriptorDistCheck
if ($pythonExitCode -ne 0) {
    throw "Packaged MuScriptor Small/Medium/Large assets failed exact identity validation."
}

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
