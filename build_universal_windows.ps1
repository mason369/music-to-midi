param(
    [Parameter(Mandatory = $true)]
    [string]$CudaDistRoot,
    [Parameter(Mandatory = $true)]
    [string]$XpuDistRoot,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [switch]$Clean
)

# Assemble one Windows delivery tree while keeping CUDA and Intel XPU native
# runtimes isolated. Identical files are converted to NTFS hard links, so the
# model/source payload occupies one physical copy without making any role depend
# on a sibling directory.

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

function Get-NormalizedPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return [System.IO.Path]::GetFullPath($Path).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
}

function Assert-SafeOutputRoot {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = Get-NormalizedPath $Path
    if ((Split-Path -Leaf $resolved) -ne "MusicToMidi-Universal") {
        throw "Universal output leaf must be MusicToMidi-Universal: $resolved"
    }
    if ($resolved -eq [System.IO.Path]::GetPathRoot($resolved)) {
        throw "Universal output cannot be a drive root."
    }
    return $resolved
}

function Assert-Role {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Executable
    )

    $resolved = Get-NormalizedPath $Root
    $executablePath = Join-Path $resolved $Executable
    $internalPath = Join-Path $resolved "_internal"
    if (-not (Test-Path -LiteralPath $executablePath -PathType Leaf)) {
        throw "Portable role executable is missing: $executablePath"
    }
    if ((Get-Item -LiteralPath $executablePath).Length -le 0) {
        throw "Portable role executable is empty: $executablePath"
    }
    if (-not (Test-Path -LiteralPath $internalPath -PathType Container)) {
        throw "Portable role internal directory is missing: $internalPath"
    }
    return $resolved
}

function Get-FileMap {
    param([Parameter(Mandatory = $true)][string]$Root)

    $resolved = Get-NormalizedPath $Root
    $prefix = $resolved + [System.IO.Path]::DirectorySeparatorChar
    $map = @{}
    foreach ($file in Get-ChildItem -LiteralPath $resolved -Recurse -File -Force) {
        $relative = $file.FullName.Substring($prefix.Length).Replace("\", "/")
        if ($map.ContainsKey($relative)) {
            throw "Duplicate relative path in portable tree: $relative"
        }
        $map[$relative] = $file
    }
    return ,$map
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    $stream = [System.IO.File]::OpenRead((Get-NormalizedPath $Path))
    try {
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        try {
            $digest = $sha256.ComputeHash($stream)
            return [System.BitConverter]::ToString($digest).Replace("-", "").ToLowerInvariant()
        }
        finally {
            $sha256.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

function Copy-TreeWithHardLinks {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $sourcePath = Get-NormalizedPath $Source
    $destinationPath = Get-NormalizedPath $Destination
    if (
        -not [System.IO.Path]::GetPathRoot($sourcePath).Equals(
            [System.IO.Path]::GetPathRoot($destinationPath),
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "Universal hard-link staging requires one NTFS volume: $sourcePath -> $destinationPath"
    }
    [System.IO.Directory]::CreateDirectory($destinationPath) | Out-Null
    $sourcePrefix = $sourcePath + [System.IO.Path]::DirectorySeparatorChar
    foreach ($item in Get-ChildItem -LiteralPath $sourcePath -Recurse -Force) {
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Portable source contains an unsupported reparse point: $($item.FullName)"
        }
        $relative = $item.FullName.Substring($sourcePrefix.Length)
        $target = Join-Path $destinationPath $relative
        if ($item.PSIsContainer) {
            [System.IO.Directory]::CreateDirectory($target) | Out-Null
            continue
        }
        [System.IO.Directory]::CreateDirectory((Split-Path -Parent $target)) | Out-Null
        New-Item -ItemType HardLink -Path $target -Target $item.FullName -ErrorAction Stop | Out-Null
    }
}

function Assert-HardLinkedTrees {
    param(
        [Parameter(Mandatory = $true)][string]$LeftRoot,
        [Parameter(Mandatory = $true)][string]$RightRoot,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $left = Get-FileMap $LeftRoot
    $right = Get-FileMap $RightRoot
    if ($left.Count -ne $right.Count) {
        throw "$Label file count mismatch: left=$($left.Count), right=$($right.Count)"
    }
    foreach ($relative in $left.Keys) {
        if (-not $right.ContainsKey($relative)) {
            throw "$Label missing relative file: $relative"
        }
        if ($left[$relative].Length -ne $right[$relative].Length) {
            throw "$Label file size mismatch: $relative"
        }
        $leftId = [MusicToMidiPortable.FileIdentity]::Get($left[$relative].FullName)
        $rightId = [MusicToMidiPortable.FileIdentity]::Get($right[$relative].FullName)
        if ($leftId -ne $rightId) {
            throw "$Label files are not hard linked: $relative"
        }
    }
}

$FrontendContractFiles = @(
    "_internal\config\web-frontend.json",
    "_internal\LICENSE",
    "_internal\THIRD_PARTY_NOTICES.md",
    "_internal\web\app.js",
    "_internal\web\assets\app_icon.png",
    "_internal\web\index.html",
    "_internal\web\locales\en_US.json",
    "_internal\web\locales\zh_CN.json",
    "_internal\web\README.md",
    "_internal\web\runtime-config.json",
    "_internal\web\styles.css"
)

function Assert-FrontendContract {
    param(
        [Parameter(Mandatory = $true)][string]$LeftRoot,
        [Parameter(Mandatory = $true)][string]$RightRoot,
        [Parameter(Mandatory = $true)][string]$Label
    )

    foreach ($relative in $FrontendContractFiles) {
        $leftPath = Join-Path $LeftRoot $relative
        $rightPath = Join-Path $RightRoot $relative
        if (-not (Test-Path -LiteralPath $leftPath -PathType Leaf)) {
            throw "$Label left contract file is missing: $relative"
        }
        if (-not (Test-Path -LiteralPath $rightPath -PathType Leaf)) {
            throw "$Label right contract file is missing: $relative"
        }
        $left = Get-Item -LiteralPath $leftPath
        $right = Get-Item -LiteralPath $rightPath
        if ($left.Length -ne $right.Length) {
            throw "$Label file size mismatch: $relative"
        }
        $leftHash = Get-Sha256 $left.FullName
        $rightHash = Get-Sha256 $right.FullName
        if ($leftHash -ne $rightHash) {
            throw "$Label SHA-256 mismatch: $relative"
        }
    }
}

function Merge-IdenticalRuntimeFiles {
    param(
        [Parameter(Mandatory = $true)][string]$CanonicalRoot,
        [Parameter(Mandatory = $true)][string]$CandidateRoot
    )

    $canonical = Get-FileMap $CanonicalRoot
    $candidate = Get-FileMap $CandidateRoot
    $canonicalHashes = @{}
    [Int64]$sharedBytes = 0
    [Int64]$sharedFiles = 0
    foreach ($relative in $candidate.Keys) {
        if (-not $canonical.ContainsKey($relative)) {
            continue
        }
        $source = $canonical[$relative]
        $target = $candidate[$relative]
        if ($source.Length -ne $target.Length) {
            continue
        }
        if (-not $canonicalHashes.ContainsKey($relative)) {
            $canonicalHashes[$relative] = Get-Sha256 $source.FullName
        }
        $candidateHash = Get-Sha256 $target.FullName
        if ($canonicalHashes[$relative] -ne $candidateHash) {
            continue
        }
        Remove-Item -LiteralPath $target.FullName -Force
        New-Item -ItemType HardLink -Path $target.FullName -Target $source.FullName -ErrorAction Stop | Out-Null
        $sharedFiles++
        $sharedBytes += $source.Length
    }
    return [pscustomobject]@{
        FileCount = $sharedFiles
        Bytes = $sharedBytes
    }
}

function Relink-RoleInternalTree {
    param(
        [Parameter(Mandatory = $true)][string]$AppInternalRoot,
        [Parameter(Mandatory = $true)][string]$BackendInternalRoot,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $app = Get-FileMap $AppInternalRoot
    $backend = Get-FileMap $BackendInternalRoot
    if ($app.Count -ne $backend.Count) {
        throw "$Label output file count mismatch."
    }
    foreach ($relative in $app.Keys) {
        if (-not $backend.ContainsKey($relative)) {
            throw "$Label output is missing: $relative"
        }
        if ($app[$relative].Length -ne $backend[$relative].Length) {
            throw "$Label output size mismatch: $relative"
        }
        Remove-Item -LiteralPath $backend[$relative].FullName -Force
        New-Item `
            -ItemType HardLink `
            -Path $backend[$relative].FullName `
            -Target $app[$relative].FullName `
            -ErrorAction Stop | Out-Null
    }
    Assert-HardLinkedTrees -LeftRoot $AppInternalRoot -RightRoot $BackendInternalRoot -Label $Label
}

if (-not ("MusicToMidiPortable.FileIdentity" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Runtime.InteropServices;

namespace MusicToMidiPortable
{
    public static class FileIdentity
    {
        [StructLayout(LayoutKind.Sequential)]
        private struct ByHandleFileInformation
        {
            public uint FileAttributes;
            public System.Runtime.InteropServices.ComTypes.FILETIME CreationTime;
            public System.Runtime.InteropServices.ComTypes.FILETIME LastAccessTime;
            public System.Runtime.InteropServices.ComTypes.FILETIME LastWriteTime;
            public uint VolumeSerialNumber;
            public uint FileSizeHigh;
            public uint FileSizeLow;
            public uint NumberOfLinks;
            public uint FileIndexHigh;
            public uint FileIndexLow;
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool GetFileInformationByHandle(
            IntPtr handle,
            out ByHandleFileInformation information
        );

        public static string Get(string path)
        {
            using (FileStream stream = new FileStream(
                path,
                FileMode.Open,
                FileAccess.Read,
                FileShare.ReadWrite | FileShare.Delete
            ))
            {
                ByHandleFileInformation information;
                if (!GetFileInformationByHandle(
                    stream.SafeFileHandle.DangerousGetHandle(),
                    out information
                ))
                {
                    throw new System.ComponentModel.Win32Exception(
                        Marshal.GetLastWin32Error(),
                        "Unable to read NTFS file identity: " + path
                    );
                }
                return string.Format(
                    "{0:X8}:{1:X8}:{2:X8}",
                    information.VolumeSerialNumber,
                    information.FileIndexHigh,
                    information.FileIndexLow
                );
            }
        }
    }
}
'@
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$cudaRoot = Get-NormalizedPath $CudaDistRoot
$xpuRoot = Get-NormalizedPath $XpuDistRoot
$universalRoot = Assert-SafeOutputRoot $OutputRoot
$sourceRoots = @($cudaRoot, $xpuRoot)
foreach ($sourceRoot in $sourceRoots) {
    if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
        throw "Portable source root is missing: $sourceRoot"
    }
    if (
        $universalRoot.StartsWith(
            $sourceRoot + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "Universal output cannot be nested under a source root: $universalRoot"
    }
}

$cudaAppSource = Assert-Role `
    -Root (Join-Path $cudaRoot "MusicToMidi-App") `
    -Executable "MusicToMidi.exe"
$cudaBackendSource = Assert-Role `
    -Root (Join-Path $cudaRoot "MusicToMidi-WebBackend") `
    -Executable "MusicToMidiBackend.exe"
$cudaFrontendSource = Join-Path $cudaRoot "MusicToMidi-WebFrontend"
$xpuAppSource = Assert-Role `
    -Root (Join-Path $xpuRoot "MusicToMidi-XPU-App") `
    -Executable "MusicToMidiXpu.exe"
$xpuBackendSource = Assert-Role `
    -Root (Join-Path $xpuRoot "MusicToMidi-XPU-WebBackend") `
    -Executable "MusicToMidiBackendXpu.exe"
$xpuFrontendSource = Join-Path $xpuRoot "MusicToMidi-WebFrontend"
foreach ($frontend in @($cudaFrontendSource, $xpuFrontendSource)) {
    if (-not (Test-Path -LiteralPath (Join-Path $frontend "MusicToMidiFrontend.exe") -PathType Leaf)) {
        throw "Portable Web frontend is missing: $frontend"
    }
}

Assert-HardLinkedTrees `
    -LeftRoot (Join-Path $cudaAppSource "_internal") `
    -RightRoot (Join-Path $cudaBackendSource "_internal") `
    -Label "CUDA App/WebBackend source"
Assert-HardLinkedTrees `
    -LeftRoot (Join-Path $xpuAppSource "_internal") `
    -RightRoot (Join-Path $xpuBackendSource "_internal") `
    -Label "XPU App/WebBackend source"
Assert-FrontendContract `
    -LeftRoot $cudaFrontendSource `
    -RightRoot $xpuFrontendSource `
    -Label "CUDA/XPU WebFrontend"

if (Test-Path -LiteralPath $universalRoot) {
    if (-not $Clean) {
        throw "Universal output already exists; pass -Clean to replace it: $universalRoot"
    }
    Remove-Item -LiteralPath $universalRoot -Recurse -Force
}
[System.IO.Directory]::CreateDirectory($universalRoot) | Out-Null
$incompleteMarker = Join-Path $universalRoot ".incomplete"
[System.IO.File]::WriteAllText($incompleteMarker, "Universal publication is incomplete.")

$appRoot = Join-Path $universalRoot "MusicToMidi-App"
$backendRoot = Join-Path $universalRoot "MusicToMidi-WebBackend"
$frontendRoot = Join-Path $universalRoot "MusicToMidi-WebFrontend"
$cudaAppRuntime = Join-Path $appRoot "runtimes\cuda"
$xpuAppRuntime = Join-Path $appRoot "runtimes\xpu"
$cudaBackendRuntime = Join-Path $backendRoot "runtimes\cuda"
$xpuBackendRuntime = Join-Path $backendRoot "runtimes\xpu"

Copy-TreeWithHardLinks -Source $cudaAppSource -Destination $cudaAppRuntime
Copy-TreeWithHardLinks -Source $xpuAppSource -Destination $xpuAppRuntime
Copy-TreeWithHardLinks -Source $cudaBackendSource -Destination $cudaBackendRuntime
Copy-TreeWithHardLinks -Source $xpuBackendSource -Destination $xpuBackendRuntime
Copy-TreeWithHardLinks -Source $cudaFrontendSource -Destination $frontendRoot

$mergeResult = Merge-IdenticalRuntimeFiles `
    -CanonicalRoot $cudaAppRuntime `
    -CandidateRoot $xpuAppRuntime
Relink-RoleInternalTree `
    -AppInternalRoot (Join-Path $cudaAppRuntime "_internal") `
    -BackendInternalRoot (Join-Path $cudaBackendRuntime "_internal") `
    -Label "CUDA Universal App/WebBackend"
Relink-RoleInternalTree `
    -AppInternalRoot (Join-Path $xpuAppRuntime "_internal") `
    -BackendInternalRoot (Join-Path $xpuBackendRuntime "_internal") `
    -Label "XPU Universal App/WebBackend"

$launcherBuildRoot = Join-Path $universalRoot ".launcher-build"
& (Join-Path $repoRoot "scripts\build_universal_windows_launchers.ps1") `
    -OutputDirectory $launcherBuildRoot
if ($LASTEXITCODE -notin @(0, $null)) {
    throw "Universal launcher build failed with exit code $LASTEXITCODE."
}
Move-Item `
    -LiteralPath (Join-Path $launcherBuildRoot "MusicToMidi.exe") `
    -Destination (Join-Path $appRoot "MusicToMidi.exe")
Move-Item `
    -LiteralPath (Join-Path $launcherBuildRoot "MusicToMidiBackend.exe") `
    -Destination (Join-Path $backendRoot "MusicToMidiBackend.exe")
Remove-Item -LiteralPath $launcherBuildRoot -Recurse -Force

$usageCandidates = @(
    Get-ChildItem `
        -LiteralPath (Join-Path $repoRoot "resources\universal") `
        -File `
        -Filter "README-*.txt"
)
if ($usageCandidates.Count -ne 1) {
    throw "Expected exactly one Universal usage document under resources/universal."
}
$usageSource = $usageCandidates[0].FullName
Copy-Item -LiteralPath $usageSource -Destination (Join-Path $appRoot "README.txt")
Copy-Item -LiteralPath $usageSource -Destination (Join-Path $backendRoot "README.txt")
Copy-Item -LiteralPath $usageSource -Destination (Join-Path $frontendRoot "README-Universal.txt")

$requiredOutputs = @(
    (Join-Path $appRoot "MusicToMidi.exe"),
    (Join-Path $backendRoot "MusicToMidiBackend.exe"),
    (Join-Path $frontendRoot "MusicToMidiFrontend.exe"),
    (Join-Path $cudaAppRuntime "MusicToMidi.exe"),
    (Join-Path $xpuAppRuntime "MusicToMidiXpu.exe"),
    (Join-Path $cudaBackendRuntime "MusicToMidiBackend.exe"),
    (Join-Path $xpuBackendRuntime "MusicToMidiBackendXpu.exe")
)
foreach ($path in $requiredOutputs) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Universal required executable is missing: $path"
    }
    if ((Get-Item -LiteralPath $path).Length -le 0) {
        throw "Universal required executable is empty: $path"
    }
}

$buildInfo = [ordered]@{
    schema_version = 1
    generated_at = [DateTimeOffset]::Now.ToString("o")
    layout = @(
        "MusicToMidi-App",
        "MusicToMidi-WebBackend",
        "MusicToMidi-WebFrontend"
    )
    accelerator_priority = @("cuda", "xpu")
    failure_fallback = $false
    explicit_accelerator_variable = "MUSIC_TO_MIDI_ACCELERATOR"
    frontend_canonical_source = "cuda"
    frontend_contract_files = @($FrontendContractFiles)
    cross_accelerator_shared_files = [Int64]$mergeResult.FileCount
    cross_accelerator_shared_bytes = [Int64]$mergeResult.Bytes
    launchers = [ordered]@{
        app_sha256 = Get-Sha256 (Join-Path $appRoot "MusicToMidi.exe")
        backend_sha256 = Get-Sha256 (Join-Path $backendRoot "MusicToMidiBackend.exe")
    }
}
$buildInfoJson = $buildInfo | ConvertTo-Json -Depth 8
$utf8 = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText(
    (Join-Path $universalRoot "UNIVERSAL_BUILD_INFO.json"),
    $buildInfoJson + [Environment]::NewLine,
    $utf8
)

Remove-Item -LiteralPath $incompleteMarker -Force
if (Test-Path -LiteralPath $incompleteMarker) {
    throw "Universal incomplete marker was not removed."
}

Write-Host "[ok] Universal Windows root: $universalRoot"
Write-Host "[ok] Cross-accelerator shared files: $($mergeResult.FileCount)"
Write-Host "[ok] Cross-accelerator shared bytes: $($mergeResult.Bytes)"
