param(
    [Parameter(Mandatory = $true)]
    [string]$CombinedRoot,
    [Parameter(Mandatory = $true)]
    [string]$DistRoot,
    [ValidateSet("cuda", "xpu")]
    [string]$Accelerator = "cuda"
)

# Split PyInstaller's shared Windows collection into two independently movable
# role directories. Files are hard-linked while publishing so the GPU host does
# not store a second copy of the large model/runtime set. The Windows release
# preserves those links in one WIM; copying or selectively extracting either
# role materializes all files required by that role. Each directory carries an
# explicit incomplete marker until validation succeeds; this avoids a
# large-directory rename that can be denied by real-time scanners.

$ErrorActionPreference = "Stop"

function Get-NormalizedPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return [System.IO.Path]::GetFullPath($Path).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
}

function Assert-DirectChildPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Parent,
        [Parameter(Mandatory = $true)][string[]]$AllowedNames
    )

    $resolvedPath = Get-NormalizedPath $Path
    $resolvedParent = Get-NormalizedPath $Parent
    $actualParent = Get-NormalizedPath (Split-Path -Parent $resolvedPath)
    $actualName = Split-Path -Leaf $resolvedPath
    if (
        -not $actualParent.Equals($resolvedParent, [System.StringComparison]::OrdinalIgnoreCase) -or
        $AllowedNames -notcontains $actualName
    ) {
        throw "Refusing to modify an unexpected portable output path: $resolvedPath"
    }
    return $resolvedPath
}

function Remove-CheckedDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Parent,
        [Parameter(Mandatory = $true)][string[]]$AllowedNames
    )

    $checkedPath = Assert-DirectChildPath -Path $Path -Parent $Parent -AllowedNames $AllowedNames
    if (Test-Path -LiteralPath $checkedPath) {
        Remove-Item -LiteralPath $checkedPath -Recurse -Force
    }
}

function Copy-DirectoryWithHardLinks {
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
        throw "Hard-link staging requires source and destination on the same volume."
    }

    [System.IO.Directory]::CreateDirectory($destinationPath) | Out-Null
    $sourcePrefix = $sourcePath + [System.IO.Path]::DirectorySeparatorChar
    foreach ($item in Get-ChildItem -LiteralPath $sourcePath -Recurse -Force) {
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Portable collection contains an unsupported reparse point: $($item.FullName)"
        }
        $relativePath = $item.FullName.Substring($sourcePrefix.Length)
        $targetPath = Join-Path $destinationPath $relativePath
        if ($item.PSIsContainer) {
            [System.IO.Directory]::CreateDirectory($targetPath) | Out-Null
            continue
        }
        [System.IO.Directory]::CreateDirectory((Split-Path -Parent $targetPath)) | Out-Null
        New-Item -ItemType HardLink -Path $targetPath -Target $item.FullName -ErrorAction Stop | Out-Null
    }
}

function Assert-PortableRole {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$RequiredExecutable,
        [Parameter(Mandatory = $true)][string]$ForbiddenExecutable
    )

    $requiredPath = Join-Path $Root $RequiredExecutable
    $forbiddenPath = Join-Path $Root $ForbiddenExecutable
    $internalPath = Join-Path $Root "_internal"
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Portable role is missing its executable: $requiredPath"
    }
    if ((Get-Item -LiteralPath $requiredPath).Length -le 0) {
        throw "Portable role executable is empty: $requiredPath"
    }
    if (Test-Path -LiteralPath $forbiddenPath) {
        throw "Portable role still contains the other role's executable: $forbiddenPath"
    }
    if (-not (Test-Path -LiteralPath $internalPath -PathType Container)) {
        throw "Portable role is missing its private _internal directory: $internalPath"
    }
}

$Accelerator = $Accelerator.ToLowerInvariant()
$resolvedDistRoot = Get-NormalizedPath $DistRoot
$collectionName = if ($Accelerator -eq "xpu") { "MusicToMidi-XPU" } else { "MusicToMidi" }
$appName = if ($Accelerator -eq "xpu") { "MusicToMidi-XPU-App" } else { "MusicToMidi-App" }
$backendName = if ($Accelerator -eq "xpu") { "MusicToMidi-XPU-WebBackend" } else { "MusicToMidi-WebBackend" }
$guiExecutable = if ($Accelerator -eq "xpu") { "MusicToMidiXpu.exe" } else { "MusicToMidi.exe" }
$backendExecutable = if ($Accelerator -eq "xpu") { "MusicToMidiBackendXpu.exe" } else { "MusicToMidiBackend.exe" }
$allowedNames = @($collectionName, $appName, $backendName)

$resolvedCombinedRoot = Assert-DirectChildPath `
    -Path $CombinedRoot `
    -Parent $resolvedDistRoot `
    -AllowedNames @($collectionName)
if (-not (Test-Path -LiteralPath $resolvedCombinedRoot -PathType Container)) {
    throw "Combined PyInstaller collection is missing: $resolvedCombinedRoot"
}

$appRoot = Assert-DirectChildPath `
    -Path (Join-Path $resolvedDistRoot $appName) `
    -Parent $resolvedDistRoot `
    -AllowedNames $allowedNames
$backendRoot = Assert-DirectChildPath `
    -Path (Join-Path $resolvedDistRoot $backendName) `
    -Parent $resolvedDistRoot `
    -AllowedNames $allowedNames
Remove-CheckedDirectory -Path $appRoot -Parent $resolvedDistRoot -AllowedNames $allowedNames
Remove-CheckedDirectory -Path $backendRoot -Parent $resolvedDistRoot -AllowedNames $allowedNames
[System.IO.Directory]::CreateDirectory($appRoot) | Out-Null
[System.IO.Directory]::CreateDirectory($backendRoot) | Out-Null
$appIncompleteMarker = Join-Path $appRoot ".incomplete"
$backendIncompleteMarker = Join-Path $backendRoot ".incomplete"
[System.IO.File]::WriteAllText($appIncompleteMarker, "Portable role publication is incomplete.")
[System.IO.File]::WriteAllText($backendIncompleteMarker, "Portable role publication is incomplete.")

Copy-DirectoryWithHardLinks -Source $resolvedCombinedRoot -Destination $appRoot
Copy-DirectoryWithHardLinks -Source $resolvedCombinedRoot -Destination $backendRoot
Remove-Item -LiteralPath (Join-Path $appRoot $backendExecutable) -Force
Remove-Item -LiteralPath (Join-Path $backendRoot $guiExecutable) -Force
Assert-PortableRole -Root $appRoot -RequiredExecutable $guiExecutable -ForbiddenExecutable $backendExecutable
Assert-PortableRole -Root $backendRoot -RequiredExecutable $backendExecutable -ForbiddenExecutable $guiExecutable
Remove-Item -LiteralPath $appIncompleteMarker -Force
Remove-Item -LiteralPath $backendIncompleteMarker -Force
if (
    (Test-Path -LiteralPath $appIncompleteMarker) -or
    (Test-Path -LiteralPath $backendIncompleteMarker)
) {
    throw "Portable role publication markers were not removed after validation."
}

Remove-CheckedDirectory -Path $resolvedCombinedRoot -Parent $resolvedDistRoot -AllowedNames $allowedNames
Write-Host "[ok] Desktop App package: $appRoot"
Write-Host "[ok] Web backend package: $backendRoot"
