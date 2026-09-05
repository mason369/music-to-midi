# Music to MIDI - Windows native CLI launcher
# Examples:
#   .\cli.ps1 song.wav
#   .\cli.ps1 batch D:\Audio --recursive --output D:\MidiOutput

#Requires -Version 5.1
# Keep this a simple script: advanced parameter binding consumes native flags
# such as -o (OutVariable/OutBuffer) before Python can receive them. Launcher
# options precede the Python command; everything after them is forwarded intact.
# Example: .\cli.ps1 -Accelerator xpu batch D:\Audio -o D:\MidiOutput
$Accelerator = "cuda"
$VenvName = ""
$argumentIndex = 0
while ($argumentIndex -lt $args.Count) {
    $option = [string]$args[$argumentIndex]
    if ($option -notmatch '^-(Accelerator|VenvName)(?::(.*))?$') {
        break
    }
    $optionName = $Matches[1]
    if ($option.Contains(":")) {
        $optionValue = $Matches[2]
    } else {
        $argumentIndex++
        if ($argumentIndex -ge $args.Count) {
            [Console]::Error.WriteLine("Missing value for launcher option $option")
            exit 2
        }
        $optionValue = [string]$args[$argumentIndex]
    }
    if ([string]::IsNullOrWhiteSpace($optionValue) -or $optionValue.StartsWith("-")) {
        [Console]::Error.WriteLine("Invalid value for launcher option $option")
        exit 2
    }
    if ($optionName -eq "Accelerator") {
        $Accelerator = $optionValue
    } else {
        $VenvName = $optionValue
    }
    $argumentIndex++
}
$CliArgs = @($args | Select-Object -Skip $argumentIndex)
if ($Accelerator -notin @("cuda", "xpu")) {
    [Console]::Error.WriteLine("Unsupported accelerator: $Accelerator (expected cuda or xpu)")
    exit 2
}

[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$env:PYTHONIOENCODING = "utf-8"

$repoDir = $PSScriptRoot
$Accelerator = $Accelerator.ToLowerInvariant()
if ([string]::IsNullOrWhiteSpace($VenvName)) {
    $VenvName = if ($Accelerator -eq "xpu") { "venv-xpu" } else { "venv" }
}
if ($VenvName -notmatch '^[A-Za-z0-9._-]+$') {
    [Console]::Error.WriteLine("Invalid virtual environment name: $VenvName")
    exit 2
}

$venvPython = Join-Path $repoDir "$VenvName\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    $installer = if ($Accelerator -eq "xpu") { ".\install_xpu.ps1" } else { ".\install.ps1" }
    [Console]::Error.WriteLine("Music to MIDI CLI environment is missing: $venvPython")
    [Console]::Error.WriteLine("Run $installer in the project directory first; global Python and accelerator fallback are not used.")
    exit 1
}

$arguments = @()
if ($CliArgs.Count -eq 0) {
    $arguments += "--help"
} else {
    $arguments += $CliArgs
}

$previousPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
$env:MUSIC_TO_MIDI_ACCELERATOR = $Accelerator
$env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($previousPythonPath)) {
    $repoDir
} else {
    "$repoDir$([System.IO.Path]::PathSeparator)$previousPythonPath"
}

function ConvertTo-NativeArgument {
    param([AllowEmptyString()][string]$Value)

    # Use the MS C runtime rules consumed by Python: double backslashes before
    # a quote or the closing delimiter, and escape embedded double quotes.
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

$process = New-Object System.Diagnostics.Process
$process.StartInfo.FileName = $venvPython
$process.StartInfo.UseShellExecute = $false
# Set-Location and PSDrives do not update the PowerShell process's native cwd.
$process.StartInfo.WorkingDirectory = $ExecutionContext.SessionState.Path.CurrentFileSystemLocation.ProviderPath
$quotedArguments = foreach ($nativeArgument in (@("-m", "src.cli") + $arguments)) {
    ConvertTo-NativeArgument -Value ([string]$nativeArgument)
}
$process.StartInfo.Arguments = $quotedArguments -join " "
# Inherit the caller's console and all three streams. A second shell would
# reinterpret ampersands, percent signs and quoted directory paths.
try {
    if (-not $process.Start()) {
        throw "Could not start the Music to MIDI CLI process."
    }
    $process.WaitForExit()
    $cliExitCode = $process.ExitCode
} finally {
    $process.Dispose()
    if ($null -eq $previousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $previousPythonPath
    }
}
exit $cliExitCode
