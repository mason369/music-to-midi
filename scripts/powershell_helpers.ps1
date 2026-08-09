# Shared helpers for Windows PowerShell 5.1 and PowerShell 7+ scripts.

function Invoke-PythonScript {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExecutable,
        [Parameter(Mandatory = $true)]
        [string]$Script
    )

    # Windows PowerShell reconstructs native command lines and can remove the
    # quotes inside a multiline `python -c $Script` argument. Execute the exact
    # UTF-8 source from a short-lived file so Python receives it byte-for-byte.
    $tempScript = Join-Path ([System.IO.Path]::GetTempPath()) (
        "music-to-midi-python-{0}.py" -f [System.Guid]::NewGuid().ToString("N")
    )
    $previousPythonPath = $env:PYTHONPATH
    $workingDirectory = (Get-Location).Path
    $exitCode = 1
    try {
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($tempScript, $Script, $utf8NoBom)
        if ([string]::IsNullOrWhiteSpace($previousPythonPath)) {
            $env:PYTHONPATH = $workingDirectory
        }
        else {
            $env:PYTHONPATH = $workingDirectory + [System.IO.Path]::PathSeparator + $previousPythonPath
        }
        & $PythonExecutable $tempScript | Out-Host
        $exitCode = $LASTEXITCODE
    }
    finally {
        if ($null -eq $previousPythonPath) {
            if (Test-Path Env:PYTHONPATH) {
                Remove-Item Env:PYTHONPATH -ErrorAction Stop
            }
        }
        else {
            $env:PYTHONPATH = $previousPythonPath
        }
        if (Test-Path -LiteralPath $tempScript) {
            Remove-Item -LiteralPath $tempScript -Force -ErrorAction Stop
        }
    }

    return [int]$exitCode
}
