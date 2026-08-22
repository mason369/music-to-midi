# Music to MIDI - Windows 安装脚本
# 支持 Windows 10/11 (x64)
# 用法: powershell -ExecutionPolicy Bypass -File install.ps1
# 或双击 install.bat

#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet("cuda", "xpu")]
    [string]$Accelerator = "cuda",
    [string]$VenvName = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
# 设置控制台输出编码为 UTF-8，避免中文乱码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$REPO_DIR = $PSScriptRoot
$Accelerator = $Accelerator.ToLowerInvariant()
if ([string]::IsNullOrWhiteSpace($VenvName)) {
    $VenvName = if ($Accelerator -eq "xpu") { "venv-xpu" } else { "venv" }
}
if ($VenvName -notmatch '^[A-Za-z0-9._-]+$') {
    throw "非法虚拟环境目录名: $VenvName"
}
$env:MUSIC_TO_MIDI_ACCELERATOR = $Accelerator
$VENV_DIR = Join-Path $REPO_DIR $VenvName
$PIP      = Join-Path $VENV_DIR "Scripts\pip.exe"
$PYTHON   = Join-Path $VENV_DIR "Scripts\python.exe"
. (Join-Path $REPO_DIR "scripts\powershell_helpers.ps1")

# --- 辅助输出函数 ---
function Write-Info  { param($msg) Write-Host "[信息]  $msg" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "[完成]  $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "[警告]  $msg" -ForegroundColor Yellow }
function Write-Err   { param($msg) Write-Host "[错误] $msg" -ForegroundColor Red; exit 1 }

function Test-PythonPackageInstalled {
    param([Parameter(Mandatory = $true)][string]$PackageName)

    if ($PackageName -notmatch '^[A-Za-z0-9_.-]+$') {
        throw "非法 Python 包名: $PackageName"
    }
    $packagePresenceCheck = @"
from importlib import metadata
try:
    metadata.version('$PackageName')
except metadata.PackageNotFoundError:
    raise SystemExit(1)
"@
    $exitCode = Invoke-PythonScript -PythonExecutable $PYTHON -Script $packagePresenceCheck
    return $exitCode -eq 0
}

# --- 通用下载函数（实时进度显示）---
# 优先使用 curl.exe（内置于 Windows 10 1803+ / Server 2019+），自带实时进度条。
# 若 curl.exe 不可用，回退到 Invoke-WebRequest（显示 PowerShell 进度条）。
function Invoke-Download {
    param(
        [string]$Url,
        [string]$OutFile,
        [string]$Description
    )
    Write-Info "  $Description"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

    $curlOk = $false
    try {
        $null = & curl.exe --version 2>&1
        if ($LASTEXITCODE -eq 0) { $curlOk = $true }
    } catch { }

    if ($curlOk) {
        # -L 跟随重定向  --fail 失败时非零退出  -# 显示 # 进度条  -o 输出文件
        & curl.exe -L --fail --show-error -# -o $OutFile $Url
        if ($LASTEXITCODE -ne 0) { throw "curl 下载失败，退出码: $LASTEXITCODE" }
    } else {
        # 回退：Invoke-WebRequest（显示 PowerShell 内置进度条）
        $ProgressPreference = 'Continue'
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
    }
}

# --- 标题 ---
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Music to MIDI  -  Windows $($Accelerator.ToUpperInvariant()) 安装程序" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# --- 检测安装路径是否包含特殊字符（中文/日文/空格/括号等）---
# PyTorch 在 Windows 上无法正确加载含特殊字符路径下的 DLL（c10.dll 等）
$nonAscii = [regex]::IsMatch($REPO_DIR, '[^\x00-\x7F]')
$hasSpaceOrParen = [regex]::IsMatch($REPO_DIR, '[\s\(\)]')
if ($nonAscii -or $hasSpaceOrParen) {
    Write-Host ""
    Write-Host "  !! 警告: 安装路径包含特殊字符 !!" -ForegroundColor Red
    Write-Host "  当前路径: $REPO_DIR" -ForegroundColor Yellow
    Write-Host ""
    if ($nonAscii) {
        Write-Host "  检测到非 ASCII 字符（如中文用户名）。" -ForegroundColor Yellow
    }
    if ($hasSpaceOrParen) {
        Write-Host "  检测到空格或括号（如路径中的 '(1)'）。" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "  PyTorch 在 Windows 上可能无法加载此路径下的 DLL，" -ForegroundColor Yellow
    Write-Host "  导致运行时出现 'DLL 初始化例程失败' 错误。" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  建议将项目移动到纯英文且无空格的路径，例如:" -ForegroundColor Green
    Write-Host "    C:\MusicToMidi" -ForegroundColor Green
    Write-Host "    D:\Projects\music-to-midi" -ForegroundColor Green
    Write-Host ""
    $continue = Read-Host "  是否仍要继续安装？(y/N)"
    if ($continue -ne 'y' -and $continue -ne 'Y') {
        Write-Host "  安装已取消。支持的项目路径不包含特殊字符。" -ForegroundColor Yellow
        exit 1
    }
    Write-Warn "继续安装（路径问题可能导致运行失败）..."
}

# --- 第 1 步/共 12 步：检测 Python 版本 ---
Write-Info "第 1 步/共 12 步  检测 Python 版本..."

$PYTHON_BIN = $null

# 优先使用 py launcher 指定兼容版本（3.11~3.12），避免选中过新的 Python（3.13+ 尚未被 PyTorch 完全支持）
foreach ($ver in @("-3.12", "-3.11")) {
    try {
        $verStr = & py $ver -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))" 2>&1
        if ($LASTEXITCODE -eq 0 -and ("$verStr" -match '^(\d+)\.(\d+)')) {
            $PYTHON_BIN = "py $ver"
            Write-Ok "找到 Python $verStr (py $ver)"
            break
        }
    }
    catch {}
}

# 回退：检测 py / python 命令，但限制版本 3.11~3.12
if (-not $PYTHON_BIN) {
    foreach ($cmd in @("py", "python")) {
        try {
            $verStr = & $cmd -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))" 2>&1
            if ($LASTEXITCODE -eq 0 -and ("$verStr" -match '^(\d+)\.(\d+)')) {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -eq 3 -and $minor -ge 11 -and $minor -le 12) {
                    $PYTHON_BIN = $cmd
                    Write-Ok "找到 Python $verStr ($cmd)"
                    break
                } elseif ($major -eq 3 -and $minor -gt 12) {
                    Write-Warn "检测到 Python $verStr，版本过新（PyTorch 尚未支持 3.13+）"
                    Write-Warn "  建议安装 Python 3.11 或 3.12: winget install Python.Python.3.11"
                }
            }
        }
        catch {}
    }
}

if (-not $PYTHON_BIN) {
    Write-Host "[错误] 未找到兼容的 Python 版本（需要 3.11~3.12）。" -ForegroundColor Red
    Write-Host "  当前系统可能安装了过旧或过新的 Python，Aria-AMT 需要 3.11+。" -ForegroundColor Yellow
    Write-Host "  Python 3.11 安装命令: winget install Python.Python.3.11" -ForegroundColor Yellow
    Write-Host "  或访问 https://www.python.org/downloads/ 下载 3.11 或 3.12" -ForegroundColor Yellow
    exit 1
}

$PYTHON_CMD = $PYTHON_BIN -split ' '

# --- 第 2 步/共 12 步：检测 git ---
Write-Info "第 2 步/共 12 步  检测 git..."

try {
    $gitVer = & git --version 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Err "git 执行失败" }
    Write-Ok "找到 $gitVer"
}
catch {
    Write-Host "[错误] 未找到 git。安装命令: winget install Git.Git" -ForegroundColor Red
    exit 1
}

# --- 第 3 步/共 12 步：检测 ffmpeg ---
Write-Info "第 3 步/共 12 步  检测 ffmpeg..."

$FFMPEG_OK = $false
try {
    $null = & ffmpeg -version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "ffmpeg 已安装"
        $FFMPEG_OK = $true
    } else {
        throw "not found"
    }
}
catch {
    Write-Warn "未找到 ffmpeg，正在尝试自动安装..."

    $installed = $false

    # 方法 1：winget（Windows 10/11 消费版自带，Server 版可能无）
    $wingetAvailable = $false
    try {
        $null = & winget --version 2>&1
        if ($LASTEXITCODE -eq 0) { $wingetAvailable = $true }
    } catch { }

    if ($wingetAvailable) {
        Write-Info "  尝试通过 winget 安装 ffmpeg..."
        try {
            $null = & winget install Gyan.FFmpeg --silent --accept-package-agreements --accept-source-agreements 2>&1
            if ($LASTEXITCODE -eq 0) { $installed = $true; Write-Ok "  winget 安装成功" }
        } catch { }
        if (-not $installed) {
            try {
                $null = & winget install Gyan.FFmpeg --silent --accept-package-agreements --accept-source-agreements --scope user 2>&1
                if ($LASTEXITCODE -eq 0) { $installed = $true; Write-Ok "  winget (user scope) 安装成功" }
            } catch { }
        }
    } else {
        Write-Info "  winget 不可用（Windows Server / 旧版系统），跳过"
    }

    # 方法 2：Scoop（如已安装）
    if (-not $installed) {
        $scoopAvailable = $false
        try {
            $null = & scoop --version 2>&1
            if ($LASTEXITCODE -eq 0) { $scoopAvailable = $true }
        } catch { }

        if ($scoopAvailable) {
            Write-Info "  尝试通过 Scoop 安装 ffmpeg..."
            try {
                # 不抑制输出，让 Scoop 的进度信息正常显示
                & scoop install ffmpeg
                if ($LASTEXITCODE -eq 0) { $installed = $true; Write-Ok "  Scoop 安装成功" }
            } catch { }
        }
    }

    # 方法 3：直接下载解压（适用于 Windows Server 等无包管理器的环境）
    if (-not $installed) {
        Write-Info "  尝试直接下载 ffmpeg（适用于 Windows Server）..."
        $ffmpegDir = Join-Path $REPO_DIR "tools\ffmpeg"
        $ffmpegBin = Join-Path $ffmpegDir "bin"
        $ffmpegExe = Join-Path $ffmpegBin "ffmpeg.exe"

        if (-not (Test-Path $ffmpegExe)) {
            $stagingRoot = $null
            try {
                # BtbN retains the final build of each month for two years. Use
                # that immutable monthly tag instead of an expiring daily build.
                $ffmpegReleaseTag = "autobuild-2026-07-31-14-10"
                $ffmpegAssetName = "ffmpeg-n7.1.5-12-g1fdbca85aa-win64-gpl-7.1.zip"
                $ffmpegExpectedSha256 = "c067a1ca58f4fc4449f4bab0890fbcd65cbb3e5f46e066cf9c768e06c0c1d4d9"
                $zipUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/$ffmpegReleaseTag/$ffmpegAssetName"
                $stagingRoot = Join-Path $env:TEMP (
                    "music-to-midi-ffmpeg-{0}" -f [System.Guid]::NewGuid().ToString("N")
                )
                New-Item -ItemType Directory -Path $stagingRoot -ErrorAction Stop | Out-Null
                $zipPath = Join-Path $stagingRoot $ffmpegAssetName
                $extractPath = Join-Path $stagingRoot "extract"

                # 使用 Invoke-Download 显示实时进度
                Invoke-Download -Url $zipUrl -OutFile $zipPath -Description "正在下载固定版本 ffmpeg（约 151 MB）..."
                $ffmpegActualSha256 = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($ffmpegActualSha256 -ne $ffmpegExpectedSha256) {
                    throw "FFmpeg 压缩包 SHA-256 不匹配：期望 $ffmpegExpectedSha256，实际 $ffmpegActualSha256"
                }

                Write-Info "  正在解压..."
                Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force

                # 找到解压后含 ffmpeg.exe 的 bin 目录（子文件夹名随版本变化）
                $binFolder = Get-ChildItem $extractPath -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
                if (-not $binFolder) {
                    throw "FFmpeg 压缩包中未找到 ffmpeg.exe"
                }
                $srcBin = $binFolder.DirectoryName
                if (-not (Test-Path $ffmpegBin)) { New-Item $ffmpegBin -ItemType Directory -Force | Out-Null }
                Copy-Item "$srcBin\*.exe" $ffmpegBin -Force
                if (-not (Test-Path -LiteralPath $ffmpegExe)) {
                    throw "FFmpeg 文件复制后仍缺少 $ffmpegExe"
                }
                $installed = $true
                Write-Ok "  ffmpeg 已解压至 $ffmpegBin"
            }
            catch {
                Write-Warn "  直接下载失败: $_"
            }
            finally {
                if ($stagingRoot -and (Test-Path -LiteralPath $stagingRoot)) {
                    $resolvedStagingRoot = [System.IO.Path]::GetFullPath($stagingRoot)
                    $resolvedTempRoot = [System.IO.Path]::GetFullPath($env:TEMP)
                    if (
                        [System.IO.Path]::GetDirectoryName($resolvedStagingRoot) -ne $resolvedTempRoot -or
                        -not [System.IO.Path]::GetFileName($resolvedStagingRoot).StartsWith("music-to-midi-ffmpeg-")
                    ) {
                        throw "拒绝清理未经验证的 FFmpeg 临时目录: $resolvedStagingRoot"
                    }
                    Remove-Item -LiteralPath $resolvedStagingRoot -Recurse -Force -ErrorAction Stop
                }
            }
        } else {
            $installed = $true
            Write-Ok "  ffmpeg 已存在于 $ffmpegBin"
        }

        if ($installed -and (Test-Path $ffmpegBin)) {
            # 将本地 ffmpeg 加入当前进程 PATH
            $env:PATH = "$ffmpegBin;" + $env:PATH
            # 持久化到用户级 PATH
            $userPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
            if ($userPath -notlike "*$ffmpegBin*") {
                [System.Environment]::SetEnvironmentVariable("PATH", "$ffmpegBin;$userPath", "User")
                Write-Info "  已将 $ffmpegBin 添加到用户 PATH（重启终端后全局生效）"
            }
        }
    }

    # 刷新 PATH 后再次验证
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";" +
                $env:PATH
    try {
        $null = & ffmpeg -version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "ffmpeg 可用"
            $FFMPEG_OK = $true
        } else { throw "still not found" }
    }
    catch {
        Write-Err "ffmpeg 安装后仍无法在当前进程执行；当前安装路径不可用。根因: $_"
    }
}

# --- 第 4 步/共 12 步：检测并安装 Visual C++ Redistributable 2022 ---
Write-Info "第 4 步/共 12 步  检测 Visual C++ Redistributable 2022 x64..."

$vcRedistOk = $false
$vcRegPaths = @(
    "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
)
foreach ($regPath in $vcRegPaths) {
    try {
        $key = Get-ItemProperty -Path $regPath -ErrorAction SilentlyContinue
        # Bld >= 29000 对应 VS2019/2022（14.2x/14.3x），均可满足 PyTorch 需求
        if ($key -and $key.Bld -ge 29000) {
            Write-Ok "Visual C++ Redistributable 已安装 (v$($key.Major).$($key.Minor).$($key.Bld))"
            $vcRedistOk = $true
            break
        }
    }
    catch { }
}

if (-not $vcRedistOk) {
    Write-Warn "未检测到 Visual C++ Redistributable 2022 x64（PyTorch 加载 fbgemm.dll 必需）"
    $vcTempRoot = $null
    try {
        $vcUrl  = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
        $vcTempRoot = Join-Path $env:TEMP (
            "music-to-midi-vcredist-{0}" -f [System.Guid]::NewGuid().ToString("N")
        )
        New-Item -ItemType Directory -Path $vcTempRoot -ErrorAction Stop | Out-Null
        $vcPath = Join-Path $vcTempRoot "vc_redist.x64.exe"

        # 使用 Invoke-Download 显示实时进度
        Invoke-Download -Url $vcUrl -OutFile $vcPath -Description "正在下载 VC++ 2022 Redistributable（约 25 MB）..."
        $vcSignature = Get-AuthenticodeSignature -LiteralPath $vcPath
        $vcSignerSubject = if ($vcSignature.SignerCertificate) {
            $vcSignature.SignerCertificate.Subject
        } else {
            "<none>"
        }
        if (
            $vcSignature.Status -ne [System.Management.Automation.SignatureStatus]::Valid -or
            -not $vcSignature.SignerCertificate -or
            $vcSignerSubject -notmatch '(^|,\s*)O=Microsoft Corporation(,|$)'
        ) {
            throw "VC++ Redistributable Authenticode 签名无效或签名者不是 Microsoft Corporation: status=$($vcSignature.Status), subject=$vcSignerSubject"
        }

        Write-Info "  正在静默安装..."
        $proc = Start-Process -FilePath $vcPath -ArgumentList "/install /quiet /norestart" -Wait -PassThru

        # 退出码 0 = 成功，3010 = 成功但需重启
        if ($proc.ExitCode -eq 0 -or $proc.ExitCode -eq 3010) {
            Write-Ok "Visual C++ Redistributable 2022 安装成功"
            if ($proc.ExitCode -eq 3010) {
                Write-Warn "  建议重启系统以使 DLL 完全生效（可稍后重启）"
            }
            $vcRedistOk = $true
        } else {
            throw "VC++ Redistributable 安装程序返回非成功退出码: $($proc.ExitCode)"
        }
    }
    catch {
        Write-Warn "VC++ Redistributable 自动安装失败: $_"
        Write-Warn "后续 PyTorch CUDA DLL 导入和 GPU 张量实测将作为最终门禁；若运行时确实缺失，安装会明确停止。"
    }
    finally {
        if ($vcTempRoot -and (Test-Path -LiteralPath $vcTempRoot)) {
            $resolvedVcTempRoot = [System.IO.Path]::GetFullPath($vcTempRoot)
            $resolvedTempRoot = [System.IO.Path]::GetFullPath($env:TEMP)
            if (
                [System.IO.Path]::GetDirectoryName($resolvedVcTempRoot) -ne $resolvedTempRoot -or
                -not [System.IO.Path]::GetFileName($resolvedVcTempRoot).StartsWith("music-to-midi-vcredist-")
            ) {
                throw "拒绝清理未经验证的 VC++ 临时目录: $resolvedVcTempRoot"
            }
            Remove-Item -LiteralPath $resolvedVcTempRoot -Recurse -Force -ErrorAction Stop
        }
    }
}

# --- 第 5 步/共 12 步：创建 Python 虚拟环境 ---
Write-Info "第 5 步/共 12 步  创建 Python 虚拟环境..."

if (-not (Test-Path $VENV_DIR) -or -not (Test-Path $PYTHON)) {
    if ((Test-Path $VENV_DIR) -and -not (Test-Path $PYTHON)) {
        Write-Warn "虚拟环境目录存在但缺少 python.exe，正在重新创建..."
    }
    if ($PYTHON_CMD.Count -eq 1) {
        & $PYTHON_CMD[0] -m venv "$VENV_DIR"
    } else {
        # e.g. py -3.11 → & py -3.11 -m venv ...
        & $PYTHON_CMD[0] $PYTHON_CMD[1] -m venv "$VENV_DIR"
    }
    if ($LASTEXITCODE -ne 0) { Write-Err "创建虚拟环境失败" }
    Write-Ok "虚拟环境创建成功: $VENV_DIR"
} else {
    Write-Ok "虚拟环境已存在: $VENV_DIR"
}

$venvPythonIdentityCheck = @"
import struct
import sys
version = sys.version_info[:2]
if not ((3, 11) <= version <= (3, 12)):
    raise RuntimeError(f'unsupported venv Python: {version[0]}.{version[1]}')
if struct.calcsize('P') * 8 != 64:
    raise RuntimeError('the project venv must use 64-bit Python')
print(f'Venv Python: {version[0]}.{version[1]} ({struct.calcsize("P") * 8}-bit)')
"@
$pythonExitCode = Invoke-PythonScript -PythonExecutable $PYTHON -Script $venvPythonIdentityCheck
if ($pythonExitCode -ne 0) {
    Write-Warn "现有 venv 的解释器损坏或不符合 Python 3.11-3.12/x64 契约，正在使用兼容解释器 --clear 重建可复现环境..."
    if ($PYTHON_CMD.Count -eq 1) {
        & $PYTHON_CMD[0] -m venv --clear "$VENV_DIR"
    } else {
        & $PYTHON_CMD[0] $PYTHON_CMD[1] -m venv --clear "$VENV_DIR"
    }
    if ($LASTEXITCODE -ne 0) { Write-Err "重建不兼容的虚拟环境失败" }
    $pythonExitCode = Invoke-PythonScript -PythonExecutable $PYTHON -Script $venvPythonIdentityCheck
    if ($pythonExitCode -ne 0) { Write-Err "重建后的虚拟环境仍未通过 Python 3.11-3.12/x64 校验" }
    Write-Ok "虚拟环境已按固定解释器重建: $VENV_DIR"
}


# 清理 pip 中断安装遗留的损坏包目录（~ 前缀，如 ~ransformers）
$sitePackages = Join-Path $VENV_DIR "Lib\site-packages"
if (Test-Path $sitePackages) {
    $brokenPkgs = Get-ChildItem $sitePackages -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like '~*' }
    foreach ($pkg in $brokenPkgs) {
        Write-Warn "清理损坏包目录: $($pkg.Name)"
        Remove-Item $pkg.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
}
# --- 第 6 步/共 12 步：升级 pip ---
Write-Info "第 6 步/共 12 步  升级 pip..."

& "$PYTHON" -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Write-Err "pip 升级失败" }
Write-Ok "pip 升级成功"

# --- 第 7 步/共 12 步：严格验证隔离的 PyTorch 加速器运行时 ---
Write-Info "第 7 步/共 12 步  验证完整七模式 $($Accelerator.ToUpperInvariant()) 运行时..."

if ($Accelerator -eq "cuda") {
$nvidiaOk = $false
$nvOutput = $null
try {
    $nvOutput = & nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) { $nvidiaOk = $true }
} catch { }

if (-not $nvidiaOk) {
    $amdGpu = $null
    try {
        $amdGpu = Get-CimInstance -ClassName Win32_VideoController -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match 'AMD|Radeon' }
    } catch { }
    if ($amdGpu) {
        Write-Err "检测到 AMD GPU；完整七模式当前需要 NVIDIA CUDA 12.8 与 ONNX Runtime CUDAExecutionProvider。Windows 不会静默改用 CPU。"
    }
    Write-Err "未检测到可用的 NVIDIA 驱动 (nvidia-smi)；完整七模式不支持 CPU/Intel 降级运行。"
}

$nvText = $nvOutput -join " "
$cudaMatch = [regex]::Match($nvText, 'CUDA Version:\s*([\d.]+)')
if (-not $cudaMatch.Success) {
    Write-Err "nvidia-smi 未报告 CUDA 驱动版本，无法确认 CUDA 12.8 运行时兼容性。"
}
$driverCuda = [version]$cudaMatch.Groups[1].Value
if ($driverCuda -lt [version]"12.8") {
    Write-Err "NVIDIA 驱动仅报告 CUDA $driverCuda；完整七模式的固定 PyTorch/ONNX Runtime 组合需要 CUDA 12.8 兼容驱动。"
}
Write-Ok "NVIDIA 驱动检查通过（报告 CUDA $driverCuda）"

function Repair-TorchOpenMPRuntime {
    $torchLib = Join-Path $VENV_DIR "Lib\site-packages\torch\lib"
    $libompDll = Join-Path $torchLib "libomp140.x86_64.dll"
    if ((Test-Path $torchLib) -and -not (Test-Path $libompDll)) {
        Write-Info "修复 PyTorch 必需的 libomp140.x86_64.dll..."
        & "$PIP" install zstandard
        if ($LASTEXITCODE -ne 0) { Write-Err "zstandard 安装失败，无法修复 PyTorch OpenMP 运行时" }
        & "$PYTHON" (Join-Path $REPO_DIR "tools\repair_torch_openmp.py") --torch-lib-dir "$torchLib"
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $libompDll)) {
            Write-Err "libomp140.x86_64.dll 修复失败，不能继续验证 PyTorch"
        }
        Write-Ok "libomp140.x86_64.dll 已修复"
    }
}

$torchCudaRuntimeCheck = @"
import sys
from importlib import metadata
expected = {'torch': '2.7.0', 'torchaudio': '2.7.0', 'torchvision': '0.22.0'}
try:
    actual = {name: metadata.version(name) for name in expected}
except metadata.PackageNotFoundError as exc:
    print(f'Missing pinned PyTorch runtime package: {exc.name}', file=sys.stderr)
    raise SystemExit(2)
base = {name: version.split('+', 1)[0] for name, version in actual.items()}
if base != expected:
    raise RuntimeError(f'PyTorch trio mismatch: expected={expected}, actual={actual}')
import torch, torchaudio, torchvision
if getattr(torch.version, 'hip', None):
    raise RuntimeError(f'ROCm runtime is unsupported for the complete seven-mode stack: HIP={torch.version.hip}')
if torch.version.cuda != '12.8':
    raise RuntimeError(f'Expected PyTorch CUDA 12.8 runtime, got {torch.version.cuda!r}')
if not torch.cuda.is_available():
    raise RuntimeError('torch.cuda.is_available() is False; a working NVIDIA CUDA GPU is required')
probe = torch.ones(1, device='cuda')
probe.add_(1)
torch.cuda.synchronize()
print('PyTorch trio:', actual)
print('PyTorch CUDA runtime:', torch.version.cuda)
print('NVIDIA device:', torch.cuda.get_device_name(0))
"@

Repair-TorchOpenMPRuntime
$pythonExitCode = Invoke-PythonScript -PythonExecutable $PYTHON -Script $torchCudaRuntimeCheck
if ($pythonExitCode -ne 0) {
    Write-Info "当前 PyTorch 三件套或 CUDA flavor 不符合固定运行时，正在强制安装 cu128..."
    & "$PIP" install "torch==2.7.0" "torchaudio==2.7.0" "torchvision==0.22.0" `
        --index-url "https://download.pytorch.org/whl/cu128" --force-reinstall
    if ($LASTEXITCODE -ne 0) { Write-Err "PyTorch 2.7.0 / torchaudio 2.7.0 / torchvision 0.22.0 cu128 安装失败" }
    Repair-TorchOpenMPRuntime
    $pythonExitCode = Invoke-PythonScript -PythonExecutable $PYTHON -Script $torchCudaRuntimeCheck
    if ($pythonExitCode -ne 0) {
        Write-Err "PyTorch 三件套安装后仍未通过精确版本/CUDA 12.8/GPU 张量验证"
    }
}
Write-Ok "PyTorch 三件套与 NVIDIA CUDA 12.8 实测通过"
} else {
    $intelGpu = @()
    try {
        $intelGpu = @(
            Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop |
                Where-Object { $_.Name -match 'Intel.*(Arc|Graphics|UHD|Iris)' }
        )
    } catch {
        Write-Err "无法读取 Intel 显卡信息，不能验证 XPU 目标硬件: $_"
    }
    if ($intelGpu.Count -eq 0) {
        Write-Err "未检测到 Intel GPU；XPU 安装不会静默改用 CUDA、DirectML 或 CPU。"
    }
    foreach ($adapter in $intelGpu) {
        Write-Info "Intel GPU: $($adapter.Name)；驱动: $($adapter.DriverVersion)"
    }

    $torchCudaRuntimeCheck = @"
import os
from importlib import metadata
os.environ['PYTORCH_DEBUG_XPU_FALLBACK'] = '1'
expected = {'torch': '2.11.0', 'torchaudio': '2.11.0', 'torchvision': '0.26.0'}
actual = {name: metadata.version(name) for name in expected}
base = {name: version.split('+', 1)[0] for name, version in actual.items()}
if base != expected or any('+xpu' not in version for version in actual.values()):
    raise RuntimeError(f'Expected the pinned PyTorch XPU trio: {actual}')
try:
    metadata.version('intel-extension-for-pytorch')
except metadata.PackageNotFoundError:
    pass
else:
    raise RuntimeError('intel-extension-for-pytorch must not be installed; native torch.xpu is required')
import torch, torchaudio, torchvision
if getattr(torch.version, 'cuda', None) is not None or getattr(torch.version, 'hip', None) is not None:
    raise RuntimeError(f'XPU venv contains a CUDA/ROCm wheel: CUDA={torch.version.cuda!r}, HIP={torch.version.hip!r}')
if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
    raise RuntimeError('torch.xpu.is_available() is False; check the Intel graphics driver')
if torch.xpu.device_count() < 1:
    raise RuntimeError('torch.xpu reports no Intel GPU')
value = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='xpu:0')
result = torch.mm(value, value)
if result.device.type != 'xpu' or result.device.index != 0:
    raise RuntimeError(f'XPU probe ran on the wrong device: {result.device}')
if result.cpu().tolist() != [[7.0, 10.0], [15.0, 22.0]]:
    raise RuntimeError('XPU matrix multiplication returned an invalid result')
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    signal = torch.linspace(-1.0, 1.0, steps=4096, device='xpu:0')
    spectrum = torch.fft.rfft(signal)
    restored = torch.fft.irfft(spectrum, n=signal.shape[-1])
    torch.xpu.synchronize(0)
fallbacks = [
    str(item.message) for item in caught
    if 'xpu' in str(item.message).lower()
    and 'cpu' in str(item.message).lower()
    and ('fallback' in str(item.message).lower() or 'falling back' in str(item.message).lower())
]
if fallbacks:
    raise RuntimeError(f'XPU FFT fell back to CPU: {fallbacks}')
if float((signal - restored).abs().max().cpu()) > 1e-4:
    raise RuntimeError('XPU FFT numerical probe failed')
torch.xpu.synchronize(0)
print('PyTorch trio:', actual)
print('Intel XPU device:', torch.xpu.get_device_name(0))
"@

    $pythonExitCode = Invoke-PythonScript -PythonExecutable $PYTHON -Script $torchCudaRuntimeCheck
    if ($pythonExitCode -ne 0) {
        Write-Info "正在安装固定的原生 PyTorch XPU 三件套（不安装 IPEX）..."
        & "$PIP" install "torch==2.11.0" "torchaudio==2.11.0" "torchvision==0.26.0" `
            --index-url "https://download.pytorch.org/whl/xpu" --force-reinstall
        if ($LASTEXITCODE -ne 0) {
            Write-Err "PyTorch 2.11.0 / torchaudio 2.11.0 / torchvision 0.26.0 XPU 安装失败"
        }
        $pythonExitCode = Invoke-PythonScript -PythonExecutable $PYTHON -Script $torchCudaRuntimeCheck
        if ($pythonExitCode -ne 0) {
            Write-Err "PyTorch XPU 安装后仍未通过版本、设备驻留与真实张量运算门禁"
        }
    }
    Write-Ok "原生 PyTorch XPU 三件套与 Intel GPU 实测通过"
}

# --- 第 8 步/共 12 步：锁定完整运行时 ---
if ($Accelerator -eq "xpu") {
    Write-Info "第 8 步/共 12 步  完整七模式固定使用原生 PyTorch XPU + OpenVINO GPU；不安装 IPEX，不允许 CPU 降级。"
} else {
    Write-Info "第 8 步/共 12 步  完整七模式固定使用 NVIDIA CUDA，不安装 IPEX/CPU 降级运行时。"
}

# --- 第 9 步/共 12 步：安装项目依赖 ---
Write-Info "第 9 步/共 12 步  安装项目 Python 依赖..."

Set-Location $REPO_DIR
if (Test-PythonPackageInstalled -PackageName "audio-separator") {
    Write-Info "检测到旧的 audio-separator，先卸载以避免 NumPy 解析冲突..."
    & "$PIP" uninstall audio-separator -y
    if ($LASTEXITCODE -ne 0) { Write-Err "卸载旧 audio-separator 失败" }
}

$ortPackagesToRemove = @()
foreach ($ortPackage in @("onnxruntime", "onnxruntime-gpu", "onnxruntime-openvino", "openvino")) {
    if (Test-PythonPackageInstalled -PackageName $ortPackage) {
        $ortPackagesToRemove += $ortPackage
    }
}
if ($ortPackagesToRemove.Count -gt 0) {
    & "$PIP" uninstall @ortPackagesToRemove -y
    if ($LASTEXITCODE -ne 0) { Write-Err "清理冲突的 ONNX Runtime 包失败" }
}

$tmpReq = Join-Path $env:TEMP (
    "music-to-midi-requirements-{0}.txt" -f [System.Guid]::NewGuid().ToString("N")
)
$requirementsExitCode = 1
try {
    $requirementLines = @(
        Get-Content -Encoding UTF8 (Join-Path $REPO_DIR "requirements.txt") |
            Where-Object { $_ -notmatch '^\s*aria-amt\s*@' }
    )
    if ($Accelerator -eq "xpu") {
        $requirementLines = @(
            $requirementLines | Where-Object {
                $_ -notmatch '^\s*(torch|torchaudio|torchvision|onnxruntime|onnxruntime-gpu)(?:\s|[=<>!~@;\[])'
            }
        )
    }
    $requirementLines | Set-Content -Encoding UTF8 $tmpReq

    & "$PIP" install -r $tmpReq
    $requirementsExitCode = $LASTEXITCODE
}
finally {
    if (Test-Path -LiteralPath $tmpReq) {
        Remove-Item -LiteralPath $tmpReq -Force -ErrorAction Stop
    }
}
if ($requirementsExitCode -ne 0) { Write-Err "requirements.txt 安装失败" }
Write-Ok "Python 依赖安装成功"

# audio-separator 0.44.1 声明 numpy>=2，但当前 Windows CUDA 2.7 / XPU 2.11 桌面栈
# 上需要 NumPy 1.26.x。按发布脚本的做法，先安装其运行依赖，再 no-deps 安装包本体。
Write-Info "安装 audio-separator 运行依赖（固定兼容 NumPy 1.26）..."
$acceleratorOrtPackages = if ($Accelerator -eq "xpu") {
    @("onnxruntime-openvino==1.24.1", "openvino==2025.4.1")
} else {
    @("onnxruntime-gpu==1.23.2")
}
& "$PIP" install `
    "numpy==1.26.4" `
    "beartype==0.18.5" `
    "diffq-fixed==0.2.4" `
    "julius==0.2.7" `
    "ml_collections==1.1.0" `
    "onnx-weekly==1.21.0.dev20260223" `
    "onnx2torch-py313==1.6.0" `
    "pydub==0.25.1" `
    "requests>=2.32.5,<3" `
    "chardet>=5,<6" `
    @acceleratorOrtPackages `
    "resampy==0.4.3" `
    "rotary-embedding-torch==0.6.5" `
    "samplerate==0.1.0" `
    "h5py>=3.10,<4" `
    "mirdata>=0.3.8,<1" `
    "six==1.17.0"
if ($LASTEXITCODE -ne 0) { Write-Err "audio-separator 运行依赖安装失败" }

$pythonExitCode = Invoke-PythonScript -PythonExecutable $PYTHON -Script $torchCudaRuntimeCheck
if ($pythonExitCode -ne 0) {
    Write-Err "requirements.txt 安装后固定的 $($Accelerator.ToUpperInvariant()) PyTorch 运行时被改变"
}
& "$PYTHON" (Join-Path $REPO_DIR "tools\validate_accelerator_runtime.py") --accelerator $Accelerator
if ($LASTEXITCODE -ne 0) {
    Write-Err "$($Accelerator.ToUpperInvariant()) PyTorch/ONNX Runtime 实际执行门禁失败"
}

& "$PIP" install "audio-separator==0.44.1" --no-deps
if ($LASTEXITCODE -ne 0) { Write-Err "audio-separator 安装失败" }
Write-Ok "audio-separator 安装成功"

Write-Info "下载其它大型模型前预检 MuScriptor Small/Medium/Large gated 访问权限..."
& "$PYTHON" -c "from src.utils.muscriptor_downloader import preflight_muscriptor_download_access; preflight_muscriptor_download_access()"
if ($LASTEXITCODE -ne 0) {
    Write-Err "MuScriptor gated 访问预检失败；访问条件是浏览器已逐项接受三个仓库条款，并在项目 venv 中登录 Hugging Face。"
}
Write-Ok "MuScriptor Small/Medium/Large gated 访问权限预检通过"

Write-Info "安装并严格校验官方 MuScriptor v0.3.0（不改写固定 PyTorch 运行时）..."
$muscriptorRequirement = & "$PYTHON" -c "from src.utils.muscriptor_source_identity import MUSCRIPTOR_SOURCE_REQUIREMENT; print(MUSCRIPTOR_SOURCE_REQUIREMENT)"
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($muscriptorRequirement)) {
    Write-Err "无法从项目源码读取 MuScriptor 固定安装身份"
}
$muscriptorRequirement = ([string]$muscriptorRequirement).Trim()
& "$PIP" install $muscriptorRequirement --no-deps --force-reinstall
if ($LASTEXITCODE -ne 0) { Write-Err "MuScriptor v0.3.0 固定公共源码安装失败" }
& "$PYTHON" -m src.utils.source_runtime
if ($LASTEXITCODE -ne 0) { Write-Err "MuScriptor v0.3.0 官方源码身份校验失败" }
& "$PYTHON" -c "from src.core.muscriptor_transcriber import MuscriptorTranscriber; reason=MuscriptorTranscriber._runtime_unavailable_reason(); print(reason or 'MuScriptor public runtime identity/API verified'); raise SystemExit(0 if reason == '' else 1)"
if ($LASTEXITCODE -ne 0) { Write-Err "MuScriptor v0.3.0 官方运行时 API 校验失败" }

Write-Info "准备 MuScriptor 结果工作台所需的 FluidSynth 2.5.6..."
& "$PYTHON" (Join-Path $REPO_DIR "download_fluidsynth_runtime.py")
if ($LASTEXITCODE -ne 0) { Write-Err "FluidSynth 2.5.6 下载或身份校验失败" }
Write-Ok "MuScriptor 运行时与结果音频合成器准备完成"

# --- 第 9.4 步：严格验证 TransKun 默认 V2 包与随包资源 ---
Write-Info "第 9.4 步/共 12 步  验证 TransKun 2.0.1 默认 V2 运行时身份..."

$transkunIdentityScript = @"
import sys
sys.path.insert(0, r'$REPO_DIR')
from download_sota_models import validate_default_transkun_runtime
identity = validate_default_transkun_runtime()
print('TransKun default runtime:', identity)
"@

$pythonExitCode = Invoke-PythonScript -PythonExecutable $PYTHON -Script $transkunIdentityScript
if ($pythonExitCode -ne 0) {
    Write-Info "TransKun 默认 V2 身份不完整，正在强制重装 transkun==2.0.1..."
    & "$PIP" install "transkun==2.0.1" --no-deps --force-reinstall
    if ($LASTEXITCODE -ne 0) { Write-Err "TransKun 2.0.1 强制重装失败" }
    $pythonExitCode = Invoke-PythonScript -PythonExecutable $PYTHON -Script $transkunIdentityScript
    if ($pythonExitCode -ne 0) {
        Write-Err "TransKun 2.0.1 重装后包内 V2 资源仍未通过大小/SHA256 身份校验"
    }
}
Write-Ok "TransKun 默认 V2 严格身份检查通过"

# --- 第 9.5 步：验证 Aria-AMT 可用性并补装模型权重 ---
Write-Info "第 9.5 步/共 12 步  验证 Aria-AMT 钢琴后端..."

$ariaCheckScript = @"
import sys
sys.path.insert(0, r'$REPO_DIR')
from src.core.aria_amt_transcriber import AriaAmtTranscriber
reason = AriaAmtTranscriber.get_unavailable_reason()
print(reason or 'Aria-AMT source identity verified')
print('Aria-AMT package:', AriaAmtTranscriber.is_available())
print('Aria-AMT model:', AriaAmtTranscriber().is_model_available())
sys.exit(0 if reason == '' else 1)
"@

$pythonExitCode = Invoke-PythonScript -PythonExecutable $PYTHON -Script $ariaCheckScript
if ($pythonExitCode -ne 0) {
    Write-Info "Aria-AMT 缺失或源码身份不匹配，正在强制安装固定 GitHub archive..."
    & "$PIP" install "aria-amt @ https://github.com/EleutherAI/aria-amt/archive/a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip" --no-deps --force-reinstall
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Aria-AMT 安装校验失败：需要 Python 3.11+ 和可访问的 GitHub 仓库。"
    }
    $pythonExitCode = Invoke-PythonScript -PythonExecutable $PYTHON -Script $ariaCheckScript
    if ($pythonExitCode -ne 0) {
        Write-Err "Aria-AMT 安装后仍不可用"
    }
}

& "$PYTHON" (Join-Path $REPO_DIR "download_aria_amt_model.py")
if ($LASTEXITCODE -eq 0) {
    Write-Ok "Aria-AMT 模型准备完成"
} else {
    Write-Err "Aria-AMT 模型下载失败"
}

# --- 第 9.6 步：验证 ByteDance Piano 带踏板后端并补装模型权重 ---
Write-Info "第 9.6 步/共 12 步  验证 ByteDance Piano 带踏板钢琴后端..."

$byteDanceCheckScript = @"
import sys
sys.path.insert(0, r'$REPO_DIR')
from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber
print('ByteDance Piano package:', ByteDancePianoTranscriber.is_available())
print('ByteDance Piano model:', ByteDancePianoTranscriber().is_model_available())
sys.exit(0 if ByteDancePianoTranscriber.is_available() else 1)
"@

$pythonExitCode = Invoke-PythonScript -PythonExecutable $PYTHON -Script $byteDanceCheckScript
if ($pythonExitCode -ne 0) {
    Write-Err "ByteDance Piano 安装校验失败：piano-transcription-inference、torchlibrosa 或 matplotlib 不完整。"
}

& "$PYTHON" (Join-Path $REPO_DIR "download_bytedance_piano_model.py")
if ($LASTEXITCODE -eq 0) {
    Write-Ok "ByteDance Piano 模型准备完成"
} else {
    Write-Err "ByteDance Piano 模型下载失败"
}

# --- 第 9.7 步：准备 MIROS 实验后端源码和权重 ---
Write-Info "第 9.7 步/共 12 步  准备 MIROS 多乐器后端（可选实验）..."

$mirosDir = Join-Path $REPO_DIR "external\ai4m-miros"
& "$PYTHON" (Join-Path $REPO_DIR "download_miros_model.py") --repo-dir "$mirosDir"
if ($LASTEXITCODE -ne 0) { Write-Err "MIROS 权重准备失败" }

$mirosHasSource = (Test-Path (Join-Path $mirosDir "main.py")) -and (Test-Path (Join-Path $mirosDir "transcribe.py"))
if ((-not (Test-Path (Join-Path $mirosDir ".git"))) -and (-not $mirosHasSource)) {
    Write-Info "MIROS 源码不存在，正在克隆 ai4m-miros..."
    if (-not (Test-Path (Split-Path -Parent $mirosDir))) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $mirosDir) | Out-Null
    }
    & git clone https://github.com/amt-os/ai4m-miros.git "$mirosDir"
    if ($LASTEXITCODE -ne 0) { Write-Err "MIROS 源码克隆失败" }
} elseif (Test-Path (Join-Path $mirosDir ".git")) {
    Write-Info "MIROS 源码已存在，检查 main 分支更新..."
    & git -C "$mirosDir" fetch --depth=1 origin main
    if ($LASTEXITCODE -ne 0) { Write-Err "MIROS 源码更新失败" }
} else {
    Write-Info "MIROS 源码已存在，使用当前目录。"
}

& "$PIP" install "h5py>=3.10,<4" "mirdata>=0.3.8,<1"
if ($LASTEXITCODE -ne 0) { Write-Err "MIROS 基础依赖安装失败" }

$mirosPrepScript = @"
import pathlib
import sys
sys.path.insert(0, r'$REPO_DIR')
from src.core.miros_transcriber import MirosTranscriber

repo = pathlib.Path(r'$mirosDir')
pretrained = repo / MirosTranscriber.PRETRAINED_REL_PATH
checkpoint = repo / MirosTranscriber.CHECKPOINT_REL_PATH
print('MIROS source:', repo)
print('MIROS pretrained:', pretrained.exists())
print('MIROS checkpoint:', checkpoint.exists())
sys.exit(0 if repo.exists() and (repo / 'main.py').exists() and (repo / 'transcribe.py').exists() else 1)
"@

$pythonExitCode = Invoke-PythonScript -PythonExecutable $PYTHON -Script $mirosPrepScript
if ($pythonExitCode -ne 0) { Write-Err "MIROS 源码目录不完整" }

$mirosMain = Join-Path $mirosDir "main.py"
if ((-not (Test-Path (Join-Path $mirosDir "model\musicfm\data\pretrained_msd.pt"))) -or
    (-not (Test-Path (Join-Path $mirosDir "logs\Multi_longer_seq_length_frozen_enc_silu\le2bzt53\checkpoints\last.ckpt")))) {
    Write-Info "MIROS 权重缺失，调用上游 main.py 准备 Google Drive 权重..."
    $dummyInput = Join-Path $REPO_DIR "output\miros_weight_probe.wav"
    $dummyOutput = Join-Path $REPO_DIR "output\miros_weight_probe.mid"
    if (-not (Test-Path (Split-Path -Parent $dummyInput))) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dummyInput) | Out-Null
    }
    & "$PYTHON" -c "import wave; p=r'$dummyInput'; f=wave.open(p, 'wb'); f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(b'\0\0' * 1600); f.close()"
    if ($LASTEXITCODE -ne 0) { Write-Err "MIROS 权重探测音频生成失败" }
    Push-Location $mirosDir
    try {
        & "$PYTHON" "$mirosMain" -i "$dummyInput" -o "$dummyOutput"
        if ($LASTEXITCODE -ne 0) { Write-Err "MIROS 权重准备失败" }
    } finally {
        Pop-Location
    }
    if ((-not (Test-Path (Join-Path $mirosDir "model\musicfm\data\pretrained_msd.pt"))) -or
        (-not (Test-Path (Join-Path $mirosDir "logs\Multi_longer_seq_length_frozen_enc_silu\le2bzt53\checkpoints\last.ckpt")))) {
        Write-Err "MIROS 权重准备失败"
    }
}

& "$PYTHON" -c "import sys; sys.path.insert(0, r'$REPO_DIR'); from src.core.miros_transcriber import MirosTranscriber; reason=MirosTranscriber.get_unavailable_reason(); print(reason or 'MIROS available'); print('MIROS model:', MirosTranscriber.is_model_available()); sys.exit(0 if reason == '' else 1)"
if ($LASTEXITCODE -ne 0) { Write-Err "MIROS 后端检查失败" }

# --- 第 10 步/共 12 步：验证核心依赖 ---
Write-Info "第 10 步/共 12 步  验证核心依赖..."

foreach ($dep in @("PyQt6", "torch", "numpy", "librosa", "mido", "soundfile", "pytorch_lightning", "beat_this", "fastapi", "amt.run", "audio_separator.separator", "h5py", "mirdata")) {
    Write-Info "  正在验证 $dep..."
    & "$PYTHON" -c "import importlib; m=importlib.import_module('$dep'); print(getattr(m, '__version__', 'unknown'))"
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "  $dep 正常"
    } else {
        Write-Err "  $dep 导入失败（上方已显示完整错误输出）"
    }
}

if ($FFMPEG_OK) {
    Write-Ok "  ffmpeg 正常"
} else {
    Write-Err "  ffmpeg 未安装；音频转换所需运行时不完整"
}

Write-Info "验证受控 YourMT3+ 源码树身份..."
$yourMt3SourceIdentityScript = @"
import sys
from pathlib import Path
sys.path.insert(0, r'$REPO_DIR')
from src.utils.yourmt3_source_identity import validate_patched_yourmt3_source
source_dir = Path(r'$REPO_DIR') / 'YourMT3' / 'amt' / 'src'
manifest, file_count = validate_patched_yourmt3_source(source_dir)
print('YourMT3+ patched source manifest:', manifest)
print('YourMT3+ patched source files:', file_count)
"@
$pythonExitCode = Invoke-PythonScript -PythonExecutable $PYTHON -Script $yourMt3SourceIdentityScript
if ($pythonExitCode -ne 0) {
    Write-Err "YourMT3+ 源码树缺失或身份不匹配；需要恢复当前项目版本，可变上游源码与本版本不兼容。"
}
Write-Ok "YourMT3+ 源码身份检查通过"

# --- 第 11 步/共 12 步：下载统一模型集合 ---
Write-Info "第 11 步/共 12 步  下载并校验全部公开工作流模型（含 MuScriptor Small/Medium/Large 与 SoundFont）..."

Set-Location $REPO_DIR
& "$PYTHON" (Join-Path $REPO_DIR "download_sota_models.py")
if ($LASTEXITCODE -eq 0) {
    Write-Ok "YourMT3+、MIROS、分离/钢琴模型、MuScriptor Small/Medium/Large 与 SoundFont 全部准备成功"
} else {
    Write-Err "统一模型集合下载或校验失败"
}

# --- 第 12 步/共 12 步：逐项验证新模型路线 ---
Write-Info "第 12 步/共 12 步  验证 Leap XE 人声、PolarFormer 伴奏与 TransKun V2 Aug 模型..."

Set-Location $REPO_DIR
& "$PYTHON" (Join-Path $REPO_DIR "download_vocal_model.py")
if ($LASTEXITCODE -eq 0) {
    Write-Ok "Leap XE 90-band 人声模型准备完成"
} else {
    Write-Err "Leap XE 90-band 人声模型下载或校验失败"
}

& "$PYTHON" (Join-Path $REPO_DIR "download_accompaniment_model.py")
if ($LASTEXITCODE -eq 0) {
    Write-Ok "PolarFormer 伴奏模型准备完成"
} else {
    Write-Err "PolarFormer 伴奏模型下载或校验失败"
}

& "$PYTHON" (Join-Path $REPO_DIR "download_transkun_v2_aug_model.py")
if ($LASTEXITCODE -eq 0) {
    Write-Ok "TransKun V2 Aug 模型准备完成"
} else {
    Write-Err "TransKun V2 Aug 模型下载或校验失败"
}

# --- 完成 ---
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  运行方式：" -ForegroundColor White
if ($Accelerator -eq "xpu") {
    Write-Host "  .\run_xpu.ps1             （Intel XPU）" -ForegroundColor Green
} else {
    Write-Host "  .\run.bat                 （NVIDIA CUDA，推荐）" -ForegroundColor Green
    Write-Host "  .\run.ps1" -ForegroundColor Green
}
Write-Host ""
Write-Host "  模型维护命令：" -ForegroundColor White
Write-Host "  $VenvName\Scripts\python.exe download_sota_models.py" -ForegroundColor Yellow
Write-Host "  $VenvName\Scripts\python.exe download_beat_this_model.py" -ForegroundColor Yellow
Write-Host "  $VenvName\Scripts\python.exe download_multistem_model.py" -ForegroundColor Yellow
Write-Host "  $VenvName\Scripts\python.exe download_vocal_model.py" -ForegroundColor Yellow
Write-Host "  $VenvName\Scripts\python.exe download_accompaniment_model.py" -ForegroundColor Yellow
Write-Host "  $VenvName\Scripts\python.exe download_transkun_v2_aug_model.py" -ForegroundColor Yellow
Write-Host "  $VenvName\Scripts\python.exe download_bytedance_piano_model.py" -ForegroundColor Yellow
Write-Host "  $VenvName\Scripts\python.exe download_muscriptor_model.py --size all" -ForegroundColor Yellow
Write-Host "  $VenvName\Scripts\python.exe download_fluidsynth_runtime.py" -ForegroundColor Yellow
Write-Host ""
