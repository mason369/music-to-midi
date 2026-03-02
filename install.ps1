# Music to MIDI - Windows 安装脚本
# 支持 Windows 10/11 (x64)
# 用法: powershell -ExecutionPolicy Bypass -File install.ps1
# 或双击 install.bat

#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
# 设置控制台输出编码为 UTF-8，避免中文乱码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$REPO_DIR = $PSScriptRoot
$VENV_DIR = Join-Path $REPO_DIR "venv"
$PIP      = Join-Path $VENV_DIR "Scripts\pip.exe"
$PYTHON   = Join-Path $VENV_DIR "Scripts\python.exe"

# --- 辅助输出函数 ---
function Write-Info  { param($msg) Write-Host "[信息]  $msg" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "[完成]  $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "[警告]  $msg" -ForegroundColor Yellow }
function Write-Err   { param($msg) Write-Host "[错误] $msg" -ForegroundColor Red; exit 1 }

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
Write-Host "  Music to MIDI  -  Windows 安装程序" -ForegroundColor Cyan
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
        Write-Host "  已取消安装。请将项目移动到无特殊字符的路径后重试。" -ForegroundColor Yellow
        exit 0
    }
    Write-Warn "继续安装（路径问题可能导致运行失败）..."
}

# --- 第 1 步/共 12 步：检测 Python 版本 ---
Write-Info "第 1 步/共 12 步  检测 Python 版本..."

$PYTHON_BIN = $null

# 优先使用 py launcher 指定兼容版本（3.10~3.12），避免选中过新的 Python（3.13+ 尚未被 PyTorch 完全支持）
foreach ($ver in @("-3.12", "-3.11", "-3.10")) {
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

# 回退：检测 py / python 命令，但限制版本 3.10~3.12
if (-not $PYTHON_BIN) {
    foreach ($cmd in @("py", "python")) {
        try {
            $verStr = & $cmd -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))" 2>&1
            if ($LASTEXITCODE -eq 0 -and ("$verStr" -match '^(\d+)\.(\d+)')) {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -eq 3 -and $minor -ge 10 -and $minor -le 12) {
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
    Write-Host "[错误] 未找到兼容的 Python 版本（需要 3.10~3.12）。" -ForegroundColor Red
    Write-Host "  当前系统可能安装了过新的 Python（3.13+），PyTorch 尚不支持。" -ForegroundColor Yellow
    Write-Host "  请安装 Python 3.11: winget install Python.Python.3.11" -ForegroundColor Yellow
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
    Write-Host "[错误] 未找到 git，请执行: winget install Git.Git" -ForegroundColor Red
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
            try {
                $zipUrl  = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
                $zipPath = Join-Path $env:TEMP "ffmpeg_dl.zip"
                $extractPath = Join-Path $env:TEMP "ffmpeg_extract"

                # 使用 Invoke-Download 显示实时进度
                Invoke-Download -Url $zipUrl -OutFile $zipPath -Description "正在下载 ffmpeg（约 90 MB）..."

                Write-Info "  正在解压..."
                if (Test-Path $extractPath) { Remove-Item $extractPath -Recurse -Force }
                Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
                Remove-Item $zipPath -Force

                # 找到解压后含 ffmpeg.exe 的 bin 目录（子文件夹名随版本变化）
                $binFolder = Get-ChildItem $extractPath -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
                if ($binFolder) {
                    $srcBin = $binFolder.DirectoryName
                    if (-not (Test-Path $ffmpegBin)) { New-Item $ffmpegBin -ItemType Directory -Force | Out-Null }
                    Copy-Item "$srcBin\*.exe" $ffmpegBin -Force
                    Remove-Item $extractPath -Recurse -Force
                    $installed = $true
                    Write-Ok "  ffmpeg 已解压至 $ffmpegBin"
                }
            }
            catch {
                Write-Warn "  直接下载失败: $_"
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
        if ($installed) {
            Write-Warn "ffmpeg 已安装，重启终端后完全生效"
            $FFMPEG_OK = $true
        } else {
            Write-Warn "ffmpeg 自动安装失败，请手动安装："
            Write-Warn "  Scoop: scoop install ffmpeg"
            Write-Warn "  或从 https://www.gyan.dev/ffmpeg/builds/ 下载后加入 PATH"
        }
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
    try {
        $vcUrl  = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
        $vcPath = Join-Path $env:TEMP "vc_redist.x64.exe"

        # 使用 Invoke-Download 显示实时进度
        Invoke-Download -Url $vcUrl -OutFile $vcPath -Description "正在下载 VC++ 2022 Redistributable（约 25 MB）..."

        Write-Info "  正在静默安装..."
        $proc = Start-Process -FilePath $vcPath -ArgumentList "/install /quiet /norestart" -Wait -PassThru
        Remove-Item $vcPath -Force -ErrorAction SilentlyContinue

        # 退出码 0 = 成功，3010 = 成功但需重启
        if ($proc.ExitCode -eq 0 -or $proc.ExitCode -eq 3010) {
            Write-Ok "Visual C++ Redistributable 2022 安装成功"
            if ($proc.ExitCode -eq 3010) {
                Write-Warn "  建议重启系统以使 DLL 完全生效（可稍后重启）"
            }
            $vcRedistOk = $true
        } else {
            # 非零非 3010 可能是"已安装更高版本"（如 1638），视为成功继续
            Write-Warn "  安装程序返回码: $($proc.ExitCode)（可能已有更高版本，继续执行）"
            $vcRedistOk = $true
        }
    }
    catch {
        Write-Warn "VC++ Redistributable 自动安装失败: $_"
        Write-Warn "  请手动下载安装后重新运行本脚本："
        Write-Warn "  https://aka.ms/vs/17/release/vc_redist.x64.exe"
    }
}

# --- 第 5 步/共 12 步：创建 Python 虚拟环境 ---
Write-Info "第 5 步/共 12 步  创建 Python 虚拟环境..."

if (-not (Test-Path $VENV_DIR) -or -not (Test-Path $PYTHON)) {
    if ((Test-Path $VENV_DIR) -and -not (Test-Path $PYTHON)) {
        Write-Warn "虚拟环境目录存在但缺少 python.exe，正在重新创建..."
    }
    & @PYTHON_CMD -m venv "$VENV_DIR"
    if ($LASTEXITCODE -ne 0) { Write-Err "创建虚拟环境失败" }
    Write-Ok "虚拟环境创建成功: $VENV_DIR"
} else {
    Write-Ok "虚拟环境已存在: $VENV_DIR"
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

# --- 第 7 步/共 12 步：检测 GPU / 安装 PyTorch ---
Write-Info "第 7 步/共 12 步  检测 GPU / 安装 PyTorch..."

$TORCH_INSTALLED = $false
$torchDistInfo = Get-ChildItem (Join-Path $VENV_DIR "Lib\site-packages") -Directory `
    -Filter "torch-*.dist-info" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($torchDistInfo -and $torchDistInfo.Name -match '^torch-([\d.]+)') {
    $torchVer = $Matches[1]
    $tvParts = $torchVer.Split('.')
    if ([int]$tvParts[0] -ge 2 -and ([int]$tvParts[0] -gt 2 -or [int]$tvParts[1] -ge 4)) {
        Write-Ok "PyTorch $torchVer 已安装（dist-info 验证，跳过重新安装）"
        $TORCH_INSTALLED = $true
        # 检查 torchvision 是否缺失，缺失则从 PyTorch 官方源补装（版本必须匹配 torch）
        $tvisionDistInfo = Get-ChildItem (Join-Path $VENV_DIR "Lib\site-packages") -Directory `
            -Filter "torchvision-*.dist-info" -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $tvisionDistInfo) {
            # torch 2.x.y → torchvision 0.(x+15).y（PyTorch 官方版本映射）
            $tvMinor = [int]$tvParts[1] + 15
            Write-Info "torchvision 未安装，正在从 PyTorch 官方源补装 (0.$tvMinor.*)..."
            $TV_INDEX = "https://download.pytorch.org/whl/cpu"
            try {
                $nvOut = & nvidia-smi 2>&1
                if ($LASTEXITCODE -eq 0) {
                    $nvS = $nvOut -join " "
                    $cm = [regex]::Match($nvS, 'CUDA Version:\s*([\d.]+)')
                    if ($cm.Success) {
                        $cmaj = [int]($cm.Groups[1].Value -split '\.')[0]
                        if ($cmaj -ge 12) { $TV_INDEX = "https://download.pytorch.org/whl/cu121" }
                        elseif ($cmaj -ge 11) { $TV_INDEX = "https://download.pytorch.org/whl/cu118" }
                    }
                }
            } catch {}
            $tvNextMinor = $tvMinor + 1
            & "$PIP" install "torchvision>=0.$tvMinor.0,<0.$tvNextMinor.0" --index-url "$TV_INDEX"
        }
    }
}
if (-not $TORCH_INSTALLED) {
    $TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
    $TORCH_LABEL = "CPU"

    # 检测 NVIDIA GPU (CUDA)
    try {
        $nvOutput = & nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            $nvStr = $nvOutput -join " "
            $cudaMatch = [regex]::Match($nvStr, 'CUDA Version:\s*([\d.]+)')
            if ($cudaMatch.Success) {
                $cudaMajor = [int]($cudaMatch.Groups[1].Value -split '\.')[0]
                if ($cudaMajor -ge 12) {
                    $TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
                    $TORCH_LABEL = "CUDA 12.1 (NVIDIA)"
                } elseif ($cudaMajor -ge 11) {
                    $TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
                    $TORCH_LABEL = "CUDA 11.8 (NVIDIA)"
                }
            }
        }
    }
    catch {
        # nvidia-smi 不可用
    }

    # 检测 AMD GPU（Windows 暂无原生 ROCm wheel）
    try {
        $gpuInfo = Get-CimInstance -ClassName Win32_VideoController -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match 'AMD|Radeon' }
        if ($gpuInfo) {
            Write-Warn "检测到 AMD GPU，Windows 暂无原生 PyTorch ROCm wheel。"
            Write-Warn "  将使用 CPU 模式，如需 GPU 加速请使用 Linux/WSL2。"
        }
    }
    catch {
        # 无法获取 GPU 信息
    }

    Write-Info "正在安装 PyTorch ($TORCH_LABEL)..."
    & "$PIP" install "torch==2.4.0" "torchaudio==2.4.0" "torchvision==0.19.0" --index-url "$TORCH_INDEX"
    if ($LASTEXITCODE -ne 0) { Write-Err "PyTorch 安装失败" }
    Write-Ok "PyTorch ($TORCH_LABEL) 安装成功"
}

# --- 第 7.5 步：确保 libomp140.x86_64.dll 存在（fbgemm.dll 依赖）---
$torchLib = Join-Path $VENV_DIR "Lib\site-packages\torch\lib"
$libompDll = Join-Path $torchLib "libomp140.x86_64.dll"

if ((Test-Path $torchLib) -and -not (Test-Path $libompDll)) {
    Write-Info "正在修复 libomp140.x86_64.dll 缺失问题（fbgemm.dll 依赖 LLVM OpenMP 运行时）..."
    try {
        & "$PIP" install zstandard
        & "$PYTHON" -c @"
import zipfile, tarfile, io, os, sys
try:
    import zstandard
except ImportError:
    print('ERROR: zstandard not available'); sys.exit(1)
tmp = os.environ['TEMP']
conda_url = 'https://api.anaconda.org/download/conda-forge/llvm-openmp/19.1.7/win-64/llvm-openmp-19.1.7-h30eaf37_1.conda'
conda_path = os.path.join(tmp, 'llvm-openmp.conda')
# Download
import urllib.request
urllib.request.urlretrieve(conda_url, conda_path)
# Extract tar.zst from conda zip
extract_dir = os.path.join(tmp, 'llvm_omp_extract')
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(conda_path) as z:
    zst_names = [n for n in z.namelist() if n.startswith('pkg-') and n.endswith('.tar.zst')]
    if not zst_names: print('ERROR: no pkg tar.zst in conda'); sys.exit(1)
    z.extract(zst_names[0], extract_dir)
# Decompress zst -> tar -> extract libomp.dll
zst_path = os.path.join(extract_dir, zst_names[0])
dctx = zstandard.ZstdDecompressor()
with open(zst_path, 'rb') as f:
    tar_data = dctx.decompress(f.read(), max_output_size=50*1024*1024)
torch_lib = sys.argv[1]
with tarfile.open(fileobj=io.BytesIO(tar_data)) as tf:
    member = tf.getmember('Library/bin/libomp.dll')
    data = tf.extractfile(member).read()
    dest = os.path.join(torch_lib, 'libomp140.x86_64.dll')
    with open(dest, 'wb') as out: out.write(data)
    print(f'OK: wrote {len(data)} bytes to {dest}')
# Cleanup
import shutil
os.remove(conda_path)
shutil.rmtree(extract_dir, ignore_errors=True)
"@ "$torchLib"
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "libomp140.x86_64.dll 已修复"
        } else {
            Write-Warn "libomp140.x86_64.dll 自动修复失败，torch 可能无法正常导入"
        }
    }
    catch {
        Write-Warn "libomp140.x86_64.dll 修复失败: $_"
    }
} elseif (Test-Path $libompDll) {
    Write-Ok "libomp140.x86_64.dll 已存在"
}

# --- 第 8 步/共 12 步：Intel GPU 加速（可选）---
Write-Info "第 8 步/共 12 步  检测 Intel GPU 加速（可选）..."

# 预先检查：若已安装 IPEX 但 torch 无法正常导入，说明版本不兼容，自动卸载
$preIpexDir = Join-Path $VENV_DIR "Lib\site-packages\intel_extension_for_pytorch"
if (Test-Path $preIpexDir) {
    $prevEAP = $ErrorActionPreference; $ErrorActionPreference = "SilentlyContinue"
    $null = & "$PYTHON" -c "import torch" 2>&1
    $ErrorActionPreference = $prevEAP
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "已安装的 IPEX 与当前 torch 版本不兼容（torch 导入失败），正在自动卸载 IPEX..."
        & "$PIP" uninstall intel_extension_for_pytorch -y
        Write-Info "已卸载 IPEX，torch 将以 CPU 模式正常运行"
    }
}

$intelGpuFound = $false
try {
    $gpuInfo = Get-CimInstance -ClassName Win32_VideoController -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'Intel.*Arc|Intel.*Xe' }
    if ($gpuInfo) {
        $intelGpuFound = $true
        Write-Info "检测到 Intel GPU: $($gpuInfo.Name)"
    }
}
catch {
    # 无法获取 GPU 信息
}

if ($intelGpuFound) {
    $ipexInstalled = $false
    $ipexSiteDir = Join-Path $VENV_DIR "Lib\site-packages\intel_extension_for_pytorch"
    if (Test-Path $ipexSiteDir) { $ipexInstalled = $true }

    if (-not $ipexInstalled) {
        Write-Info "正在尝试安装 intel_extension_for_pytorch（Intel GPU 加速）..."
        try {
            & "$PIP" install intel_extension_for_pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "intel_extension_for_pytorch 安装成功（Intel GPU 加速已启用）"
                # 验证 IPEX 安装后 torch 是否仍可正常导入（防止版本不兼容破坏 torch）
                $prevEAP2 = $ErrorActionPreference; $ErrorActionPreference = "SilentlyContinue"
                $null = & "$PYTHON" -c "import torch" 2>&1
                $ErrorActionPreference = $prevEAP2
                if ($LASTEXITCODE -ne 0) {
                    Write-Warn "IPEX 与 torch 版本不兼容（IPEX XPU 需要匹配的 torch 版本），正在自动卸载 IPEX..."
                    & "$PIP" uninstall intel_extension_for_pytorch -y
                    Write-Info "已卸载 IPEX，将以 CPU 模式运行"
                }            } else {
                Write-Warn "intel_extension_for_pytorch 安装失败，将使用 CPU 模式。"
            }
        }
        catch {
            Write-Warn "intel_extension_for_pytorch 安装失败（可选）"
        }
    } else {
        Write-Ok "intel_extension_for_pytorch 已安装"
    }
} else {
    Write-Info "未检测到 Intel GPU（Arc/Xe/UHD/Iris），跳过 IPEX 安装"
}

# --- 第 9 步/共 12 步：安装项目依赖 ---
Write-Info "第 9 步/共 12 步  安装项目 Python 依赖..."

Set-Location $REPO_DIR
& "$PIP" install -r (Join-Path $REPO_DIR "requirements.txt")
if ($LASTEXITCODE -ne 0) { Write-Err "requirements.txt 安装失败" }
Write-Ok "Python 依赖安装成功"

# --- 第 10 步/共 12 步：验证核心依赖 ---
Write-Info "第 10 步/共 12 步  验证核心依赖..."

foreach ($dep in @("PyQt6", "torch", "librosa", "mido", "soundfile", "pytorch_lightning")) {
    Write-Info "  正在验证 $dep..."
    & "$PYTHON" -c "import importlib; m=importlib.import_module('$dep'); print(getattr(m, '__version__', 'unknown'))"
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "  $dep 正常"
    } else {
        Write-Warn "  $dep 导入失败（上方已显示完整错误输出）"
    }
}

if ($FFMPEG_OK) {
    Write-Ok "  ffmpeg 正常"
} else {
    Write-Warn "  ffmpeg 未安装（已在第 3 步提示）"
}

# --- 第 11 步/共 12 步：下载 SOTA 模型权重 ---
Write-Info "第 11 步/共 12 步  下载 YourMT3+ SOTA 模型权重（约 800 MB）..."
Write-Info "按 Ctrl+C 可跳过，稍后手动执行: venv\Scripts\python.exe download_sota_models.py"

Set-Location $REPO_DIR
try {
    & "$PYTHON" (Join-Path $REPO_DIR "download_sota_models.py")
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "SOTA 模型权重下载成功"
    } else {
        Write-Host ""
        Write-Host "  !! 模型下载失败 !!" -ForegroundColor Red
        Write-Host "  可能原因：网络问题 / SSL 证书验证失败 / 代理环境" -ForegroundColor Yellow
        Write-Host "  请稍后手动执行: venv\Scripts\python.exe download_sota_models.py" -ForegroundColor Yellow
        Write-Host "  （上方已输出完整下载日志）" -ForegroundColor Yellow
        Write-Host ""
    }
}
catch {
    Write-Host ""
    Write-Host "  !! 模型下载失败 !!" -ForegroundColor Red
    Write-Host "  请稍后手动执行: venv\Scripts\python.exe download_sota_models.py" -ForegroundColor Yellow
    Write-Host ""
}

# --- 第 12 步/共 12 步：下载 BS-RoFormer 人声分离模型 ---
Write-Info "第 12 步/共 12 步  下载 BS-RoFormer ep368 模型（约 600 MB）..."
Write-Info "按 Ctrl+C 可跳过，稍后手动执行: venv\Scripts\python.exe download_vocal_model.py"

Set-Location $REPO_DIR
try {
    & "$PYTHON" (Join-Path $REPO_DIR "download_vocal_model.py")
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "BS-RoFormer 模型下载成功"
    } else {
        Write-Host ""
        Write-Host "  !! BS-RoFormer 模型下载失败 !!" -ForegroundColor Red
        Write-Host "  可能原因：网络问题 / HuggingFace 连接失败 / 代理环境" -ForegroundColor Yellow
        Write-Host "  请稍后手动执行: venv\Scripts\python.exe download_vocal_model.py" -ForegroundColor Yellow
        Write-Host "  （上方已输出完整下载日志）" -ForegroundColor Yellow
        Write-Host ""
    }
}
catch {
    Write-Host ""
    Write-Host "  !! BS-RoFormer 模型下载失败 !!" -ForegroundColor Red
    Write-Host "  请稍后手动执行: venv\Scripts\python.exe download_vocal_model.py" -ForegroundColor Yellow
    Write-Host ""
}

# --- 完成 ---
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  运行方式：" -ForegroundColor White
Write-Host "  .\run.bat                 （推荐）" -ForegroundColor Green
Write-Host "  .\run.ps1" -ForegroundColor Green
Write-Host ""
Write-Host "  如果模型下载失败，请手动执行：" -ForegroundColor White
Write-Host "  venv\Scripts\python.exe download_sota_models.py" -ForegroundColor Yellow
Write-Host "  venv\Scripts\python.exe download_vocal_model.py" -ForegroundColor Yellow
Write-Host ""
