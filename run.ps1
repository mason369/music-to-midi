# Music to MIDI - Windows 启动脚本
# 检查依赖，如需要则自动运行安装程序，然后启动应用
# 用法: powershell -ExecutionPolicy Bypass -File run.ps1
# 或双击 run.bat

#Requires -Version 5.1

$REPO_DIR    = $PSScriptRoot
$VENV_PYTHON = Join-Path $REPO_DIR "venv\Scripts\python.exe"

function Write-Info { param($msg) Write-Host "[信息]  $msg" -ForegroundColor Cyan }
function Write-Ok   { param($msg) Write-Host "[完成]  $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[警告]  $msg" -ForegroundColor Yellow }

# --- 依赖检查 ---
$NEED_INSTALL = $false

# 检查 1：虚拟环境
if (-not (Test-Path $VENV_PYTHON)) {
    Write-Warn "未找到虚拟环境"
    $NEED_INSTALL = $true
}

# 检查 2：核心 Python 包
if (-not $NEED_INSTALL) {
    Write-Info "检查核心 Python 包（PyQt6/librosa/mido）..."
    & "$VENV_PYTHON" -c "import PyQt6, librosa, mido; print('核心依赖导入成功')"
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "核心 Python 包缺失（上方已显示完整错误输出）"
        $NEED_INSTALL = $true
    } else {
        Write-Ok "核心 Python 包检查通过"
    }
}

# 检查 3：YourMT3+ 模型权重
if (-not $NEED_INSTALL) {
    Write-Info "检查 YourMT3+ 模型权重..."
    $checkScript = @"
import sys
sys.path.insert(0, r'$REPO_DIR')
from src.utils.yourmt3_downloader import get_model_path
model_path = get_model_path()
print('YourMT3+ model:', model_path if model_path else 'missing')
sys.exit(0 if model_path else 1)
"@
    & "$VENV_PYTHON" -c $checkScript
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "未找到 YourMT3+ 模型权重"
        $NEED_INSTALL = $true
    } else {
        Write-Ok "YourMT3+ 模型权重检查通过"
    }
}

# 检查 4：BS-RoFormer 人声分离模型
if (-not $NEED_INSTALL) {
    Write-Info "检查 BS-RoFormer 人声分离模型..."
    $checkVocalScript = @"
import sys
sys.path.insert(0, r'$REPO_DIR')
from download_vocal_model import is_vocal_model_available, resolve_vocal_model_path
target = resolve_vocal_model_path()
print('BS-RoFormer model:', target)
sys.exit(0 if is_vocal_model_available() else 1)
"@
    & "$VENV_PYTHON" -c $checkVocalScript
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "未找到 BS-RoFormer 模型权重"
        $NEED_INSTALL = $true
    } else {
        Write-Ok "BS-RoFormer 模型权重检查通过"
    }
}

# 检查 5：BS-RoFormer SW 六声部分离模型
if (-not $NEED_INSTALL) {
    Write-Info "检查 BS-RoFormer SW 六声部分离模型..."
    $checkMultistemScript = @"
import sys
sys.path.insert(0, r'$REPO_DIR')
from download_multistem_model import is_multistem_model_available, resolve_multistem_model_paths
model_path, config_path = resolve_multistem_model_paths()
print('BS-RoFormer SW model:', model_path)
print('BS-RoFormer SW config:', config_path)
sys.exit(0 if is_multistem_model_available() else 1)
"@
    & "$VENV_PYTHON" -c $checkMultistemScript
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "未找到 BS-RoFormer SW 六声部模型资源"
        $NEED_INSTALL = $true
    } else {
        Write-Ok "BS-RoFormer SW 六声部模型检查通过"
    }
}

# 检查 6：Aria-AMT 包
if (-not $NEED_INSTALL) {
    Write-Info "检查 Aria-AMT 包..."
    & "$VENV_PYTHON" -c "import importlib.util; ok = importlib.util.find_spec('amt.run') is not None; print('Aria-AMT package:', 'ok' if ok else 'missing'); raise SystemExit(0 if ok else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "未安装 Aria-AMT 包（可选，钢琴专用模式不可用）"
    } else {
        Write-Ok "Aria-AMT 包检查通过"
    }
}

# 检查 7：Aria-AMT 钢琴模型
if (-not $NEED_INSTALL) {
    Write-Info "检查 Aria-AMT 钢琴模型..."
    $checkAriaScript = @"
import sys
sys.path.insert(0, r'$REPO_DIR')
from download_aria_amt_model import is_aria_model_available, resolve_aria_model_path
model_path = resolve_aria_model_path()
print('Aria-AMT model:', model_path)
sys.exit(0 if is_aria_model_available() else 1)
"@
    & "$VENV_PYTHON" -c $checkAriaScript
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "未找到 Aria-AMT 钢琴模型权重（可选，钢琴专用模式不可用）"
    } else {
        Write-Ok "Aria-AMT 钢琴模型检查通过"
    }
}

# --- 如需要则运行安装程序 ---
if ($NEED_INSTALL) {
    Write-Info "依赖不完整，正在运行安装程序..."
    $installScript = Join-Path $REPO_DIR "install.ps1"
    & powershell -ExecutionPolicy Bypass -File "$installScript"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[错误] 安装失败，请查看上方错误信息。" -ForegroundColor Red
        Read-Host "按回车键退出"
        exit 1
    }
    Write-Ok "安装完成，正在启动应用..."
}

# --- 启动应用 ---
Write-Ok "依赖已就绪，正在启动 Music to MIDI..."
Set-Location $REPO_DIR
& "$VENV_PYTHON" -m src.main @args

