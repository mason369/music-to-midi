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
from src.utils.yourmt3_downloader import OFFICIAL_YOURMT3_MODEL_KEYS, YOURMT3_MODELS
missing = []
for model_key in OFFICIAL_YOURMT3_MODEL_KEYS:
    model_info = YOURMT3_MODELS[model_key]
    label = model_info.get('ui_label', model_key)
    model_path = get_model_path(model_key)
    print(f'YourMT3+ {label}:', model_path if model_path else 'missing')
    if model_path is None:
        missing.append(label)
if missing:
    print('missing YourMT3+ official model modes:', ', '.join(missing))
sys.exit(0 if not missing else 1)
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
from src.core.vocal_separator import VocalSeparator
target = resolve_vocal_model_path()
print('BS-RoFormer model:', target)
print('audio-separator package:', VocalSeparator.is_available())
print('BS-RoFormer model available:', VocalSeparator.is_model_available())
sys.exit(0 if VocalSeparator.is_available() and is_vocal_model_available() else 1)
"@
    & "$VENV_PYTHON" -c $checkVocalScript
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "未找到 BS-RoFormer 模型权重"
        $NEED_INSTALL = $true
    } else {
        Write-Ok "BS-RoFormer 模型权重检查通过"
    }
}

# 检查 5：Aria-AMT 钢琴后端
if (-not $NEED_INSTALL) {
    Write-Info "检查 Aria-AMT 钢琴后端..."
    $checkAriaScript = @"
import sys
import importlib.util
sys.path.insert(0, r'$REPO_DIR')
from src.core.aria_amt_transcriber import AriaAmtTranscriber
transcriber = AriaAmtTranscriber()
print('amt.run spec:', importlib.util.find_spec('amt.run'))
print('Aria-AMT package:', AriaAmtTranscriber.is_available())
print('Aria-AMT model:', transcriber.is_model_available())
sys.exit(0 if AriaAmtTranscriber.is_available() and transcriber.is_model_available() else 1)
"@
    & "$VENV_PYTHON" -c $checkAriaScript
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "未找到 Aria-AMT 包或模型权重"
        Write-Warn "  请先运行: python download_aria_amt_model.py"
        $NEED_INSTALL = $true
    } else {
        Write-Ok "Aria-AMT 后端检查通过"
    }
}

# 检查 6：ByteDance Piano 带踏板钢琴后端
if (-not $NEED_INSTALL) {
    Write-Info "检查 ByteDance Piano 带踏板钢琴后端..."
    $checkByteDanceScript = @"
import sys
sys.path.insert(0, r'$REPO_DIR')
from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber
transcriber = ByteDancePianoTranscriber()
print('ByteDance Piano package:', ByteDancePianoTranscriber.is_available())
print('ByteDance Piano model:', transcriber.is_model_available())
sys.exit(0 if ByteDancePianoTranscriber.is_available() and transcriber.is_model_available() else 1)
"@
    & "$VENV_PYTHON" -c $checkByteDanceScript
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "未找到 ByteDance Piano 包或模型权重"
        Write-Warn "  请先运行: python download_bytedance_piano_model.py"
        $NEED_INSTALL = $true
    } else {
        Write-Ok "ByteDance Piano 后端检查通过"
    }
}

# 检查 7：MIROS 多乐器后端
if (-not $NEED_INSTALL) {
    Write-Info "检查 MIROS 多乐器后端..."
    $checkMirosScript = @"
import sys
sys.path.insert(0, r'$REPO_DIR')
from src.core.miros_transcriber import MirosTranscriber
reason = MirosTranscriber.get_unavailable_reason()
print(reason or 'MIROS available')
print('MIROS package:', MirosTranscriber.is_available())
print('MIROS model:', MirosTranscriber.is_model_available())
sys.exit(0 if MirosTranscriber.is_available() and MirosTranscriber.is_model_available() else 1)
"@
    & "$VENV_PYTHON" -c $checkMirosScript
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "未找到 MIROS 源码、运行依赖或模型权重"
        Write-Warn "  请先运行: python download_miros_model.py"
        $NEED_INSTALL = $true
    } else {
        Write-Ok "MIROS 后端检查通过"
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

