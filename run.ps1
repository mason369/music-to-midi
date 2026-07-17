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

# 检查 3：精确 PyTorch/CUDA/ONNX Runtime GPU 契约
if (-not $NEED_INSTALL) {
    Write-Info "检查完整七模式 NVIDIA CUDA 12.8 运行时..."
    $checkGpuRuntimeScript = @"
from importlib import metadata
expected = {'torch': '2.7.0', 'torchaudio': '2.7.0', 'torchvision': '0.22.0'}
actual = {name: metadata.version(name) for name in expected}
base = {name: version.split('+', 1)[0] for name, version in actual.items()}
if base != expected:
    raise RuntimeError(f'PyTorch trio mismatch: expected={expected}, actual={actual}')
import onnxruntime as ort
import torch, torchaudio, torchvision
if getattr(torch.version, 'hip', None):
    raise RuntimeError(f'ROCm runtime is unsupported for the complete seven-mode stack: HIP={torch.version.hip}')
if torch.version.cuda != '12.8':
    raise RuntimeError(f'Expected PyTorch CUDA 12.8 runtime, got {torch.version.cuda!r}')
if not torch.cuda.is_available():
    raise RuntimeError('torch.cuda.is_available() is False; a working NVIDIA CUDA GPU is required')
providers = ort.get_available_providers()
if 'CUDAExecutionProvider' not in providers:
    raise RuntimeError(f'ONNX Runtime CUDAExecutionProvider is missing: {providers}')
probe = torch.ones(1, device='cuda')
probe.add_(1)
torch.cuda.synchronize()
print('PyTorch trio:', actual)
print('PyTorch CUDA runtime:', torch.version.cuda)
print('ONNX Runtime providers:', providers)
print('NVIDIA device:', torch.cuda.get_device_name(0))
"@
    & "$VENV_PYTHON" -c $checkGpuRuntimeScript
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "完整七模式的精确 PyTorch/CUDA/ONNX Runtime GPU 契约未通过"
        $NEED_INSTALL = $true
    } else {
        Write-Ok "NVIDIA CUDA 12.8 运行时实测通过"
    }
}

# 检查 4：受控 YourMT3+ 源码身份
if (-not $NEED_INSTALL) {
    Write-Info "检查受控 YourMT3+ 源码树..."
    $checkYourMt3SourceScript = @"
import sys
from pathlib import Path
sys.path.insert(0, r'$REPO_DIR')
from src.utils.yourmt3_source_identity import validate_patched_yourmt3_source
source_dir = Path(r'$REPO_DIR') / 'YourMT3' / 'amt' / 'src'
manifest, file_count = validate_patched_yourmt3_source(source_dir)
print('YourMT3+ patched source manifest:', manifest)
print('YourMT3+ patched source files:', file_count)
"@
    & "$VENV_PYTHON" -c $checkYourMt3SourceScript
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[错误] YourMT3+ 源码树缺失或身份不匹配；请重新取得当前项目版本，不能用可变上游源码替代。" -ForegroundColor Red
        exit 1
    }
    Write-Ok "YourMT3+ 源码身份检查通过"
}

# 检查 5：YourMT3+ 模型权重
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

# 检查 6：BS-RoFormer SW Fixed 六声部分离模型
if (-not $NEED_INSTALL) {
    Write-Info "检查 BS-RoFormer SW Fixed 六声部分离模型..."
    $checkMultiStemScript = @"
import sys
sys.path.insert(0, r'$REPO_DIR')
from download_multistem_model import validate_multistem_assets
model_path, config_path = validate_multistem_assets()
print('BS-RoFormer SW Fixed checkpoint:', model_path)
print('BS-RoFormer SW Fixed config:', config_path)
sys.exit(0)
"@
    & "$VENV_PYTHON" -c $checkMultiStemScript
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "未找到 BS-RoFormer SW Fixed 六声部模型，或文件校验失败"
        Write-Warn "  请先运行: python download_multistem_model.py"
        $NEED_INSTALL = $true
    } else {
        Write-Ok "BS-RoFormer SW Fixed 六声部模型检查通过"
    }
}

# 检查 7：Leap XE 人声 + PolarFormer 伴奏分离模型
if (-not $NEED_INSTALL) {
    Write-Info "检查 Leap XE 90-band 人声与 PolarFormer 伴奏分离模型..."
    $checkVocalScript = @"
import sys
sys.path.insert(0, r'$REPO_DIR')
from download_accompaniment_model import is_accompaniment_model_available, resolve_accompaniment_model_path
from download_vocal_model import is_vocal_model_available, resolve_vocal_model_paths
from src.core.vocal_separator import VocalSeparator
print('Leap XE vocals assets:', [str(path) for path in resolve_vocal_model_paths()])
print('PolarFormer accompaniment model:', resolve_accompaniment_model_path())
print('audio-separator package:', VocalSeparator.is_available())
print('Leap XE vocals available:', is_vocal_model_available())
print('PolarFormer accompaniment available:', is_accompaniment_model_available())
print('Vocal split route available:', VocalSeparator.is_model_available())
sys.exit(0 if VocalSeparator.is_available() and is_vocal_model_available() and is_accompaniment_model_available() and VocalSeparator.is_model_available() else 1)
"@
    & "$VENV_PYTHON" -c $checkVocalScript
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Leap XE 人声或 PolarFormer 伴奏模型缺失/校验失败"
        $NEED_INSTALL = $true
    } else {
        Write-Ok "Leap XE + PolarFormer 人声分离模型检查通过"
    }
}

# 检查 8：TransKun 默认 V2 钢琴后端
if (-not $NEED_INSTALL) {
    Write-Info "检查 TransKun 2.0.1 默认 V2 包与随包资源身份..."
    $checkDefaultTransKunScript = @"
import sys
sys.path.insert(0, r'$REPO_DIR')
from download_sota_models import validate_default_transkun_runtime
identity = validate_default_transkun_runtime()
print('TransKun default runtime:', identity)
"@
    & "$VENV_PYTHON" -c $checkDefaultTransKunScript
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "TransKun 2.0.1 默认 V2 包或随包资源身份校验失败"
        Write-Warn "  安装程序将强制重装并再次校验 transkun==2.0.1"
        $NEED_INSTALL = $true
    } else {
        Write-Ok "TransKun 默认 V2 严格身份检查通过"
    }
}

# 检查 9：TransKun V2 Aug 钢琴后端
if (-not $NEED_INSTALL) {
    Write-Info "检查 TransKun V2 Aug 钢琴模型..."
    $checkTransKunV2AugScript = @"
import sys
sys.path.insert(0, r'$REPO_DIR')
from download_transkun_v2_aug_model import (
    is_transkun_v2_aug_model_available,
    resolve_transkun_v2_aug_model_dir,
)
model_dir = resolve_transkun_v2_aug_model_dir()
print('TransKun V2 Aug model directory:', model_dir)
print('TransKun V2 Aug model available:', is_transkun_v2_aug_model_available())
sys.exit(0 if is_transkun_v2_aug_model_available() else 1)
"@
    & "$VENV_PYTHON" -c $checkTransKunV2AugScript
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "TransKun V2 Aug 模型缺失或校验失败"
        Write-Warn "  请先运行: python download_transkun_v2_aug_model.py"
        $NEED_INSTALL = $true
    } else {
        Write-Ok "TransKun V2 Aug 模型检查通过"
    }
}

# 检查 10：Aria-AMT 钢琴后端
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

# 检查 11：ByteDance Piano 带踏板钢琴后端
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

# 检查 12：MIROS 多乐器后端
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
    Write-Ok "安装完成，正在重新执行完整依赖与身份校验..."
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$PSCommandPath" @args
    exit $LASTEXITCODE
}

# --- 启动应用 ---
Write-Ok "依赖已就绪，正在启动 Music to MIDI..."
Set-Location $REPO_DIR
& "$VENV_PYTHON" -m src.main @args

