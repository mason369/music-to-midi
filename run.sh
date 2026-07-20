#!/usr/bin/env bash
# Music to MIDI 启动脚本
# 自动检查依赖，不完整时运行 install.sh，然后启动应用

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${REPO_DIR}/venv/bin/python"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info() { echo -e "${BLUE}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }

if grep -qi microsoft /proc/version 2>/dev/null; then
    [ -z "${DISPLAY:-}" ] && export DISPLAY=:0
    WAYLAND_SOCK="/run/user/$(id -u)/wayland-0"
    [ -z "${WAYLAND_DISPLAY:-}" ] && [ -e "$WAYLAND_SOCK" ] && \
        export WAYLAND_DISPLAY=wayland-0
fi

[ -d "/run/user/$(id -u)" ] && chmod 700 "/run/user/$(id -u)" 2>/dev/null || true

NEED_INSTALL=false

if [ ! -f "$VENV_PYTHON" ]; then
    warn "Python virtual environment missing"
    NEED_INSTALL=true
fi

if ! $NEED_INSTALL; then
    info "Checking core Python packages..."
    if ! "$VENV_PYTHON" -c "import PyQt6, librosa, mido; print('core imports OK')"; then
        warn "Core Python packages missing (full error shown above)"
        NEED_INSTALL=true
    fi
fi

if ! $NEED_INSTALL; then
    info "Checking exact NVIDIA CUDA 12.8 / PyTorch / ONNX Runtime contract..."
    if ! "$VENV_PYTHON" -c '
from importlib import metadata
expected = {"torch": "2.7.0", "torchaudio": "2.7.0", "torchvision": "0.22.0"}
actual = {name: metadata.version(name) for name in expected}
base = {name: version.split("+", 1)[0] for name, version in actual.items()}
if base != expected:
    raise RuntimeError(f"PyTorch trio mismatch: expected={expected}, actual={actual}")
import onnxruntime as ort
import torch, torchaudio, torchvision
if getattr(torch.version, "hip", None):
    raise RuntimeError(f"ROCm runtime is unsupported for the complete seven-mode stack: HIP={torch.version.hip}")
if torch.version.cuda != "12.8":
    raise RuntimeError(f"Expected PyTorch CUDA 12.8 runtime, got {torch.version.cuda!r}")
if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is False; a working NVIDIA CUDA GPU is required")
providers = ort.get_available_providers()
if "CUDAExecutionProvider" not in providers:
    raise RuntimeError(f"ONNX Runtime CUDAExecutionProvider is missing: {providers}")
probe = torch.ones(1, device="cuda")
probe.add_(1)
torch.cuda.synchronize()
print("PyTorch trio:", actual)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("ONNX Runtime providers:", providers)
print("NVIDIA device:", torch.cuda.get_device_name(0))
'; then
        warn "Complete seven-mode exact PyTorch/CUDA/ONNX Runtime GPU contract failed"
        NEED_INSTALL=true
    else
        ok "NVIDIA CUDA 12.8 runtime probe passed"
    fi
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${REPO_DIR}')
from src.utils.yourmt3_source_identity import validate_patched_yourmt3_source
source_dir = Path('${REPO_DIR}') / 'YourMT3' / 'amt' / 'src'
manifest, file_count = validate_patched_yourmt3_source(source_dir)
print('YourMT3+ patched source manifest:', manifest)
print('YourMT3+ patched source files:', file_count)
"; then
    warn "YourMT3+ source tree is missing or has the wrong identity. Re-check out the current project version; mutable upstream source is not accepted."
    exit 1
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '${REPO_DIR}')
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
exit(0 if not missing else 1)
"; then
    warn "YourMT3+ model weights missing"
    NEED_INSTALL=true
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '${REPO_DIR}')
from download_multistem_model import validate_multistem_assets
model_path, config_path = validate_multistem_assets()
print('BS-RoFormer SW Fixed checkpoint:', model_path)
print('BS-RoFormer SW Fixed config:', config_path)
exit(0)
"; then
    warn "BS-RoFormer SW Fixed six-stem model missing or checksum validation failed"
    warn "  先运行: python download_multistem_model.py"
    NEED_INSTALL=true
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '${REPO_DIR}')
from download_accompaniment_model import is_accompaniment_model_available, resolve_accompaniment_model_path
from download_vocal_model import is_vocal_model_available, resolve_vocal_model_paths
from src.core.vocal_separator import VocalSeparator
print('Leap XE vocals assets:', [str(path) for path in resolve_vocal_model_paths()])
print('PolarFormer accompaniment model:', resolve_accompaniment_model_path())
print('audio-separator package:', VocalSeparator.is_available())
print('Leap XE vocals available:', is_vocal_model_available())
print('PolarFormer accompaniment available:', is_accompaniment_model_available())
print('Vocal split route available:', VocalSeparator.is_model_available())
exit(0 if VocalSeparator.is_available() and is_vocal_model_available() and is_accompaniment_model_available() and VocalSeparator.is_model_available() else 1)
"; then
    warn "Leap XE vocals or PolarFormer accompaniment assets missing/invalid"
    NEED_INSTALL=true
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '${REPO_DIR}')
from download_sota_models import validate_default_transkun_runtime
identity = validate_default_transkun_runtime()
print('TransKun default runtime:', identity)
"; then
    warn "TransKun 2.0.1 default V2 package/resources failed exact identity validation"
    warn "  The installer will force-reinstall and revalidate transkun==2.0.1"
    NEED_INSTALL=true
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '${REPO_DIR}')
from download_transkun_v2_aug_model import is_transkun_v2_aug_model_available, resolve_transkun_v2_aug_model_dir
model_dir = resolve_transkun_v2_aug_model_dir()
print('TransKun V2 Aug model directory:', model_dir)
print('TransKun V2 Aug model available:', is_transkun_v2_aug_model_available())
exit(0 if is_transkun_v2_aug_model_available() else 1)
"; then
    warn "TransKun V2 Aug model missing or checksum validation failed"
    warn "  先运行: python download_transkun_v2_aug_model.py"
    NEED_INSTALL=true
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '${REPO_DIR}')
from src.core.aria_amt_transcriber import AriaAmtTranscriber
transcriber = AriaAmtTranscriber()
import importlib.util
print('amt.run spec:', importlib.util.find_spec('amt.run'))
print('Aria-AMT package:', AriaAmtTranscriber.is_available())
print('Aria-AMT model:', transcriber.is_model_available())
exit(0 if AriaAmtTranscriber.is_available() and transcriber.is_model_available() else 1)
"; then
    warn "Aria-AMT backend or model missing"
    warn "  先运行: python download_aria_amt_model.py"
    NEED_INSTALL=true
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '${REPO_DIR}')
from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber
transcriber = ByteDancePianoTranscriber()
print('ByteDance Piano package:', ByteDancePianoTranscriber.is_available())
print('ByteDance Piano model:', transcriber.is_model_available())
exit(0 if ByteDancePianoTranscriber.is_available() and transcriber.is_model_available() else 1)
"; then
    warn "ByteDance Piano backend or model missing"
    warn "  先运行: python download_bytedance_piano_model.py"
    NEED_INSTALL=true
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '${REPO_DIR}')
from src.core.miros_transcriber import MirosTranscriber
reason = MirosTranscriber.get_unavailable_reason()
print(reason or 'MIROS available')
print('MIROS package:', MirosTranscriber.is_available())
print('MIROS model:', MirosTranscriber.is_model_available())
exit(0 if MirosTranscriber.is_available() and MirosTranscriber.is_model_available() else 1)
"; then
    warn "MIROS backend or model missing"
    warn "  先运行: python download_miros_model.py"
    NEED_INSTALL=true
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '${REPO_DIR}')
from src.core.muscriptor_transcriber import MuscriptorTranscriber
from src.utils.fluidsynth_runtime import get_fluidsynth_executable
from src.utils.muscriptor_downloader import get_cached_muscriptor_paths
from src.utils.muscriptor_soundfont_downloader import download_muscriptor_soundfont
reason = MuscriptorTranscriber._runtime_unavailable_reason()
if reason:
    raise RuntimeError(reason)
# The actual model load performs the full SHA-256 gate. Avoid reading the 5.4 GB
# checkpoint twice by keeping launcher preflight to exact path/size validation.
weights, config = get_cached_muscriptor_paths(validate_hashes=False)
soundfont = download_muscriptor_soundfont(printer=print)
fluidsynth = get_fluidsynth_executable()
print('MuScriptor model:', weights)
print('MuScriptor config:', config)
print('MuScriptor SoundFont:', soundfont)
print('FluidSynth:', fluidsynth)
"; then
    warn "MuScriptor-large or its real SoundFont playback assets are missing/invalid"
    warn "  Accept the Hugging Face model terms, then run: python download_sota_models.py"
    NEED_INSTALL=true
fi

if $NEED_INSTALL; then
    info "Dependencies are incomplete, running installer..."
    bash "${REPO_DIR}/install.sh"
    ok "Install complete, restarting launcher..."
    exec bash "${BASH_SOURCE[0]}" "$@"
fi

ok "All dependencies are ready"
source "${REPO_DIR}/venv/bin/activate"
cd "$REPO_DIR"
exec python -m src.main "$@"
