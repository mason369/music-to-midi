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

if ! $NEED_INSTALL && [ ! -d "${REPO_DIR}/YourMT3/amt/src" ]; then
    warn "YourMT3 source tree missing"
    NEED_INSTALL=true
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
from download_vocal_model import is_vocal_model_available, resolve_vocal_model_path
from src.core.vocal_separator import VocalSeparator
target = resolve_vocal_model_path()
print('BS-RoFormer model:', target)
print('audio-separator package:', VocalSeparator.is_available())
print('BS-RoFormer model available:', VocalSeparator.is_model_available())
exit(0 if VocalSeparator.is_available() and is_vocal_model_available() else 1)
"; then
    warn "BS-RoFormer model weights missing"
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
