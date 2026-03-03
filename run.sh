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
model_path = get_model_path()
print('YourMT3+ model:', model_path if model_path else 'missing')
exit(0 if model_path else 1)
"; then
    warn "YourMT3+ model weights missing"
    NEED_INSTALL=true
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '${REPO_DIR}')
from download_vocal_model import is_vocal_model_available, resolve_vocal_model_path
target = resolve_vocal_model_path()
print('BS-RoFormer model:', target)
exit(0 if is_vocal_model_available() else 1)
"; then
    warn "BS-RoFormer model weights missing"
    NEED_INSTALL=true
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '${REPO_DIR}')
from download_multistem_model import is_multistem_model_available, resolve_multistem_model_paths
model_path, config_path = resolve_multistem_model_paths()
print('BS-RoFormer SW model:', model_path)
print('BS-RoFormer SW config:', config_path)
exit(0 if is_multistem_model_available() else 1)
"; then
    warn "BS-RoFormer SW six-stem assets missing"
    NEED_INSTALL=true
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "
import importlib.util
ok = importlib.util.find_spec('amt.run') is not None
print('Aria-AMT package:', 'ok' if ok else 'missing')
exit(0 if ok else 1)
"; then
    warn "Aria-AMT package missing (optional; piano-only mode disabled)"
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '${REPO_DIR}')
from download_aria_amt_model import is_aria_model_available, resolve_aria_model_path
model_path = resolve_aria_model_path()
print('Aria-AMT model:', model_path)
exit(0 if is_aria_model_available() else 1)
"; then
    warn "Aria-AMT model missing (optional; piano-only mode disabled)"
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
