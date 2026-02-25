#!/usr/bin/env bash
# Music to MIDI 启动脚本
# 自动检测依赖完整性，缺失时运行 install.sh 后再启动

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${REPO_DIR}/venv/bin/python"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }

# WSL / Linux 显示环境初始化
if grep -qi microsoft /proc/version 2>/dev/null; then
    [ -z "${DISPLAY:-}" ] && export DISPLAY=:0
    WAYLAND_SOCK="/run/user/$(id -u)/wayland-0"
    [ -z "${WAYLAND_DISPLAY:-}" ] && [ -e "$WAYLAND_SOCK" ] && \
        export WAYLAND_DISPLAY=wayland-0
fi

# 修复运行时目录权限（避免 Qt 警告）
[ -d "/run/user/$(id -u)" ] && chmod 700 "/run/user/$(id -u)" 2>/dev/null || true

# ───────────────────────── 依赖完整性检查 ─────────────────────────
NEED_INSTALL=false

if [ ! -f "$VENV_PYTHON" ]; then
    warn "虚拟环境不存在"; NEED_INSTALL=true
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "import PyQt6, librosa, mido" 2>/dev/null; then
    warn "核心 Python 包缺失"; NEED_INSTALL=true
fi

if ! $NEED_INSTALL && ! "$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '${REPO_DIR}')
from src.utils.yourmt3_downloader import is_model_available
exit(0 if is_model_available() else 1)
" 2>/dev/null; then
    warn "YourMT3+ 模型权重不存在"; NEED_INSTALL=true
fi

# ───────────────────────── 按需安装 ─────────────────────────
if $NEED_INSTALL; then
    info "依赖不完整，正在运行安装脚本..."
    bash "${REPO_DIR}/install.sh"
    ok "安装完成，正在启动应用..."
    # install.sh 会覆盖 run.sh，必须重新执行以读取新文件
    exec bash "${BASH_SOURCE[0]}" "$@"
fi

ok "所有依赖已就绪"
source "${REPO_DIR}/venv/bin/activate"
cd "$REPO_DIR"
exec python -m src.main "$@"
