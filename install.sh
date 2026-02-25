#!/usr/bin/env bash
# Music to MIDI - Linux/WSL 自动安装脚本
# 支持 Ubuntu 20.04+ / Debian 11+ / WSL2 (WSLg)
# 用法: bash install.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_DIR}/venv"

# ───────────────────────── 颜色输出 ─────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ───────────────────────── 横幅 ─────────────────────────
echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════╗"
echo "║        Music to MIDI - 安装脚本              ║"
echo "║        支持 Linux / WSL2                     ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# ───────────────────────── 检测环境 ─────────────────────────
IS_WSL=false
if grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=true
    info "检测到 WSL2 环境"
else
    info "检测到原生 Linux 环境"
fi

# ───────────────────────── 检查 sudo ─────────────────────────
if ! sudo -n true 2>/dev/null; then
    warn "需要 sudo 权限安装系统依赖，请输入密码..."
fi

# ───────────────────────── 检查 Python ─────────────────────────
info "检查 Python 版本..."
PYTHON_BIN=""
for cmd in python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        MAJOR="${VER%%.*}"; MINOR="${VER##*.}"
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_BIN="$cmd"
            success "找到 Python $VER ($cmd)"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    warn "未找到 Python 3.10+，尝试安装..."
    info "更新包列表..."
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip
    PYTHON_BIN="python3.10"
fi

# ───────────────────────── 安装系统依赖 ─────────────────────────
info "安装系统依赖..."
info "更新包列表..."
sudo apt-get update

SYSTEM_PKGS=(
    # 版本控制
    git
    # 音频处理
    ffmpeg
    libsndfile1
    libportaudio2
    portaudio19-dev
    # Python 开发
    python3-dev
    python3-venv
    python3.10-venv
    python3.11-venv
    # PyQt6 / X11 运行时库
    libxcb-xinerama0
    libxcb-cursor0
    libxkbcommon-x11-0
    libxcb-icccm4
    libxcb-image0
    libxcb-keysyms1
    libxcb-randr0
    libxcb-render-util0
    libxcb-shape0
    libxcb-util1
    libxrender1
    libgl1
    libglib2.0-0
    # UI 字体（英文）
    fonts-ubuntu
    fonts-dejavu-core
    # 中文字体（修复中文显示为方块的问题）
    fonts-noto-cjk
    fonts-wqy-zenhei
    fonts-wqy-microhei
    # Emoji/图标字体（修复图标显示为方块的问题）
    fonts-noto-color-emoji
    fonts-symbola
)

MISSING_PKGS=()
for pkg in "${SYSTEM_PKGS[@]}"; do
    if ! dpkg -s "$pkg" &>/dev/null 2>&1; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    info "安装缺失的系统包 (${#MISSING_PKGS[@]} 个)..."
    sudo apt-get install -y "${MISSING_PKGS[@]}"
    success "系统包安装完成"
else
    success "所有系统依赖已就绪"
fi

# ───────────────────────── 更新字体缓存 ─────────────────────────
info "更新字体缓存..."
fc-cache -f 2>/dev/null || true
success "字体缓存已更新"

# ───────────────────────── 配置 fontconfig（emoji回退）─────────────────────────
info "配置字体回退（Emoji/图标支持）..."
mkdir -p "${HOME}/.config/fontconfig"
cat > "${HOME}/.config/fontconfig/fonts.conf" << 'FONTCONF_EOF'
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <!-- 优先使用 Noto CJK 渲染中文，Noto Color Emoji 渲染图标 -->
  <alias>
    <family>sans-serif</family>
    <prefer>
      <family>Noto Sans CJK SC</family>
      <family>WenQuanYi Micro Hei</family>
      <family>Noto Color Emoji</family>
      <family>Symbola</family>
    </prefer>
  </alias>
  <alias>
    <family>serif</family>
    <prefer>
      <family>Noto Serif CJK SC</family>
      <family>Noto Color Emoji</family>
      <family>Symbola</family>
    </prefer>
  </alias>
  <alias>
    <family>monospace</family>
    <prefer>
      <family>Noto Sans Mono CJK SC</family>
      <family>Ubuntu Mono</family>
      <family>Noto Color Emoji</family>
      <family>Symbola</family>
    </prefer>
  </alias>
</fontconfig>
FONTCONF_EOF
fc-cache -f 2>/dev/null || true
success "字体配置完成"

# ───────────────────────── WSL 显示检查 ─────────────────────────
if $IS_WSL; then
    info "检查 WSL 显示环境..."
    if [ -z "${DISPLAY:-}" ]; then
        if [ -d "/mnt/wslg" ]; then
            warn "WSLg 检测到。请确保 ~/.bashrc 包含: export DISPLAY=:0"
        else
            warn "未检测到 WSLg。"
            warn "  - Windows 11: 确保 WSLg 已启用（通常默认开启）"
            warn "  - Windows 10: 需要安装 VcXsrv 并设置 DISPLAY"
        fi
    else
        success "显示环境 DISPLAY=$DISPLAY 已就绪"
    fi
fi

# ───────────────────────── 修复运行时目录权限 ─────────────────────────
if [ -d "/run/user/$(id -u)" ]; then
    chmod 700 "/run/user/$(id -u)" 2>/dev/null || true
fi

# ───────────────────────── 创建虚拟环境 ─────────────────────────
info "创建 Python 虚拟环境..."
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "虚拟环境已创建: $VENV_DIR"
else
    success "虚拟环境已存在: $VENV_DIR"
fi

PIP="${VENV_DIR}/bin/pip"
PYTHON="${VENV_DIR}/bin/python"

# ───────────────────────── 升级 pip ─────────────────────────
info "升级 pip..."
"$PIP" install --upgrade pip setuptools wheel
success "pip 升级完成"

# ───────────────────────── 安装 PyTorch ─────────────────────────
info "检测 GPU / 加速器..."

TORCH_INSTALLED=false
info "检查 PyTorch 是否已安装（首次检查可能需要几秒）..."
if "$PYTHON" -c "import torch; v=torch.__version__; assert tuple(int(x) for x in v.split('+')[0].split('.')[:2]) >= (2,1)" 2>/dev/null; then
    success "PyTorch 已安装且版本满足要求"
    TORCH_INSTALLED=true
fi

if ! $TORCH_INSTALLED; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_LABEL="CPU"

    # 检测 NVIDIA GPU (CUDA)
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+' | head -1 || echo "")
        CUDA_MAJOR="${CUDA_VER%%.*}"
        if [ "${CUDA_MAJOR:-0}" -ge 12 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            TORCH_LABEL="CUDA 12.1 (NVIDIA)"
        elif [ "${CUDA_MAJOR:-0}" -ge 11 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
            TORCH_LABEL="CUDA 11.8 (NVIDIA)"
        fi
    fi

    # 检测 AMD GPU (ROCm) - 仅 Linux
    if command -v rocm-smi &>/dev/null && rocm-smi &>/dev/null 2>&1; then
        TORCH_INDEX="https://download.pytorch.org/whl/rocm5.7"
        TORCH_LABEL="ROCm 5.7 (AMD)"
    fi

    info "安装 PyTorch ($TORCH_LABEL)..."
    "$PIP" install torch==2.4.0 torchaudio==2.4.0 \
        --index-url "$TORCH_INDEX"
    success "PyTorch ($TORCH_LABEL) 安装完成"
fi

# ───────────────────────── Intel 加速（可选）─────────────────────────
if ! "$PYTHON" -c "import intel_extension_for_pytorch" 2>/dev/null; then
    INTEL_FOUND=false
    INTEL_HAS_GPU=false

    # 方式1: lspci 检测 Intel GPU（原生 Linux，WSL2 通常不可见）
    if command -v lspci &>/dev/null && lspci 2>/dev/null | grep -qi \
        "intel.*graphics\|intel.*display\|intel.*uhd\|intel.*iris\|intel.*arc"; then
        INTEL_FOUND=true
        INTEL_HAS_GPU=true
        info "检测到 Intel GPU（lspci）"
    fi

    # 方式2: /proc/cpuinfo 检测 Intel 处理器（WSL2/虚拟机/所有环境通用）
    if ! $INTEL_FOUND && grep -qi "genuine intel\|intel(r)" /proc/cpuinfo 2>/dev/null; then
        INTEL_FOUND=true
        info "检测到 Intel 处理器（/proc/cpuinfo）"
    fi

    if $INTEL_FOUND; then
        # 原生 Linux：尝试安装 Intel Level Zero GPU 计算驱动（启用 XPU 模式）
        if ! $IS_WSL && $INTEL_HAS_GPU; then
            if ! dpkg -s intel-level-zero-gpu &>/dev/null 2>&1; then
        info "检测到 Intel GPU，安装 Intel Level Zero GPU 驱动..."
                info "  → 安装 gpg-agent、wget..."
                sudo apt-get install -y --no-install-recommends \
                    gpg-agent wget || true
                info "  → 下载并导入 Intel GPU GPG 密钥..."
                wget -O - https://repositories.intel.com/gpu/intel-graphics.key | \
                    sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg || true
                info "  → 添加 Intel GPU 软件仓库..."
                echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
https://repositories.intel.com/gpu/ubuntu jammy unified" | \
                    sudo tee /etc/apt/sources.list.d/intel-gpu.list || true
                info "  → 更新包列表..."
                sudo apt-get update -q || true
                info "  → 安装 Level Zero 驱动..."
                sudo apt-get install -y --no-install-recommends \
                    intel-level-zero-gpu libze-intel-gpu1 libze-dev && \
                    success "Intel Level Zero GPU 驱动安装完成（XPU 加速已就绪）" || \
                    warn "Level Zero 驱动安装失败（Intel Arc/Iris Xe XPU 不可用，仍可使用 CPU 优化）"
            else
                success "Intel Level Zero GPU 驱动已安装"
            fi
        fi

        if $IS_WSL; then
            warn "WSL2 环境：Intel iGPU XPU 模式不可用（驱动限制）"
            info "  → 将安装 IPEX 启用 Intel CPU 指令集优化（AVX-512/AMX）"
        fi

        info "安装 intel_extension_for_pytorch（Intel GPU/CPU 加速）..."
        "$PIP" install intel_extension_for_pytorch && \
            success "intel_extension_for_pytorch 安装完成" || \
            warn "intel_extension_for_pytorch 安装失败（可选，将使用标准 PyTorch）"
    fi
fi

# ───────────────────────── 安装项目依赖 ─────────────────────────
info "安装项目 Python 依赖..."
cd "$REPO_DIR"

info "安装 tflite-runtime（可选，Linux 专用 TensorFlow Lite 后端）..."
"$PIP" install tflite-runtime || warn "tflite-runtime 安装失败（可选，不影响主要功能）"

"$PIP" install -r requirements.txt
success "Python 依赖安装完成"

# ───────────────────────── 验证核心依赖 ─────────────────────────
info "验证核心依赖..."
DEPS_OK=true

declare -A DEP_CHECKS=(
    ["PyQt6"]="PyQt6"
    ["torch"]="torch"
    ["librosa"]="librosa"
    ["mido"]="mido"
    ["soundfile"]="soundfile"
    ["pytorch_lightning"]="pytorch_lightning"
)

for name in "${!DEP_CHECKS[@]}"; do
    mod="${DEP_CHECKS[$name]}"
    info "  正在验证 $name..."
    if "$PYTHON" -c "import $mod" 2>/dev/null; then
        success "  $name OK"
    else
        warn "  $name 导入失败（可选依赖，基础功能不受影响）"
    fi
done

if ! command -v ffmpeg &>/dev/null; then
    warn "  ffmpeg 未找到 - 部分音频格式可能无法处理"
    DEPS_OK=false
else
    FFMPEG_VER=$(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')
    success "  ffmpeg $FFMPEG_VER OK"
fi

# ───────────────────────── YourMT3 代码库安装 ─────────────────────────
cd "$REPO_DIR"

if [ -d "YourMT3" ] && [ -d "YourMT3/amt/src" ]; then
    success "YourMT3 代码库已存在，跳过克隆"
else
    if ! command -v git >/dev/null 2>&1; then
        warn "未找到 git，跳过 YourMT3 代码库安装（可手动运行 bash install_yourmt3_code.sh）"
    else
        info "克隆 YourMT3 代码库（仅代码，不含模型权重）..."
        GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --progress \
            https://huggingface.co/spaces/mimbres/YourMT3 YourMT3 || true

        if [ -d "YourMT3" ]; then
            success "YourMT3 代码库克隆完成"
        else
            warn "YourMT3 克隆失败（无法连接 huggingface.co），下载模型时会自动包含所需代码"
        fi
    fi
fi

# 安装 YourMT3 Python 依赖（如果代码库存在）
if [ -f "YourMT3/requirements.txt" ]; then
    info "安装 YourMT3 Python 依赖..."
    "$PIP" install -r YourMT3/requirements.txt && \
        success "YourMT3 依赖安装完成" || \
        warn "YourMT3 依赖安装部分失败（可稍后手动运行: bash install_yourmt3_code.sh）"
fi

# ───────────────────────── 下载 SOTA 模型权重 ─────────────────────────
info "下载 YourMT3+ SOTA 模型权重（YPTF.MoE+Multi PS，约 800MB）..."
info "如需跳过，按 Ctrl+C 后手动运行: venv/bin/python download_sota_models.py"

"$PYTHON" "${REPO_DIR}/download_sota_models.py" && \
    success "SOTA 模型权重下载完成" || \
    warn "模型下载失败，可稍后手动运行: venv/bin/python download_sota_models.py"

# 如果 YourMT3 目录不存在，从缓存创建符号链接
YOURMT3_DIR="${REPO_DIR}/YourMT3"
CACHE_DIR="${HOME}/.cache/music_ai_models/yourmt3_all"
if [ ! -d "$YOURMT3_DIR" ] && [ -d "$CACHE_DIR/amt/src" ]; then
    info "YourMT3 仓库未克隆成功，从模型缓存创建符号链接..."
    if ln -s "$CACHE_DIR" "$YOURMT3_DIR" 2>/dev/null; then
        success "已创建符号链接: YourMT3 -> $CACHE_DIR"
    else
        warn "创建符号链接失败，尝试复制文件..."
        cp -r "$CACHE_DIR" "$YOURMT3_DIR" && \
            success "已从缓存复制 YourMT3 代码到项目目录" || \
            warn "复制失败"
    fi
fi

# 补装 YourMT3 依赖（如果第 12 步跳过了）
YOURMT3_AMT_SRC="${REPO_DIR}/YourMT3/amt/src"
if [ -d "$YOURMT3_AMT_SRC" ]; then
    if ! "$PYTHON" -c "
import sys
sys.path.insert(0, '$YOURMT3_AMT_SRC')
from model.ymt3 import YourMT3
" 2>/dev/null; then
        info "补装 YourMT3 Python 依赖..."
        "$PIP" install einops "transformers>=4.30.0" deprecated smart-open --quiet && \
            success "YourMT3 依赖补装成功" || \
            warn "YourMT3 依赖补装部分失败"
    fi
fi

# ───────────────────────── 创建启动脚本 ─────────────────────────
info "创建启动脚本..."
cat > "${REPO_DIR}/run.sh" << 'LAUNCH_EOF'
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

if ! $NEED_INSTALL && [ ! -d "${REPO_DIR}/YourMT3/amt/src" ]; then
    warn "YourMT3 代码库不存在"; NEED_INSTALL=true
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
LAUNCH_EOF
chmod +x "${REPO_DIR}/run.sh"
success "启动脚本已创建: run.sh"

# ───────────────────────── 添加 ~/.bashrc 配置 ─────────────────────────
if $IS_WSL; then
    BASHRC="${HOME}/.bashrc"
    if ! grep -q "export DISPLAY=:0" "$BASHRC" 2>/dev/null; then
        echo "" >> "$BASHRC"
        echo "# WSLg 显示配置（Music to MIDI）" >> "$BASHRC"
        echo "export DISPLAY=:0" >> "$BASHRC"
        info "已将 DISPLAY=:0 添加到 ~/.bashrc"
    fi
fi

# ───────────────────────── 完成 ─────────────────────────
echo ""
echo -e "${BOLD}${GREEN}══════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  安装完成！${NC}"
echo -e "${BOLD}${GREEN}══════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}运行方式：${NC}"
echo -e "  ${GREEN}./run.sh${NC}                    # 推荐：直接运行"
echo -e "  ${GREEN}source venv/bin/activate && python -m src.main${NC}"
echo ""
echo -e "  ${BOLD}已自动安装：${NC}"
echo -e "  ${GREEN}✔${NC} Python 依赖"
echo -e "  ${GREEN}✔${NC} YourMT3+ 代码库"
echo -e "  ${GREEN}✔${NC} YPTF.MoE+Multi (PS) 模型权重"
echo ""
echo -e "  ${BOLD}若模型下载失败，可手动补下：${NC}"
echo -e "  ${YELLOW}venv/bin/python download_sota_models.py${NC}"
echo ""
if $IS_WSL; then
    echo -e "  ${YELLOW}WSL 提示：${NC}如果首次运行，请先执行 source ~/.bashrc"
    echo ""
fi
