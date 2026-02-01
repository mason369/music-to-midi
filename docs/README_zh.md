# 音乐转MIDI转换器

<p align="center">
  中文 | <a href="./README.md">English</a>
</p>

将音频文件转换为多轨道MIDI，自动嵌入歌词。支持 128 种 GM 乐器精确识别。

## 功能特点

- **双模式处理**：
  - **钢琴模式**：跳过音源分离，使用 ByteDance 专业钢琴模型直接转换为多轨钢琴MIDI（适合纯钢琴曲）
  - **智能模式**：使用 YourMT3+ MoE 模型直接识别多种乐器，支持 128 种 GM 乐器精确识别
- **音源分离**：使用 Demucs v4 自动将音频分离为 6 个轨道（人声、鼓、贝斯、吉他、钢琴、其他）
- **乐器识别**：使用 PANNs 进行智能乐器检测和分类
- **多乐器转写**：
  - **YourMT3+ MoE**（2025 AMT Challenge SOTA）：层次化注意力 Transformer + 混合专家架构，支持 128 种 GM 乐器
  - **Basic Pitch**（Spotify）：多音高检测备选方案
  - **ByteDance Piano Transcription**：专业钢琴转写，支持踏板检测
- **MIDI后处理**：音符量化、力度平滑、去重、复音限制等优化
- **歌词识别**：识别人声中的歌词，并以单词级时间戳嵌入MIDI
- **多语言界面**：支持中文和英文界面切换
- **专业深色主题**：现代化音频软件风格界面设计

## 平台支持

| 平台 | 状态 | 说明 |
|------|------|------|
| Windows | ✅ 已支持 | 完整功能，推荐使用 CUDA |
| Linux | ✅ 已支持 | 完整功能，推荐 Ubuntu 22.04+ |
| macOS | 🚧 计划中 | Apple Silicon MPS 支持开发中 |

## 截图

即将推出...

## 安装

### 前置要求

- **Python 3.10+**（推荐 3.10 或 3.11，3.12 可能存在兼容性问题）
- **FFmpeg**：音频处理必需
  - Windows: `choco install ffmpeg` 或从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载
  - Linux: `sudo apt install ffmpeg` (Ubuntu/Debian) 或 `sudo dnf install ffmpeg` (Fedora)
  - macOS: `brew install ffmpeg`
- **Git LFS**（可选，用于安装 YourMT3+ 代码）
- **NVIDIA GPU + CUDA**（推荐）：使用 CUDA 加速处理，显著提升性能

### Git LFS 安装

- Windows：
  - `choco install git-lfs` 或 `winget install GitHub.GitLFS`
  - 安装后执行：`git lfs install`
- macOS：
  - `brew install git-lfs`
  - 安装后执行：`git lfs install`
- Linux：
  - Ubuntu/Debian：`sudo apt-get install git-lfs`
  - Fedora：`sudo dnf install git-lfs`
  - 安装后执行：`git lfs install`

### 依赖环境要求

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| PyTorch | 2.1.0 - 2.4.x | pyannote.audio 兼容性要求 |
| torchaudio | 2.1.0 - 2.4.x | 与 PyTorch 版本对应 |
| NumPy | < 2.0 | numba/JAX 兼容性 |
| TensorFlow | 2.15.x（Windows） | Basic Pitch 后端（Windows 用于替代 tflite-runtime） |
| CUDA | 11.8 或 12.1 | GPU 加速（可选） |

### Linux 安装（推荐）

Linux 是运行本项目的推荐平台，环境配置更简单，GPU 加速更稳定。

```bash
# 1. 克隆仓库
git clone https://github.com/mason369/music-to-midi.git
cd music-to-midi

# 2. 创建虚拟环境（推荐使用 conda）
conda create -n music2midi python=3.10
conda activate music2midi

# 或使用 venv
python -m venv venv
source venv/bin/activate

# 3. 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 仅 CPU（不推荐，速度较慢）
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 安装 YourMT3+ 代码库（可选，用于 128 种乐器识别）
git lfs install
git clone https://huggingface.co/spaces/mimbres/YourMT3
cd YourMT3
pip install -r requirements.txt
# Linux 可选：仅 GuitarSet 预处理需要
sudo apt-get install sox
cd ..
# 或运行安装脚本
bash install_yourmt3_code.sh

# 6. 下载 YourMT3+ 模型（可选）
python download_sota_models.py

# 7. 运行应用
python -m src.main
```

### Windows 安装

注意：Windows 上没有 `tflite-runtime` 发行版，`requirements.txt` 已通过平台条件自动改用 `tensorflow` 作为 Basic Pitch 后端，请确保使用较新的 `pip`。

```bash
# 克隆仓库
git clone https://github.com/mason369/music-to-midi.git
cd music-to-midi

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装 PyTorch（以下三选一）
# CUDA 11.8
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 仅 CPU（无独显或不需要 CUDA）
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# 安装依赖
pip install -r requirements.txt

# 可选：安装 YourMT3+ 代码（用于 128 乐器识别）
git lfs install
git clone https://huggingface.co/spaces/mimbres/YourMT3
cd YourMT3
pip install -r requirements.txt
cd ..

# 可选：下载 YourMT3+ 模型
python download_sota_models.py

# 运行应用
python -m src.main
```

### CUDA 安装指南

#### Linux (Ubuntu/Debian)

```bash
# 方法 1: 使用 NVIDIA 官方仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-1

# 方法 2: 使用 conda（推荐，自动管理）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 验证 CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

#### Windows

1. 从 [NVIDIA 官网](https://developer.nvidia.com/cuda-downloads) 下载 CUDA Toolkit
2. 安装时选择自定义安装，确保勾选 cuDNN
3. 重启后验证：`nvidia-smi`

### 从发布版安装

从 [Releases](https://github.com/mason369/music-to-midi/releases) 页面下载预编译版本。

## 使用方法

1. **打开音频文件**：拖放音频文件（MP3、WAV、FLAC、OGG）或点击浏览选择
2. **配置输出**：选择输出目录和选项（MIDI、歌词、分离音轨）
3. **开始处理**：点击"开始"按钮开始转换
4. **获取结果**：在输出目录中找到MIDI文件、LRC歌词和分离的音轨

## 支持的格式

### 输入
- MP3, WAV, FLAC, OGG, M4A, AAC, WMA

### 输出
- MIDI (.mid) - 嵌入歌词的多轨道MIDI
- LRC (.lrc) - 同步歌词文件
- WAV - 分离的音频轨道

## 技术细节

### 使用的AI模型

| 模型 | 来源 | 用途 | 说明 |
|------|------|------|------|
| **YourMT3+ MoE** | KAIST | 多乐器转写 | 2025 AMT Challenge SOTA，混合专家架构，支持 128 种 GM 乐器 |
| **Demucs v4** | Meta | 音源分离 | 最先进的音源分离，支持 4/6 轨模式 |
| **PANNs** | KAIST | 乐器识别 | 音频模式分析与乐器分类 |
| **Basic Pitch** | Spotify | 多音高检测 | 轻量级音频转 MIDI |
| **Piano Transcription** | ByteDance | 钢琴转写 | 专业钢琴转写，支持踏板检测 |
| **Whisper + WhisperX** | OpenAI | 语音识别 | 带单词级对齐的歌词识别 |

### 处理模式

| 模式 | 说明 | 适用场景 | 输出轨道数 |
|------|------|----------|-----------|
| 钢琴模式 | 跳过分离，生成多轨钢琴MIDI | 纯钢琴曲、简单旋律 | 1-6 轨（自动检测） |
| 智能模式（标准） | 6轨分离 + 乐器识别 | 完整编曲、多乐器作品 | 最多 6 轨 |
| 智能模式（精确） | YourMT3+ 直接转写 | 复杂多乐器作品 | 最多 128 种 GM 乐器 |

### 架构

```
音频输入
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 模式选择                                              │
│  ├─ 钢琴模式 ──→ ByteDance Piano Transcription       │
│  └─ 智能模式 ──→ YourMT3+ MoE 或 Demucs+Basic Pitch  │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 智能模式处理流程                                       │
│  ├──→ YourMT3+ MoE（首选）: 直接多乐器转写            │
│  │    └── 支持 128 种 GM 乐器精确识别                 │
│  ├──→ 备选方案: Demucs 6轨分离 + Basic Pitch         │
│  ├──→ 节拍检测 (librosa)                              │
│  └──→ 歌词识别 (Whisper + WhisperX)                   │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ MIDI后处理                                            │
│  ├──→ 音符量化（可选）                                │
│  ├──→ 力度平滑                                        │
│  ├──→ 智能去重（处理重叠分段）                        │
│  └──→ 复音数限制                                      │
└─────────────────────────────────────────────────────┘
    │
    ▼
输出: MIDI + LRC + WAV
```

## 开发

### 设置开发环境

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest

# 运行特定测试
pytest tests/test_yourmt3_integration.py -v

# 可选：生成覆盖率报告（会生成 htmlcov/ 与 .coverage）
pytest --cov=src --cov-report=html

# 格式化代码
black src/
isort src/

# 类型检查
mypy src/
```

### 构建可执行文件

```bash
# 安装PyInstaller
pip install pyinstaller

# 使用项目配置文件构建（推荐）
pyinstaller MusicToMidi.spec

# 构建产物在 dist/MusicToMidi/ 目录下
```

### GPU 诊断

```bash
# 检查 GPU 状态
python -c "from src.utils.gpu_utils import print_gpu_diagnosis; print_gpu_diagnosis()"

# 检查 YourMT3+ 可用性
python -c "from src.core.yourmt3_transcriber import YourMT3Transcriber; print(YourMT3Transcriber.is_available())"
```

## 常见问题

### Linux 环境问题

**Q: 提示找不到 libGL.so.1**
```bash
# Ubuntu/Debian
sudo apt install libgl1-mesa-glx

# CentOS/RHEL
sudo yum install mesa-libGL
```

**Q: PyQt6 无法启动，显示 "could not load Qt platform plugin"**
```bash
# 安装 Qt 依赖
sudo apt install libxcb-xinerama0 libxkbcommon-x11-0

# 如果在无头服务器上运行，需要虚拟显示
sudo apt install xvfb
xvfb-run python -m src.main
```

**Q: CUDA 不可用**
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回 False，重新安装正确版本的 PyTorch
pip uninstall torch torchaudio
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

**Q: YourMT3+ 不可用**
```bash
# 确保 YourMT3 代码库存在
ls YourMT3/

# 如果不存在，克隆仓库（需要 Git LFS）
git lfs install
git clone https://huggingface.co/spaces/mimbres/YourMT3
cd YourMT3
pip install -r requirements.txt
cd ..

# 下载模型
python download_sota_models.py
```

## 贡献

欢迎贡献！请随时提交Pull Request。

1. Fork本仓库
2. 创建功能分支（`git checkout -b feature/amazing-feature`）
3. 提交更改（`git commit -m 'Add amazing feature'`）
4. 推送到分支（`git push origin feature/amazing-feature`）
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](../LICENSE) 文件。

## 致谢

- [YourMT3+](https://huggingface.co/spaces/mimbres/YourMT3) - 2025 AMT Challenge SOTA 多乐器转写
- [Demucs](https://github.com/facebookresearch/demucs) - 音乐源分离
- [PANNs](https://github.com/qiuqiangkong/panns_inference) - 音频模式分析与乐器识别
- [Basic Pitch](https://github.com/spotify/basic-pitch) - 音频转MIDI转录
- [Piano Transcription](https://github.com/bytedance/piano_transcription) - ByteDance 钢琴转写
- [Whisper](https://github.com/openai/whisper) - 语音识别
- [WhisperX](https://github.com/m-bain/whisperX) - 单词级对齐
- [mido](https://github.com/mido/mido) - MIDI文件处理

## 支持

如果您遇到任何问题，请 [创建issue](https://github.com/mason369/music-to-midi/issues)。
