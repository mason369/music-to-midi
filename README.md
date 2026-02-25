# 音乐转MIDI转换器

<p align="center">
  中文 | <a href="./docs/README.md">English</a>
</p>

将音频文件转换为多轨道MIDI，支持 128 种 GM 乐器精确识别。

**平台支持：Windows / Linux / WSL2**

## 截图演示

| Windows | Linux |
|---------|-------|
| ![Windows 演示](resources/icons/Windows演示.png) | ![Linux 演示](resources/icons/Linux演示.png) |

## 功能特点

- **多乐器转写**：使用 YourMT3+ MoE（2025 AMT Challenge 顶级模型）直接识别混音中的多种乐器
- **128 种 GM 乐器**：输出标准 General MIDI 多轨道 MIDI，精确区分鼓、贝斯、吉他、钢琴等
- **MIDI 后处理**：音符量化、力度平滑、去重、复音限制等优化
- **GPU 加速**：自动检测并使用 CUDA（NVIDIA）/ ROCm（AMD）/ CPU
- **多语言界面**：支持中文和英文界面切换
- **深色主题**：现代化音频软件风格界面

## 平台支持

| 平台 | 状态 | 说明 |
|------|------|------|
| Windows 10/11 (x64) | ✅ 已支持 | 双击 `run.bat` 一键启动 |
| Linux (Ubuntu/Debian) | ✅ 已支持 | 推荐 Ubuntu 22.04+，完整功能 |
| WSL2 (Windows 11) | ✅ 已支持 | 需要 WSLg（Win11 内置）|
| macOS | 🚧 计划中 | Apple Silicon MPS 支持开发中 |

## 快速开始

### Windows

```
1. 克隆或下载仓库
2. 双击 run.bat（首次自动安装所有依赖）
```

或使用 PowerShell：
```powershell
powershell -ExecutionPolicy Bypass -File run.ps1
```

### Linux / WSL2

```bash
# 1. 克隆仓库
git clone https://github.com/mason369/music-to-midi.git
cd music-to-midi

# 2. 直接运行（首次自动安装所有依赖）
./run.sh
```

启动脚本会自动检测依赖完整性，首次运行或依赖缺失时自动完成：
- 安装系统包（FFmpeg、PyQt6 所需库、中文/Emoji 字体等）
- 创建 Python 虚拟环境
- 安装 PyTorch（根据 GPU 自动选择 CUDA/ROCm/CPU 版本）
- 安装所有 Python 依赖
- 克隆 YourMT3+ 代码库
- 下载 YPTF.MoE+Multi (PS) 模型权重（约 2.5GB）

## 手动安装

如需手动安装，请按以下步骤操作：

### 1. 系统依赖

```bash
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libxkbcommon-x11-0 \
    libgl1 \
    fonts-ubuntu \
    fonts-dejavu-core \
    fonts-noto-cjk \
    fonts-noto-color-emoji
```

### 2. Python 环境

```bash
# 创建虚拟环境（推荐 Python 3.10 或 3.11）
python3.10 -m venv venv
source venv/bin/activate

# 安装 PyTorch（按 GPU 选择）
# CUDA 12.1
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 仅 CPU（速度较慢）
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# 安装项目依赖
pip install -r requirements.txt
```

### 3. YourMT3+ 代码库与模型

```bash
# 克隆 YourMT3 代码库
bash install_yourmt3_code.sh

# 下载模型权重（约 2.5GB）
python download_sota_models.py
```

### 4. 运行

```bash
source venv/bin/activate
python -m src.main
```

## WSL2 配置说明

WSL2 下运行 GUI 需要 WSLg（Windows 11 内置），环境变量会自动设置：

```bash
# 如果 DISPLAY 未自动设置，添加到 ~/.bashrc
export DISPLAY=:0
```

验证显示环境：

```bash
echo $DISPLAY        # 应显示 :0
echo $WAYLAND_DISPLAY  # 应显示 wayland-0（WSLg 环境）
```

如果 WSLg 不可用（Windows 10），可安装 VcXsrv：
1. 在 Windows 上安装 [VcXsrv](https://sourceforge.net/projects/vcxsrv/)
2. 启动 XLaunch（勾选 "Disable access control"）
3. 在 WSL 中：`export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0`

## 依赖版本说明

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| PyTorch | 2.1.0 - 2.4.x | YourMT3+ 兼容性要求 |
| torchaudio | 2.1.0 - 2.4.x | 与 PyTorch 版本对应 |
| NumPy | < 2.0 | numba 兼容性 |
| CUDA | 11.8 或 12.1 | GPU 加速（可选） |
| Python | 3.10+ | 推荐 3.10 或 3.11 |

## 使用方法

1. **打开音频文件**：拖放音频文件（MP3、WAV、FLAC、OGG 等）或点击浏览选择
2. **配置输出**：选择输出目录
3. **开始处理**：点击"开始"按钮
4. **获取结果**：在输出目录中找到 MIDI 文件

## 支持的格式

**输入**：MP3, WAV, FLAC, OGG, M4A, AAC, WMA

**输出**：MIDI (.mid)

## 技术架构

```
音频输入 → MusicToMidiPipeline
              ↓
          YourMT3+ MoE（YPTF.MoE+Multi PS）
          直接对完整混音进行多乐器转写
              ↓
          MIDI 后处理（量化 / 去重 / 复音限制）
              ↓
          多轨道 MIDI 输出（最多 128 种 GM 乐器）
```

### 使用的 AI 模型

| 模型 | 来源 | 用途 |
|------|------|------|
| **YourMT3+ MoE** | KAIST | 多乐器转写，128 种 GM 乐器（唯一转写引擎） |

## 开发

```bash
source venv/bin/activate

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/ -v

# 代码格式化
black src/
isort src/

# 类型检查
mypy src/

# 代码检查
flake8 src/ --max-line-length=100
```

## 常见问题

**Q: 中文/图标显示为方块（字体渲染问题）**
```bash
sudo apt-get install -y fonts-noto-cjk fonts-wqy-zenhei fonts-noto-color-emoji fonts-symbola
fc-cache -f
```
`./run.sh` 首次运行时会自动处理此问题。

**Q: PyQt6 提示 "could not load Qt platform plugin"**
```bash
sudo apt-get install libxcb-xinerama0 libxkbcommon-x11-0 libxcb-cursor0
```

**Q: 提示找不到 libGL.so.1**
```bash
sudo apt-get install libgl1-mesa-glx
```

**Q: WSL2 窗口无法显示**

确认 WSLg 已启用（Windows 11 22000+ 默认内置），并检查：
```bash
echo $DISPLAY   # 应为 :0
ls /mnt/wslg    # 应有文件
```

**Q: CUDA 不可用**
```bash
nvidia-smi                                    # 检查驱动
python -c "import torch; print(torch.cuda.is_available())"
```

**Q: YourMT3+ 不可用**
```bash
ls YourMT3/                    # 检查代码库是否存在
bash install_yourmt3_code.sh   # 重新安装代码库
python download_sota_models.py # 重新下载模型
```

## GPU 诊断

```bash
python -c "from src.utils.gpu_utils import print_gpu_diagnosis; print_gpu_diagnosis()"
python -c "from src.core.yourmt3_transcriber import YourMT3Transcriber; print(YourMT3Transcriber.is_available())"
```

## 贡献

欢迎提交 Pull Request。

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'feat: add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

## 许可证

MIT License - 详见 [LICENSE](./LICENSE) 文件。

## 致谢

- [YourMT3+](https://huggingface.co/spaces/mimbres/YourMT3) - SOTA 多乐器转写
- [mido](https://github.com/mido/mido) - MIDI 文件处理
- [librosa](https://librosa.org/) - 音频分析
