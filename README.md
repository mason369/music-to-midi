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
- 下载 YPTF.MoE+Multi (PS) 模型权重（约 800MB）

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

### 3. 下载模型权重

```bash
# 下载模型权重（约 800MB）
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

本项目使用 **YPTF.MoE+Multi (PS)** — YourMT3+ 系列中的最高性能变体。

| 项目 | 详情 |
|------|------|
| 模型全称 | YPTF.MoE+Multi (PS) |
| 检查点 | `mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2` |
| 来源 | [KAIST - YourMT3+](https://huggingface.co/spaces/mimbres/YourMT3)（[arXiv:2407.04822](https://arxiv.org/abs/2407.04822)） |
| 架构 | Perceiver Transformer 编码器 + Multi-T5 解码器 |
| MoE | 8 专家, Top-2 路由, SiLU 激活 |
| 位置编码 | RoPE（部分旋转位置编码） |
| 归一化 | RMSNorm |
| 训练增强 | Pitch Shift 音高偏移增强（PS） |
| 模型大小 | ~724 MB |
| 任务类型 | `mt3_full_plus`（128 种 GM 乐器 + 鼓） |

#### 性能基准（Slakh2100 数据集）

| 指标 | YPTF.MoE+Multi (PS) | MT3 (Google 基线) |
|------|---------------------|-------------------|
| Multi F1 | **0.7484** | 0.62 |
| Frame F1 | 0.8487 | — |
| Onset F1 | 0.8419 | — |
| Offset F1 | 0.6961 | — |
| Drum Onset F1 | 0.9113 | — |

各乐器 Onset F1：Bass 0.93 / Piano 0.88 / Guitar 0.82 / Synth Lead 0.82 / Brass 0.73 / Strings 0.73

#### 仓库内可用模型变体

| 模型 | MoE | Pitch Shift | 大小 | 说明 |
|------|-----|-------------|------|------|
| YPTF.MoE+Multi (PS) | ✅ 8专家 | ✅ | 724 MB | **默认，最高性能** |
| YPTF.MoE+Multi (noPS) | ✅ 8专家 | ❌ | 724 MB | 无音高偏移增强 |
| YPTF+Multi (PS) | ❌ | ✅ | 2.0 GB | 标准 Perceiver，较轻量 |
| YPTF+Multi (noPS) | ❌ | ❌ | 2.0 GB | 标准 Perceiver，无增强 |
| YourMT3+ 传统版 | ❌ | ❌ | 2.0 GB | 旧版兼容 |

#### 未来可关注的转写模型

以下为 2025 年后出现的新兴模型和研究方向：

| 模型/方向 | 来源 | 类型 | 状态 | 说明 |
|-----------|------|------|------|------|
| [Aria-AMT5](https://github.com/EleutherAI/aria-amt) | EleutherAI | 钢琴 | ✅ 已开源 | 基于 Whisper 架构的钢琴转写，2025 年用于生成 100 万+ MIDI 数据集，钢琴领域新 SOTA |
| Streaming AMT | arXiv 2025 | 多乐器 | 📄 论文阶段 | 卷积编码器 + 自回归 Transformer 解码器，支持实时流式转写，性能接近离线 SOTA |
| 2025 AMT Challenge 冠军方案 | ISMIR 2025 | 多乐器 | 📄 论文阶段 | 8 支队伍参赛，2 支超越 MT3 基线，聚焦合成古典音乐转写 |
| CVC 评估框架 | ISMIR 2025 | 评估方法 | 📄 论文阶段 | 跨版本一致性（Cross-Version Consistency），无需标注的评估方法，适用于管弦乐场景 |

#### 同领域已有模型对比

| 模型 | 来源 | 类型 | 说明 |
|------|------|------|------|
| [MT3](https://github.com/magenta/mt3) | Google Magenta | 多乐器 | Transformer 编码-解码，YourMT3+ 的基础架构，Multi F1=0.62 (Slakh) |
| [Omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) | MCT Lab | 多任务 | 支持钢琴/鼓/人声/和弦转写，2025 年无重大更新 |
| [Basic Pitch](https://github.com/spotify/basic-pitch) | Spotify | 通用 | 轻量级单音/复音转写，适合快速推理，精度低于 MT3 系列 |

> **趋势总结**：2025 年多乐器 AMT 仍以 MT3/YourMT3+ 系 Transformer 架构为主导。钢琴转写最为成熟（Aria-AMT5），多乐器和吉他指法转写仍是活跃研究方向。实时流式转写和大规模无标注评估是新兴热点。

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
