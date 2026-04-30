# 音乐转 MIDI 转换器

<p align="center">
  中文 | <a href="./docs/README.md">English</a>
</p>

一个基于 AI 的音频转 MIDI 工具，提供 PyQt6 桌面版、Gradio Web 版和 Google Colab 运行入口。当前版本专注两条稳定流程：完整混音多乐器转写，以及人声/伴奏分离后分别转写。

## 截图

| Windows | Linux |
|---------|-------|
| ![Windows 演示](resources/icons/Windows演示.png) | ![Linux 演示](resources/icons/Linux演示.png) |

## 当前能力

- **完整混音转写**：`SMART` 模式直接读取整首音频，用多乐器后端生成 MIDI。
- **人声/伴奏分离转写**：`VOCAL_SPLIT` 模式先分离人声与伴奏，再分别生成 MIDI；可选额外输出一个合并 MIDI。
- **YourMT3+ 默认后端**：默认使用 YourMT3+ MoE，多乐器、GM 程序号、鼓轨和多轨 MIDI 输出均由该链路承担。
- **MIROS 可选后端**：桌面版可切换到本地 `ai4m-miros` 仓库作为实验性多乐器后端。
- **MIDI 输出布局**：YourMT3+ 支持“按 GM 乐器分轨”和“非鼓合并单轨、鼓独立”两种布局。
- **节拍与后处理**：自动检测 BPM，生成带速度信息的 MIDI；支持量化、去重、力度平滑、复音限制等后处理。
- **多格式输入**：支持 `MP3`、`WAV`、`FLAC`、`OGG`、`M4A`。非 WAV 会优先通过 FFmpeg 转为 44.1 kHz PCM WAV。
- **多平台入口**：桌面版、Space、Colab 均只暴露当前保留的两种处理模式。

## 不同入口的功能范围

| 入口 | 处理模式 | 后端选择 | 适合场景 |
|------|----------|----------|----------|
| PyQt6 桌面版 | `SMART`、`VOCAL_SPLIT` | `YourMT3+`、`MIROS` | 本地长期使用、GPU 推理、批量输出文件 |
| Gradio Space | `SMART`、`VOCAL_SPLIT` | 默认 `YourMT3+` | 浏览器中快速试用或部署 |
| Google Colab | `SMART`、`VOCAL_SPLIT` | 默认 `YourMT3+` | 临时使用 Colab GPU |

## 处理模式

| 模式 | 内部流程 | 主要输出 | 说明 |
|------|----------|----------|------|
| `SMART` | 音频 -> 多乐器后端 -> MIDI 生成 | `<歌曲名>.mid` | 不做音源分离，适合大多数混音歌曲、纯音乐和多乐器片段。 |
| `VOCAL_SPLIT` | 音频 -> 人声/伴奏分离 -> 伴奏转写 -> 人声转写 -> MIDI 生成 | `<歌曲名>_accompaniment.mid`、`<歌曲名>_vocal.mid`，可选 `<歌曲名>_vocal_accompaniment_merged.mid` | 分离后会把人声 MIDI 尽量收敛到旋律轨，减少伴奏乐器幻觉。 |

## 输出文件

桌面版默认输出到：

```text
MidiOutput/<音频文件名>/
```

如果同名目录已存在，会自动使用 `<音频文件名>_2`、`<音频文件名>_3` 等后缀。

常见输出：

```text
song.mid
song_accompaniment.mid
song_vocal.mid
song_vocal_accompaniment_merged.mid
song_(Vocals).wav
song_(Instrumental).wav
```

实际文件数量取决于所选模式、是否启用合并 MIDI，以及分离器输出。

## 后端说明

### YourMT3+

YourMT3+ 是默认后端。项目使用 `download_sota_models.py` 下载默认 checkpoint，并通过 `src/core/yourmt3_transcriber.py` 调用本地 `YourMT3/amt/src` 源码。

需要满足：

```text
YourMT3/amt/src/model/ymt3.py
YourMT3/amt/src/utils/task_manager.py
YourMT3/amt/src/config/config.py
```

如果你的仓库副本没有 `YourMT3/amt/src`，可以手动放置 YourMT3 源码：

```bash
git clone https://github.com/mimbres/YourMT3.git
```

模型权重下载：

```bash
python download_sota_models.py
```

默认搜索模型位置包括：

```text
~/.cache/music_ai_models/yourmt3_all
runtime/models/yourmt3_all          # 便携版
models/yourmt3_all                  # 打包资源
```

### MIROS

MIROS 是桌面版可选实验后端。它不是 PyPI 包接入方式，而是要求本地存在上游仓库目录，并由包装器调用其入口生成临时 MIDI 后再转换为项目内部音符结构。

支持路径：

```text
ai4m-miros/
external/ai4m-miros/
MIROS/
external/MIROS/
```

包装器会检查：

```text
main.py
transcribe.py
model/musicfm/data/pretrained_msd.pt
logs/Multi_longer_seq_length_frozen_enc_silu/le2bzt53/checkpoints/last.ckpt
```

MIROS 还需要其上游运行依赖。`requirements.txt` 保证本项目运行，不保证完整安装 MIROS 上游环境。

## MIDI 轨道布局

YourMT3+ 后端提供两个输出布局：

| 布局 | 行为 |
|------|------|
| 多轨（按 GM 乐器分轨） | 每个识别到的 GM 程序号尽量独立成轨；鼓统一使用 GM 鼓通道。 |
| 单轨（旋律合并，鼓独立） | 非鼓音符合并到一条旋律轨，但保留原始通道与音色变化；鼓仍独立成轨。 |

当识别到的乐器数量超过 MIDI 非鼓通道上限时，生成器会按乐器族进行合并，避免直接丢弃音符。

## 转写质量

桌面版和 Web 版均提供：

```text
fast
balanced
best
```

- 对 `YourMT3+`：质量档位会影响后处理策略。
- 对 `MIROS`：当前包装路径使用固定 checkpoint 质量，档位不会改变 MIROS 推理本身。

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.10+，Windows 安装脚本优先使用 3.10-3.12 |
| PyTorch | 2.4.0 或更高，建议与 `torchaudio`、`torchvision` 版本匹配 |
| FFmpeg | 必需；用于可靠处理 MP3/M4A/FLAC/OGG 等格式 |
| GPU | 推荐 NVIDIA CUDA；CPU 可运行但速度慢 |
| 系统 | Windows 10/11、Linux、WSL2 |

Windows 建议把项目放在纯英文且无空格的路径，例如：

```text
C:\MusicToMidi
D:\Projects\music-to-midi
```

含中文、空格或括号的路径可能导致 PyTorch DLL 加载失败。

## 快速开始

### Windows

推荐：

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

或双击：

```text
run.bat
```

`run.ps1` 会检查虚拟环境、核心依赖、YourMT3+ 权重和人声分离模型；缺失时会调用 `install.ps1`。

### Linux / WSL2

```bash
chmod +x run.sh
./run.sh
```

`run.sh` 会检查虚拟环境、核心依赖、YourMT3+ 源码、YourMT3+ 权重和人声分离模型；缺失时会调用 `install.sh`。

### 源码直接运行

```bash
python -m src.main
```

## 手动安装

### 1. 创建虚拟环境

Windows:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

Linux:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2. 安装 PyTorch

CUDA 12.1:

```bash
pip install torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

CUDA 11.8:

```bash
pip install torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
```

CPU:

```bash
pip install torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 4. 准备 YourMT3+ 源码与模型

```bash
git clone https://github.com/mimbres/YourMT3.git
python download_sota_models.py
```

如果 `YourMT3/` 已经存在，可以只执行模型下载。

### 5. 准备人声分离模型

```bash
python download_vocal_model.py
```

模型默认缓存到：

```text
~/.music-to-midi/models/audio-separator
```

### 6. 启动

```bash
python -m src.main
```

## Google Colab

Colab 入口：

```text
colab_notebook.ipynb
```

使用步骤：

1. 打开笔记本。
2. 选择 GPU 运行时。
3. 依次运行单元格。
4. 最后一个单元格会启动 Gradio，并输出公开访问链接。

Colab 版本会保留预装 PyTorch，避免重装 torch 导致 CUDA 运行库冲突。

## Gradio Space

Space 入口：

```text
space/app.py
```

本地启动：

```bash
cd space
python app.py
```

Space 版会尝试从 Hugging Face Space 仓库同步 YourMT3 源码，并自动检查默认模型权重。

## 便携版打包

Windows 目录式便携包：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_portable.ps1
```

指定 Python 或 FFmpeg：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_portable.ps1 `
  -PythonExe .\venv\Scripts\python.exe `
  -FfmpegDir C:\ffmpeg\bin
```

打包脚本会尽量收集：

```text
YourMT3/amt/src
YourMT3 模型缓存
audio-separator 模型缓存
可选 MIROS 本地仓库
ffmpeg.exe / ffprobe.exe
```

分发时请分发整个目录：

```text
dist/MusicToMidi/
```

不要只复制单个 exe。

## 项目结构

```text
src/
  core/
    pipeline.py              # 主处理流水线
    yourmt3_transcriber.py   # YourMT3+ 后端
    miros_transcriber.py     # MIROS 本地包装器
    vocal_separator.py       # 人声/伴奏分离
    midi_generator.py        # MIDI 生成与后处理
    beat_detector.py         # BPM/节拍检测
  gui/
    main_window.py           # PyQt6 主窗口
    widgets/track_panel.py   # 模式、后端、轨道布局选择
    workers/processing_worker.py
  models/
    data_models.py           # Config、ProcessingResult、NoteEvent 等
    gm_instruments.py        # GM 128 乐器映射
  utils/
    runtime_paths.py         # 运行时资源路径
    yourmt3_downloader.py    # YourMT3+ 模型路径与下载辅助

space/app.py                 # Gradio Web 界面
colab_notebook.ipynb         # Colab 运行入口
download_sota_models.py      # YourMT3+ 默认模型下载
download_vocal_model.py      # 人声分离模型下载
MusicToMidi.spec             # PyInstaller 配置
```

## 开发命令

```bash
pytest
pytest tests/test_yourmt3_integration.py -v
black src/
isort src/
flake8 src/
mypy src/
pyinstaller MusicToMidi.spec
```

常用自检：

```bash
python -m src.main --self-test
python -c "from src.utils.gpu_utils import print_gpu_diagnosis; print_gpu_diagnosis()"
python -c "from src.core.yourmt3_transcriber import YourMT3Transcriber; print(YourMT3Transcriber.is_available())"
```

## 常见问题

### PyTorch DLL 加载失败

优先检查：

- 项目路径是否含中文、空格或括号。
- 是否已安装 Visual C++ Redistributable 2022 x64。
- PyTorch、torchaudio、torchvision 版本是否匹配。

Windows 可重新运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

### FFmpeg 不可用

Windows 可使用安装脚本自动安装，或手动安装后加入 PATH。Linux:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

### YourMT3+ 不可用

检查源码目录：

```text
YourMT3/amt/src
```

检查模型：

```bash
python -c "from src.utils.yourmt3_downloader import get_model_path; print(get_model_path())"
```

缺失时：

```bash
git clone https://github.com/mimbres/YourMT3.git
python download_sota_models.py
```

### 人声分离不可用

确认依赖和模型：

```bash
pip install "audio-separator>=0.38.0" "onnxruntime>=1.16.0,<2"
python download_vocal_model.py
```

### MIROS 不可用

确认本地仓库位置和文件完整性：

```text
ai4m-miros/main.py
ai4m-miros/transcribe.py
```

若提示缺少 Python 模块，请按 MIROS 上游仓库说明补齐依赖。

## 许可证

本项目使用 MIT License。第三方模型、数据和上游仓库遵循各自许可证与使用条款。
