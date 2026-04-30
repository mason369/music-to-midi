# AGENTS.md

此文件为 Codex 在本仓库中工作时提供指导。

## 项目概述

音乐转 MIDI 转换器是一个基于 AI 的桌面与 Web 应用，可将音频文件转换为 MIDI。当前产品流程只保留：

- `SMART`：完整混音多乐器转写。
- `VOCAL_SPLIT`：人声/伴奏分离后分别转写，可选生成合并 MIDI。

**技术栈**: Python 3.10+, PyQt6, PyTorch 2.1-2.4, PyInstaller, Gradio

## 开发命令

```bash
# 运行应用程序
python -m src.main

# 运行所有测试
pytest

# 运行单个测试文件
pytest tests/test_yourmt3_integration.py -v

# 代码格式化和检查
black src/
isort src/
flake8 src/
mypy src/

# 构建 Windows 可执行文件
pyinstaller MusicToMidi.spec

# GPU 诊断
python -c "from src.utils.gpu_utils import print_gpu_diagnosis; print_gpu_diagnosis()"

# 检查 YourMT3+ 可用性
python -c "from src.core.yourmt3_transcriber import YourMT3Transcriber; print(YourMT3Transcriber.is_available())"

# 下载 YourMT3+ 模型
python download_sota_models.py
```

## 架构

### 处理模式

1. **SMART**：使用 `YourMT3+` 或本地 `MIROS` 后端直接转写完整混音，支持 128 种 GM 乐器识别。
2. **VOCAL_SPLIT**：使用 `audio-separator` 分离人声与伴奏，再用所选多乐器后端分别转写。

### 处理流水线

```text
音频输入 -> 模式选择
    -> SMART: 多乐器后端直接转写
    -> VOCAL_SPLIT: 人声/伴奏分离 -> 分别转写
    -> 节拍检测
    -> MIDI 生成与后处理
    -> 输出 MIDI 文件
```

### 核心类

| 类名 | 文件 | 用途 |
|------|------|------|
| `MusicToMidiPipeline` | `src/core/pipeline.py` | 协调整个工作流程 |
| `YourMT3Transcriber` | `src/core/yourmt3_transcriber.py` | YourMT3+ 多乐器转写 |
| `MirosTranscriber` | `src/core/miros_transcriber.py` | MIROS 实验后端包装 |
| `VocalSeparator` | `src/core/vocal_separator.py` | 人声/伴奏分离 |
| `MidiGenerator` | `src/core/midi_generator.py` | MIDI 文件生成 |
| `BeatDetector` | `src/core/beat_detector.py` | 节拍检测 |

### 数据模型

关键枚举：

- `ProcessingMode`: `SMART`, `VOCAL_SPLIT`, legacy `PIANO` -> `SMART`
- `MultiInstrumentModel`: `YOURMT3`, `MIROS`
- `MidiTrackMode`: `MULTI_TRACK`, `SINGLE_TRACK`
- `TranscriptionQuality`: `FAST`, `BALANCED`, `BEST`

关键数据类：

- `Config`
- `ProcessingResult`
- `NoteEvent`
- `BeatInfo`
- `Project`

### GUI 结构

- `src/gui/main_window.py`：PyQt6 主窗口
- `src/gui/widgets/track_panel.py`：模式、后端、MIDI 轨道布局选择
- `src/gui/widgets/progress_widget.py`：处理进度显示
- `src/gui/workers/processing_worker.py`：后台处理线程

### 平台界面

- 桌面版：`python -m src.main`
- Gradio Space：`space/app.py`
- Colab：`colab_notebook.ipynb`

三者应保持相同的处理模式集合：`SMART` 与 `VOCAL_SPLIT`。

## AI 模型

| 模型 | 用途 | 位置 |
|------|------|------|
| `YourMT3+ MoE` | 默认多乐器转写 | `YourMT3/` + 模型缓存 |
| `MIROS` | 可选实验多乐器转写 | 本地 `ai4m-miros/` 或 `external/ai4m-miros/` |
| `audio-separator` 模型 | 人声/伴奏分离 | 运行时模型缓存 |

## 约束条件

- 需要 PyTorch 2.1.0-2.4.x。
- 需要 NumPy `<2.0`。
- 需要外部安装或打包 FFmpeg。
- 推荐 NVIDIA GPU + CUDA。
- YourMT3+ 需要源码目录 `YourMT3/amt/src` 可用。

## 国际化

翻译文件位于 `src/i18n/`：

- `zh_CN.json`
- `en_US.json`
