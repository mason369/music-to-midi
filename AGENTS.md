# AGENTS.md

此文件为 Codex 在本仓库中工作时提供指导。

## 项目概述

音乐转 MIDI 转换器是一个基于 AI 的桌面与 Web 应用，可将音频文件转换为 MIDI。当前产品流程包含：

- `SMART`：完整混音多乐器转写。
- `VOCAL_SPLIT`：人声/伴奏分离后分别转写，可选生成合并 MIDI。
- `SIX_STEM_SPLIT`：六声部分离后分别转写，可选只转写指定 stem。
- `PIANO_TRANSKUN`：Transkun 钢琴专用转写。
- `PIANO_ARIA_AMT`：Aria-AMT 钢琴专用转写。

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
3. **SIX_STEM_SPLIT**：使用 BS-RoFormer SW 分离 bass/drums/guitar/piano/vocals/other 六个 stem，再生成 stem MIDI 和合并 MIDI。
4. **PIANO_TRANSKUN**：使用 Transkun 钢琴专用模型直接转写纯钢琴音频。
5. **PIANO_ARIA_AMT**：使用 Aria-AMT 钢琴专用模型直接转写纯钢琴音频。

### 处理流水线

```text
音频输入 -> 模式选择
    -> SMART: 多乐器后端直接转写
    -> VOCAL_SPLIT: 人声/伴奏分离 -> 分别转写
    -> SIX_STEM_SPLIT: 六声部分离 -> stem MIDI -> 合并 MIDI
    -> PIANO_TRANSKUN: Transkun 钢琴专用转写
    -> PIANO_ARIA_AMT: Aria-AMT 钢琴专用转写
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
| `TranskunTranscriber` | `src/core/transkun_transcriber.py` | Transkun 钢琴专用转写 |
| `AriaAmtTranscriber` | `src/core/aria_amt_transcriber.py` | Aria-AMT 钢琴专用转写 |
| `SixStemSeparator` | `src/core/multi_stem_separator.py` | 六声部分离 |
| `VocalSeparator` | `src/core/vocal_separator.py` | 人声/伴奏分离 |
| `MidiGenerator` | `src/core/midi_generator.py` | MIDI 文件生成 |
| `BeatDetector` | `src/core/beat_detector.py` | 节拍检测 |

### 数据模型

关键枚举：

- `ProcessingMode`: `SMART`, `VOCAL_SPLIT`, `SIX_STEM_SPLIT`, `PIANO_TRANSKUN`, `PIANO_ARIA_AMT`, legacy `PIANO` -> `SMART`
- `TranscriptionBackend`: `ARIA_AMT`, `YOURMT3`, `MIROS`
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

三者应保持相同的处理模式集合：`SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT`、`PIANO_TRANSKUN`、`PIANO_ARIA_AMT`。

## AI 模型

| 模型 | 用途 | 位置 |
|------|------|------|
| `YourMT3+ MoE` | 默认多乐器转写 | `YourMT3/` + 模型缓存 |
| `MIROS` | 可选实验多乐器转写 | 本地 `ai4m-miros/` 或 `external/ai4m-miros/` |
| `audio-separator` 模型 | 人声/伴奏分离 | 运行时模型缓存 |
| `BS-RoFormer SW` | 六声部分离 | `~/.music-to-midi/models/audio-separator/` 或打包资源 |
| `Transkun` | 钢琴专用转写 | `transkun` 包随附资源 |
| `Aria-AMT` | 钢琴专用转写 | `~/.cache/music_ai_models/aria_amt/` 或打包资源 |

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

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **music-to-midi** (3331 symbols, 7593 relationships, 284 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "master"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/music-to-midi/context` | Codebase overview, check index freshness |
| `gitnexus://repo/music-to-midi/clusters` | All functional areas |
| `gitnexus://repo/music-to-midi/processes` | All execution flows |
| `gitnexus://repo/music-to-midi/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
