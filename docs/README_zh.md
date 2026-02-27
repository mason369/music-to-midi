# 音乐转MIDI转换器

<p align="center">
  中文 | <a href="./README.md">English</a>
</p>

将音频文件转换为多轨道MIDI，支持 128 种 GM 乐器精确识别。

**平台支持：Windows / Linux / WSL2**

## 截图演示

| Windows | Linux |
|---------|-------|
| ![Windows 演示](../resources/icons/Windows演示.png) | ![Linux 演示](../resources/icons/Linux演示.png) |

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

## 手动安装

### 依赖环境要求

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| PyTorch | 2.1.0 - 2.4.x | YourMT3+ 兼容性要求 |
| torchaudio | 2.1.0 - 2.4.x | 与 PyTorch 版本对应 |
| NumPy | < 2.0 | numba 兼容性 |
| CUDA | 11.8 或 12.1 | GPU 加速（可选） |
| Python | 3.10+ | 推荐 3.10 或 3.11 |

### Linux 安装

```bash
# 1. 克隆仓库
git clone https://github.com/mason369/music-to-midi.git
cd music-to-midi

# 2. 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 3. 安装 PyTorch（根据 GPU 选择）
# CUDA 12.1
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# 仅 CPU
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 下载模型权重
python download_sota_models.py

# 6. 运行
python -m src.main
```

### Windows 安装

```powershell
# 克隆仓库
git clone https://github.com/mason369/music-to-midi.git
cd music-to-midi

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装 PyTorch
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install -r requirements.txt

# 运行
python -m src.main
```

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
音频输入 (16kHz mono)
    ↓
MusicToMidiPipeline
    ├── BeatDetector (librosa) → BPM / 节拍网格
    └── YourMT3Transcriber
            ↓
        ┌─────────────────────────────────────────────┐
        │  YourMT3+ MoE (YPTF.MoE+Multi PS)          │
        │                                             │
        │  音频 → Mel Spectrogram (256×512)            │
        │    ↓                                        │
        │  Pre-Encoder (Conv2d Res3B, 1→C, /8 freq)   │
        │    ↓                                        │
        │  PerceiverTF Encoder (26 latents, MoE×8)    │
        │    ↓                                        │
        │  Multi-T5 Decoder (13 channels, AR greedy)  │
        │    ↓                                        │
        │  Token → Note Events (per channel)          │
        └─────────────────────────────────────────────┘
            ↓
        智能去重 (重叠分段合并)
            ↓
        MIDI 后处理 (量化 / 力度平滑 / 复音限制)
            ↓
        多轨道 MIDI 输出 (最多 128 种 GM 乐器)
```

### 项目源码结构

```
src/
├── main.py                          # 应用入口
├── core/                            # 核心处理引擎
│   ├── pipeline.py                  # 主处理流水线 (MusicToMidiPipeline)
│   ├── yourmt3_transcriber.py       # YourMT3+ 转写器封装
│   ├── beat_detector.py             # 节拍/BPM 检测
│   ├── midi_generator.py            # MIDI 生成与后处理
│   └── vocal_separator.py           # BS-RoFormer 人声分离
├── gui/                             # PyQt6 图形界面
│   ├── main_window.py               # 主窗口 (MainWindow)
│   ├── widgets/
│   │   ├── dropzone.py              # 文件拖放区
│   │   ├── track_panel.py           # 轨道/模式配置面板
│   │   └── progress_widget.py       # 进度显示组件
│   └── workers/
│       └── processing_worker.py     # QThread 后台处理线程
├── models/                          # 数据模型
│   ├── data_models.py               # 所有核心数据类 (Config, NoteEvent, BeatInfo...)
│   └── gm_instruments.py            # 128 种 GM 乐器定义与映射
├── i18n/                            # 国际化
│   └── translator.py                # 翻译系统 (zh_CN / en_US)
└── utils/                           # 工具模块
    ├── gpu_utils.py                 # GPU 检测/设备选择/内存管理
    ├── yourmt3_downloader.py        # 模型权重下载与缓存
    ├── audio_utils.py               # 音频加载/重采样/格式转换
    ├── logger.py                    # 彩色日志系统
    └── warnings_filter.py           # 第三方库警告过滤
```

---

### 应用入口与启动流程

`src/main.py` (~150 行) 负责环境初始化和 GUI 启动：

```
main()
├── 环境变量设置
│   ├── OMP_NUM_THREADS = 物理核心数 (动态检测)
│   ├── MKL_NUM_THREADS = 同上
│   └── TF_CPP_MIN_LOG_LEVEL = 3 (抑制 TensorFlow 日志)
├── Windows 特殊处理
│   ├── DLL 路径修复 (中文路径 → 8.3 短路径)
│   └── PyTorch DLL 预加载 (torch.dll, torch_cpu.dll)
├── 第三方库警告抑制 (warnings_filter.py)
├── 日志系统初始化 (logger.py)
└── PyQt6 应用启动
    ├── 高 DPI 缩放: Floor 策略 (防止界面过大)
    ├── 样式: Fusion (跨平台一致)
    ├── 中文字体: Microsoft YaHei / Noto Sans CJK
    ├── MainWindow 创建
    └── 事件循环 (app.exec())
```

---

### 处理流水线：MusicToMidiPipeline

`src/core/pipeline.py` (~400 行) 是核心调度器，协调所有处理阶段。

支持两种处理模式：

#### 模式一：SMART（默认）

```
音频文件
    ↓
BeatDetector.detect() → BPM / 节拍网格        [0-10%]
    ↓
YourMT3Transcriber.transcribe_precise()        [10-85%]
    → instrument_notes: Dict[program, List[NoteEvent]]
    → drum_notes: Dict[program, List[NoteEvent]]
    ↓
MidiGenerator.generate_from_precise_instruments_v2()  [85-95%]
    ↓
多轨道 MIDI 文件                                [95-100%]
```

#### 模式二：VOCAL_SPLIT（人声分离）

```
音频文件
    ↓
BeatDetector.detect() → BPM                    [0-5%]
    ↓
VocalSeparator.separate()                       [5-35%]
    → vocals.wav + no_vocals.wav
    ↓
YourMT3Transcriber.transcribe_precise(伴奏)     [35-60%]
    ↓
YourMT3Transcriber.transcribe_precise(人声)     [60-85%]
    ↓
MidiGenerator × 2 (伴奏 MIDI + 人声 MIDI)       [85-95%]
    ↓
两个 MIDI 文件                                   [95-100%]
```

关键设计：
- **取消传播**：`pipeline.cancel()` 向所有子模块传播，支持任意阶段中止
- **进度回调**：每个阶段通过 `_report()` 发送 `ProcessingProgress`，包含阶段进度和总体进度
- **错误隔离**：每个阶段独立 try/catch，finally 块确保模型卸载和 GPU 内存清理

---

### 节拍检测：BeatDetector

`src/core/beat_detector.py` (~380 行) 使用 librosa 的多算法融合策略检测 BPM。

```
音频文件
    ↓
librosa.load(sr=22050, mono=True)
    ↓
4 种算法并行检测:
    ├── librosa.beat.beat_track() → tempo₁
    ├── librosa.beat.tempo(onset_envelope) → tempo₂
    ├── librosa.beat.tempo(aggregate=np.mean) → tempo₃
    └── librosa.feature.tempogram() → tempo₄
    ↓
倍频修正: _correct_octave_error()
    将候选值约束到 60-200 BPM 范围
    (如 240 BPM → 120 BPM, 50 BPM → 100 BPM)
    ↓
聚类投票: _vote_best_tempo()
    聚类阈值 8 BPM
    选择包含最多候选值的聚类中心
    ↓
BeatInfo(bpm, beat_times, downbeats, time_signature)
```

设计特点：
- 多算法投票比单一算法更鲁棒，避免极端值
- 倍频修正解决常见的"双倍/半倍 BPM"误检
- 失败时优雅降级到默认 120 BPM

---

### 人声分离：VocalSeparator

`src/core/vocal_separator.py` (~200 行) 通过 audio-separator 库封装 BS-RoFormer 模型。

```
音频文件
    ↓
BS-RoFormer 模型 (SDR 12.97)
    ↓
2 个 stem:
    ├── vocals.wav
    └── instrumental.wav (→ 重命名为 accompaniment.wav)
    ↓
输出: {"vocals": path, "no_vocals": path}
```

关键参数：
- 模型: `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (首次使用时自动下载)
- 模型缓存: `~/.music-to-midi/models/audio-separator/`
- 后台进度线程：每 3 秒更新一次进度（audio-separator 不提供细粒度回调）

---

### MIDI 生成与后处理：MidiGenerator

`src/core/midi_generator.py` (~1100 行) 将音符事件转换为标准 MIDI 文件。

#### 核心方法：generate_from_precise_instruments_v2()

```
instrument_notes: Dict[int, List[NoteEvent]]  (program → 音符列表)
drum_notes: Dict[int, List[NoteEvent]]        (program → 鼓音符列表)
    ↓
后处理链 (按质量模式选择):
    ↓
MIDI 通道分配:
    ├── 通道 0-8, 10-15: 旋律乐器 (最多 15 种)
    ├── 通道 9: 鼓 (GM 标准, 固定)
    └── 超过 15 种乐器时: 智能合并同族乐器
    ↓
人声特殊处理:
    program 100/101 (YourMT3 人声) → 映射为钢琴音色
    ↓
mido.MidiFile 写入:
    ├── Track 0: 速度轨 (tempo meta event)
    └── Track 1-N: 各乐器轨道
        ├── program_change (音色选择)
        ├── note_on / note_off (音符事件)
        └── control_change (踏板等)
    ↓
.mid 文件输出
```

#### 后处理链（7 步）

根据 `transcription_quality` 选择不同的后处理强度：

| 步骤 | 方法 | best | balanced | fast |
|------|------|------|----------|------|
| 1 | `_smooth_vibrato()` 平滑颤音 | ❌ | ✅ | ❌ |
| 2 | `_remove_duplicate_notes(25ms)` 去重 | ❌ | ✅ | ❌ |
| 3 | `_merge_close_notes(10ms)` 合并碎片 | ❌ | ✅ | ❌ |
| 4 | `_quantize_notes(1/32)` 量化到网格 | ❌ | ✅ | ❌ |
| 5 | `_smooth_velocity(window=5)` 力度平滑 | ❌ | ✅ | ❌ |
| 6 | `_normalize_velocity(mean=80)` 力度归一化 | ❌ | ✅ | ❌ |
| 7 | `_limit_polyphony(max=40)` 复音限制 | ❌ | ✅ | ❌ |
| 特殊 | `post_process_minimal()` 仅移除 <10ms | ✅ | ❌ | ❌ |

`best` 模式有意跳过大部分后处理，保留 AI 模型的原始输出，仅移除明显的噪音音符（<10ms）。这是因为 YourMT3+ 的输出质量已经很高，过度后处理反而会损失细节。

---

### GUI 层

#### 主窗口：MainWindow

`src/gui/main_window.py` (~900 行) 基于 PyQt6，深色主题 Fusion 样式。

```
┌─────────────────────────────────────────┐
│  🎵 Music to MIDI                       │  ← _create_header()
│  将音频转换为多轨道 MIDI                  │
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────┐    │
│  │  🎵 拖放音频文件到此处           │    │  ← DropZoneWidget
│  │     或点击浏览选择文件           │    │
│  └─────────────────────────────────┘    │
├─────────────────────────────────────────┤
│  处理模式: [智能模式 ▼]                  │  ← TrackPanel
│  YourMT3+ 直接对完整混音进行多乐器转写    │
├─────────────────────────────────────────┤
│  ● 预处理 → ● 转写 → ○ 生成             │  ← ProgressWidget
│  ████████████████░░░░░░░░  65%          │     (StageIndicator × N)
│  正在转写: 第 15/28 段...               │
├─────────────────────────────────────────┤
│  输出目录: [/path/to/output] [浏览]      │  ← _create_output_settings()
├─────────────────────────────────────────┤
│  [  开始处理  ]  [  停止  ]              │  ← _create_action_buttons()
├─────────────────────────────────────────┤
│  GPU: NVIDIA RTX 4090 | 显存: 8.2/24GB  │  ← 状态栏
└─────────────────────────────────────────┘
```

关键设计：
- **后台 GPU 检测**：启动时在独立线程中检测 GPU，避免阻塞 UI
- **信号-槽通信**：`ProcessingWorker` 通过 Qt 信号发送进度/结果/错误，线程安全
- **阴影效果**：`QGraphicsDropShadowEffect` 为卡片组件添加立体感
- **菜单栏**：文件（打开/退出）、编辑（设置）、视图（语言切换）、帮助（关于）

#### 后台处理线程：ProcessingWorker

`src/gui/workers/processing_worker.py` (~70 行) 继承 `QThread`。

```
MainWindow._start_processing()
    ↓
ProcessingWorker(QThread)
    ├── 信号:
    │   ├── progress_updated(ProcessingProgress)  → 更新进度条
    │   ├── processing_finished(ProcessingResult) → 显示完成对话框
    │   └── error_occurred(str)                   → 显示错误对话框
    ├── run():
    │   ├── MusicToMidiPipeline(config)
    │   ├── pipeline.process(audio_path, output_dir)
    │   └── finally: clear_gpu_memory()
    └── cancel():
        └── pipeline.cancel()
```

#### 自定义组件

| 组件 | 文件 | 功能 |
|------|------|------|
| `DropZoneWidget` | `widgets/dropzone.py` (~200 行) | 拖放 + 浏览双通道文件输入，支持 MP3/WAV/FLAC/OGG/M4A/AAC/WMA |
| `TrackPanel` | `widgets/track_panel.py` (~120 行) | 处理模式选择（智能/人声分离），发射 `mode_changed` 信号 |
| `ProgressWidget` | `widgets/progress_widget.py` (~280 行) | 阶段流程指示器 + 进度条，根据模式动态重建阶段 |
| `StageIndicator` | `widgets/progress_widget.py` | 单个阶段的状态指示（pending/current/done），带颜色和图标变化 |

---

### 数据模型

`src/models/data_models.py` (~650 行) 定义所有核心数据结构。

#### 枚举类型

```python
ProcessingMode        # SMART | VOCAL_SPLIT | PIANO(已弃用→SMART)
TranscriptionQuality  # FAST | BALANCED | BEST
ProcessingStage       # PREPROCESSING | SEPARATION | TRANSCRIPTION |
                      # VOCAL_TRANSCRIPTION | SYNTHESIS | COMPLETE
InstrumentType        # PIANO | DRUMS | BASS | GUITAR | VOCALS | STRINGS |
                      # BRASS | WOODWIND | SYNTH | ORGAN | ... (25种)
                      # 提供 to_program_number() / get_display_name(lang)
```

#### 核心数据类

```python
@dataclass
class NoteEvent:
    pitch: int          # MIDI 音高 (0-127)
    start_time: float   # 起始时间 (秒)
    end_time: float     # 结束时间 (秒)
    velocity: int       # 力度 (0-127)
    program: int        # GM 程序号 (0-127, 128=鼓)

@dataclass
class BeatInfo:
    bpm: float                          # 检测到的 BPM
    beat_times: List[float]             # 节拍时间点列表
    downbeats: Optional[List[float]]    # 强拍时间点
    time_signature: tuple               # 拍号 (4, 4)

@dataclass
class ProcessingProgress:
    stage: ProcessingStage      # 当前阶段
    stage_progress: float       # 阶段内进度 (0-1)
    overall_progress: float     # 总体进度 (0-1)
    message: str                # 显示消息

@dataclass
class ProcessingResult:
    midi_path: str                              # 主 MIDI 文件路径
    tracks: List[Track]                         # 轨道列表
    beat_info: Optional[BeatInfo]               # 节拍信息
    processing_time: float                      # 处理耗时 (秒)
    total_notes: int                            # 总音符数
    vocal_midi_path: Optional[str]              # 人声 MIDI (仅 VOCAL_SPLIT)
    accompaniment_midi_path: Optional[str]      # 伴奏 MIDI (仅 VOCAL_SPLIT)
    separated_audio: Optional[Dict[str, str]]   # 分离音频路径

@dataclass
class Config:
    # 界面
    language: str = "zh_CN"             # 界面语言
    theme: str = "dark"                 # 主题
    # GPU
    use_gpu: bool = True                # 启用 GPU
    gpu_device: int = 0                 # GPU 设备索引
    # 处理
    processing_mode: str = "smart"      # 处理模式
    transcription_quality: str = "best" # 转写质量
    # MIDI 后处理
    quantize_notes: bool = True         # 音符量化
    quantize_grid: str = "1/32"         # 量化网格
    max_polyphony: int = 40             # 最大复音数
    aggressive_post_processing: bool = False  # 激进后处理
    # ... 更多字段见源码
```

#### GM 乐器映射

`src/models/gm_instruments.py` (~450 行) 定义完整的 128 种 General MIDI 乐器：

```
16 个乐器族 (GMFamily):
├── PIANO (0-7):       Acoustic Grand → Clavinet
├── CHROMATIC (8-15):  Celesta → Dulcimer
├── ORGAN (16-23):     Drawbar Organ → Tango Accordion
├── GUITAR (24-31):    Nylon Guitar → Guitar Harmonics
├── BASS (32-39):      Acoustic Bass → Synth Bass 2
├── STRINGS (40-47):   Violin → Tremolo Strings
├── ENSEMBLE (48-55):  String Ensemble → Orchestra Hit
├── BRASS (56-63):     Trumpet → Tuba
├── REED (64-71):      Soprano Sax → Bassoon
├── PIPE (72-79):      Piccolo → Ocarina
├── SYNTH_LEAD (80-87): Square → Charang
├── SYNTH_PAD (88-95): New Age → Sweep
├── SYNTH_FX (96-103): Rain → Sci-fi
├── ETHNIC (104-111):  Sitar → Shanai
├── PERCUSSIVE (112-119): Tinkle Bell → Reverse Cymbal
└── SOUND_FX (120-127): Guitar Fret → Gunshot

YourMT3 扩展:
├── Program 100: Singing Voice (女声)
└── Program 101: Singing Voice (男声)
```

提供双语乐器名称查询：`get_instrument_name(program=0, language="zh_CN")` → `"原声大钢琴"`

---

### 工具模块

#### GPU 管理：gpu_utils.py

`src/utils/gpu_utils.py` (~900 行) 管理多种加速器的检测、选择和内存优化。

```
支持的加速器:
├── CUDA (NVIDIA)     — 主要支持，自动检测 CUDA 版本
├── ROCm (AMD)        — 通过 torch.cuda (ROCm 兼容层)
├── MPS (Apple)       — macOS Apple Silicon
├── XPU (Intel)       — 需要 intel_extension_for_pytorch
├── DirectML          — Windows 通用 GPU (实验性)
└── CPU               — 回退方案

关键函数:
├── get_device(prefer_gpu, gpu_index) → "cuda:0" / "cpu" / ...
├── get_optimal_batch_size(n_segments, quality, device, ultra_quality)
│   ├── 性能档位检测: high (≥8GB) / medium (4-8GB) / low (<4GB)
│   ├── best 模式: bsz = 4/2/1
│   ├── balanced 模式: bsz = 8/4/2
│   └── fast 模式: bsz = 16/8/4
├── clear_gpu_memory()
│   ├── torch.cuda.empty_cache()
│   ├── gc.collect()
│   └── 平台特定清理
└── diagnose_gpu() → 完整诊断报告 (dict)
```

特殊处理：
- Windows DLL 路径修复：中文用户名/特殊字符路径 → 8.3 短路径
- OOM 自动回退：推理时 CUDA OOM → batch size 减半重试
- 动态线程数：`OMP_NUM_THREADS` 设为物理核心数（非逻辑核心数）

#### 模型下载：yourmt3_downloader.py

`src/utils/yourmt3_downloader.py` (~550 行) 管理 YourMT3+ 模型权重的下载和缓存。

```
模型缓存结构:
~/.cache/music_ai_models/yourmt3_all/
└── amt/logs/2024/{checkpoint_name}/checkpoints/model.ckpt

下载流程:
├── 检查本地缓存 → 命中则直接返回路径
├── 尝试 Hugging Face 主站下载
├── 失败 → 尝试 HF 镜像站 (hf-mirror.com, 国内加速)
├── SSL 证书修复 (企业网络/代理环境)
├── 断点续传支持
└── 递归路径搜索 (兼容仓库结构变化)

Checkpoint 名称映射:
├── "YPTF.MoE+Multi (PS)" → mc13_256_g4_all_v7_mt3f_...
├── "YPTF.MoE+Multi (noPS)" → ...
├── "YPTF+Multi (PS)" → ...
└── "YPTF+Multi (noPS)" → ...
```

#### 国际化：translator.py

`src/i18n/translator.py` (~200 行) 提供全局翻译函数 `t(key)`。

```
设计:
├── 单例模式: get_translator() 返回全局实例
├── JSON 存储: src/i18n/zh_CN.json, src/i18n/en_US.json
├── 嵌套键: t("menu.file.open") → 查找 {"menu": {"file": {"open": "打开"}}}
├── 参数替换: t("progress.segment", current=5, total=28) → "第 5/28 段"
└── 回退机制: 当前语言缺失 → 尝试 en_US → 返回原始 key

支持语言:
├── zh_CN: 简体中文 (默认)
└── en_US: English
```

#### 其他工具

| 模块 | 行数 | 功能 |
|------|------|------|
| `audio_utils.py` | ~100 | 音频加载/保存/重采样/格式检测，支持 7 种格式 |
| `logger.py` | ~150 | ANSI 彩色控制台日志 + 纯文本文件日志，按级别着色 |
| `warnings_filter.py` | ~250 | 抑制 TensorFlow/Keras/librosa 等库的警告，自定义 stderr 过滤器 |

---

### 核心 AI 模型：YourMT3+ YPTF.MoE+Multi (PS)

本项目唯一的转写引擎。由韩国科学技术院 (KAIST) 的 Sungkyun Chang 开发，论文发表于 IEEE MLSP 2024。

| 项目 | 详情 |
|------|------|
| 全称 | YPTF.MoE+Multi (PS) |
| Checkpoint | `mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2` |
| 来源 | [KAIST - YourMT3+](https://huggingface.co/spaces/mimbres/YourMT3) ([arXiv:2407.04822](https://arxiv.org/abs/2407.04822)) |
| 许可证 | Apache 2.0 |
| 模型大小 | ~724 MB |
| 任务类型 | `mc13_full_plus_256` (34 种 MT3 乐器类 → 13 解码通道, 最大 256 tokens/通道) |

#### 编码器：PerceiverTF + MoE

PerceiverTF 是一种层次化注意力 Transformer，核心思想是用少量可学习的 latent 向量通过交叉注意力从高维输入中提取信息，避免了标准 Transformer 对长序列的二次复杂度。

```
Mel Spectrogram (B, 256, 512)
    ↓
Pre-Encoder: Conv2d Res3B
    3 个残差卷积块, kernel=(3,3), 频率维度 /8
    输出: (B, 256, 64, C)  →  reshape → (B, 256, 64*C)
    ↓
PerceiverTF Encoder
    ├── 26 个可学习 Latent 向量 (d_latent)
    ├── 3 个 PerceiverTF Block, 每个包含:
    │   ├── SCA (Stochastic Cross-Attention)
    │   │   Latent 作为 Query, 频谱特征作为 Key/Value
    │   │   attention_to_channel=True: 注意力沿频率通道维度
    │   │   sca_use_query_residual=True: Query 残差连接
    │   ├── 2 × Local Self-Attention
    │   │   Latent 之间的局部自注意力
    │   └── 2 × Temporal Self-Attention
    │       时间维度上的全局自注意力
    ├── 位置编码: RoPE (Rotary Position Embedding)
    │   rope_partial_pe=True: 仅对部分维度应用旋转
    ├── 归一化: RMSNorm (比 LayerNorm 更高效)
    └── 前馈层: MoE (Mixture of Experts)
        ├── 8 个专家网络 (num_experts=8)
        ├── Top-2 路由 (topk=2): 每个 token 激活 2 个专家
        ├── 扩展因子: 4 (ff_widening_factor=4)
        └── 激活函数: SiLU (Sigmoid Linear Unit)
```

MoE 的关键优势：8 个专家中每次只激活 2 个，参数量大但计算量与单专家相当。不同专家自然学会处理不同乐器族的特征，实现了隐式的乐器专业化。

#### 解码器：Multi-Channel T5

Multi-T5 解码器将 128 种 GM 乐器映射到 13 个独立解码通道，每个通道负责一个乐器族：

| 通道 | 乐器族 | GM 程序号范围 |
|------|--------|-------------|
| 0 | Piano (钢琴) | 0-7 |
| 1 | Chromatic Percussion (半音阶打击乐) | 8-15 |
| 2 | Organ (风琴) | 16-23 |
| 3 | Guitar (吉他) | 24-31 |
| 4 | Bass (贝斯) | 32-39 |
| 5 | Strings (弦乐 + 合奏) | 40-55 |
| 6 | Brass (铜管) | 56-63 |
| 7 | Reed (簧管) | 64-71 |
| 8 | Pipe (哨笛) | 72-79 |
| 9 | Synth Lead (主奏合成器) | 80-87 |
| 10 | Synth Pad (铺底合成器) | 88-95 |
| 11 | Singing Voice (人声) | 100-101 |
| 12 | Drums (鼓) | 128 (内部) |

每个通道独立进行自回归解码，最大 token 长度 256。解码策略为 greedy（`logits.argmax(-1)`），使用 KV-cache 加速。

```
Encoder Hidden States (B, T, D)
    ↓
Multi-Channel T5 Decoder
    ├── 13 个独立解码通道, 共享权重
    ├── 基于 T5-small 架构 (google/t5-v1_1-small)
    ├── 自回归生成: <BOS> → token₁ → token₂ → ... → <EOS>
    ├── 每步: embed → decoder(+KV-cache) → lm_head → argmax
    └── 最大长度: 256 tokens/通道
        ↓
    Token 序列 (B, 13, ≤256)
        ↓
    TaskManager.detokenize() → NoteEvent / TieEvent
        ↓
    merge_zipped_note_events_and_ties_to_notes()
        ↓
    mix_notes() → 合并 13 通道
```

#### 训练配置

```
训练命令 (作者公开):
python train.py mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2 \
  -p slakh2024 -d all_cross_final -it 320000 -vit 20000 \
  -enc perceiver-tf -dec multi-t5 -nl 26 \
  -ff moe -wf 4 -nmoe 8 -kmoe 2 -act silu \
  -epe rope -rp 1 -sqr 1 -atc 1 \
  -ac spec -hop 300 -bsz 10 10 -xk 5 \
  -tk mc13_full_plus_256 \
  -edr 0.05 -ddr 0.05 -sb 1 -ps -2 2 \
  -st ddp -wb online
```

| 参数 | 值 | 含义 |
|------|-----|------|
| 数据集 | Slakh2024 + all_cross_final | 多数据集交叉训练 |
| 迭代次数 | 320,000 | 总训练步数 |
| 全局 batch size | 80 | 10×10 (2 GPU × 5 accumulation) |
| 音高偏移 | [-2, +2] 半音 | 数据增强 |
| 精度 | bf16-mixed | 混合精度训练 |
| 采样率 | 16,000 Hz | 输入音频 |
| Hop length | 300 | 频谱图帧步长 |
| 输入帧数 | 32,767 (~2.05s) | 每段音频长度 |

#### 推理流程（本项目实现）

```
完整音频 (任意长度)
    ↓
重采样到 16kHz, 转单声道
    ↓
分段: slice_padded_array()
    25% 重叠 (best 模式), 每段 32,767 帧 (~2.05s)
    ↓
批量推理: inference_file(bsz=auto)
    自动混合精度 (bf16/fp16)
    OOM 自动回退 (bsz 减半重试)
    ↓
13 通道 token 解码
    ↓
智能去重: _deduplicate_overlapping_notes_smart()
    按 (pitch, program, is_drum) 分组
    聚类合并 onset 差 < 10ms 的音符
    保留持续时间最长的
    ↓
MIDI 后处理 (按质量模式):
    best:     仅移除 <10ms 噪音音符
    balanced: 轻度去重 + 力度平滑 + 复音限制
    fast:     无后处理
```

#### Benchmark (Slakh2100 数据集)

| 指标 | YPTF.MoE+Multi (PS) | MT3 (Google Baseline) |
|------|---------------------|----------------------|
| Multi F1 | **0.7484** | 0.62 |
| Frame F1 | 0.8487 | — |
| Onset F1 | 0.8419 | — |
| Offset F1 | 0.6961 | — |
| Drum Onset F1 | 0.9113 | — |

各乐器 Onset F1: Bass 0.93 / Piano 0.88 / Guitar 0.82 / Synth Lead 0.82 / Brass 0.73 / Strings 0.73

#### 可用模型变体

| 模型 | MoE | Pitch Shift | 大小 | 说明 |
|------|-----|-------------|------|------|
| YPTF.MoE+Multi (PS) | 8 专家 | 有 | 724 MB | **默认，最高性能** |
| YPTF.MoE+Multi (noPS) | 8 专家 | 无 | 724 MB | 无音高偏移增强 |
| YPTF+Multi (PS) | 无 | 有 | 2.0 GB | 标准 Perceiver |
| YPTF+Multi (noPS) | 无 | 无 | 2.0 GB | 标准 Perceiver，无增强 |

---

### 编码器架构对比：PerceiverTF vs MusicFM

2025 年 AI4Musician AMT Challenge 的冠军方案使用了 MusicFM 编码器替代 PerceiverTF。以下是两种编码器的深入对比。

#### MusicFM 编码器

MusicFM 是由 Minz Won 等人提出的音乐基础模型（ICASSP 2024, [arXiv:2311.03318](https://arxiv.org/abs/2311.03318)），在 Million Song Dataset 的 16 万小时无标注音乐上通过自监督学习预训练。

```
音频 (24kHz)
    ↓
128-band Mel Spectrogram
    ↓
2 层残差 Conv2dSubsampling (降采样到 25Hz 帧率)
    ↓
12 层 Wav2Vec2 Conformer
    ├── 每层: Multi-Head Self-Attention + Convolution Module
    ├── RoPE 位置编码
    ├── 输出维度: 1024
    └── 帧率: 25 Hz
    ↓
密集音乐嵌入 (B, T, 1024) @ 25Hz
```

训练方式：BEST-RQ 风格的掩码 token 建模 + 随机投影量化器。模型学会从被掩码的音频片段中预测原始特征，从而捕获丰富的音乐结构信息。

#### 竞赛冠军完整架构 (amt-os/ai4m-miros)

```
音频 (16kHz → 重采样 24kHz)
    ↓
MusicFM 25Hz 编码器 (冻结, 12层 Conformer, 1024-dim)
    ↓
MusicFMAdapter (创新组件)
    ├── 13 个可学习视图嵌入 (512-dim)
    ├── 4 轮迭代循环注意力:
    │   concat(encoder_output, recurrent_state)
    │   → Linear → 3层 Self-Attention (RoPE, QK-norm, SiLU)
    └── 输出: (B, 13, T, 512) — 13 个乐器视图
    ↓
TemporalUpsample (2× ConvTranspose1d, 25Hz → 100Hz)
    ↓
Multi-Dec 解码器 (Llama 风格, 非 T5)
    ├── 13 通道, 8 层, 8 头
    ├── RoPE + RMSNorm + SiLU
    ├── 支持 torch.compile 加速
    └── 最大 token 长度: 1024
    ↓
LM Head → Token 预测
```

| 维度 | PerceiverTF + MoE (当前) | MusicFM + Multi-Dec (冠军) |
|------|------------------------|--------------------------|
| 编码器类型 | 从零端到端训练 | 自监督预训练 (16万小时) |
| 编码器架构 | PerceiverTF (交叉注意力 + MoE) | Conformer (自注意力 + 卷积) |
| 编码器参数 | 较小 (MoE 稀疏激活) | ~300M (全量激活) |
| 输出维度 | ~512 | 1024 |
| 解码器 | Multi-T5 (T5-small 架构) | Multi-Dec (Llama 风格) |
| 输入窗口 | 32,767 帧 (~2.05s) | 87,381 帧 (~5.46s) |
| Token 长度 | 256 / 通道 | 1024 / 通道 |
| 帧率 | ~53 Hz | 25 Hz → 100 Hz (上采样) |
| 推理速度 | 快 (MoE 稀疏 + 短窗口) | 慢 (全量 + 长窗口) |
| 显存需求 | 低 (~2GB) | 高 (~6GB+) |
| 音乐理解深度 | 仅转写任务 | 广泛音乐结构 |
| 密集复音处理 | 256 tokens 可能截断 | 1024 tokens 充裕 |

#### 核心差异分析

**PerceiverTF + MoE（专才模型）：**
- 优势：效率极高，MoE 稀疏激活使得 8 专家的参数量只需 2 专家的计算量；短窗口推理快；显存友好
- 劣势：知识面窄，只从标注数据学习；2.05s 窗口在长音符和慢速乐曲中可能产生边界错误；256 tokens 在密集复音段落可能截断

**MusicFM（通才模型）：**
- 优势：16 万小时预训练带来深层音乐理解（和弦结构、音色特征、节奏模式）；5.46s 窗口大幅减少边界错误；1024 tokens 处理任意密度
- 劣势：推理慢（全量 Conformer）；显存需求高；需要额外的 Adapter 层桥接编码器和解码器

**类比：** PerceiverTF 像一个只学过速记的专业速记员——效率高但知识面窄；MusicFM 像一个音乐学院毕业生——音乐素养深厚但需要再学速记才能工作。

---

### AI4Musician 2025 AMT Challenge

| 项目 | 详情 |
|------|------|
| 名称 | 2025 Automatic Music Transcription (AMT) Challenge |
| 组织方 | AI4Musicians (Purdue University 关联) |
| 时间 | 2025 年 4 月 |
| 论文 | "Advancing Multi-Instrument Music Transcription: Results from the 2025 AMT Challenge", NeurIPS 2025 Datasets & Benchmarks Track |
| 参赛规模 | 8 支队伍提交有效方案，2 支超过 baseline (MT3) |
| 冠军 | amt-os (University of Osnabrück) |
| 冠军仓库 | [amt-os/ai4m-miros](https://github.com/amt-os/ai4m-miros) |

#### 冠军模型集成可行性：当前不可行

| 阻碍因素 | 状态 | 详情 |
|----------|------|------|
| 模型权重 | ❌ 不存在 | 仓库 checkpoint 目录为空，LFS 文件未上传，无外部下载链接 |
| 许可证 | ❌ 无 | 仓库无 LICENSE 文件，法律上默认保留所有权利 |
| 文档 | ❌ 无 | 无 README，无使用说明 |
| 性能对比 | ❓ 未知 | 比赛 baseline 是 MT3 而非 YourMT3+，无直接对比数据 |
| 社区 | ⚠️ 极低 | 1 star, 0 fork |
| MusicFM 预训练权重 | ✅ 可用 | [HuggingFace](https://huggingface.co/minzwon/MusicFM), MIT 许可, ~1.3GB |

MusicFM 编码器本身可用（MIT 许可），但仅有编码器无法做转写——还需要 Adapter + 解码器 + 在 AMT 数据集上 fine-tune，这是研究项目级别的工作量。

---

### 前沿模型与研究方向

| 模型 / 方向 | 来源 | 类型 | 状态 | 说明 |
|-------------|------|------|------|------|
| [Aria-AMT5](https://github.com/EleutherAI/aria-amt) | EleutherAI | 钢琴 | 开源 | Whisper 架构钢琴转写，2025 新 SOTA |
| Streaming AMT | arXiv 2025 | 多乐器 | 论文 | Conv 编码器 + AR 解码器，实时流式，接近离线 SOTA |
| 2025 AMT Challenge | NeurIPS 2025 | 多乐器 | 论文 | 8 支队伍，2 支超过 MT3 baseline |
| CVC Framework | ISMIR 2025 | 评估 | 论文 | 跨版本一致性，无标注评估管弦乐场景 |

> **趋势总结**：截至 2025 年，多乐器 AMT 仍由 MT3/YourMT3+ Transformer 架构主导。预训练音乐基础模型（MusicFM）是最有前景的编码器升级方向，但完整的开源 fine-tuned checkpoint 尚未出现。钢琴转写最成熟（Aria-AMT5），多乐器和吉他谱转写仍是活跃研究前沿。

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

**Q: PyQt6 提示 "could not load Qt platform plugin"**
```bash
sudo apt-get install libxcb-xinerama0 libxkbcommon-x11-0 libxcb-cursor0
```

**Q: CUDA 不可用**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Q: YourMT3+ 不可用**
```bash
python download_sota_models.py
```

## 贡献

欢迎提交 Pull Request。

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'feat: add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

## 许可证

MIT License - 详见 [LICENSE](../LICENSE) 文件。

## 致谢

- [YourMT3+](https://huggingface.co/spaces/mimbres/YourMT3) - SOTA 多乐器转写
- [mido](https://github.com/mido/mido) - MIDI 文件处理
- [librosa](https://librosa.org/) - 音频分析
