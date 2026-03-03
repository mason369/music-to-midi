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

- **多乐器转写**：使用 YourMT3+ MoE 高性能模型直接识别混音中的多种乐器
- **人声分离模式**：BS-RoFormer 分离人声与伴奏，默认输出两个独立 MIDI（可选额外输出 1 个合并 MIDI，默认 checkpoint：`model_bs_roformer_ep_368_sdr_12.9628.ckpt`）
- **六声部分离模式**：BS-RoFormer SW 分离 `bass/drums/guitar/piano/vocals/other`，支持“仅转写选中 stem”，输出 N 个 stem MIDI + 1 个合并 MIDI
- **主唱/和声（实验近似）**：六声部分离模式可将 `vocals` 再拆为主唱 + 和声代理 stem（公开 male/female 模型）
- **钢琴专用模式**：Aria-AMT 钢琴专用转写，输入钢琴曲直接输出钢琴 MIDI
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
python download_vocal_model.py
python download_multistem_model.py
python download_aria_amt_model.py

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
2. **选择处理模式**：SMART / VOCAL_SPLIT / SIX_STEM_SPLIT / PIANO_ARIA_AMT
3. **可选（VOCAL_SPLIT）**：勾选“输出 1 个人声+伴奏合并 MIDI”
4. **可选（SIX_STEM_SPLIT）**：勾选“仅转写选中的 stem”并选择目标 stem
5. **可选（SIX_STEM_SPLIT）**：勾选“将 vocals 进一步分离为主唱 + 和声（实验近似）”
6. **开始处理**：点击"开始"按钮
7. **获取结果**：在输出目录中找到 MIDI 与分离音频文件

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
│   ├── aria_amt_transcriber.py      # Aria-AMT 钢琴转写封装
│   ├── beat_detector.py             # 节拍/BPM 检测
│   ├── midi_generator.py            # MIDI 生成与后处理
│   ├── vocal_separator.py           # BS-RoFormer 人声分离
│   ├── multi_stem_separator.py      # BS-RoFormer SW 六声部分离
│   └── vocal_harmony_separator.py   # 主唱/和声（实验近似）分离
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

支持四种处理模式：

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
MidiGenerator × 2 (伴奏 MIDI + 人声 MIDI)       [85-92%]
    ↓
可选合并：生成 1 个 merged MIDI                  [92-95%]
    ↓
默认两个 MIDI（或 +1 合并 MIDI）                 [95-100%]
```

#### 模式三：SIX_STEM_SPLIT（六声部分离）

```
音频文件
    ↓
BeatDetector.detect() → BPM                    [0-5%]
    ↓
SixStemSeparator.separate()                     [5-30%]
    → bass.wav / drums.wav / guitar.wav / piano.wav / vocals.wav / other.wav
    ↓
可选开关：vocals 继续拆分为主唱/和声（实验近似）
    ↓
可选开关：仅转写选中的 stem
    ↓
YourMT3Transcriber.transcribe_precise() × N     [30-75%]
    ↓
MidiGenerator × N                               [75-93%]
    ↓
合并所选 stem MIDI 生成 1 个 merged MIDI         [93-100%]
```

#### 模式四：PIANO_ARIA_AMT（钢琴专用）

```
音频文件（钢琴为主）
    ↓
BeatDetector.detect() → BPM                    [0-10%]
    ↓
AriaAmtTranscriber.transcribe()                [10-85%]
    ↓
钢琴 MIDI 单文件输出                             [85-100%]
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

### 人声分离模型：BS-RoFormer

本项目使用 **BS-RoFormer**（Band-Split Rotary Transformer）进行人声/伴奏分离，通过 [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 库封装调用。

| 项目 | 详情 |
|------|------|
| 全称 | Band-Split RoFormer |
| 论文 | [Music Source Separation with Band-Split RoFormer](https://arxiv.org/abs/2309.02612) (ISMIR 2023 Workshop) |
| Checkpoint | `model_bs_roformer_ep_368_sdr_12.9628.ckpt` (epoch 368) |
| 训练者 | [ZFTurbo](https://github.com/ZFTurbo) / [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) |
| 许可证 | MIT |
| 指标说明 | checkpoint 文件名包含 `sdr_12.9628`（训练过程分数标签，不是统一评测口径） |
| 公开对比（人声 SDR） | Multisong: BS-RoFormer 10.87 / MelBand-RoFormer(Kim) 10.98 / HTDemucs_ft 8.38 |
| 模型大小 | ~500 MB |
| 首次使用 | 自动从 HuggingFace 下载到 `~/.music-to-midi/models/audio-separator/` |
| 输出选项 | 默认 2 个 MIDI（伴奏 + 人声）；可勾选额外输出 1 个合并 MIDI |

#### 架构概览

BS-RoFormer 的核心思想是将频谱按频段（band）拆分后独立建模，再用 Rotary Position Embedding (RoPE) 增强时序建模能力：

```
音频波形 (44.1kHz stereo)
    ↓
STFT → 复数频谱 (F × T)
    ↓
Band-Split 模块
    将频率轴按预定义子带划分为 K 个频段
    每个频段独立通过 MLP 映射到 D 维嵌入
    → (K, T, D)
    ↓
N 层 Band-Split RoFormer Block
    ├── Band-level Self-Attention (频段间交互)
    │   K 个频段作为 token，捕获跨频段谐波关系
    │   使用 RoPE 编码时间位置
    ├── Temporal Self-Attention (时间维度建模)
    │   T 个时间帧作为 token，捕获时序依赖
    │   使用 RoPE 编码时间位置
    └── Feed-Forward Network
    ↓
Band-Merge 模块
    将 K 个频段嵌入映射回频率维度
    → 复数掩码 (F × T)
    ↓
掩码 × 原始频谱 → iSTFT
    ↓
分离后的波形 (vocals / instrumental)
```

#### 为什么选择 BS-RoFormer（替代 Demucs）

项目最初使用 Meta 的 Demucs (htdemucs) 进行人声分离，后在 `d6309de` 提交中整体替换为 BS-RoFormer，原因：

| 维度 | Demucs (htdemucs) | BS-RoFormer |
|------|-------------------|-------------|
| 公开对比（Multisong 人声 SDR） | 8.38 (HTDemucs_ft) | 10.87 (BS-RoFormer_ep317，公开参考口径) |
| 架构 | 混合 U-Net + Transformer | 纯 Transformer (频段拆分) |
| 集成方式 | 需手动管理模型加载 | audio-separator 三步调用 |
| 依赖链 | 较重（demucs 自带依赖） | 轻量（audio-separator + onnxruntime） |
| 打包兼容 | PyInstaller 兼容性差 | 良好 |

分离质量直接决定下游 YourMT3+ 转写准确性。当前文档统一采用“来源 + 数据集 + 指标”标注，避免跨数据集数字误读。

#### 处理流程

`src/core/vocal_separator.py` (~200 行)：

```
音频文件
    ↓
Separator(output_dir, model_file_dir, output_format="WAV")
    ↓
separator.load_model("model_bs_roformer_ep_368_sdr_12.9628.ckpt")
    ↓
separator.separate(audio_path)
    ↓
2 个 stem:
    ├── {stem}_(Vocals).wav   → 重命名为 {stem}_vocals.wav
    └── {stem}_(Instrumental).wav → 重命名为 {stem}_accompaniment.wav
    ↓
输出: {"vocals": vocals_path, "no_vocals": accompaniment_path}
```

关键设计：
- **GPU 自动检测**：audio-separator 的 Separator 类自行检测 GPU 设备，不接受外部 device 参数
- **阻塞调用**：`separator.separate()` 是阻塞调用，分离过程中无法响应取消，仅在调用前后检查取消状态
- **后台进度线程**：每 3 秒基于已用时间估算进度（audio-separator 不提供细粒度回调）
- **速度估算**：GPU 约 3x 实时，CPU 约 0.15x 实时

#### 前沿人声分离模型（2026-03-01 核实）

| 模型/方向 | 来源 | 类型 | 状态 | 说明 |
|-----------|------|------|------|------|
| BS-RoFormer ep368（当前） | 当前项目默认 checkpoint | 本地直替（audio-separator） | ✅ 使用中 | 默认模型：`model_bs_roformer_ep_368_sdr_12.9628.ckpt`；Multisong 公开常见参考值多来自 ep317 口径 |
| SCNet XL IHF | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) | 本地直替（audio-separator） | ✅ 可替换 | Multisong 人声 SDR 11.11（`model_scnet_xl_2ep_..._musdb18hq.ckpt`），本地开源可落地中更强候选 |
| MelBand-RoFormer (Kim) | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) | 本地直替（audio-separator） | ✅ 可替换 | Multisong 人声 SDR 10.98（`Kim_Vocal_2.onnx`） |
| Ensemble (vocals,instrum) | [MVSEP Algorithms](https://mvsep.com/algorithms) | 榜单前沿（服务/集成） | 🌐 可用（非本地直替） | MVSEP vocals SDR 11.93（ver 2025.06），当前公开榜单最高之一 |
| BS Roformer (vocals,instrum) | [MVSEP Algorithms](https://mvsep.com/algorithms) | 榜单前沿（服务/单模型） | 🌐 可用（非本地直替） | MVSEP vocals SDR 11.89（ver 2025.07） |
| BS Roformer SW (vocals,instrum, 6 stem) | [MVSEP Quality Checker](https://mvsep.ru/quality_checker/synth_mutlitrack/create_table) | 榜单前沿（服务/多stem） | 🌐 可用（非本地直替） | MVSEP vocals SDR 11.30；偏多 stem 场景，不等同本项目 2-stem 本地替换路径 |
| Mel-RoFormer (ISMIR 2024) | [arXiv:2409.04702](https://arxiv.org/abs/2409.04702) / [ar5iv 表2](https://ar5iv.org/html/2409.04702v1) | 论文阶段（研究模型） | 📄 论文已发表 | MUSDB18-HQ（表2, 场景ⓑ, 含额外数据）Vocals SDR 13.29；同文中 BS-RoFormer 为 12.82 |
| Mamba2 Meets Silence (v2, 2025) | [arXiv:2508.14556](https://arxiv.org/abs/2508.14556) | 论文阶段（研究模型） | 📄 论文 | 摘要报告 cSDR 11.03 dB（作者称 best reported），强调稀疏人声段鲁棒性 |
| Windowed Sink Attention (2025) | [arXiv:2510.25745](https://arxiv.org/abs/2510.25745) | 论文阶段（效率优化方向） | 📄 论文 + 开源代码 | 在微调设定下恢复原模型约 92% SDR，同时 FLOPs 降低约 44.5x（偏效率收益） |

> **结论（按口径）**：  
> - 若只看 **本地开源可直接替换**：SCNet XL IHF（11.11）在公开 Multisong 口径下通常高于 ep317 参考值；与当前默认 ep368 需按同口径复测。  
> - 若看 **MVSEP 榜单**：Ensemble（11.93）和 BS Roformer（11.89）更高。  
> - 若看 **论文特定协议（MUSDB18-HQ 表2 场景ⓑ）**：Mel-RoFormer 报告 13.29。  
> **口径提醒**：不同榜单/数据集/评测协议（Multisong、MUSDB、MVSEP、cSDR/uSDR）不可直接横比。

#### 六声部与细粒度乐器分离：前沿模型（2026-03-03 核实）

> 注：此处聚焦 **6-stem / 乐器细分**，并严格区分“本地可下载”与“在线服务/API”。

| 模型/方向 | 来源 | 类型 | 状态 | 说明 |
|-----------|------|------|------|------|
| BS Roformer SW（当前六声部主后端） | [MVSEP Algorithms #77](https://mvsep.com/algorithms/77) / [HF: jarredou/BS-ROFO-SW-Fixed](https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/blob/main/BS-Rofo-SW-Fixed.ckpt) / [openmirlab config](https://raw.githubusercontent.com/openmirlab/bs-roformer-infer/main/src/bs_roformer/configs/config_bs_roformer_sw.yaml) | 本地可下载（6-stem） | ✅ 使用中 | MVSEP 页面给出：**vocals 11.30 / instrum 17.50 / bass 14.62 / drums 14.11 / guitar 9.05 / piano 7.83 / other 8.71**；本项目已接入为 `SIX_STEM_SPLIT` 主后端 |
| HTDemucs4 (6 stems) | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) | 本地可下载（6-stem 基线） | ✅ 可替换（回退） | Multisong（同表）示例：bass 11.22 / drums 10.22 / vocals 8.05 |
| DrumSep mdx23c（jarredou，5 stems） | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) / [jarredou models](https://github.com/jarredou/models/releases) | 本地可下载（鼓细分） | ✅ 可用 | 鼓细分公开指标：kick 16.66 / snare 11.53 / toms 12.33 / hh 4.04 / cymbals 6.36 |
| DrumSep mdx23c（aufr33+jarredou，6 stems） | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) / [jarredou models](https://github.com/jarredou/models/releases) | 本地可下载（鼓细分） | ✅ 可用（实验） | 鼓细分公开指标：kick 14.54 / snare 9.79 / toms 10.63 / hh 3.19 / ride+crash 6.08 |
| MVSep Bass/Drums/Piano/Guitar 专项路线 | [Bass #37](https://mvsep.com/algorithms/37) / [Drums #43](https://mvsep.com/algorithms/43) / [Piano #14](https://mvsep.com/algorithms/14) / [Guitar #17](https://mvsep.com/algorithms/17) | 在线服务/榜单（专项） | 🌐 服务可用（非本地直替） | 页面显示 BS Roformer SW 专项单模型值较高（如 Bass 14.62、Drums 14.11、Piano 7.83、Guitar 9.05），更高值通常来自融合 |
| Ensemble All-In（vocals,bass,drums,piano,guitar,lead/back,other） | [MVSEP Algorithms #47](https://mvsep.com/algorithms/47) | 在线融合 | 🌐 服务可用（非开源） | 覆盖面最广，但当前未见完整公开 checkpoint 对应 |
| 主唱/和声（lead/back）前沿路线 | [MVSEP Karaoke #76](https://mvsep.com/algorithms/76) / [QC #8211](https://mvsep.com/quality_checker/entry/8211) / [QC #7845](https://mvsep.com/quality_checker/entry/7845) | 在线+部分开源 | 🔬 混合（开源 + 服务） | 公开可下载常见为 karaoke 或 male/female 路线；更高 lead/back 组合多在服务侧或融合流程中 |

> **落地建议（本地开源优先）**：  
> - 6-stem 主链路：`BS Roformer SW`。  
> - 细分鼓：在 `drums` 后接 `DrumSep mdx23c`。  
> - 主唱/和声：在 `vocals` 后接 karaoke/chorus 路线，并注明“近似 lead/back、非统一 benchmark”。  
> - API/服务模型需单独标注“非公开权重、不可本地直替”。

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
| `TrackPanel` | `widgets/track_panel.py` (~120 行) | 处理模式选择（智能/人声分离/六声部），支持人声合并 MIDI 开关与六声部目标选择 |
| `ProgressWidget` | `widgets/progress_widget.py` (~280 行) | 阶段流程指示器 + 进度条，根据模式动态重建阶段 |
| `StageIndicator` | `widgets/progress_widget.py` | 单个阶段的状态指示（pending/current/done），带颜色和图标变化 |

---

### 数据模型

`src/models/data_models.py` (~650 行) 定义所有核心数据结构。

#### 枚举类型

```python
ProcessingMode        # SMART | VOCAL_SPLIT | SIX_STEM_SPLIT | PIANO(已弃用→SMART)
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
    stem_midi_paths: Optional[Dict[str, str]]   # stem MIDI 路径 (仅 SIX_STEM_SPLIT)
    merged_midi_path: Optional[str]             # 合并 MIDI 路径 (SIX_STEM_SPLIT 或可选 VOCAL_SPLIT)

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
    #   可选值: smart / vocal_split / six_stem_split / piano_aria_amt
    vocal_split_merge_midi: bool        # VOCAL_SPLIT 是否额外输出合并 MIDI
    six_stem_targets: List[str]         # SIX_STEM_SPLIT 指定转写 stem（空=全部）
    six_stem_split_vocal_harmony: bool  # SIX_STEM_SPLIT 下 vocals 是否额外拆主唱/和声（实验近似）
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
| 模型权重 | ⚠️ 待核实 | 仓库包含 `checkpoints/` 路径，但完整性、可用性和可复现实测仍需手动验证 |
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
| YPTF.MoE+Multi (PS)（当前） | [YourMT3+ 论文](https://arxiv.org/abs/2407.04822) / [KAIST HF](https://huggingface.co/spaces/mimbres/YourMT3) | 多乐器 | ✅ 使用中 | Slakh2100: Multi F1 0.7484（本 README 同口径基准） |
| AI4Musician 2025 冠军方案（ai4m-miros） | [AI4Musicians 2025](https://ai4musicians.org/transcription/2025transcription.html) / [amt-os/ai4m-miros](https://github.com/amt-os/ai4m-miros) | 多乐器 | 🔬 论文/仓库阶段 | 冠军方案已公开仓库；与 YourMT3+ 的公开同口径对比仍有限 |
| [Aria-AMT](https://github.com/EleutherAI/aria-amt) | EleutherAI | 钢琴 | ✅ 开源可用 | seq-to-seq 钢琴 AMT 实现，提供可下载权重 |
| MusicFM 编码器 + 转写解码器 | [MusicFM 论文](https://arxiv.org/abs/2311.03318) / [MusicFM HF](https://huggingface.co/minzwon/MusicFM) | 多乐器（研究方向） | 📄 论文/组件可用 | 编码器可用，但完整 AMT 仍需 Adapter+Decoder+再训练 |
| CountEM（弱监督 AMT） | [arXiv:2511.14250](https://arxiv.org/abs/2511.14250) | AMT 训练方法 | 📄 论文阶段 | 以音符直方图监督替代对齐监督，降低标注依赖 |

> **趋势总结**：截至 2026 年初，多乐器 AMT 主要沿两条路线发展：MT3/YourMT3+ 这类任务特化模型，以及预训练编码器增强路线。钢琴 AMT 的开源成熟度高于多乐器冠军方案的工程可复现度。

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

- [YourMT3+](https://huggingface.co/spaces/mimbres/YourMT3) - 多乐器转写核心模型
- [BS-RoFormer](https://arxiv.org/abs/2309.02612) - 人声分离模型架构论文
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) - 音源分离推理框架
- [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) - BS-RoFormer 预训练权重
- [mido](https://github.com/mido/mido) - MIDI 文件处理
- [librosa](https://librosa.org/) - 音频分析
