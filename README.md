# 音乐转 MIDI 转换器

<p align="center">
  中文 | <a href="./docs/README.md">English</a>
</p>

一个基于 AI 的音频转 MIDI 工具，提供 PyQt6 桌面版、Gradio Web 版和 Google Colab 运行入口。当前版本同步六种处理模式：完整混音多乐器转写、人声/伴奏分离后分别转写、六声部分离后分别转写，以及 Transkun / Aria-AMT / ByteDance Pedal 三条钢琴专用转写流程。

## 截图

| Windows | Linux |
|---------|-------|
| ![Windows 演示](resources/icons/Windows演示.png) | ![Linux 演示](resources/icons/Linux演示.png) |

## 当前能力

- **完整混音转写**：`SMART` 模式直接读取整首音频，用多乐器后端生成 MIDI。
- **人声/伴奏分离转写**：`VOCAL_SPLIT` 模式先分离人声与伴奏，再分别生成 MIDI；可选额外输出一个合并 MIDI。
- **六声部分离转写**：`SIX_STEM_SPLIT` 模式先分离 `bass / drums / guitar / piano / vocals / other` 六个 stem，再输出各 stem MIDI 和合并 MIDI。
- **钢琴专用转写**：`PIANO_TRANSKUN`、`PIANO_ARIA_AMT` 与 `PIANO_BYTEDANCE_PEDAL` 面向纯钢琴音频，分别调用 Transkun、Aria-AMT 和 ByteDance 带踏板模型。
- **默认后端语义**：配置默认偏好为 `Aria-AMT` 钢琴后端；`SMART`、`VOCAL_SPLIT` 和非钢琴 stem 仍由 YourMT3+ 或 MIROS 这类多乐器后端承担。
- **MIROS 可选后端**：桌面版可切换到本地 `ai4m-miros` 仓库作为实验性多乐器后端。
- **MIDI 输出布局**：YourMT3+ 支持“按 GM 乐器分轨”和“非鼓合并单轨、鼓独立”两种布局。
- **节拍与后处理**：BPM 检测成功后生成带速度信息的 MIDI；检测失败会停止，不写入默认 tempo。支持量化、去重、力度平滑、复音限制等后处理。
- **多格式输入**：支持 `MP3`、`WAV`、`FLAC`、`OGG`、`M4A`。非 WAV 必须通过 FFmpeg 转为 44.1 kHz PCM WAV；FFmpeg 失败会直接停止并显示 stderr 根因。
- **多平台入口**：桌面版、Space、Colab 均同步暴露六种处理模式。

## 不同入口的功能范围

| 入口 | 处理模式 | 后端选择 | 适合场景 |
|------|----------|----------|----------|
| PyQt6 桌面版 | `SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT`、`PIANO_TRANSKUN`、`PIANO_ARIA_AMT`、`PIANO_BYTEDANCE_PEDAL` | `Aria-AMT`、`ByteDance Pedal`、`YourMT3+`、`MIROS` | 本地长期使用、GPU 推理、批量输出文件、钢琴专用转写 |
| Gradio Space | 同桌面六种模式 | 默认 `Aria-AMT` / `ByteDance Pedal` 钢琴后端 + `YourMT3+` 多乐器后端 | 浏览器中快速试用或部署 |
| Google Colab | 同桌面六种模式 | 默认 `Aria-AMT` / `ByteDance Pedal` 钢琴后端 + `YourMT3+` 多乐器后端 | 临时使用 Colab GPU |

## 入口与依赖同步状态

本仓库把“处理工作流”和“YourMT3+ 官方 checkpoint 模式”分开维护：

- `SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT`、`PIANO_TRANSKUN`、`PIANO_ARIA_AMT`、`PIANO_BYTEDANCE_PEDAL` 是本项目的六种处理工作流。
- `YMT3+`、`YPTF+Single (noPS)`、`YPTF+Multi (PS)`、`YPTF.MoE+Multi (noPS)`、`YPTF.MoE+Multi (PS)` 是官方 YourMT3 demo 暴露的五种 checkpoint / 架构模式。
- 桌面版、Gradio Space 和 Colab 都暴露同一组六种处理工作流；其中需要多乐器转写的工作流会再选择 YourMT3+ 官方 checkpoint 模式。

当前同步覆盖如下：

| 位置 | 已同步内容 | 说明 |
|------|------------|------|
| `download_sota_models.py` | 下载全部五种官方 YourMT3+ checkpoint 模式 | 使用 `OFFICIAL_YOURMT3_MODEL_KEYS`，下载后会校验 checkpoint 可解析。 |
| `run.ps1` / `run.sh` | 启动前检查全部官方 YourMT3+ 模式、Aria-AMT、ByteDance Pedal、MIROS 与分离器可用性 | 缺少必需模型时显式报错，不把缺失资源当作可运行状态。 |
| `install.ps1` / `install.sh` | 安装 PyTorch 2.4、NumPy 1.26、audio-separator 0.41.1 运行依赖，并下载必需模型 | `audio-separator` 使用 `--no-deps`，避免其 NumPy 2 解析要求破坏当前 PyTorch/桌面栈。 |
| `.github/workflows/build.yml` | CI 构建和测试安装同一组 pinned runtime 依赖 | 测试失败不会用 `|| true` 掩盖。 |
| `.github/workflows/release.yml` | 发布包下载并打包全部官方 YourMT3+ 模式、BS-RoFormer、Aria-AMT、ByteDance Pedal 和 MIROS | GPU 版使用 PyTorch 2.4 + CUDA 12.1 wheel。 |
| `colab_notebook.ipynb` | 保留 Colab 预装 Torch，安装 pinned Web/runtime 依赖，下载全部官方 YourMT3+ 模式 | Colab UI 提供 YourMT3 官方模型选择器，并在选择后显示说明。 |

## 处理模式

| 模式 | 内部流程 | 主要输出 | 说明 |
|------|----------|----------|------|
| `SMART` | 音频 -> 多乐器后端 -> MIDI 生成 | `<歌曲名>.mid` | 不做音源分离，适合大多数混音歌曲、纯音乐和多乐器片段。 |
| `VOCAL_SPLIT` | 音频 -> 人声/伴奏分离 -> 伴奏转写 -> 人声转写 -> MIDI 生成 | `<歌曲名>_accompaniment.mid`、`<歌曲名>_vocal.mid`，可选 `<歌曲名>_vocal_accompaniment_merged.mid` | 分离后会把人声 MIDI 尽量收敛到旋律轨，减少伴奏乐器幻觉。 |
| `SIX_STEM_SPLIT` | 音频 -> 六声部分离 -> stem 转写 -> stem MIDI 合并 | `<歌曲名>_<stem>.mid`、`<歌曲名>_all_stems_merged.mid` 或 `<歌曲名>_selected_stems_merged.mid` | 可只转写选中的 stem；piano stem 在偏好 Aria-AMT 且权重可用时会优先走 Aria-AMT。 |
| `PIANO_TRANSKUN` | 音频 -> Transkun 钢琴模型 -> MIDI | `<歌曲名>_piano_transkun.mid` | 适合纯钢琴音频；使用 Transkun checkpoint 直接推理。 |
| `PIANO_ARIA_AMT` | 音频 -> Aria-AMT 钢琴模型 -> MIDI | `<歌曲名>_piano_aria_amt.mid` | 适合纯钢琴音频；需要先准备 Aria-AMT checkpoint。 |
| `PIANO_BYTEDANCE_PEDAL` | 音频 -> ByteDance 带踏板钢琴模型 -> MIDI | `<歌曲名>_piano_bytedance_pedal.mid` | 适合纯钢琴音频；会保留延音踏板 CC64，需要先准备 ByteDance Piano checkpoint。 |

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
song_bass.mid
song_drums.mid
song_guitar.mid
song_piano.mid
song_vocals.mid
song_other.mid
song_all_stems_merged.mid
song_piano_transkun.mid
song_piano_aria_amt.mid
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

### Transkun

Transkun 是钢琴专用转写后端，适合纯钢琴或以钢琴为主的音频。项目通过 `src/core/transkun_transcriber.py` 调用 `transkun` PyPI 包随附的预训练资源：

```bash
pip install "transkun>=2.0.1"
```

可用性检查会确认 `transkun.transcribe`、`pretrained/2.0.pt` 和 `pretrained/2.0.conf` 是否存在。缺失时请重新安装：

```bash
python -m pip install --force-reinstall transkun
```

### Aria-AMT

Aria-AMT 是另一条钢琴专用后端。上游官方 README 使用 `aria-amt transcribe` CLI；本项目包装器 `src/core/aria_amt_transcriber.py` 当前通过 Python 模块入口调用 `amt.run transcribe`。默认 checkpoint 为：

```text
piano-medium-double-1.0.safetensors
```

安装依赖：

```bash
python -m pip install git+https://github.com/EleutherAI/aria-amt.git
```

下载模型：

```bash
python download_aria_amt_model.py
```

默认搜索模型位置包括：

```text
~/.cache/music_ai_models/aria_amt
runtime/models/aria_amt
models/aria_amt
```

### ByteDance Pedal

ByteDance Pedal 是钢琴专用的带踏板转写后端，适合独奏钢琴或清晰的钢琴 stem。它来自 ByteDance 的 High-Resolution Piano Transcription with Pedals 系统，本项目通过 `piano-transcription-inference` 包装，并保留上游 MIDI 中的延音踏板 `CC64`。

安装依赖：

```bash
python -m pip install "piano-transcription-inference>=0.0.6,<0.1" "torchlibrosa>=0.1.0,<0.2" "matplotlib>=3.7.0,<4"
```

下载模型：

```bash
python download_bytedance_piano_model.py
```

默认搜索模型位置包括：

```text
~/.cache/music_ai_models/bytedance_piano
runtime/models/bytedance_piano
models/bytedance_piano
```

## 钢琴后端选择建议

三个钢琴后端都是“钢琴专精模型”，不参与完整混音的多乐器识别。选择时可以按目标来分：

| 目标 | 推荐模式 | 说明 |
|------|----------|------|
| 追求纯钢琴音符转写，优先使用当前默认钢琴路线 | `PIANO_ARIA_AMT` | 现代钢琴 AMT 后端，适合作为常规纯钢琴候选。 |
| 需要一个成熟、轻量、随包权重的钢琴后端 | `PIANO_TRANSKUN` | 依赖和 checkpoint 边界清楚，适合快速本地验证。 |
| 需要踏板 CC64，尤其是古典、抒情、连奏明显的钢琴音频 | `PIANO_BYTEDANCE_PEDAL` | 保留 sustain pedal 控制事件；上游主仓库已归档，建议在目标环境做一次实际音频验证。 |

这三者的结果不能和 `YourMT3+` / `MIROS` 的多乐器结果直接横比：钢琴后端只解决 88 键钢琴及其演奏细节，多乐器后端负责完整混音中的乐器识别和多轨输出。

## 模型与公开对比

本节恢复自历史 README 中的模型对比内容，并按当前版本的实际能力重新标注：当前已发布入口同步开放 `SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT`、`PIANO_TRANSKUN`、`PIANO_ARIA_AMT` 与 `PIANO_BYTEDANCE_PEDAL` 六种模式。下列表格把“公开 benchmark”和“项目内入口状态”分开写，避免把研究指标误写成产品能力。

#### 当前已集成后端总览

| 后端/模型 | 类型 | 项目入口 | 公开质量口径 | 选择说明 |
|-----------|------|----------|--------------|----------|
| YourMT3+ | 多乐器 AMT | `SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT` 的多乐器/stem 转写 | Slakh2100 `Multi (Onset-Offset) F1 = 74.84` | 默认完整混音主线，适合多乐器和 GM 分轨输出。 |
| MIROS | 多乐器 AMT | 桌面版可选多乐器后端 | 上游仓库标注为 2025 AI4Musician winning model | 实验性本地后端，适合与 YourMT3+ 做同任务 A/B。 |
| Transkun | 钢琴专精 | `PIANO_TRANSKUN` | Transkun V2 / pip checkpoint 在 MAESTRO V3 上有公开 F1 | 成熟轻量，随包权重边界清楚，但当前 pip checkpoint 不保留踏板延音扩展。 |
| Aria-AMT | 钢琴专精 | `PIANO_ARIA_AMT` | 公开 checkpoint；README 不写入未发布的同口径统一 F1 | 当前默认钢琴候选，适合常规纯钢琴音符转写。 |
| ByteDance Pedal | 钢琴专精 / 踏板感知 | `PIANO_BYTEDANCE_PEDAL` | MAESTRO note onset F1 / 踏板 onset F1 = 96.72% / 91.86% | 需要踏板 CC64 时优先选择；不会作为其他钢琴后端的静默替代。 |
| BS-RoFormer | 人声/伴奏分离 | `VOCAL_SPLIT` 前置分离 | checkpoint 文件名含训练分数标签，不作为统一横向 benchmark | 只负责分离，后续 MIDI 质量还取决于转写后端。 |
| BS-RoFormer SW | 六声部分离 | `SIX_STEM_SPLIT` 前置分离 | MVSEP 6-stem SDR 口径 | 分离指标不是端到端 MIDI F1。 |

YourMT3+ / MIROS 属于多乐器后端，Transkun / Aria-AMT / ByteDance Pedal 属于钢琴专精后端，BS-RoFormer 系列属于音源分离后端；三类模型的公开指标不能混成同一张排行榜。

### 当前默认转写模型：YourMT3+

本项目默认使用 **YPTF.MoE+Multi (PS)**，它是 YourMT3+ 系列中的高性能多乐器变体。

| 项目 | 详情 |
|------|------|
| 模型全称 | YPTF.MoE+Multi (PS) |
| 检查点 | `mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2` |
| 来源 | [KAIST - YourMT3+](https://huggingface.co/spaces/mimbres/YourMT3)（[arXiv:2407.04822](https://arxiv.org/abs/2407.04822)） |
| 架构 | Perceiver Transformer 编码器 + Multi-T5 解码器 |
| MoE | 8 专家，Top-2 路由，SiLU 激活 |
| 位置编码 | RoPE（部分旋转位置编码） |
| 归一化 | RMSNorm |
| 训练增强 | Pitch Shift 音高偏移增强（PS） |
| 模型大小 | ~724 MB（本地 checkpoint 约 723.8 MiB） |
| 任务类型 | `mt3_full_plus`（128 种 GM 乐器 + 鼓） |

#### 性能基准（Slakh2100 数据集）

下表只保留可在 YourMT3+ 论文中核到的同口径总分。README 历史版本里的 Frame / Onset / Offset / Drum Onset 单项数值没有在已核验来源中逐项定位到，因此不再写入文档。

| 指标 | YPTF.MoE+Multi (PS) | MT3 (Google 基线) | 来源口径 |
|------|---------------------|-------------------|----------|
| Multi (Onset-Offset) F1 | **74.84** | 62.0 | YourMT3+ 论文 Slakh2100 对比表 |

#### YourMT3+ 可用模型变体

官方 Hugging Face / Colab demo 的模型选择项是 checkpoint / 架构选择，不是本项目的处理工作流。当前项目中的 YourMT3+ 选择器与官方列表对齐；区别是本项目会把该模型嵌入 `SMART`、`VOCAL_SPLIT` 和 `SIX_STEM_SPLIT` 等工作流中，并在 MIDI 生成阶段提供多轨/单轨输出布局选项。

| 模型 | MoE | Pitch Shift | 说明 |
|------|-----|-------------|------|
| YMT3+ | 无 | 无 | 官方 Colab 模型族中的基线 YourMT3+ checkpoint |
| YPTF+Single (noPS) | 无 | 无 | Perceiver-TF + 单解码器 checkpoint |
| YPTF+Multi (PS) | 无 | 有 | Perceiver-TF + multi-t5 多通道解码 |
| YPTF.MoE+Multi (noPS) | 8 专家 | 无 | 官方 Hugging Face Space 默认模型，更适合复现官方 Space 输出 |
| YPTF.MoE+Multi (PS) | 8 专家 | 有 | 本项目默认模型，综合表现最好；本地 checkpoint 约 723.8 MiB |
| YourMT3+ 传统版 | 无 | 无 | 旧配置兼容项，不在界面默认显示 |

与官方 demo 的主要对齐点：

- 五个官方模式名称、checkpoint 目录映射和 UI 顺序一致。
- `YPTF.MoE+Multi (noPS)` 对齐官方 Hugging Face Space 默认展示模型。
- `YMT3+`、`YPTF+Single (noPS)`、`YPTF+Multi (PS)` 这类旧 checkpoint 缺少 `TOKENIZER.max_shift_steps` 元数据时，本项目按官方 MT3 tokenizer 默认 `206` 处理。
- 旧 T5 checkpoint 缺少 `ff_layer_type` 元数据时，本项目按标准 T5 前馈层 `t5_gmlp` 处理。

与官方 demo 的差异也需要明确：

- 官方 demo 默认只运行单一 YourMT3 checkpoint；本项目还提供音源分离、钢琴专用模型、MIDI 后处理和输出布局控制。
- 官方 GPU Space 通常使用 16-bit 推理；本项目默认使用 full precision，以降低不同 Windows / CUDA 环境中的不稳定因素。
- 为避免长音频在大 batch 下长时间无进度，本项目对 YourMT3+ 推理 batch 做了保守上限：旧官方模式默认最多 1，MoE 模式默认最多 2；需要复现实验时可用 `MUSIC_TO_MIDI_YOURMT3_BATCH_SIZE` 显式指定。

### 当前可选后端：MIROS

| 后端 | 类型 | 集成方式 | 当前语义 | 说明 |
|------|------|----------|----------|------|
| MIROS (MusicFM) | 多乐器 | 本地 `ai4m-miros` 仓库 + 当前工程包装器 | 固定 checkpoint 质量 | 官方仓库标注为 Music Transcription Challenge winning model，可作为桌面版 `SMART` 与 `VOCAL_SPLIT` 的多乐器后端 |

处理语义：

- 所有入口默认使用固定高质量处理策略。
- `MIROS` 当前为固定 checkpoint 推理，可用于与 YourMT3+ 做同任务 A/B。

### 当前人声分离模型：BS-RoFormer

`VOCAL_SPLIT` 模式使用 **BS-RoFormer**（Band-Split Rotary Transformer）进行人声/伴奏分离。

| 项目 | 详情 |
|------|------|
| 模型全称 | Band-Split RoFormer |
| 论文 | [Music Source Separation with Band-Split RoFormer](https://arxiv.org/abs/2309.02612) (ISMIR 2023 Workshop) |
| 检查点 | `model_bs_roformer_ep_368_sdr_12.9628.ckpt` (epoch 368) |
| 训练者 | [ZFTurbo](https://github.com/ZFTurbo) / [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) |
| 许可证 | MIT |
| 指标说明 | checkpoint 文件名包含 `sdr_12.9628`（训练过程分数标签，不是统一评测口径） |
| 公开对比说明 | 人声分离指标来源混杂，常见口径包括 Multisong、MUSDB18-HQ、MVSEP 与 audio-separator 内置分数；这些分数不能直接合并成一张排行榜。 |
| 模型大小 | ~610 MB（本地 checkpoint 约 609.7 MiB） |
| 调用方式 | 通过 [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 库封装 |
| 首次使用 | 自动从 HuggingFace 下载到 `~/.music-to-midi/models/audio-separator/` |
| 输出选项 | 默认 2 个 MIDI（伴奏 + 人声）；可勾选额外输出 1 个合并 MIDI |

核心思想：将频谱按频段（band）拆分后独立建模，用 RoPE 增强时序建模，通过 Band-level 和 Temporal Self-Attention 交替捕获跨频段谐波关系和时序依赖。

#### 人声分离模型对比

> 注：本表只保留这次重新核验时能找到公开来源支撑的结论。若写明“未写入数值”，表示没有找到与当前 checkpoint 明确绑定、且口径足够清晰的公开数值。

| 模型/方向 | 来源 | 类型 | 状态 | 说明 |
|-----------|------|------|------|------|
| BS-RoFormer ep368（当前） | [TRvlvr download_checks](https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json) / [TRvlvr model_repo](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models) / [audio-separator models-scores](https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/models-scores.json) | 本地直替（audio-separator） | 使用中 | 当前工程默认 checkpoint：`model_bs_roformer_ep_368_sdr_12.9628.ckpt`；文件名自带 `sdr_12.9628`，但这不是统一公开 benchmark，README 不再把它当作横向对比分数。 |
| BS-RoFormer ep317（公开可下载） | [MVSEP News](https://www.mvsep.com/news) / [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) | 本地直替（audio-separator） | 可替换（权衡） | 公开页面能确认其为可下载 BS-RoFormer 系列模型；具体 SDR 数值因榜单与 checkpoint 映射不够清晰，未写入 README。 |
| MelBand-RoFormer (KimberleyJensen) | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) / [Hugging Face](https://huggingface.co/KimberleyJSN/melbandroformer) / [MVSEP Full API](https://mvsep.ru/full_api) | 本地可用（vocals/other） | 可用（偏人声） | 公开权重 `MelBandRoformer.ckpt` 可核；MVSEP/API 分数未找到与该公开 checkpoint 的一一对应说明，未写入具体数值。 |
| SCNet XL IHF（开源权重） | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) / [ZFTurbo Release v1.0.15](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.15) | 开源可下载（4-stem） | 需改造接入 | 公开权重是 4-stem 模型，不是本项目现有 2-stem 直替；README 不写入未能绑定到当前用法的具体分数。 |
| MVSEP Ensemble / BS Roformer 服务模型 | [MVSEP News](https://www.mvsep.com/news) / [MVSEP Full API](https://mvsep.ru/full_api) | API 调用（权重未公开） | 非本地直替 | 榜单服务可用，但公开下载清单未见对应 checkpoint；不作为本地可复现模型写入具体数值。 |
| Mel-RoFormer (ISMIR 2024) | [arXiv:2409.04702](https://arxiv.org/abs/2409.04702) / [ar5iv 表2](https://ar5iv.org/html/2409.04702v1) | 论文阶段（研究模型） | 论文已发表 | MUSDB18-HQ（论文表2，场景 b，含额外数据）仅报告 Vocals SDR；这是论文特定协议，不与 Multisong / MVSEP 数字混排。 |
| Mamba2 Meets Silence (v2, 2025) | [arXiv:2508.14556](https://arxiv.org/abs/2508.14556) | 论文阶段（研究模型） | 论文 | 摘要报告 cSDR 11.03 dB（作者称 best reported），强调稀疏人声段鲁棒性 |
| Windowed Sink Attention (2025) | [arXiv:2510.25745](https://arxiv.org/abs/2510.25745) | 论文阶段（效率优化方向） | 论文 + 开源代码 | 在微调设定下恢复原模型约 92% SDR，同时 FLOPs 降低约 44.5x（偏效率收益） |

结论（按口径）：

- 当前 README 不再把不同来源的人声分离分数混成排行榜。
- 若来源是 API/服务模型、没有公开 checkpoint 映射，文档只标注“非本地直替”，不写成可直接替换的本地模型。
- 若来源是论文特定协议，文档只说明协议，不与工程默认 checkpoint 的文件名分数横比。
- **口径提醒**：不同榜单/数据集/评测协议（Multisong、MUSDB、MVSEP、cSDR/uSDR）不可直接横比。

### 已恢复流程对比

下表覆盖已恢复到桌面版、Space 和 Colab 的额外流程。注意：公开数据通常只覆盖“分离”或“钢琴 AMT”单项任务，不等于本项目端到端音频转 MIDI 的统一评分。

| 流程 | 当前仓库状态 | 上游模型/实现 | 可核验公开数据 | 与当前 `SMART` / `VOCAL_SPLIT` 的关系 |
|------|--------------|---------------|----------------|---------------------------------------|
| 六声部分离 + 分别转写 | `six_stem_split` 已在 pipeline、桌面 UI、Space 和 Colab 中开放 | BS Roformer SW（vocals, bass, drums, guitar, piano, other）+ 后续 AMT 转写 | MVSEP Algorithms #77 给出 6-stem SDR：vocals 11.30 / instrum 17.50 / bass 14.62 / drums 14.11 / guitar 9.05 / piano 7.83 / other 8.71 | 这些是音源分离 SDR，不是最终 MIDI 转写 F1；“分离后分别转写”的端到端 MIDI 质量没有找到公开统一 benchmark。 |
| 钢琴专用转写（Transkun） | `piano_transkun` 已在 pipeline、桌面 UI、Space 和 Colab 中开放 | `pip install transkun`，命令行支持 `transkun input.mp3 output.mid` | 官方 model cards：Transkun V2 在 MAESTRO V3 上 Note Onset / Onset+Offset / Onset+Offset+Velocity F1 为 0.9832 / 0.9349 / 0.9296；pip 随包 No Ext checkpoint 为 0.9833 / 0.8149 / 0.8109 | 这是钢琴专精协议，适合纯钢琴；不能与 YourMT3+ 的 Slakh2100 多乐器 F1 直接横比。 |
| 钢琴专用转写（Aria-AMT） | `piano_aria_amt` 已在 pipeline、桌面 UI、Space 和 Colab 中开放 | EleutherAI `aria-amt`，公开 preliminary piano v1 checkpoint `piano-medium-double-1.0.safetensors` | 官方 README 提供安装、checkpoint 下载和 CLI 用法；未给出与 Transkun 同口径的 MAESTRO/MAPS benchmark。本地打包资源中的 checkpoint 约 425.9 MiB。 | 可作为钢琴转写候选，但 README 不写入不存在的统一分数；如需比较，建议使用同一批本地音频做 A/B 评测。 |
| 钢琴专用转写（ByteDance Pedal） | `piano_bytedance_pedal` 已在 pipeline、桌面 UI、Space 和 Colab 中开放 | `piano-transcription-inference`，checkpoint `note_F1=0.9677_pedal_F1=0.9186.pth` | 论文报告 MAESTRO onset F1 96.72% 与 pedal onset F1 91.86%；本项目保留上游 MIDI 中的 sustain pedal CC64。 | 上游 ByteDance 主仓库已归档，推理包兼容性需在目标环境验证；不作为 Transkun / Aria-AMT 的静默替代。 |

### 未来可关注的转写模型

下列对比只保留历史 README 中“公开可核实”的数据。`Slakh2100 Multi (Onset-Offset) F1`、`MAESTRO onset F1`、官方挑战名次、以及主观听感/下游任务增益并不是同一协议，不能当成同一张排行榜直接横比。

#### 多乐器模型（公开可核实）

| 模型 | 公开来源 | Benchmark / 协议 | 公开结果 | 状态 | 说明 |
|------|----------|------------------|----------|------|------|
| YPTF.MoE+Multi (PS)（当前） | [YourMT3+ 论文](https://arxiv.org/abs/2407.04822) / [KAIST HF](https://huggingface.co/spaces/mimbres/YourMT3) | Slakh2100 `Multi (Onset-Offset) F1` | **74.84**；同表 `MT3 = 62.0` | 使用中 | 当前项目主线后端，且这是 README 中与 MT3 最同口径的一组公开数字 |
| [MT3](https://github.com/magenta/mt3) | [YourMT3+ 论文](https://arxiv.org/abs/2407.04822) / [Magenta 仓库](https://github.com/magenta/mt3) | Slakh2100 `Multi (Onset-Offset) F1` | **62.0** | 开源基线 | YourMT3+ 继承并扩展的 token-based 多乐器基线 |
| 2025 AI4Musician 冠军路线（[ai4m-miros](https://github.com/amt-os/ai4m-miros)） | [ICME 2025 Workshop](https://ai4musicians.org/2025icme.html) / [Challenge 页](https://ai4musicians.org/transcription/2025transcription.html) / [代码仓库](https://github.com/amt-os/ai4m-miros) | 官方仓库描述 | winning model | 冠军路线 / 代码可见 | 这是赛事/仓库描述，不是与 Slakh Multi F1 同口径的数值榜；公开资料显示该路线基于 MusicFM 编码器与多解码器思路 |

#### 钢琴专精模型对比（公开可核实）

#### 钢琴同类模型质量对比

| 模型 | 当前项目入口 | 同类型质量口径 | 公开结果 | 使用判断 |
|------|--------------|----------------|----------|----------|
| Transkun V2 | 研究 checkpoint，不是当前 pip 默认入口 | MAESTRO V3 `note onset F1 / onset+offset F1 / onset+offset+velocity F1` | **0.9832 / 0.9349 / 0.9296** | 公开钢琴 AMT 分数很强，适合作为钢琴专精质量参考。 |
| Transkun pip checkpoint（No Ext） | `PIANO_TRANSKUN` | MAESTRO V3 No Ext 同口径三项指标 | **0.9833 / 0.8149 / 0.8109** | 当前随包权重轻量可用，但仓库说明它不带 pedal extension。 |
| Aria-AMT | `PIANO_ARIA_AMT` | 官方公开 checkpoint，但未发布与 Transkun 完全同口径的统一榜单 | 未写入统一 F1 | 作为当前默认钢琴后端候选；建议用同一批本地钢琴音频 A/B。 |
| ByteDance Pedal | `PIANO_BYTEDANCE_PEDAL` | MAESTRO `note onset F1 / 踏板 onset F1` | **96.72% / 91.86%** | 同类中优势是踏板事件，输出 MIDI 会保留 sustain pedal `CC64`。 |

YourMT3+ / MIROS 属于多乐器后端，不能与上表的钢琴专精 F1 横向混排；ByteDance Pedal 的 `踏板 onset F1` 也不能直接等同于 Transkun 的 `onset+offset+velocity F1`。

| 模型 | 公开来源 | Benchmark / 协议 | 公开结果 | 状态 | 说明 |
|------|----------|------------------|----------|------|------|
| [Transkun V2（论文 checkpoint）](https://github.com/Yujia-Yan/Transkun) | [Transkun 官方仓库 / model cards](https://github.com/Yujia-Yan/Transkun) | MAESTRO V3 `note onset F1 / onset+offset F1 / onset+offset+velocity F1` | **0.9832 / 0.9349 / 0.9296** | 开源 | 这是论文公开 checkpoint 的模型卡结果，不是当前项目入口后端 |
| [Transkun pip 随包 checkpoint（No Ext）](https://github.com/Yujia-Yan/Transkun) | [Transkun 官方仓库 / model cards](https://github.com/Yujia-Yan/Transkun) | MAESTRO V3 No Ext 同口径三项指标 | **0.9833 / 0.8149 / 0.8109** | 开源 | 仓库明确写明随 pip 包 checkpoint 为 `without pedal extension of notes`；适合作为后续钢琴后端候选 |
| [Aria-AMT](https://github.com/EleutherAI/aria-amt) | [EleutherAI 官方仓库](https://github.com/EleutherAI/aria-amt) | 公开 checkpoint 发布 | 仓库公开 `piano-medium-double-1.0.safetensors`；但仓库页未给出与上表完全同口径的统一 MAESTRO/MAPS 榜单 | 开源 | 可作为钢琴后端候选，但这里不伪造不存在的统一 benchmark 行 |
| [High-Resolution Piano Transcription with Pedals by Regressing Onset and Offset Times](https://arxiv.org/abs/2010.01815) | [论文](https://arxiv.org/abs/2010.01815) / [ByteDance 仓库](https://github.com/bytedance/piano_transcription) | MAESTRO `onset F1 / pedal onset F1` | **96.72% / 91.86%** | 论文 + 代码 | 代表性踏板感知钢琴论文；协议是钢琴专精口径，不应与多乐器 Slakh 分数混排 |

#### 论文阶段 / 协议不一致的研究方向

| 模型/方向 | 公开来源 | 公开协议 / 任务 | 可核实的公开信息 | 为什么不与上表混成同一分数榜 |
|-----------|----------|-----------------|------------------|------------------------------|
| [MR-MT3](https://arxiv.org/abs/2403.10024) | [论文](https://arxiv.org/abs/2403.10024) / [代码](https://github.com/gudgud96/MR-MT3) | Slakh2100；重点看 `onset F1`、`instrument leakage ratio`、`instrument detection F1` | 摘要明确写的是“improved onset F1 scores and reduced instrument leakage” | 它主打 leakage 抑制，并引入了新指标；不等于上面的 Slakh `Multi (Onset-Offset) F1` |
| [Jointist](https://arxiv.org/abs/2302.00286) | [论文](https://arxiv.org/abs/2302.00286) | 流行音乐联合转写 + 分离 | 摘要给出的公开结果是：转写提升 `>1 ppt`、分离提升 `+5 SDR`、downbeat `+1.8 ppt`、和弦/调性各 `+1.4 ppt` | 它是 joint transcription + separation 路线，公开协议与 Slakh / MAESTRO 完全不同 |
| MusicFM 编码器 + AMT 解码器 | [MusicFM 论文](https://arxiv.org/abs/2311.03318) / [仓库](https://github.com/minzwon/musicfm) / [HF 权重](https://huggingface.co/minzwon/MusicFM) | 预训练编码器迁移 | 公开的是基础编码器权重；通用可复现的完整 AMT decoder / 微调流水线并未作为现成后端发布 | 它更像 MIROS 这类路线背后的表示学习部件，不是拿来就能切换的通用后端 |
| [CountEM / Count The Notes](https://arxiv.org/abs/2511.14250) | [论文](https://arxiv.org/abs/2511.14250) / [项目页](https://yoni-yaffe.github.io/count-the-notes) / [代码](https://github.com/Yoni-Yaffe/count-the-notes) | 弱监督 AMT 训练方法 | 公开论文、代码和模型，核心贡献是“用音符直方图 + EM”替代精确对齐监督 | 这是训练范式创新，不是固定 checkpoint 的 turnkey 后端 |
| [PerceiverTF](https://arxiv.org/abs/2306.10785) | [论文](https://arxiv.org/abs/2306.10785) | 多乐器公开数据集（论文自有协议） | 摘要只明确说其在多个公开数据集上优于 MT3 / SpecTNT | 它更适合作为 YourMT3+ 的架构祖先来理解，不应和上表的统一数值行硬拼 |

补充说明：

- [Basic Pitch](https://github.com/spotify/basic-pitch) 依然是很有价值的轻量方案，但它不发布与上表同口径的 Slakh/MAESTRO 综合榜单。
- [Omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) 仍是有参考价值的多任务工具链，但其 GitHub latest release 仍为 `0.5.0`（2021-12-09），与当前多乐器/钢琴专精 SOTA 的公开比较协议并不一致。

趋势总结：截至 2026 年初，多乐器 AMT 的公开强路线主要沿两条线发展：一条是 `MT3 / YourMT3+ / MR-MT3` 这种 token-based 专用模型演进，另一条是 `MusicFM` 这类预训练编码器增强路线。钢琴 AMT 的公开成熟度仍然更高，但 `Transkun pip 权重`、`论文 checkpoint`、`pedal-aware 论文系统` 之间的协议差异必须写清楚，不能简单合并成一个“钢琴榜单”。

## MIDI 轨道布局

YourMT3+ 后端提供两个输出布局：

| 布局 | 行为 |
|------|------|
| 多轨（按 GM 乐器分轨） | 每个识别到的 GM 程序号尽量独立成轨；鼓统一使用 GM 鼓通道。 |
| 单轨（旋律合并，鼓独立） | 非鼓音符合并到一条旋律轨，但保留原始通道与音色变化；鼓仍独立成轨。 |

当识别到的乐器数量超过 MIDI 非鼓通道上限时，生成器会按乐器族进行合并，避免直接丢弃音符。

## 默认处理策略

桌面版、Space 和 Colab 不再提供可调质量入口。所有流程默认走固定高质量策略；YourMT3+ 仍使用重叠分段、智能去重和 MIDI 后处理来保留细节，其它后端按各自 checkpoint 直接推理。

## 失败处理原则

本项目的默认原则是：**真实失败必须显式暴露，不用静默回退、假成功或默认值制造“看起来完成”的结果**。

应直接失败并提示根因的情况：

- 模型、checkpoint、上游源码或必需 Python 包缺失。
- FFmpeg / 音频解码失败。
- YourMT3+、MIROS、Transkun、Aria-AMT 或 audio-separator 推理失败。
- 分离器没有生成完整目标文件。
- 模型下载遇到证书、网络或完整性问题。

当前实现：

| 场景 | 当前行为 |
|------|----------|
| FFmpeg 转 WAV 失败 | 直接报错，保留 FFmpeg stderr；不再尝试 librosa/soundfile 替代转换 |
| audio-separator CUDA 不兼容 | 直接报错并说明 GPU/CUDA/PyTorch 不兼容；不再自动改用 CPU 分离 |
| BPM 检测失败 | 直接报错；不再写入默认 120 BPM |
| YourMT3+ 显存不足 | 允许降低 batch size 重试，因为只影响性能不改变语义；日志和进度会显示初始 batch 与最终 batch |
| 配置非法模式/后端/轨道布局 | 未知值直接抛出校验错误；仅旧 `piano` 模式会兼容映射为 `smart` |
| 人声转写未识别 vocal 程序号 | 输出空人声 MIDI 并记录警告；不再用音符最多的乐器冒充人声 |
| SSL 证书验证失败 | 默认停止下载并提示配置 CA/代理；只有显式设置 `ALLOW_INSECURE_HF_DOWNLOAD=1` 才会跳过验证 |

如果未来实现新的 fallback，必须同时说明：

- 触发条件。
- 用户可见行为。
- 输出结果风险。
- 日志或 UI 中的明确标记。

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.10+，Windows 安装脚本优先使用 3.10-3.12 |
| PyTorch | 2.4.0 或更高，建议与 `torchaudio`、`torchvision` 版本匹配 |
| FFmpeg | 必需；用于可靠处理 MP3/M4A/FLAC/OGG 等格式 |
| GPU | 发布包要求 NVIDIA CUDA；CI/CD 和便携包只生成 CUDA 12.1 GPU 版本 |
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

CI/CD 和便携发布包不再生成 CPU 版本；本地源码开发如需 CPU-only PyTorch，应自行承担模型速度和依赖兼容性差异。

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 4. 准备 YourMT3+ 源码与模型

```bash
git clone https://github.com/mimbres/YourMT3.git
python download_sota_models.py
```

如果 `YourMT3/` 已经存在，可以只执行模型下载。`download_sota_models.py` 会准备全部五种官方 YourMT3+ checkpoint 模式，而不是只下载默认 MoE 模型。

### 5. 准备分离与钢琴模型

```bash
python download_vocal_model.py
python download_multistem_model.py
python download_vocal_harmony_model.py
python download_aria_amt_model.py
python download_bytedance_piano_model.py
python download_miros_model.py
```

模型默认缓存到：

```text
~/.music-to-midi/models/audio-separator
~/.cache/music_ai_models/aria_amt
~/.cache/music_ai_models/bytedance_piano
external/ai4m-miros
```

Transkun 的模型资源随 `transkun` 包安装；若钢琴专用 Transkun 模式提示资源缺失，请执行 `python -m pip install --force-reinstall transkun`。

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

Space 版会尝试从 Hugging Face Space 仓库同步 YourMT3 源码，并自动检查默认 YourMT3+ 与 Aria-AMT 模型权重。

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
Aria-AMT 模型缓存
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
    transkun_transcriber.py  # Transkun 钢琴专用后端
    aria_amt_transcriber.py  # Aria-AMT 钢琴专用后端
    vocal_separator.py       # 人声/伴奏分离
    multi_stem_separator.py  # 六声部分离
    vocal_harmony_separator.py # 主唱/和声实验分离
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
download_multistem_model.py  # 六声部分离模型下载
download_aria_amt_model.py   # Aria-AMT 模型下载
download_bytedance_piano_model.py # ByteDance Pedal 模型下载
download_vocal_harmony_model.py # 主唱/和声实验模型下载
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

如果下载模型时报 SSL 证书错误，默认会停止下载。请先配置系统/代理 CA，或设置：

```bash
export SSL_CERT_FILE=/path/to/ca.pem
export REQUESTS_CA_BUNDLE=/path/to/ca.pem
```

只有在你明确接受证书校验风险时，才设置：

```bash
export ALLOW_INSECURE_HF_DOWNLOAD=1
```

Windows PowerShell:

```powershell
$env:ALLOW_INSECURE_HF_DOWNLOAD = "1"
```

YourMT3+ 推理如果发生显存不足，会尝试降低 batch size 重试；日志和进度会显示初始 batch 与最终 batch。这个重试只影响性能，不改变转写语义。

### 人声分离不可用

确认依赖和模型：

```bash
pip install "audio-separator>=0.38.0" "onnxruntime>=1.16.0,<2"
python download_vocal_model.py
```

如果当前 NVIDIA GPU、PyTorch/CUDA 或 audio-separator 后端不兼容，流程会直接失败并提示根因；不会自动改用 CPU 分离。需要 CPU 路径时，应由后续 UI/配置提供显式选项并在结果中标注。

### 六声部分离不可用

确认 `audio-separator>=0.38.0` 已安装，并下载 BS-RoFormer SW 资源：

```bash
python download_multistem_model.py
```

如果六声部分离缺少任一 stem，流程应失败并提示缺失项；不要把缺失 stem 当作空音轨静默合并。

### 钢琴专用转写不可用

Transkun 模式需要 `transkun` 包和其随包预训练资源：

```bash
python -m pip install --force-reinstall transkun
```

Aria-AMT 模式需要 `aria-amt` 包和 checkpoint：

```bash
python -m pip install git+https://github.com/EleutherAI/aria-amt.git
python download_aria_amt_model.py
```

### MIROS 不可用

确认本地仓库位置和文件完整性：

```text
ai4m-miros/main.py
ai4m-miros/transcribe.py
```

若提示缺少 Python 模块，请按 MIROS 上游仓库说明补齐依赖。

### BPM 检测失败

当前版本不会用默认 120 BPM 继续生成 MIDI。若 BPM 检测失败、检测算法没有返回 BPM，或返回 0 / NaN 等无效 BPM，流程会停止并提示 librosa、音频质量或节拍检测算法的具体错误。

### 配置校验失败

未知的处理模式、转写后端、多乐器模型或 MIDI 轨道布局会直接报错，不会自动改成默认值。请确认配置值属于：

```text
processing_mode: smart / vocal_split / six_stem_split / piano_transkun / piano_aria_amt / piano_bytedance_pedal
transcription_backend: aria_amt / yourmt3 / miros
multi_instrument_model: yourmt3 / miros
midi_track_mode: multi_track / single_track
```

旧配置中的 `processing_mode=piano` 会兼容映射为 `smart`。

## 许可证

本项目使用 MIT License。第三方模型、数据和上游仓库遵循各自许可证与使用条款。
