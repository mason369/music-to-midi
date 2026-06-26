# 音乐转 MIDI 转换器

<p align="center">
  中文 | <a href="./README.md">English</a>
</p>

一个基于 AI 的音频转 MIDI 工具，提供 PyQt6 桌面版、Gradio Web 版和 Google Colab 运行入口。当前版本同步六种处理模式：完整混音多乐器转写、人声/伴奏分离后分别转写、六声部分离后分别转写，以及 Transkun / Aria-AMT / ByteDance Pedal 三条钢琴专用转写流程。

## 截图

| Windows | Linux |
|---------|-------|
| ![Windows 演示](../resources/icons/Windows演示.png) | ![Linux 演示](../resources/icons/Linux演示.png) |

## 当前能力

- **完整混音转写**：`SMART` 模式直接读取整首音频，用多乐器后端生成 MIDI。
- **人声/伴奏分离转写**：`VOCAL_SPLIT` 模式先分离人声与伴奏，再分别生成 MIDI；可选额外输出一个合并 MIDI。
- **六声部分离转写**：`SIX_STEM_SPLIT` 模式先分离 `bass / drums / guitar / piano / vocals / other` 六个 stem，再输出各 stem MIDI 和合并 MIDI。
- **钢琴专用转写**：`PIANO_TRANSKUN`、`PIANO_ARIA_AMT` 与 `PIANO_BYTEDANCE_PEDAL` 面向纯钢琴音频，分别调用 Transkun、Aria-AMT 和 ByteDance 带踏板模型。
- **默认后端语义**：多乐器默认后端为 YourMT3+ 官方 `YPTF.MoE+Multi (noPS)`；桌面版可切换 MIROS，钢琴专用模式分别使用 Transkun、Aria-AMT 或 ByteDance Pedal。
- **MIROS 可选后端**：桌面版可切换到本地 `ai4m-miros` 仓库作为实验性多乐器后端。
- **节拍与后处理**：`SMART` 模式保留 YourMT3+ / MIROS 官方 MIDI 输出；分离后再生成 MIDI 的扩展流程会先检测 BPM，检测失败会停止，不写入默认 tempo。支持量化、去重、力度平滑、复音限制等后处理。
- **多格式输入**：支持 `MP3`、`WAV`、`FLAC`、`OGG`、`M4A`。非 WAV 会优先通过 FFmpeg 转为 44.1 kHz PCM WAV。
- **多平台入口**：桌面版、Space、Colab 均同步暴露六种处理模式。

## 不同入口的功能范围

| 入口 | 处理模式 | 后端选择 | 适合场景 |
|------|----------|----------|----------|
| PyQt6 桌面版 | `SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT`、`PIANO_TRANSKUN`、`PIANO_ARIA_AMT`、`PIANO_BYTEDANCE_PEDAL` | 多乐器默认 YourMT3+ noPS，可选 MIROS；钢琴模式按入口使用 Transkun / Aria-AMT / ByteDance Pedal | 本地长期使用、GPU 推理、批量输出文件、钢琴专用转写 |
| Gradio Space | 同桌面六种模式 | 固定处理链路；多乐器默认 YourMT3+ noPS；钢琴模式按入口使用对应后端 | 浏览器中快速试用或部署 |
| Google Colab | 同桌面六种模式 | SMART 可选 YourMT3+ 官方 checkpoint；分离模式 MIDI 扩展默认 noPS；钢琴模式按入口使用对应后端 | 临时使用 Colab GPU |

## 处理模式

| 模式 | 内部流程 | 主要输出 | 说明 |
|------|----------|----------|------|
| `SMART` | 音频 -> YourMT3+ / MIROS 官方 MIDI 输出 | `<歌曲名>.mid` | 不做音源分离，适合大多数混音歌曲、纯音乐和多乐器片段。 |
| `VOCAL_SPLIT` | 音频 -> RoFormer `vocal_rvc` 人声/伴奏分离 -> RoFormer `karaoke` 主唱/和声拆分 -> 人声/伴奏 MIDI 转写 -> MIDI 生成 | `<歌曲名>_accompaniment.mid`、`<歌曲名>_vocal.mid`，可选 `<歌曲名>_vocal_accompaniment_merged.mid` | 分离阶段会额外产出 `vocals_with_harmony`、`original_vocals`、`backing_vocals`、`accompaniment_with_harmony` WAV；MIDI 阶段继续用所选多乐器后端处理人声与伴奏。 |
| `SIX_STEM_SPLIT` | 音频 -> BS-RoFormer SW 六声部 WAV 分离 -> 多乐器后端完整混音转写 -> 按 GM 乐器族分配到 stem MIDI -> stem MIDI 合并 | `<歌曲名>_<stem>.mid`、`<歌曲名>_all_stems_merged.mid` | 六个 WAV stem 来自分离模型；MIDI 不是逐个 stem 重新跑 AMT，而是由完整混音转写结果按乐器族路由生成。piano stem 在偏好 Aria-AMT 且权重可用时会优先走 Aria-AMT。 |
| `PIANO_TRANSKUN` | 音频 -> Transkun 钢琴模型 -> MIDI | `<歌曲名>_piano_transkun.mid` | 适合纯钢琴音频；使用 Transkun 随包 checkpoint 固定推理。 |
| `PIANO_ARIA_AMT` | 音频 -> Aria-AMT 钢琴模型 -> MIDI | `<歌曲名>_piano_aria.mid` | 适合纯钢琴音频；需要 Aria-AMT checkpoint 已随包或在模型目录可用。 |
| `PIANO_BYTEDANCE_PEDAL` | 音频 -> ByteDance 带踏板钢琴模型 -> MIDI | `<歌曲名>_piano_bytedance_pedal.mid` | 适合纯钢琴音频；会保留延音踏板 CC64；需要 ByteDance Piano checkpoint 已随包或在模型目录可用。 |

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
song_piano_aria.mid
song_piano_bytedance_pedal.mid
song_vocals_with_harmony.wav
song_original_vocals.wav
song_backing_vocals.wav
song_accompaniment.wav
song_accompaniment_with_harmony.wav
song_bass.wav
song_drums.wav
song_guitar.wav
song_piano.wav
song_vocals.wav
song_other.wav
```

实际文件数量取决于所选模式、是否启用人声分离合并 MIDI，以及分离器输出。六声部模式固定输出六个 stem MIDI 和一个 `all_stems` 合并 MIDI；人声分离会创建 `_vocal_rvc/`、`_karaoke/` 子目录作为中间输出目录。

## 后端说明

### YourMT3+

YourMT3+ 是默认多乐器后端。项目使用 `download_sota_models.py` 下载全部官方 YourMT3+ checkpoint 模式，并准备 BS-RoFormer SW 六轨资源与 RoFormer `vocal_rvc` / `karaoke` 人声 ensemble；YourMT3 推理通过 `src/core/yourmt3_transcriber.py` 调用本地 `YourMT3/amt/src` 源码。

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

下载脚本会克隆上游 `amt-os/ai4m-miros` 代码；`pretrained_msd.pt` 使用官方 Hugging Face `minzwon/MusicFM` 权重，`last.ckpt` 按上游 `main.py` 中的 Google Drive 官方文件 ID 获取。GitHub Actions 发布打包不依赖实时 Google Drive 配额，而是从本仓库既有 GitHub Release 中下载已校验的 `miros-last.ckpt.part*` 云端镜像资产并重组；若镜像资产缺失、大小/hash 不匹配或 checkpoint 容器不完整，发布流程会直接失败并显示真实原因，不会改用未知来源或静默跳过。

### Transkun

Transkun 是钢琴专用转写后端，适合纯钢琴或以钢琴为主的音频。项目通过 `src/core/transkun_transcriber.py` 调用 `transkun` PyPI 包随附的预训练资源：

```bash
pip install transkun>=2.0.1
```

可用性检查会确认 `transkun.transcribe`、`pretrained/2.0.pt` 和 `pretrained/2.0.conf` 是否存在。缺失时请重新安装：

```bash
python -m pip install --force-reinstall transkun
```

### Aria-AMT

Aria-AMT 是另一条钢琴专用后端。项目通过 `src/core/aria_amt_transcriber.py` 调用 `amt.run transcribe`，默认 checkpoint 为：

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
models/aria_amt
```

### ByteDance Pedal

ByteDance Pedal 是钢琴专用的带踏板转写后端，适合独奏钢琴或清晰的钢琴 stem。它来自 ByteDance 的 High-Resolution Piano Transcription with Pedals 系统，本项目通过 `piano-transcription-inference` 包装，并保留上游 MIDI 中的延音踏板 `CC64`。

安装依赖：

```bash
python -m pip install "piano-transcription-inference>=0.0.6,<0.1" "torchlibrosa>=0.1.0,<0.2" "matplotlib>=3.7.0,<4"
```

准备模型：

```bash
python download_bytedance_piano_model.py
```

默认搜索模型位置包括：

```text
~/.cache/music_ai_models/bytedance_piano
models/bytedance_piano
```

## 模型与公开对比

本节恢复自历史 README 中的模型对比内容，并按当前版本的实际能力重新标注：当前已发布入口同步开放 `SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT`、`PIANO_TRANSKUN`、`PIANO_ARIA_AMT` 与 `PIANO_BYTEDANCE_PEDAL` 六种模式。下列表格把“公开 benchmark”和“项目内入口状态”分开写，避免把研究指标误写成产品能力。

### 当前默认转写模型：YourMT3+

本项目默认使用 **YPTF.MoE+Multi (noPS)**。官方 Hugging Face Space 的 `app.py` 默认项就是 `YPTF.MoE+Multi (noPS)`；`YPTF.MoE+Multi (PS)` 仍保留为可选 pitch-shift checkpoint，但不再写成项目默认。

| 项目 | 详情 |
|------|------|
| 模型全称 | YPTF.MoE+Multi (noPS) |
| 检查点 | `mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops`，官方 Space 指向 `last.ckpt` |
| 来源 | [官方 Hugging Face Space](https://huggingface.co/spaces/mimbres/YourMT3/blob/main/app.py) / [Space noPS 评测结果](https://huggingface.co/spaces/mimbres/YourMT3/blob/main/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/result_mc13_full_plus_256_default_all_eval_final.json) / [arXiv:2407.04822](https://arxiv.org/abs/2407.04822) |
| 架构 | Perceiver Transformer 编码器 + Multi-T5 解码器 |
| MoE | 8 专家，Top-2 路由，SiLU 激活 |
| 位置编码 | RoPE（部分旋转位置编码） |
| 归一化 | RMSNorm |
| 训练增强 | 不使用 Pitch Shift 音高偏移增强（noPS） |
| 模型大小 | noPS 官方 `last.ckpt` 本地解析约 535.5 MiB；PS 本地 `model.ckpt` 约 723.8 MiB |
| 任务类型 | `mt3_full_plus`（128 种 GM 乐器 + 鼓） |

#### 性能基准（Slakh2100 数据集）

下表把“项目默认 noPS 的 Space 结果文件”和“YourMT3+ 论文表的最终模型数字”分开写，避免把论文表数字直接冒充当前默认 noPS checkpoint 的单独结果。

| 指标 | 当前默认 noPS | YourMT3+ 论文 YPTF.MoE+Multi | MT3 (Google 基线) | 来源口径 |
|------|----------------|-----------------------------|-------------------|----------|
| Multi (Onset-Offset) F1 / `multi_f` | **0.7398 / 73.98%** | **74.84** | 62.0 | Space noPS 结果文件 / YourMT3+ 论文 Slakh2100 对比表 |

#### YourMT3+ 可用模型变体

| 模型 | MoE | Pitch Shift | 说明 |
|------|-----|-------------|------|
| YMT3+ | 无 | 无 | 官方 Colab 模型族中的基线 YourMT3+ checkpoint |
| YPTF+Single (noPS) | 无 | 无 | Perceiver-TF + 单解码器 checkpoint |
| YPTF+Multi (PS) | 无 | 有 | Perceiver-TF + multi-t5 多通道解码 |
| YPTF.MoE+Multi (noPS) | 8 专家 | 无 | 本项目默认模型；官方 Hugging Face Space 默认模型；Space 结果文件中 Slakh `multi_f = 0.7398` |
| YPTF.MoE+Multi (PS) | 8 专家 | 有 | 可选 pitch-shift MoE checkpoint；YourMT3+ 论文表中最终模型 Slakh `Multi F1 = 74.84`；本地 PS checkpoint 约 723.8 MiB |

### 当前可选后端：MIROS

| 后端 | 类型 | 集成方式 | 当前语义 | 说明 |
|------|------|----------|----------|------|
| MIROS (MusicFM) | 多乐器 | 本地 `ai4m-miros` 仓库 + 当前工程包装器 | 固定 checkpoint 质量 | 官方仓库标注为 Music Transcription Challenge winning model，可作为桌面版 `SMART` 与 `VOCAL_SPLIT` 的多乐器后端 |

处理语义：

- 所有入口默认使用固定高质量处理策略。
- `MIROS` 当前为固定 checkpoint 推理，可用于与 YourMT3+ 做同任务 A/B。

### 当前人声分离模型：RoFormer vocal_rvc / karaoke ensemble

`VOCAL_SPLIT` 模式使用 TelkNet 对齐的 **audio-separator RoFormer ensemble** 链路进行人声/伴奏分离，并保留本项目“分离后继续转 MIDI”的扩展。

| 项目 | 详情 |
|------|------|
| 模型全称 | audio-separator RoFormer ensemble presets |
| 运行库 | `audio-separator==0.44.1` |
| 第一段 | `ensemble:vocal_rvc`，包含 `melband_roformer_big_beta6x.ckpt` 与 `mel_band_roformer_vocals_fv4_gabox.ckpt` |
| 第二段 | `ensemble:karaoke`，包含 `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt`、`mel_band_roformer_karaoke_gabox_v2.ckpt`、`mel_band_roformer_karaoke_becruily.ckpt` |
| 调用方式 | 创建 `Separator(..., ensemble_preset="vocal_rvc")` / `Separator(..., ensemble_preset="karaoke")` 后无参 `load_model()` |
| 模型准备 | `download_sota_models.py` 会准备完整五个 checkpoint；也可分别运行 `download_vocal_model.py` 与 `download_vocal_harmony_model.py` |
| 打包行为 | release 工作流会把 `~/.music-to-midi/models/audio-separator/` 打进便携包；运行时若缺模型会明确报错，不把库的自动下载当作已打包成功 |
| 输出选项 | 分离阶段输出 `vocals_with_harmony`、`original_vocals`、`accompaniment`、`accompaniment_with_harmony`；MIDI 阶段默认输出伴奏 + 人声两个 MIDI，可选额外输出 1 个合并 MIDI |

核心链路：`vocal_rvc` 先分离带和声的人声与伴奏；`karaoke` 再把带和声的人声拆成主唱与 backing vocal；最后把 backing vocal 混回伴奏侧生成 `accompaniment_with_harmony` 供检查，同时本项目继续使用人声/伴奏 stem 做 MIDI 转写。

#### 人声分离模型对比

> 注：本表只保留这次重新核验时能找到公开来源支撑的结论。若写明“未写入数值”，表示没有找到与当前 checkpoint 明确绑定、且口径足够清晰的公开数值。

| 模型/方向 | 来源 | 类型 | 状态 | 说明 |
|-----------|------|------|------|------|
| RoFormer vocal_rvc + karaoke ensemble（当前） | [audio-separator `ensemble_presets.json`](https://github.com/nomadkaraoke/python-audio-separator/blob/main/audio_separator/ensemble_presets.json) | 本地 ensemble（audio-separator） | 使用中 | 与 TelkNet 对齐；`vocal_rvc` 使用 `melband_roformer_big_beta6x.ckpt` + `mel_band_roformer_vocals_fv4_gabox.ckpt`；`karaoke` 使用 3 个 karaoke checkpoint，preset 描述给出 karaoke ensemble SDR 约 10.6。 |
| BS-RoFormer ep317（公开可下载） | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) | 本地直替（audio-separator） | 可替换（权衡） | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` 公开可下载；ZFTurbo 表按 Multisong 写明 `SDR vocals = 10.87`。注意文件名中的 `12.9755` 是训练标签，不等同于表中 vocals SDR。 |
| MelBand-RoFormer (KimberleyJensen) | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) / [Hugging Face](https://huggingface.co/KimberleyJSN/melbandroformer) | 本地可用（vocals/other） | 可用（偏人声） | 公开权重 `MelBandRoformer.ckpt` 可核；ZFTurbo 表按 Multisong 写明 `SDR vocals = 10.98`。 |
| SCNet XL IHF（开源权重） | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) / [ZFTurbo Release v1.0.15](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.15) | 开源可下载（4-stem） | 需改造接入 | 公开权重是 4-stem 模型，不是本项目现有 2-stem 直替；ZFTurbo 表写明 MUSDB test avg 10.08、Multisong avg 9.92。 |
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

### 未来可关注的转写模型

下列对比只保留历史 README 中“公开可核实”的数据。`Slakh2100 Multi (Onset-Offset) F1`、`MAESTRO onset F1`、官方挑战名次、以及主观听感/下游任务增益并不是同一协议，不能当成同一张排行榜直接横比。

#### 多乐器模型（公开可核实）

| 模型 | 公开来源 | Benchmark / 协议 | 公开结果 | 状态 | 说明 |
|------|----------|------------------|----------|------|------|
| YPTF.MoE+Multi (noPS)（当前默认） | [官方 Space app.py](https://huggingface.co/spaces/mimbres/YourMT3/blob/main/app.py) / [Space noPS 结果文件](https://huggingface.co/spaces/mimbres/YourMT3/blob/main/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/result_mc13_full_plus_256_default_all_eval_final.json) | Slakh `multi_f` | **0.7398 / 73.98%** | 使用中 | 当前项目默认 YourMT3+ checkpoint；对齐官方 Hugging Face Space 默认项 |
| YPTF.MoE+Multi（论文表最终模型） | [YourMT3+ 论文](https://arxiv.org/abs/2407.04822) | Slakh2100 `Multi (Onset-Offset) F1` | **74.84**；同表 `MT3 = 62.0` | 论文公开结果 | 这是论文表中的最终模型口径，不把它写成当前 noPS 默认 checkpoint 的单独成绩 |
| [MT3](https://github.com/magenta/mt3) | [YourMT3+ 论文](https://arxiv.org/abs/2407.04822) / [Magenta 仓库](https://github.com/magenta/mt3) | Slakh2100 `Multi (Onset-Offset) F1` | **62.0** | 开源基线 | YourMT3+ 继承并扩展的 token-based 多乐器基线 |
| 2025 AI4Musician 冠军路线（[ai4m-miros](https://github.com/amt-os/ai4m-miros)） | [ICME 2025 Workshop](https://ai4musicians.org/2025icme.html) / [Challenge 页](https://ai4musicians.org/transcription/2025transcription.html) / [代码仓库](https://github.com/amt-os/ai4m-miros) | 官方仓库描述 | winning model | 冠军路线 / 代码可见 | 这是赛事/仓库描述，不是与 Slakh Multi F1 同口径的数值榜；公开资料显示该路线基于 MusicFM 编码器与多解码器思路 |

#### 钢琴专精模型（公开可核实）

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

## 默认处理策略

桌面版、Space 和 Colab 不再提供可调质量入口。所有流程默认走固定高质量策略；YourMT3+ 仍使用重叠分段、智能去重和 MIDI 后处理来保留细节，其它后端按各自 checkpoint 直接推理。

`SIX_STEM_SPLIT` 中，WAV stem 来自 BS-RoFormer SW；stem MIDI 由一次完整混音多乐器转写结果按 GM 乐器族分配而来。piano stem 在偏好 Aria-AMT 且模型可用时会使用固定 checkpoint 的 Aria-AMT。

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.10+，Windows 安装脚本优先使用 3.10-3.12 |
| PyTorch | 安装脚本/发布包路径使用 `torch==2.7.0`、`torchaudio==2.7.0`、`torchvision==0.22.0` |
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

CUDA 12.8（推荐，RTX 50 系列 / sm_120 需要此类运行时）:

```bash
pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

CUDA 11.8:

```bash
pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
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

如果 `YourMT3/` 已经存在，可以只执行模型下载。`download_sota_models.py` 会同时准备全部官方 YourMT3+ checkpoint 模式、BS-RoFormer SW 六轨资源与 RoFormer `vocal_rvc` / `karaoke` 人声 ensemble。

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
~/.cache/music_ai_models/yourmt3_all
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

Space 版会尝试从 Hugging Face Space 仓库同步 YourMT3 源码。运行转换时会按所选模式检查/准备 YourMT3+ 官方 checkpoint、BS-RoFormer SW、RoFormer `vocal_rvc` / `karaoke`、Aria-AMT 或 ByteDance Pedal 资源；缺失资源准备失败会在后续流程中显式暴露。

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
YourMT3 模型缓存 -> models/yourmt3_all
audio-separator 模型缓存 -> models/audio-separator
Aria-AMT 模型缓存 -> models/aria_amt
ByteDance Piano 模型缓存 -> models/bytedance_piano
可选 MIROS 本地仓库
ffmpeg.exe / ffprobe.exe
```

便携版资源来源优先级：

```text
MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR 或 ~/.cache/music_ai_models/yourmt3_all 或 checkpoints/yourmt3_all
MUSIC_TO_MIDI_BUNDLE_AUDIO_SEPARATOR_DIR 或 ~/.music-to-midi/models/audio-separator 或 checkpoints/audio-separator
MUSIC_TO_MIDI_BUNDLE_ARIA_AMT_DIR 或 ~/.cache/music_ai_models/aria_amt 或 checkpoints/aria_amt
MUSIC_TO_MIDI_BUNDLE_BYTEDANCE_PIANO_DIR 或 ~/.cache/music_ai_models/bytedance_piano 或 checkpoints/bytedance_piano
MUSIC_TO_MIDI_BUNDLE_MIROS_DIR 或 external/ai4m-miros / ai4m-miros / .tmp/ai4m-miros
MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR 或 tools/ffmpeg / ffmpeg
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
    bytedance_piano_transcriber.py # ByteDance Pedal 钢琴专用后端
    vocal_separator.py       # 人声/伴奏分离
    multi_stem_separator.py  # 六声部分离
    midi_generator.py        # MIDI 生成与后处理
    beat_detector.py         # BPM/节拍检测
  gui/
    main_window.py           # PyQt6 主窗口
    widgets/track_panel.py   # 模式、后端、模型选择
    workers/processing_worker.py
  models/
    data_models.py           # Config、ProcessingResult、NoteEvent 等
    gm_instruments.py        # GM 128 乐器映射
  utils/
    runtime_paths.py         # 运行时资源路径
    yourmt3_downloader.py    # YourMT3+ 模型路径与下载辅助

space/app.py                 # Gradio Web 界面
colab_notebook.ipynb         # Colab 运行入口
download_sota_models.py      # YourMT3+ 官方模式 + BS-RoFormer SW 六轨资源下载
download_vocal_model.py      # 人声分离模型下载
download_multistem_model.py  # 六声部分离模型下载
download_aria_amt_model.py   # Aria-AMT 模型下载
download_bytedance_piano_model.py # ByteDance Pedal 模型下载
download_vocal_harmony_model.py # RoFormer karaoke 人声模型下载
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
pip install "audio-separator==0.44.1" "onnxruntime==1.23.2" --no-deps
python download_vocal_model.py
python download_vocal_harmony_model.py
```

### 六声部分离不可用

确认 `audio-separator==0.44.1` 已安装，并下载 BS-RoFormer SW 资源：

```bash
python download_multistem_model.py
```

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

ByteDance Pedal 模式需要 `piano-transcription-inference`、`torchlibrosa` 和 ByteDance Piano checkpoint：

```bash
python -m pip install "piano-transcription-inference>=0.0.6,<0.1" "torchlibrosa>=0.1.0,<0.2"
python download_bytedance_piano_model.py
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
