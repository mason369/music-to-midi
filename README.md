# 音乐转 MIDI 转换器 (AI Audio to MIDI)

<p align="center">
  中文 | <a href="./docs/README.md">English</a>
</p>

一个本地优先的 AI 音频转 MIDI / Audio to MIDI 转换器，面向编曲、扒谱、采样拆解、钢琴练习和自动音乐转写 (AMT) 实验。你可以把 `MP3`、`WAV`、`FLAC`、`OGG` 或 `M4A` 音频丢进来，在 PyQt6 桌面版、Gradio Web 版或 Google Colab 中生成可编辑的 MIDI 文件。

当前版本同步七种处理模式：完整混音多乐器转写、人声/伴奏分离后分别转写、六声部分离后逐 stem 转写，以及 TransKun 默认 V2 / TransKun V2 Aug / Aria-AMT / ByteDance Pedal 四条钢琴专用转写流程。它不是只抓单音旋律的小工具，而是把多乐器 AI music transcription、stem separation、钢琴转 MIDI、BPM/tempo 元数据和 MIDI 合并放进同一条工作流里。

## 统一界面演示

桌面版、Gradio Web 版和 Google Colab 采用同一套七模式工作流与操作语义。以下演示按“主界面 → 分离完成 → 逐轨处理 → MuScriptor 渐进式预览”的顺序展示核心流程。

### 1. 主界面与完整混音转写

![主界面与完整混音转写](resources/screenshots/01-main-interface.png)

### 2. 六声部分离完成

![六声部分离完成](resources/screenshots/02-six-stem-separation-result.png)

### 3. 六声部波形与逐轨转 MIDI 控件

![六声部波形与逐轨转 MIDI 控件](resources/screenshots/03-six-stem-track-controls.png)

### 4. MuScriptor 边转写边预览 MIDI

![MuScriptor 分片转写、可播放进度与钢琴卷帘预览](resources/screenshots/04-muscriptor-progressive-midi-preview.png)

## 适合场景

如果你想把一段人声旋律、钢琴录音、完整混音或分轨 stem 变成可继续编辑的 MIDI，这个项目会比“上传一个文件、下载一个结果”的在线转换器更可控。它适合音乐制作人、扒谱爱好者、钢琴学习者、MIDI 编曲用户，也适合需要在本地 GPU 上验证自动音乐转写模型的开发者。

## 当前能力

- **完整混音转写**：`SMART` 模式直接读取整首歌，可选 YourMT3+、MIROS 或 MuScriptor Large，把音符、鼓点和 GM 乐器轨道转成 MIDI。
- **人声/伴奏分离与逐轨转写**：`VOCAL_SPLIT` 模式用 Leap XE 90-band 提取 vocals、PolarFormer 提取 accompaniment，主流程先交付两条真实 WAV；随后可在音轨工作台为每条 WAV 独立选择 11 条 MIDI 路线并显式开始转换。
- **六声部分离与逐轨转写**：`SIX_STEM_SPLIT` 模式用 `BS-Rofo-SW-Fixed.ckpt` 分离 `bass / drums / guitar / piano / vocals / other` 六条真实 WAV；随后逐轨选择 11 条 MIDI 路线，不会在用户未选择时自动生成或合并 MIDI。
- **钢琴专用转写**：`PIANO_TRANSKUN`、`PIANO_TRANSKUN_V2_AUG`、`PIANO_ARIA_AMT` 与 `PIANO_BYTEDANCE_PEDAL` 面向纯钢琴音频，分别调用 TransKun 默认 V2、官方 V2 Aug、Aria-AMT 和 ByteDance 带踏板模型。
- **默认后端语义**：多乐器默认后端为 YourMT3+ 官方 `YPTF.MoE+Multi (noPS)`；`SMART` 可显式切换 MIROS 或 MuScriptor Large，分离后的每条 WAV 也可独立选择这三类多乐器路线。
- **MuScriptor 真约束**：MuScriptor 的乐器多选不是显示过滤器。空选表示模型自动检测；非空选择会传入官方 `instruments` + `prelude_forcing` 解码接口，未选乐器 token 在生成阶段被禁止，事件流和最终 MIDI 还会再次校验，发现越界就拒绝发布文件。
- **MIROS 可选后端**：`SMART` 与分离结果的逐轨多乐器菜单都可显式选择本地 `ai4m-miros` 后端。
- **官方转写结果**：YourMT3+ 与 MIROS 路线保留官方 writer 的音符、音色、力度、控制器和弯音消息，只在缺少 `set_tempo` 时按检测 BPM 保持绝对秒并补写 tempo。MuScriptor 路线直接使用官方事件与 MIDI writer，并额外执行所选乐器集合的严格一致性校验。项目不会对这些结果做量化、去重、短音符过滤、力度平滑、复音限制或 `NoteEvent` 重建。
- **多格式输入**：支持 `MP3`、`WAV`、`FLAC`、`OGG`、`M4A`。非 WAV 必须通过 FFmpeg 转为 44.1 kHz PCM WAV；FFmpeg 失败会直接停止并显示 stderr 根因。
- **多平台入口**：桌面版、Space、Colab 均同步暴露七种处理模式。

## 不同入口的功能范围

| 入口 | 处理模式 | 后端选择 | 适合场景 |
|------|----------|----------|----------|
| PyQt6 桌面版 | `SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT`、`PIANO_TRANSKUN`、`PIANO_TRANSKUN_V2_AUG`、`PIANO_ARIA_AMT`、`PIANO_BYTEDANCE_PEDAL` | SMART 可选 YourMT3+ / MIROS / MuScriptor；分离结果逐轨选择 11 条路线；钢琴模式使用各自固定后端 | 本地长期使用、GPU 推理、批量输出文件、钢琴专用转写 |
| Gradio Space | 同桌面七种模式 | 同步提供 MuScriptor 搜索式乐器多选、硬约束和官方式结果工作台 | 浏览器中快速试用或部署 |
| Google Colab | 同桌面七种模式 | 与 Space 同样传递 MuScriptor 乐器约束并显示真实结果音频工作台 | 临时使用 Colab GPU |

## 入口与依赖同步状态

本仓库把“处理工作流”和“YourMT3+ 官方 checkpoint 模式”分开维护：

- `SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT`、`PIANO_TRANSKUN`、`PIANO_TRANSKUN_V2_AUG`、`PIANO_ARIA_AMT`、`PIANO_BYTEDANCE_PEDAL` 是本项目的七种处理工作流。
- `YMT3+`、`YPTF+Single (noPS)`、`YPTF+Multi (PS)`、`YPTF.MoE+Multi (noPS)`、`YPTF.MoE+Multi (PS)` 是官方 YourMT3 demo 暴露的五种 checkpoint / 架构模式。
- 桌面版、Gradio Space 和 Colab 都暴露同一组七种处理工作流；`SMART` 可选 YourMT3+、MIROS 或 MuScriptor Large，两个分离工作流先输出 WAV，再为每条音轨独立提供 11 条显式 MIDI 路线。

当前同步覆盖如下：

| 位置 | 已同步内容 | 说明 |
|------|------------|------|
| `download_sota_models.py` | 准备全部五种官方 YourMT3+ checkpoint、固定 MIROS 源码与两组权重、`BS-Rofo-SW-Fixed.ckpt`、Leap XE、PolarFormer、TransKun V2 Aug、Aria-AMT、ByteDance，并严格验证默认 TransKun 2.0.1 包及其内置 V2 资源 | 固定来源模型按已知大小/SHA256 或明确的源码/运行时身份校验；任一必需资源失败时立即停止。 |
| `run.ps1` / `run.sh` | 启动前检查全部官方 YourMT3+ 模式、BS-RoFormer SW Fixed、Leap XE、PolarFormer、TransKun V2 Aug、Aria-AMT、ByteDance Pedal、MIROS 与分离器可用性 | 缺少必需模型或校验失败时显式报错，不把缺失资源当作可运行状态。 |
| `install.ps1` / `install.sh` | 安装 PyTorch 2.7、NumPy 1.26、audio-separator 0.44.1 运行依赖，并下载必需模型 | 完整七模式要求 NVIDIA 驱动兼容 CUDA 12.8，并使用精确 `cu128` wheel；`audio-separator` 使用 `--no-deps`，避免其 NumPy 2 解析要求破坏当前 PyTorch/桌面栈。 |
| `.github/workflows/build.yml` | push / PR 只运行 Linux、Windows 源码检查、测试和打包契约验证 | 不生成便携包，也不使用空目录或假模型绕过强制 bundle 校验。 |
| `.github/workflows/release.yml` | 完整便携发布的目标构建链；计划下载并严格校验全部官方 YourMT3+ 模式、BS-RoFormer SW Fixed、Leap XE、PolarFormer、TransKun V2 Aug、Aria-AMT、ByteDance Pedal 和 MIROS | 当前在任何构建开始前由第三方许可闭集门禁阻断；许可未清零前不会生成或发布便携包。目标 GPU 运行时为 PyTorch 2.7 + CUDA 12.8。 |
| `colab_notebook.ipynb` | 保留 Colab 预装 Torch，安装 pinned Web/runtime 依赖，并同步七种模式 | `SMART` 与逐轨工作台同步提供 YourMT3+、MIROS、MuScriptor Large；逐轨菜单另含四条钢琴路线。 |

## 处理模式

| 模式 | 内部流程 | 主要输出 | 说明 |
|------|----------|----------|------|
| `SMART` | 音频 -> 所选 YourMT3+ / MIROS / MuScriptor Large -> MIDI | `<歌曲名>.mid` | 不做音源分离；MuScriptor 非空乐器选择会成为真实解码约束。 |
| `VOCAL_SPLIT` | 音频 -> Leap XE 90-band vocals + PolarFormer accompaniment -> 两条 WAV -> 逐轨显式转 MIDI | `<歌曲名>_vocals.wav`、`<歌曲名>_accompaniment.wav`；按需生成逐轨 MIDI | 分离阶段不自动转 MIDI；每条 WAV 可独立选择五个 YourMT3+ checkpoint、MIROS、MuScriptor Large 或四个钢琴后端。 |
| `SIX_STEM_SPLIT` | 音频 -> `BS-Rofo-SW-Fixed.ckpt` -> 六条 WAV -> 逐轨显式转 MIDI | `<歌曲名>_<stem>.wav`；按需生成逐轨 MIDI | 不对原混音伪造 stem MIDI，也不自动合并；每条真实 WAV 的转写路线和是否转换均由用户明确选择。 |
| `PIANO_TRANSKUN` | 音频 -> TransKun 默认 V2 模型 -> MIDI | `<歌曲名>_piano_transkun.mid` | 适合纯钢琴音频；使用 PyPI 包随附 checkpoint。 |
| `PIANO_TRANSKUN_V2_AUG` | 音频 -> 官方 TransKun V2 Aug checkpoint -> MIDI | `<歌曲名>_piano_transkun_v2_aug.mid` | 独立模式，不会在默认 TransKun 失败时静默替代；需要先下载并校验 V2 Aug 资源。 |
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
song_piano_transkun_v2_aug.mid
song_piano_aria.mid
song_piano_bytedance_pedal.mid
song_accompaniment.wav
song_bass.wav
song_drums.wav
song_guitar.wav
song_piano.wav
song_vocals.wav
song_other.wav
```

实际文件数量取决于所选模式和用户主动执行的逐轨转换。人声分离主流程只产生规范的 `<歌曲名>_vocals.wav` 与 `<歌曲名>_accompaniment.wav`；六声部主流程必须完整产生六条真实 WAV。任一分离输出缺失时流程显式失败；MIDI 仅为用户点击转换的音轨单独生成。

## 后端说明

### YourMT3+

YourMT3+ 是默认多乐器后端。`download_sota_models.py` 准备完整公开工作流资源：全部五种官方 YourMT3+ checkpoint、固定 MIROS 源码与两组权重、`BS-Rofo-SW-Fixed.ckpt`、Leap XE、PolarFormer、TransKun V2 Aug、Aria-AMT 和 ByteDance，并严格验证默认 TransKun 2.0.1 包及其内置 V2 资源；YourMT3 推理通过 `src/core/yourmt3_transcriber.py` 调用仓库内受控的 `YourMT3/amt/src` 源码。

需要满足：

```text
YourMT3/amt/src/model/ymt3.py
YourMT3/amt/src/utils/task_manager.py
YourMT3/amt/src/config/config.py
```

完整项目 checkout 已包含经过兼容补丁并由固定 manifest 校验的 `YourMT3/amt/src`。如果该目录缺失，请重新取得当前项目版本中的受控源码；不要用可变的上游 `master` 覆盖它。直接克隆上游源码只适合独立实验，不满足本项目三端源码一致性和便携构建身份契约。

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

### MuScriptor Large

项目固定使用公开提交 `302343e8992bdfc619f77f1988168374ed5d675d`
（包版本 `0.2.2a1`）及 gated 权重仓库
[`MuScriptor/muscriptor-large`](https://huggingface.co/MuScriptor/muscriptor-large) 的
revision `8809fdfbed2affa7ade94a7059e746e3880720e7`。权重约 5.47 GB，许可为
CC BY-NC 4.0；必须先在 Hugging Face 接受仓库条款并登录：

```bash
hf auth login
python download_muscriptor_model.py
```

Windows 结果工作台还需要固定 FluidSynth 2.5.6；安装脚本会准备，也可单独运行：

```bash
python download_fluidsynth_runtime.py
```

界面和官方公开演示保持同一功能语义：可搜索的标签多选与清除、空选自动检测、
实时转写进度/音符、钢琴卷帘、播放/暂停、跟随播放头、原音↔MIDI 混合、Stereo、
逐乐器静音/独奏，以及 MIDI、合成 WAV、原音左声道/MIDI 右声道立体声下载。
这些控制连接真实后端资产：逐乐器播放来自最终 MIDI 经官方 MuseScore General
SoundFont 与 FluidSynth 合成，不是无效按钮或 UI 模拟。

#### 模型身份、公开分数与评价

MuScriptor 是 Kyutai、Mirelo AI 与 IRCAM 研究人员在 2026 年 7 月公开的完整混音多乐器 AMT 模型。Large 使用 decoder-only Transformer（48 层、隐藏维度 1536；官方模型卡写约 1.3B 参数，代码 README 四舍五入为 1.4B），以 5 秒、16 kHz 单声道音频分片生成 MT3 风格音符事件。训练链路包含约 145 万 MIDI 的合成预训练、17 万首/约 11,000 小时真实音乐微调和 300 首高质量转写的强化学习后训练。

官方模型卡在作者保留的 372 首真实多乐器 `D_Test` 上报告如下结果（完整训练链路，CFG=2）：

| 模型 | Onset F1 | Frame F1 | Offset F1 | Drums F1 | Multi F1 |
|---|---:|---:|---:|---:|---:|
| YourMT3+ `YPTF.MoE+Multi (noPS)` | 32.5 | 45.5 | 17.8 | 41.4 | 21.9 |
| MuScriptor Large | **60.4** | **72.4** | **48.6** | **49.6** | **47.8** |

这些分数表明 MuScriptor Large 是很强的公开完整混音候选，但不是“所有 benchmark 的无条件 SOTA”：`D_Test` 是作者自建留出集，尚无公共下载入口；在论文列出的 8 个公共跨域数据集上，MuScriptor 的 Multi F1 高于 YourMT3+ 其中 6 个、低于其中 2 个。它也不输出 velocity，只提供 36 组乐器分类，并受 CC BY-NC 4.0 非商用许可约束。

发布时间也应分开理解：[Hugging Face API](https://huggingface.co/api/models/MuScriptor/muscriptor-large) 记录模型仓库创建于 2026-06-30；[论文](https://arxiv.org/abs/2607.08168)与 [Mirelo 官方文章](https://mirelo.ai/blog/turning-audio-to-midi)发表于 2026-07-09；GitHub `v0.2.1` 和当前公开权重 revision 更新于 2026-07-10。仓库日期只是发布流程元数据，不表示更早已有同一套最终代码和权重。

完整训练消融、8 个公共数据集逐项分数、模型规模对比、乐器条件收益、Mirelo Studio 私有增强版边界与后续模型观察清单见 [MuScriptor 模型研究、分数与项目定位](docs/muscriptor-model.md)。

### MIROS

MIROS 是桌面版、Space 与 Colab 中 `SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT` 三个模式的可选固定版本多乐器后端，对齐项目锁定的 MusicFM / AI4Musician Challenge SOTA 路线。它不是 PyPI 包接入方式，而是要求本地存在经过身份校验的固定版本上游仓库与权重，并由包装器调用其入口生成临时 MIDI 后再转换为项目内部音符结构。

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

下载脚本会检出 `amt-os/ai4m-miros` 的固定源码 commit 并应用受控兼容补丁；`pretrained_msd.pt` 使用官方 Hugging Face `minzwon/MusicFM` 权重，`last.ckpt` 按上游 `main.py` 中的 Google Drive 官方文件 ID 获取。GitHub Actions 发布打包不依赖实时 Google Drive 配额，而是从本仓库既有 `v1.0.16` Linux 便携包中流式提取已打包验证过的 `external/ai4m-miros` 目录；若便携包资产缺失、提取失败或 checkpoint 容器不完整，发布流程会直接失败并显示真实原因，不会改用未知来源或静默跳过。

### TransKun 默认 V2

TransKun 默认路线是钢琴专用转写后端，适合纯钢琴或以钢琴为主的音频。项目通过 `src/core/transkun_transcriber.py` 调用 `transkun` PyPI 包随附的 V2 预训练资源：

```bash
python -m pip install "transkun==2.0.1"
```

可用性检查会确认 `transkun.transcribe`、`pretrained/2.0.pt` 和 `pretrained/2.0.conf` 是否存在。缺失时请重新安装：

```bash
python -m pip install --force-reinstall "transkun==2.0.1"
```

### TransKun V2 Aug

`PIANO_TRANSKUN_V2_AUG` 是与默认路线并列的独立模式，使用上游官方 `checkpointTransformerAug.zip` 中的 `checkpointMSimplerAug/checkpoint.pt` 与 `model.conf`。`Aug` 表示训练时使用数据增强；它不会在默认 TransKun 失败时自动接管，反之亦然。

下载并校验：

```bash
python download_transkun_v2_aug_model.py
```

默认搜索模型位置包括：

```text
~/.cache/music_ai_models/transkun_v2_aug/checkpointMSimplerAug
models/transkun_v2_aug/checkpointMSimplerAug     # 打包资源
```

### Aria-AMT

Aria-AMT 是另一条钢琴专用后端。上游官方 README 使用 `aria-amt transcribe` CLI；本项目包装器 `src/core/aria_amt_transcriber.py` 当前通过 Python 模块入口调用 `amt.run transcribe`。默认 checkpoint 为：

```text
piano-medium-double-1.0.safetensors
```

安装依赖：

```bash
python -m pip install --no-deps --force-reinstall "aria-amt @ https://github.com/EleutherAI/aria-amt/archive/a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip"
```

下载模型：

```bash
python download_aria_amt_model.py
```

默认搜索模型位置包括：

```text
~/.cache/music_ai_models/aria_amt
models/aria_amt                  # 打包资源
```

### ByteDance Pedal

ByteDance Pedal 是钢琴专用的带踏板转写后端，适合独奏钢琴或清晰的钢琴 stem。它来自 ByteDance 的 High-Resolution Piano Transcription with Pedals 系统，本项目通过 `piano-transcription-inference` 包装，并保留上游 MIDI 中的延音踏板 `CC64`。

安装依赖：

```bash
python -m pip install "piano-transcription-inference==0.0.6" "torchlibrosa>=0.1.0,<0.2" "matplotlib>=3.7.0,<4"
```

下载模型：

```bash
python download_bytedance_piano_model.py
```

默认搜索模型位置包括：

```text
~/.cache/music_ai_models/bytedance_piano
models/bytedance_piano           # 打包资源
```

## 钢琴后端选择建议

四个钢琴后端都是“钢琴专精模型”，不参与完整混音的多乐器识别。选择时可以按目标来分：

| 目标 | 推荐模式 | 说明 |
|------|----------|------|
| 使用项目默认 TransKun 路线 | `PIANO_TRANSKUN` | 使用 PyPI 包随附 V2 资源，依赖和 checkpoint 边界清楚。 |
| 对比官方数据增强 checkpoint | `PIANO_TRANSKUN_V2_AUG` | 使用独立下载并固定校验的 V2 Aug 资源；不会静默替代默认 V2。 |
| 使用另一种现代钢琴 AMT 后端 | `PIANO_ARIA_AMT` | 适合作为纯钢琴 A/B 候选。 |
| 需要踏板 CC64，尤其是古典、抒情、连奏明显的钢琴音频 | `PIANO_BYTEDANCE_PEDAL` | 保留 sustain pedal 控制事件；上游主仓库已归档，建议在目标环境做一次实际音频验证。 |

这四者的结果不能和 `YourMT3+` / `MIROS` 的多乐器结果直接横比：钢琴后端只解决 88 键钢琴及其演奏细节，多乐器后端负责完整混音中的乐器识别和 GM 多乐器 MIDI 输出。

## 模型与公开对比

本节恢复自历史 README 中的模型对比内容，并按当前版本的实际能力重新标注：当前入口同步开放 `SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT`、`PIANO_TRANSKUN`、`PIANO_TRANSKUN_V2_AUG`、`PIANO_ARIA_AMT` 与 `PIANO_BYTEDANCE_PEDAL` 七种模式。下列表格把“公开 benchmark”和“项目内入口状态”分开写，避免把研究指标误写成产品能力。

#### 当前已集成后端总览

| 后端/模型 | 类型 | 项目入口 | 公开质量口径 | 选择说明 |
|-----------|------|----------|--------------|----------|
| YourMT3+ | 多乐器 AMT | `SMART` 可直接使用；分离结果可逐轨选择五个官方 checkpoint，转换时保留官方 writer 音符并只补必要 tempo 元数据 | 官方 Space 默认 noPS 结果文件：Slakh `multi_f = 0.7398`；YourMT3+ 论文表：Slakh2100 `Multi F1 = 74.84`，同表 `MT3 = 62.0` | 默认多乐器后端；工程默认 checkpoint 为 `YPTF.MoE+Multi (noPS)`。 |
| MuScriptor Large | 多乐器 AMT | `SMART` 可选；分离后的 WAV 可逐轨选择，并使用真实乐器硬约束与官方 writer | 作者 `D_Test`：Onset / Frame / Offset / Drums / Multi F1 = **60.4 / 72.4 / 48.6 / 49.6 / 47.8**；同表 YourMT3+ Multi F1 = 21.9 | 强公开完整混音候选；分数来自作者自建协议，不能直接与 Slakh 榜单混排；权重仅限非商用。 |
| MIROS | 多乐器 AMT | `SMART` 可选；分离后的 WAV 可逐轨选择 | 2025 AMT Challenge：F1 **0.5998**，同表 YourMT3-YPTF-MoE-M 0.5938、MT3 0.3932 | 固定 MusicFM 后端；挑战使用 76 个受约束合成短片段，不与 MuScriptor `D_Test` 或 Slakh 分数横比。 |
| TransKun 默认 V2 | 钢琴专精 | `PIANO_TRANSKUN` | TransKun V2 / pip checkpoint 在 MAESTRO V3 上有公开 F1 | 默认 TransKun 路线，随包权重边界清楚。 |
| TransKun V2 Aug | 钢琴专精 | `PIANO_TRANSKUN_V2_AUG` | 上游官方数据增强 checkpoint；不将不同 checkpoint 的指标混写 | 独立下载、固定大小与 SHA256 校验，不是默认 V2 的 fallback。 |
| Aria-AMT | 钢琴专精 | `PIANO_ARIA_AMT` | 公开 checkpoint；README 不写入未发布的同口径统一 F1 | 适合常规纯钢琴 A/B。 |
| ByteDance Pedal | 钢琴专精 / 踏板感知 | `PIANO_BYTEDANCE_PEDAL` | MAESTRO note onset F1 / 踏板 onset F1 = 96.72% / 91.86% | 需要踏板 CC64 时优先选择；不会作为其他钢琴后端的静默替代。 |
| Leap XE + PolarFormer | 人声/伴奏分离 | `VOCAL_SPLIT` 前置分离 | 两个公开模型使用不同目标/口径，不合成单一 benchmark | Leap XE 提取 vocals，PolarFormer 提取 accompaniment；后续 MIDI 质量还取决于转写后端。 |
| BS-RoFormer SW Fixed | 六声部分离 | `SIX_STEM_SPLIT` 前置分离 | MVSEP 6-stem SDR 口径 | 分离后每个 stem 独立转写；分离指标不是端到端 MIDI F1。 |

YourMT3+ / MuScriptor / MIROS 属于多乐器后端，TransKun / Aria-AMT / ByteDance Pedal 属于钢琴专精后端，Leap XE / PolarFormer / BS-RoFormer SW Fixed 属于音源分离后端；三类模型的公开指标不能混成同一张排行榜。

### 当前默认转写模型：YourMT3+

本项目默认使用 **YPTF.MoE+Multi (noPS)**。原因不是猜测：官方 Hugging Face Space 的 `app.py` 默认 `model_name` 就是 `YPTF.MoE+Multi (noPS)`；`YPTF.MoE+Multi (PS)` 仍保留为可选 pitch-shift checkpoint，但不再写成项目默认。

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

官方 Hugging Face / Colab demo 的模型选择项是 checkpoint / 架构选择，不是本项目的处理工作流。当前项目中的 YourMT3+ 选择器与官方列表对齐；区别是本项目会把该模型嵌入 `SMART`、`VOCAL_SPLIT` 和 `SIX_STEM_SPLIT` 等工作流中，并统一输出 GM 多乐器 MIDI。

| 模型 | MoE | Pitch Shift | 说明 |
|------|-----|-------------|------|
| YMT3+ | 无 | 无 | 官方 Colab 模型族中的基线 YourMT3+ checkpoint |
| YPTF+Single (noPS) | 无 | 无 | Perceiver-TF + 单解码器 checkpoint |
| YPTF+Multi (PS) | 无 | 有 | Perceiver-TF + multi-t5 多通道解码 |
| YPTF.MoE+Multi (noPS) | 8 专家 | 无 | 本项目默认模型；官方 Hugging Face Space 默认模型；Space 结果文件中 Slakh `multi_f = 0.7398` |
| YPTF.MoE+Multi (PS) | 8 专家 | 有 | 可选 pitch-shift MoE checkpoint；YourMT3+ 论文表中最终模型 Slakh `Multi F1 = 74.84`；本地 PS checkpoint 约 723.8 MiB |
| YourMT3+ 传统版 | 无 | 无 | 旧配置兼容项，不在界面默认显示 |

与官方 demo 的主要对齐点：

- 五个官方模式名称、checkpoint 目录映射和 UI 顺序一致。
- `YPTF.MoE+Multi (noPS)` 对齐官方 Hugging Face Space 默认展示模型。
- 五种 checkpoint 都使用官方 Space 对应的参数表和官方 `update_config` 生成 tokenizer/model/audio 配置；旧 checkpoint 不再依靠缺失元数据的猜测值。
- 旧 T5 checkpoint 缺少 `ff_layer_type` 元数据时，本项目按标准 T5 前馈层 `t5_gmlp` 处理。

与官方 demo 的差异也需要明确：

- 官方 demo 默认只运行单一 YourMT3 checkpoint；本项目还提供音源分离、钢琴专用模型、tempo 元数据补齐和 stem MIDI 合并，但不会对官方 writer 的音符做二次清理。
- 官方 GPU Space 通常使用 16-bit 推理；本项目默认使用 full precision，以降低不同 Windows / CUDA 环境中的不稳定因素。
- 产品使用与官方 Space 相同的无重叠分段和 `inference_file(bsz=8)`；环境变量不再改写这条官方路线的 batch。

### 当前可选后端：MIROS

| 后端 | 类型 | 集成方式 | 当前语义 | 说明 |
|------|------|----------|----------|------|
| MIROS (MusicFM) | 多乐器 | 本地 `ai4m-miros` 仓库 + 当前工程包装器 | 固定 checkpoint 质量 | 官方仓库标注为 Music Transcription Challenge winning model，可作为 `SMART`、`VOCAL_SPLIT` 与 `SIX_STEM_SPLIT` 的多乐器后端 |

处理语义：

- 所有入口默认使用固定高质量处理策略。
- `MIROS` 当前为固定 checkpoint 推理，可用于与 YourMT3+ 做同任务 A/B。

### 当前人声分离模型：Leap XE vocals + PolarFormer accompaniment

`VOCAL_SPLIT` 的模型与输入输出契约对齐当前公开 TelkNet 工具：BS-RoFormer Leap XE 90-band 对原混音生成 vocals，BS PolarFormer public ONNX 也对原混音独立生成 accompaniment。两路规范 WAV 进入音轨工作台后，可分别选择五个 YourMT3+ checkpoint、MIROS、MuScriptor Large 或四个钢琴后端。

对齐边界：本轮经授权核验了私有 `mason369/telknet` 的 `dev` 提交 `52be6fec179be492f5229ba149545ac2833b284a`。当前工程只对齐其 YourMT3/MIROS“官方 writer 后只补 tempo、不做通用音符清理”的核心语义；本项目的两个分离主流程同样只交付 WAV，MIDI 由用户在逐轨工作台显式触发。没有证据证明该 `dev` 已部署线上，也不声称模式路由逐行一致、推理环境相同或输出文件位级一致。

| 项目 | 详情 |
|------|------|
| vocals 模型 | [BS-RoFormer Leap XE](https://huggingface.co/pcunwa/BS-Roformer-Leap)：`Xe/bs_leap_xe_voc.ckpt` + `Xe/leap_xe_config_voc.yaml` |
| accompaniment 模型 | [BS PolarFormer](https://huggingface.co/bgkb/bs_polarformer)：`bs_polarformer.onnx` + `model_bs_polarformer_float16.yaml` |
| 运行库 | Leap XE 使用 audio-separator 内的 BS-RoFormer 实现；PolarFormer 使用 ONNX Runtime |
| 调用方式 | 两个模型各自对原音频推理；Leap XE 输出 vocals，PolarFormer 的 vocals 估计从混音中相减得到 accompaniment |
| 模型准备 | `download_sota_models.py` 会准备并校验两组资源；也可分别运行 `download_vocal_model.py` 与 `download_accompaniment_model.py` |
| 兼容入口 | 历史文件 `download_vocal_harmony_model.py` 仍转发 PolarFormer accompaniment 下载；它不再表示 karaoke/和声链路 |
| 打包行为 | release 工作流会把 `~/.music-to-midi/models/audio-separator/` 打进便携包；运行时缺模型或校验失败会明确报错 |
| 输出选项 | 分离阶段输出 `<歌曲名>_vocals.wav` 与 `<歌曲名>_accompaniment.wav`；逐轨 MIDI 仅在用户勾选路线并点击转换后生成，不自动合并 |

这两个分离结果来自两次独立推理，不是用一个输出静默补出另一条路径；任一模型失败都会让 `VOCAL_SPLIT` 显式失败。

#### 人声分离模型对比

> 注：本表只保留这次重新核验时能找到公开来源支撑的结论。若写明“未写入数值”，表示没有找到与当前 checkpoint 明确绑定、且口径足够清晰的公开数值。

| 模型/方向 | 来源 | 类型 | 状态 | 说明 |
|-----------|------|------|------|------|
| Leap XE vocals + PolarFormer accompaniment（当前） | [Leap XE 模型仓库](https://huggingface.co/pcunwa/BS-Roformer-Leap) / [PolarFormer 模型仓库](https://huggingface.co/bgkb/bs_polarformer) | 本地 PyTorch + ONNX 双模型 | 使用中 | 模型与输入输出契约对齐当前公开 TelkNet 工具；不据此声称服务端源码或结果位级一致。两个模型目标不同，不把各自指标合成一个“总 SDR”。 |
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
| 六声部分离 + 逐轨显式转写 | `six_stem_split` 已在 pipeline、桌面 UI、Space 和 Colab 中开放 | `BS-Rofo-SW-Fixed.ckpt`（vocals, bass, drums, guitar, piano, other）+ 每条 WAV 独立选择 11 条转写路线 | MVSEP Algorithms #77 给出 6-stem SDR：vocals 11.30 / instrum 17.50 / bass 14.62 / drums 14.11 / guitar 9.05 / piano 7.83 / other 8.71 | 这些是音源分离 SDR，不是最终 MIDI 转写 F1；逐轨 AMT 的端到端质量没有公开统一 benchmark。 |
| 钢琴专用转写（TransKun 默认 V2） | `piano_transkun` 已在 pipeline、桌面 UI、Space 和 Colab 中开放 | `transkun==2.0.1`，使用该 wheel 随附并严格校验的资源 | 官方 model cards：TransKun V2 在 MAESTRO V3 上 Note Onset / Onset+Offset / Onset+Offset+Velocity F1 为 0.9832 / 0.9349 / 0.9296；pip 随包 No Ext checkpoint 为 0.9833 / 0.8149 / 0.8109 | 这是钢琴专精协议，适合纯钢琴；不能与 YourMT3+ 的 Slakh2100 多乐器 F1 直接横比。 |
| 钢琴专用转写（TransKun V2 Aug） | `piano_transkun_v2_aug` 已在 pipeline、桌面 UI、Space 和 Colab 中开放 | 官方 `checkpointTransformerAug.zip`，固定校验后加载 `checkpointMSimplerAug/checkpoint.pt` + `model.conf` | `Aug` 表示数据增强 checkpoint；README 不把其他 V2 checkpoint 的指标直接移植给它 | 与默认 V2 并列，供同一音频显式 A/B，不是失败回退。 |
| 钢琴专用转写（Aria-AMT） | `piano_aria_amt` 已在 pipeline、桌面 UI、Space 和 Colab 中开放 | EleutherAI `aria-amt`，公开 preliminary piano v1 checkpoint `piano-medium-double-1.0.safetensors` | 官方 README 提供安装、checkpoint 下载和 CLI 用法；未给出与 TransKun 同口径的 MAESTRO/MAPS benchmark。本地打包资源中的 checkpoint 约 425.9 MiB。 | 已集成为钢琴转写 A/B 选项，但 README 不写入不存在的统一分数；比较时应使用同一批本地音频。 |
| 钢琴专用转写（ByteDance Pedal） | `piano_bytedance_pedal` 已在 pipeline、桌面 UI、Space 和 Colab 中开放 | `piano-transcription-inference`，checkpoint `note_F1=0.9677_pedal_F1=0.9186.pth` | 论文报告 MAESTRO onset F1 96.72% 与 pedal onset F1 91.86%；本项目保留上游 MIDI 中的 sustain pedal CC64。 | 上游 ByteDance 主仓库已归档，推理包兼容性需在目标环境验证；不作为 TransKun / Aria-AMT 的静默替代。 |

### 未来可关注的转写模型

下列对比按 2026-07-19 的公开资料更新。`MuScriptor D_Test Multi F1`、`Slakh2100 Multi (Onset-Offset) F1`、`MAESTRO onset F1`、官方挑战名次、以及主观听感/下游任务增益并不是同一协议，不能当成同一张排行榜直接横比。

#### 多乐器模型（公开可核实）

| 模型 | 公开来源 | Benchmark / 协议 | 公开结果 | 状态 | 说明 |
|------|----------|------------------|----------|------|------|
| [MuScriptor Large](https://huggingface.co/MuScriptor/muscriptor-large) | [论文](https://arxiv.org/abs/2607.08168) / [代码](https://github.com/muscriptor/muscriptor) | 作者 `D_Test`，372 首真实多乐器曲目；完整训练，CFG=2 | Onset / Frame / Offset / Drums / Multi F1 = **60.4 / 72.4 / 48.6 / 49.6 / 47.8**；同表 YourMT3+ Multi F1 = 21.9 | 已集成 | 作者数据上提升很大；8 个公共跨域集的 Multi F1 赢 6、输 2，因此不写成所有协议的绝对 SOTA |
| MuScriptor Small / Medium | [官方代码与三档权重](https://github.com/muscriptor/muscriptor#models) | 论文 `D_Real` only、CFG=2 规模消融 | Small Multi F1 38.2；Medium 39.7；Large 40.5 | 未来候选 | 103M / 307M 权重已公开，适合评估低显存与 CPU；必须先做本地同音频 A/B，不会静默替代 Large |
| YPTF.MoE+Multi (noPS)（当前默认） | [官方 Space app.py](https://huggingface.co/spaces/mimbres/YourMT3/blob/main/app.py) / [Space noPS 结果文件](https://huggingface.co/spaces/mimbres/YourMT3/blob/main/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/result_mc13_full_plus_256_default_all_eval_final.json) | Slakh `multi_f` | **0.7398 / 73.98%** | 使用中 | 当前项目默认 YourMT3+ checkpoint；对齐官方 Hugging Face Space 默认项 |
| YPTF.MoE+Multi（论文表最终模型） | [YourMT3+ 论文](https://arxiv.org/abs/2407.04822) | Slakh2100 `Multi (Onset-Offset) F1` | **74.84**；同表 `MT3 = 62.0` | 论文公开结果 | 这是论文表中的最终模型口径，不把它写成当前 noPS 默认 checkpoint 的单独成绩 |
| [MT3](https://github.com/magenta/mt3) | [YourMT3+ 论文](https://arxiv.org/abs/2407.04822) / [Magenta 仓库](https://github.com/magenta/mt3) | Slakh2100 `Multi (Onset-Offset) F1` | **62.0** | 开源基线 | YourMT3+ 继承并扩展的 token-based 多乐器基线 |
| 2025 AMT Challenge 冠军 MIROS | [挑战论文](https://arxiv.org/abs/2603.27528) / [代码](https://github.com/amt-os/ai4m-miros) | 76 个约 20 秒、最多 3 种乐器的受约束合成片段；Multi Onset F1 | **0.5998**；YourMT3-YPTF-MoE-M 0.5938；MT3 0.3932 | 已集成 | MusicFM 编码器 + 多解码器；该挑战协议不能与 Slakh 或 MuScriptor `D_Test` 直接横比 |
| Mirelo Studio 改进版 | [Mirelo 官方文章](https://mirelo.ai/blog/turning-audio-to-midi) | 未公开 | 官方仅说明“使用更多数据训练、更准确” | 私有服务观察项 | 没有公开权重、revision、参数量或同协议分数；不是当前 `muscriptor-large` 权重，不能离线集成 |

#### 钢琴专精模型对比（公开可核实）

#### 钢琴同类模型质量对比

| 模型 | 当前项目入口 | 同类型质量口径 | 公开结果 | 使用判断 |
|------|--------------|----------------|----------|----------|
| TransKun V2 | 公开研究 checkpoint | MAESTRO V3 `note onset F1 / onset+offset F1 / onset+offset+velocity F1` | **0.9832 / 0.9349 / 0.9296** | 公开钢琴 AMT 分数很强，适合作为钢琴专精质量参考。 |
| TransKun pip checkpoint（No Ext） | `PIANO_TRANSKUN` | MAESTRO V3 No Ext 同口径三项指标 | **0.9833 / 0.8149 / 0.8109** | 项目默认 TransKun 路线；仓库说明它不带 pedal extension。 |
| TransKun V2 Aug | `PIANO_TRANSKUN_V2_AUG` | 官方数据增强 checkpoint；不套用其他 checkpoint 的指标 | 未写入跨 checkpoint F1 | 用同一批本地钢琴音频与默认 V2 显式 A/B。 |
| Aria-AMT | `PIANO_ARIA_AMT` | 官方公开 checkpoint，但未发布与 TransKun 完全同口径的统一榜单 | 未写入统一 F1 | 已集成的钢琴 A/B 选项；建议用同一批本地钢琴音频比较。 |
| ByteDance Pedal | `PIANO_BYTEDANCE_PEDAL` | MAESTRO `note onset F1 / 踏板 onset F1` | **96.72% / 91.86%** | 同类中优势是踏板事件，输出 MIDI 会保留 sustain pedal `CC64`。 |

YourMT3+ / MuScriptor / MIROS 属于多乐器后端，不能与上表的钢琴专精 F1 横向混排；ByteDance Pedal 的 `踏板 onset F1` 也不能直接等同于 TransKun 的 `onset+offset+velocity F1`。

| 模型 | 公开来源 | Benchmark / 协议 | 公开结果 | 状态 | 说明 |
|------|----------|------------------|----------|------|------|
| [TransKun V2（论文 checkpoint）](https://github.com/Yujia-Yan/Transkun) | [TransKun 官方仓库 / model cards](https://github.com/Yujia-Yan/Transkun) | MAESTRO V3 `note onset F1 / onset+offset F1 / onset+offset+velocity F1` | **0.9832 / 0.9349 / 0.9296** | 开源 | 这是论文公开 checkpoint 的模型卡结果；项目另设 V2 Aug 独立入口，不混用指标 |
| [TransKun pip 随包 checkpoint（No Ext）](https://github.com/Yujia-Yan/Transkun) | [TransKun 官方仓库 / model cards](https://github.com/Yujia-Yan/Transkun) | MAESTRO V3 No Ext 同口径三项指标 | **0.9833 / 0.8149 / 0.8109** | 开源 | 仓库明确写明随 pip 包 checkpoint 为 `without pedal extension of notes`；对应项目默认 `PIANO_TRANSKUN` |
| [Aria-AMT](https://github.com/EleutherAI/aria-amt) | [EleutherAI 官方仓库](https://github.com/EleutherAI/aria-amt) | 公开 checkpoint 发布 | 仓库公开 `piano-medium-double-1.0.safetensors`；但仓库页未给出与上表完全同口径的统一 MAESTRO/MAPS 榜单 | 开源 | 已集成为钢琴 A/B 选项；这里不伪造不存在的统一 benchmark 行 |
| [High-Resolution Piano Transcription with Pedals by Regressing Onset and Offset Times](https://arxiv.org/abs/2010.01815) | [论文](https://arxiv.org/abs/2010.01815) / [ByteDance 仓库](https://github.com/bytedance/piano_transcription) | MAESTRO `onset F1 / pedal onset F1` | **96.72% / 91.86%** | 论文 + 代码 | 代表性踏板感知钢琴论文；协议是钢琴专精口径，不应与多乐器 Slakh 分数混排 |

#### 论文阶段 / 协议不一致的研究方向

| 模型/方向 | 公开来源 | 公开协议 / 任务 | 可核实的公开信息 | 为什么不与上表混成同一分数榜 |
|-----------|----------|-----------------|------------------|------------------------------|
| 密集复音与乐器检测 | [2025 AMT Challenge 论文](https://arxiv.org/abs/2603.27528) | 1/2/3 乐器分组分析 | MIROS 从 1 种到 3 种乐器时 F-measure 从 0.7193 降到 0.4367；论文把 polyphony、相似音色、乐器泄漏列为主要失败模式 | 这是失败模式和未来评测方向，不是一个可下载的新 checkpoint |
| [MR-MT3](https://arxiv.org/abs/2403.10024) | [论文](https://arxiv.org/abs/2403.10024) / [代码](https://github.com/gudgud96/MR-MT3) | Slakh2100；重点看 `onset F1`、`instrument leakage ratio`、`instrument detection F1` | 摘要明确写的是“improved onset F1 scores and reduced instrument leakage” | 它主打 leakage 抑制，并引入了新指标；不等于上面的 Slakh `Multi (Onset-Offset) F1` |
| [Jointist](https://arxiv.org/abs/2302.00286) | [论文](https://arxiv.org/abs/2302.00286) | 流行音乐联合转写 + 分离 | 摘要给出的公开结果是：转写提升 `>1 ppt`、分离提升 `+5 SDR`、downbeat `+1.8 ppt`、和弦/调性各 `+1.4 ppt` | 它是 joint transcription + separation 路线，公开协议与 Slakh / MAESTRO 完全不同 |
| MusicFM 编码器 + AMT 解码器 | [MusicFM 论文](https://arxiv.org/abs/2311.03318) / [仓库](https://github.com/minzwon/musicfm) / [HF 权重](https://huggingface.co/minzwon/MusicFM) | 预训练编码器迁移 | 公开的是基础编码器权重；通用可复现的完整 AMT decoder / 微调流水线并未作为现成后端发布 | 它更像 MIROS 这类路线背后的表示学习部件，不是拿来就能切换的通用后端 |
| [CountEM / Count The Notes](https://arxiv.org/abs/2511.14250) | [论文](https://arxiv.org/abs/2511.14250) / [项目页](https://yoni-yaffe.github.io/count-the-notes) / [代码](https://github.com/Yoni-Yaffe/count-the-notes) | 弱监督 AMT 训练方法 | 公开论文、代码和模型，核心贡献是“用音符直方图 + EM”替代精确对齐监督 | 这是训练范式创新，不是固定 checkpoint 的 turnkey 后端 |
| [PerceiverTF](https://arxiv.org/abs/2306.10785) | [论文](https://arxiv.org/abs/2306.10785) | 多乐器公开数据集（论文自有协议） | 摘要只明确说其在多个公开数据集上优于 MT3 / SpecTNT | 它更适合作为 YourMT3+ 的架构祖先来理解，不应和上表的统一数值行硬拼 |

补充说明：

- [Basic Pitch](https://github.com/spotify/basic-pitch) 依然是很有价值的轻量方案，但它不发布与上表同口径的 Slakh/MAESTRO 综合榜单。
- [Omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) 仍是有参考价值的多任务工具链，但其 GitHub latest release 仍为 `0.5.0`（2021-12-09），与当前多乐器/钢琴专精 SOTA 的公开比较协议并不一致。

趋势总结：截至 2026-07-19，多乐器 AMT 已形成三条清晰路线：`MT3 / YourMT3+ / MR-MT3` 的 token-based 架构演进、MIROS 的 MusicFM 预训练编码器路线，以及 MuScriptor 依靠大规模真实数据与 RL 后训练的 decoder-only 路线。下一阶段最重要的不是再堆一个不可比的单分数，而是改善密集复音、相似音色乐器泄漏、稀有乐器和真实 jazz/pop 泛化，并同时公布可复现权重、许可、速度与显存数据。钢琴 AMT 的公开成熟度仍然更高，但 `TransKun pip 权重`、`V2 Aug checkpoint`、`论文 checkpoint`、`pedal-aware 论文系统` 之间的协议差异必须写清楚，不能简单合并成一个“钢琴榜单”。

## 默认处理策略

桌面版、Space 和 Colab 不再提供可调质量入口。YourMT3+ 产品路线使用官方无重叠分段、固定 `bsz=8`、逐解码通道 detokenize/merge、`mix_notes` 和官方 MIDI writer；MIROS 保留官方 CLI writer 输出；MuScriptor 使用官方分段生成、事件流与 MIDI writer。项目不追加重叠分段去重、稀疏音色过滤或本地 MIDI 重新生成。

## 失败处理原则

本项目的默认原则是：**真实失败必须显式暴露，不用静默回退、假成功或默认值制造“看起来完成”的结果**。

应直接失败并提示根因的情况：

- 模型、checkpoint、上游源码或必需 Python 包缺失。
- FFmpeg / 音频解码失败。
- YourMT3+、MIROS、MuScriptor、TransKun 默认 V2 / V2 Aug、Aria-AMT、ByteDance Pedal 或分离模型推理失败。
- 分离器没有生成完整目标文件。
- 模型下载遇到证书、网络或完整性问题。

当前实现：

| 场景 | 当前行为 |
|------|----------|
| FFmpeg 转 WAV 失败 | 直接报错，保留 FFmpeg stderr；不再尝试 librosa/soundfile 替代转换 |
| audio-separator CUDA 不兼容 | 直接报错并说明 GPU/CUDA/PyTorch 不兼容；不再自动改用 CPU 分离 |
| BPM 检测失败 | 直接报错；不再写入默认 120 BPM |
| YourMT3+ 显存不足 | 官方 `inference_file(bsz=8)` 路线直接报错并保留 CUDA 根因；不自动改 batch 伪装为同一次官方处理 |
| 配置非法模式/后端 | 未知值直接抛出校验错误；仅旧 `piano` 模式会兼容映射为 `smart` |
| 独立 vocal stem 的官方 writer 输出任意 program | 原样保留 program、鼓轨、控制器与音符；不再筛选 100/101/0，也不重映射为单一 vocal program |
| SSL 证书验证失败 | 默认停止下载并提示配置 CA/代理；只有显式设置 `ALLOW_INSECURE_HF_DOWNLOAD=1` 才会跳过验证 |

如果未来实现新的 fallback，必须同时说明：

- 触发条件。
- 用户可见行为。
- 输出结果风险。
- 日志或 UI 中的明确标记。

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.11+；Windows 安装脚本支持并优先选择 3.11-3.12 |
| PyTorch | 桌面/便携安装基线为 2.7.0，必须与 `torchaudio`、`torchvision` 版本匹配 |
| FFmpeg | 必需；用于可靠处理 MP3/M4A/FLAC/OGG 等格式 |
| GPU | 部分源码转写路线可在 CPU 上运行；完整七模式体验与完整便携发布包要求兼容的 GPU 运行时 |
| 系统 | Windows 10/11、Linux、WSL2 |

各平台使用各自经过固定的兼容运行时，不应把一个平台的 NumPy/Torch 版本强行覆盖到另一个平台：

| 平台 | Python / Torch | NumPy 与 GPU 运行时 | 发布边界 |
|------|----------------|---------------------|----------|
| Windows / NVIDIA 桌面与便携目标 | Python 3.11-3.12；Torch 2.7.0 / torchaudio 2.7.0 / torchvision 0.22.0 | NumPy 1.26.4；CUDA 12.8 wheel | 源码运行按此契约校验；当前 `release.yml` 因第三方许可未清零而不生成便携成品 |
| Linux / NVIDIA 源码运行 | Python 3.11+；Torch 2.7.0 / torchaudio 2.7.0 / torchvision 0.22.0 | NumPy 1.26.4；NVIDIA 驱动兼容 CUDA 12.8；仅 `cu128` | `install.sh` / `run.sh` 对完整七模式执行精确运行时校验；`build.yml` 只做源码、测试和打包契约检查 |
| Linux / AMD/ROCm | 不提供完整七模式兼容运行时 | PolarFormer 固定依赖 ONNX Runtime `CUDAExecutionProvider` | 当前不支持；安装脚本会明确停止，不静默改用 CPU |
| Hugging Face Space | Python 3.12.12；Torch 2.8.0 / torchaudio 2.8.0 / torchvision 0.23.0 | NumPy `>=2,<2.5`；ZeroGPU | 使用 `space/requirements.txt`，不可套用桌面 NumPy 1.26 环境 |
| Google Colab | Colab 当前预装 Python/Torch | 保留预装 Torch；只安装 pinned Web/runtime 依赖 | 避免替换 Torch 导致 CUDA 运行库冲突 |

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

`run.ps1` 会检查虚拟环境、核心依赖、五种 YourMT3+ checkpoint、BS-RoFormer SW Fixed、Leap XE、PolarFormer、TransKun V2 Aug、Aria-AMT、ByteDance Pedal 与 MIROS；资源缺失或身份校验失败时会调用 `install.ps1`。

### Linux / WSL2

```bash
chmod +x run.sh
./run.sh
```

`run.sh` 会检查虚拟环境、核心依赖、受控 YourMT3+ 源码与五种 checkpoint、BS-RoFormer SW Fixed、Leap XE、PolarFormer、TransKun V2 Aug、Aria-AMT、ByteDance Pedal 与 MIROS；资源缺失或身份校验失败时会调用 `install.sh`。

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

CUDA 12.8（完整七模式受支持且由启动器严格校验的运行时）:

```bash
pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

`cu118` / CUDA 11 不属于当前一键启动器和完整七模式验收契约；启动器不会把它静默当成已对齐环境。

AMD/ROCm 当前不能完成七模式：即使 PyTorch 提供 ROCm wheel，PolarFormer 仍固定依赖 ONNX Runtime `CUDAExecutionProvider`。安装脚本会明确停止，不会静默改用 CPU；完整七模式目前只验收 NVIDIA CUDA。

`release.yml` 的目标便携包只规划 CUDA 12.8 GPU 版本，不规划 CPU 版本；但当前 22 项第三方组件闭集清单中仍有 13 项 `BLOCKED`，工作流会在构建前失败，因此目前没有可声明为合规的最新便携成品。push / PR 的 `build.yml` 仅验证源码、测试与打包契约，不生成便携成品。本地源码开发如需 CPU-only PyTorch，应自行承担模型速度和依赖兼容性差异。

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
python -m pip install --no-deps "audio-separator==0.44.1"
python -m pip install --no-deps --force-reinstall "aria-amt @ https://github.com/EleutherAI/aria-amt/archive/a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip"
python -m pip install --no-deps --force-reinstall "https://github.com/muscriptor/muscriptor/archive/302343e8992bdfc619f77f1988168374ed5d675d.zip"
```

`requirements.txt` 有意不让 `audio-separator` 的 NumPy 2 元数据覆盖桌面 NumPy 1.26，也不让 Aria-AMT 或 MuScriptor 的服务端依赖覆盖当前 Torch/FastAPI 运行时。上述三项因此必须按固定版本以 `--no-deps` 单独安装；需要完整伴随依赖时优先运行 `install.ps1` / `install.sh`。

### 4. 准备 YourMT3+ 源码与模型

```bash
python download_sota_models.py
```

当前仓库已经包含受控且经过兼容补丁的 `YourMT3/amt/src`；不要在这里克隆可变上游 `master` 覆盖它。`download_sota_models.py` 会准备全部五种官方 YourMT3+ checkpoint、固定 MIROS 源码与两组权重、`BS-Rofo-SW-Fixed.ckpt`、Leap XE、PolarFormer、TransKun V2 Aug、Aria-AMT 与 ByteDance checkpoint，并严格校验默认 TransKun 2.0.1 包版本和内置 V2 资源；缺失或身份不符会直接失败。

### 5. 准备分离与钢琴模型

```bash
python download_vocal_model.py
python download_multistem_model.py
python download_accompaniment_model.py
python download_transkun_v2_aug_model.py
python download_aria_amt_model.py
python download_bytedance_piano_model.py
python download_miros_model.py
```

模型默认缓存到：

```text
~/.cache/music_ai_models/yourmt3_all
~/.music-to-midi/models/audio-separator
~/.cache/music_ai_models/transkun_v2_aug
~/.cache/music_ai_models/aria_amt
~/.cache/music_ai_models/bytedance_piano
external/ai4m-miros
```

默认 TransKun V2 的模型资源随 `transkun==2.0.1` 安装；若 `PIANO_TRANSKUN` 提示资源或身份不符，请执行 `python -m pip install --force-reinstall "transkun==2.0.1"`。`PIANO_TRANSKUN_V2_AUG` 使用独立缓存，必须运行 `python download_transkun_v2_aug_model.py`。

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

Space 版随部署包携带项目已验证的 `YourMT3/amt/src` 兼容源码，与桌面版和 Colab 使用同一棵源码树；不会在运行时改用 Hugging Face Space 的可变源码。运行转换时会按所选模式检查/准备 YourMT3+ 官方 checkpoint 或 MIROS、BS-RoFormer SW Fixed、Leap XE、PolarFormer、TransKun V2 Aug、Aria-AMT 或 ByteDance Pedal 资源；缺失或身份校验失败会显式暴露。

ZeroGPU 入口只承诺短片段试用，不承诺完整长歌端到端完成。[Hugging Face ZeroGPU 文档](https://huggingface.co/docs/hub/main/en/spaces-zerogpu) 当前公开配额为匿名用户每日 2 分钟、登录免费账户每日 5 分钟 GPU。当前保守的最小请求经 `large` GPU 平台倍率折算后已高于匿名额度，因此转换必须先登录；Space 会按模式、后端和模型估算，再按固定的 `spaces==0.51.0` 平台倍率上界折算，超过登录免费账户 300 GPU 秒窗口的请求会在下载模型前明确拒绝。该估算只是准入上限，不代表用户一定仍有足够的当日配额或队列容量；长歌请改用 Colab、桌面版或专用 GPU。

当前公式下的最大输入时长是准入阈值，不是实测耗时承诺。默认 `YPTF.MoE+Multi (noPS)` 与 MIROS 的精确阈值如下；换用其它 YourMT3 checkpoint 时会按其独立系数重新计算：

| ZeroGPU 路线 | YourMT3 默认 noPS | MIROS |
|--------------|------------------:|------:|
| `SMART` | 2.00 秒 | 1.00 秒 |
| `VOCAL_SPLIT` | 0.53 秒 | 0.27 秒 |
| `SIX_STEM_SPLIT` | 0.22 秒 | 0.11 秒 |
| 任一钢琴专用模式 | 2.50 秒 | 不适用 |

Space 的失败请求会立即删除专属输出目录；成功结果会保留给 Gradio 下载，默认在 24 小时后进入过期清理，并在后续请求删除过期结果或在 Space 实例正常退出时清理。Colab 同样在失败时立即清理，并把成功结果保留到当前运行时结束；Gradio 缓存按 24 小时进入清理。

## 便携版打包

> 当前发布状态：仅保留并验证未来便携构建流程；由于 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 中仍有模型权重、补丁源码和运行时再分发许可未解决，官方 release 门禁会显式阻断。下面的本地构建命令不等于获得再分发授权。

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

打包脚本要求并严格校验以下全部资源；缺少任一项，或大小、SHA256、源码 manifest、运行时包身份不符时，会在 PyInstaller 前立即失败：

```text
YourMT3/amt/src
YourMT3 模型缓存 -> models/yourmt3_all
audio-separator 模型缓存 -> models/audio-separator
transkun==2.0.1 包及其内置默认 V2 资源
TransKun V2 Aug 模型缓存 -> models/transkun_v2_aug
Aria-AMT 模型缓存 -> models/aria_amt
ByteDance Piano 模型缓存 -> models/bytedance_piano
固定版本且已打兼容补丁的 MIROS 源码与两组权重
ffmpeg.exe / ffprobe.exe
```

便携版资源来源优先级：

```text
MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR 或 ~/.cache/music_ai_models/yourmt3_all 或 checkpoints/yourmt3_all
MUSIC_TO_MIDI_BUNDLE_AUDIO_SEPARATOR_DIR 或 ~/.music-to-midi/models/audio-separator 或 checkpoints/audio-separator
MUSIC_TO_MIDI_BUNDLE_TRANSKUN_V2_AUG_DIR 或 ~/.cache/music_ai_models/transkun_v2_aug 或 checkpoints/transkun_v2_aug
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
    transkun_transcriber.py  # TransKun 默认 V2 钢琴专用后端
    transkun_v2_aug_transcriber.py # TransKun V2 Aug 钢琴专用后端
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
download_sota_models.py      # 全部公开工作流模型下载 + 默认 TransKun 严格校验
download_vocal_model.py      # Leap XE vocals 模型下载
download_accompaniment_model.py # PolarFormer accompaniment 下载入口
download_multistem_model.py  # BS-RoFormer SW Fixed 六声部分离模型下载
download_transkun_v2_aug_model.py # TransKun V2 Aug 下载与校验
download_aria_amt_model.py   # Aria-AMT 模型下载
download_bytedance_piano_model.py # ByteDance Pedal 模型下载
download_vocal_harmony_model.py # PolarFormer 历史兼容入口
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
python download_sota_models.py
```

如果受控的 `YourMT3/amt/src` 缺失，请重新取得当前项目版本中的该目录；不要用可变上游 `master` 覆盖，否则三端源码一致性与便携 manifest 校验均无法保证。

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

YourMT3+ 官方路线固定调用 `inference_file(bsz=8)`。如果显存不足，处理会直接停止并显示 CUDA 根因；不会在同一条“官方对齐”路线里静默改变 batch。

### 人声分离不可用

确认依赖和模型：

Windows / Linux NVIDIA CUDA 环境：

```bash
python -m pip install --no-deps "audio-separator==0.44.1" "onnxruntime-gpu==1.23.2"
python download_vocal_model.py
python download_accompaniment_model.py
```

macOS 或明确的 CPU 环境把 `onnxruntime-gpu==1.23.2` 换成 `onnxruntime==1.23.2`。

如果当前 NVIDIA GPU、PyTorch/CUDA 或 audio-separator 后端不兼容，流程会直接失败并提示根因；不会自动改用 CPU 分离。需要 CPU 路径时，应由后续 UI/配置提供显式选项并在结果中标注。

### 六声部分离不可用

确认 `audio-separator==0.44.1` 已安装，并下载 BS-RoFormer SW Fixed 资源：

```bash
python download_multistem_model.py
```

如果六声部分离缺少任一 stem，流程应失败并提示缺失项；不要把缺失 stem 当作空音轨静默合并。

### 钢琴专用转写不可用

默认 TransKun 模式需要 `transkun` 包和其随包预训练资源：

```bash
python -m pip install --force-reinstall "transkun==2.0.1"
```

TransKun V2 Aug 模式使用独立、固定校验的官方 checkpoint：

```bash
python download_transkun_v2_aug_model.py
```

Aria-AMT 模式需要 `aria-amt` 包和 checkpoint：

```bash
python -m pip install --no-deps --force-reinstall "aria-amt @ https://github.com/EleutherAI/aria-amt/archive/a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip"
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

未知的处理模式、转写后端或多乐器模型会直接报错，不会自动改成默认值。请确认配置值属于：

```text
processing_mode: smart / vocal_split / six_stem_split / piano_transkun / piano_transkun_v2_aug / piano_aria_amt / piano_bytedance_pedal
transcription_backend: aria_amt / yourmt3 / miros / muscriptor
multi_instrument_model: yourmt3 / miros / muscriptor
```

旧配置中的 `processing_mode=piano` 会兼容映射为 `smart`。

## 许可证

本项目使用 MIT License。第三方模型、数据和上游仓库遵循各自许可证与使用条款；代码改编与完整声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
