# 音乐转 MIDI 转换器

<p align="center">
  中文 | <a href="./README.md">English</a>
</p>

一个基于 AI 的音频转 MIDI 工具，提供 PyQt6 桌面版、Gradio Web 版和 Google Colab 运行入口。当前版本同步七种处理模式：完整混音多乐器转写、人声/伴奏 WAV 分离、六声部 WAV 分离，以及 TransKun 默认 V2 / TransKun V2 Aug / Aria-AMT / ByteDance Pedal 四条钢琴专用转写流程。两个分离模式会生成 WAV 分轨，每条音轨都可在同一结果工作台选择 13 条 MIDI 路线之一。

## 独立 Web API 与浏览器前端

独立 Web API 与浏览器前端调用同一个 `MusicToMidiPipeline`，提供与桌面版一致的七种模式。Windows 发布版按用途提供三个目录：

| 包 | 启动文件 | 用途 |
|----|----------|------|
| `MusicToMidi-App` | `MusicToMidi.exe` | 本机桌面 App |
| `MusicToMidi-WebBackend` | `MusicToMidiBackend.exe` | GPU 推理、任务队列、结果文件与 API |
| `MusicToMidi-WebFrontend` | `MusicToMidiFrontend.exe` | 浏览器界面和静态资源，不包含推理模型 |

源码环境在仓库根目录使用一条命令启动 Web 前后端：

```powershell
.\venv\Scripts\python.exe -m src.web
```

该入口会自动识别服务电脑的主要局域网 IPv4，依次启动 WebBackend 和 WebFrontend，确认两端就绪后自动打开 Edge 应用窗口。终端显示的 Web 地址可由同一局域网的其他电脑直接访问；按 `Ctrl+C` 或关闭 Edge 应用窗口会一并停止前后端。多网卡或 VPN 环境可明确指定服务地址：

```powershell
.\venv\Scripts\python.exe -m src.web --host 192.168.1.50
```

其他电脑通过浏览器访问终端显示的 `http://<服务电脑局域网IPv4>:5173`。服务电脑的 Windows 防火墙规则把 TCP `5173` 和 `8765` 限制在受信任局域网内。

打包版的启动顺序是 WebBackend 包中的 `MusicToMidiBackend.exe`，随后是 WebFrontend 包中的 `MusicToMidiFrontend.exe`。同机使用默认配置即可；局域网使用时，两个 EXE 旁边首次运行生成的 `MusicToMidiBackend.json` 和 `MusicToMidiFrontend.json` 保存网络配置，重启后由其他电脑访问前端地址。其他电脑只需要浏览器，不需要安装任何包。后端健康检查为 `http://<后端地址>:8765/api/v1/health`，API 文档为 `http://<后端地址>:8765/docs`。

浏览器前端通过 multipart 作业接口提交音频，查询 `GET /api/v1/jobs/<job-id>` 获取真实终态，并从响应中的 `download_url` 下载 MIDI、乐谱 ZIP 或分轨 WAV。`POST /api/v1/jobs/<job-id>/sheet-music` 为一个明确的 MIDI 制品生成对应乐谱。前后端核对 API 2.0，浏览器以 5 秒心跳更新真实在线状态；后端默认保留 30 天、最多 200 条终态任务，删除任务及相关文件前会再次确认。桌面、Web、Space 与 Colab 一次只运行一个加速器任务，其他任务会等待。失败会返回明确错误，不会用简化算法或静默回退伪装成功。

独立 Web 版本面向受信任局域网，不内置认证、授权或 TLS；直接暴露到互联网会形成无认证服务。完整的配置 JSON、防火墙命令、连通性验证、停止方式和前后端分机部署说明见 [web/README.md](../web/README.md)。

`v1.6.0` 已为 Issue #9 提供完整 Docker 自托管包，包括无外置模型的后端/网关镜像、digest 固定的 Compose、环境模板、Windows/Linux 管理脚本、镜像清单与 SHA-256。默认入口是 `http://127.0.0.1:7860`；模型按 `.env` 中选择的配置下载到持久卷，常驻推理容器离线运行。安装、配置、日志、备份、升级与故障排查见 [Docker 自托管指南](docker-deployment.md)。

## 统一界面演示

桌面版、Gradio Web 版和 Google Colab 采用同一套七模式工作流与操作语义。以下演示按“主界面 → 分离完成 → 逐轨处理 → MuScriptor 渐进式预览”的顺序展示核心流程。

### 1. 主界面与完整混音转写

![主界面与完整混音转写](../resources/screenshots/01-main-interface.png)

### 2. 六声部分离完成

![六声部分离完成](../resources/screenshots/02-six-stem-separation-result.png)

### 3. 六声部波形与逐轨转 MIDI 控件

![六声部波形与逐轨转 MIDI 控件](../resources/screenshots/03-six-stem-track-controls.png)

### 4. MuScriptor 边转写边预览 MIDI

![MuScriptor 分片转写、可播放进度与钢琴卷帘预览](../resources/screenshots/04-muscriptor-progressive-midi-preview.png)

## 当前能力

| 范围 | 当前行为 |
|------|----------|
| 完整混音 | `SMART` 读取整首音频，可选 YourMT3+、MIROS 或 MuScriptor Large / Medium / Small，输出含音符、鼓点和 GM 乐器分组的 MIDI。默认 checkpoint 是 YourMT3+ 官方 `YPTF.MoE+Multi (noPS)`。 |
| 音源分离 | `VOCAL_SPLIT` 用 Leap XE 90-band 与 PolarFormer 生成两条 WAV；`SIX_STEM_SPLIT` 用 `BS-Rofo-SW-Fixed.ckpt` 生成 `bass / drums / guitar / piano / vocals / other` 六条 WAV。分离后，每条 WAV 可独立选择 13 条 MIDI 路线并点击转换。 |
| 钢琴转写 | `PIANO_TRANSKUN`、`PIANO_TRANSKUN_V2_AUG`、`PIANO_ARIA_AMT` 和 `PIANO_BYTEDANCE_PEDAL` 分别调用 TransKun 默认 V2、官方 V2 Aug、Aria-AMT 和 ByteDance 带踏板模型。 |
| MuScriptor 乐器约束 | 空选时由模型检测乐器；非空选择会传入官方 `instruments` 与 `prelude_forcing` 接口，生成阶段屏蔽未选 token，并校验事件流和最终 MIDI。越界结果不会发布。 |
| 乐谱导出 | 每个真实 MIDI 结果（包括分离后逐轨转换）都可显式生成独立 ZIP；只对私有副本量化，不修改原 MIDI。ZIP 含量化 MIDI、MusicXML、总谱、逐乐器分谱及符合 4–9 弦识别条件的 Tab PDF。桌面、Space、Colab 使用当前编辑器网格，独立 Web/API 使用 `1/32`；MuseScore Studio 4 缺失或失败会直接报错。 |
| 节拍与速度 | 七种模式和逐轨转写都使用 Beat This `final0`。拍点清理后以全局最小二乘拟合 BPM，下拍独立决定拍号；证据不足时不写 4/4。默认自动写入检测到的唯一 BPM；也可生成稳定的段落级 tempo map。手动 30–300 BPM 会保留检测 BPM 的音乐 tick 并覆盖工程速度。模型事件映射到音乐网格，但不做量化。 |
| MIDI 内容 | YourMT3+ 与 MIROS 保留官方 writer 的音符、音色、力度、控制器和弯音消息，只在缺少 `set_tempo` 时按检测 BPM 保持绝对秒并补写 tempo。MuScriptor 使用官方事件与 writer，并校验所选乐器集合。项目不添加去重、短音符过滤、力度平滑、复音限制或 `NoteEvent` 重建。 |
| 试听与 DAW | MuScriptor 工作台以 MIDI 合成轨为主时钟，同步 MIDI、原音和乐器分轨，并校正超过 80 ms 的漂移。DAW 导入时需启用 tempo map；MuseScore 3/4 可能重新跟拍未量化演奏并改写乐谱页显示 BPM，但不会改变文件中已经校验的 tempo。 |
| 输入与入口 | 支持 `MP3`、`WAV`、`FLAC`、`OGG`、`M4A`；非 WAV 通过 FFmpeg 转为 44.1 kHz PCM WAV。桌面版、Space 与 Colab 提供相同的七种处理模式。 |

## 不同入口的功能范围

| 入口 | 处理模式 | 后端选择 | 适合场景 |
|------|----------|----------|----------|
| PyQt6 桌面版 | `SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT`、`PIANO_TRANSKUN`、`PIANO_TRANSKUN_V2_AUG`、`PIANO_ARIA_AMT`、`PIANO_BYTEDANCE_PEDAL` | SMART 可选 YourMT3+ / MIROS / MuScriptor；分离结果逐轨选择 13 条路线；钢琴模式使用各自固定后端 | 本地长期使用、GPU 推理、批量输出文件、钢琴专用转写 |
| 独立 Web API | 同桌面七种模式 | multipart 作业、终态轮询和制品下载；推理仍由同一 `MusicToMidiPipeline` 执行 | 自建 Web 前端、局域网服务或系统集成 |
| Gradio Space | 同桌面七种模式 | 同步提供 MuScriptor 乐器硬约束与 13 条逐轨路线 | 浏览器中快速试用或部署 |
| Google Colab | 同桌面七种模式 | 与 Space 同步 MuScriptor 约束和逐轨结果工作台 | 临时使用 Colab GPU |

## 处理模式

| 模式 | 处理流程 | 主要输出 | 说明 |
|------|----------|----------|------|
| `SMART` | 音频 -> 所选 YourMT3+ / MIROS / MuScriptor Large、Medium 或 Small -> MIDI | `<歌曲名>.mid` | 不做音源分离；MuScriptor 非空乐器选择会成为真实解码约束。 |
| `VOCAL_SPLIT` | 音频 -> Leap XE vocals + PolarFormer accompaniment -> 两条 WAV -> 逐轨显式转 MIDI | `<歌曲名>_vocals.wav`、`<歌曲名>_accompaniment.wav`；按需生成逐轨 MIDI | 分离阶段不自动转 MIDI，每条 WAV 可独立选择 13 条路线。 |
| `SIX_STEM_SPLIT` | 音频 -> `BS-Rofo-SW-Fixed.ckpt` -> 六条 WAV -> 逐轨显式转 MIDI | `<歌曲名>_<stem>.wav`；按需生成逐轨 MIDI | 每条真实 WAV 的路线和是否转换均由用户明确选择，不自动合并 MIDI。 |
| `PIANO_TRANSKUN` | 音频 -> TransKun 默认 V2 模型 -> MIDI | `<歌曲名>_piano_transkun.mid` | 适合纯钢琴音频；使用 PyPI 包随附 checkpoint。 |
| `PIANO_TRANSKUN_V2_AUG` | 音频 -> 官方 TransKun V2 Aug checkpoint -> MIDI | `<歌曲名>_piano_transkun_v2_aug.mid` | 独立模式，不会在默认 V2 失败时静默接管；需要独立下载并校验 V2 Aug 资源。 |
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
song_vocals.wav
song_accompaniment.wav
song_bass.wav
song_drums.wav
song_guitar.wav
song_piano.wav
song_other.wav
```

实际文件数量取决于所选模式和用户主动执行的逐轨转换。人声分离主流程只暴露规范的 `vocals` 与 `accompaniment` WAV；六声部主流程只交付六个真实分离 WAV。MIDI 仅为用户点击转换的音轨单独生成。

## 后端说明

### YourMT3+

YourMT3+ 是默认多乐器后端。`download_sota_models.py` 会准备 Beat This `final0`、五种官方 YourMT3+ checkpoint、固定 MIROS 源码与两组权重、MuScriptor Large / Medium / Small、`BS-Rofo-SW-Fixed.ckpt`、Leap XE、PolarFormer、TransKun V2 Aug、Aria-AMT、ByteDance、MuseScore General SoundFont、FluidSynth 与固定 MuseScore Studio 4.7.4，并严格校验默认 TransKun 2.0.1 包及其内置 V2 资源；YourMT3 推理通过 `src/core/yourmt3_transcriber.py` 调用仓库内受控的 `YourMT3/amt/src` 源码。

需要满足：

```text
YourMT3/amt/src/model/ymt3.py
YourMT3/amt/src/utils/task_manager.py
YourMT3/amt/src/config/config.py
```

完整项目 checkout 已包含经过兼容补丁并由固定 manifest 校验的 `YourMT3/amt/src`。该目录缺失时，受支持的恢复来源是当前项目版本中的受控源码；可变的上游 `master` 只适合独立实验，不满足三端源码一致性和便携构建身份契约。

模型权重下载：

以下源码维护命令以仓库内 `venv` 为运行环境；未激活时，Windows 的完整解释器路径是 `.\venv\Scripts\python.exe` / `.\venv\Scripts\hf.exe`，Linux/WSL2 为 `./venv/bin/python` / `./venv/bin/hf`。全局 Python 不属于受支持的源码运行环境。

```bash
python download_sota_models.py
```

默认搜索模型位置包括：

```text
~/.cache/music_ai_models/yourmt3_all
runtime/models/yourmt3_all          # 便携版
models/yourmt3_all                  # 打包资源
```

### MuScriptor Large / Medium / Small

项目严格固定上游官方 `v0.3.0` 提交 `d73147e75e5b9b0c0a79ebe154587db4fd603e0c`，并对七个运行时源码文件执行 SHA-256 身份校验，不叠加未合并分支。全项目唯一的 Beat This `final0` 分析把完整拍点、BPM、可靠拍号和首个强拍传给官方 v0.3.0 BeatGrid；官方 onset 相位估计只在满足样本数与集中度阈值时校正，否则明确记录 0 校正。项目不启动第二套节拍检测，也不接受占位 120 BPM。tempo 归一化后复制到每个含音符轨以兼容 MuseScore，桌面、Space 和 Colab 使用相同网格显示拍线、强拍和交替小节。

三档 gated 权重分别固定为 [`muscriptor-large`](https://huggingface.co/MuScriptor/muscriptor-large) revision `8809fdfbed2affa7ade94a7059e746e3880720e7`、[`muscriptor-medium`](https://huggingface.co/MuScriptor/muscriptor-medium) revision `f32236969308476e01fd3aae67357de5feb05a2d`、[`muscriptor-small`](https://huggingface.co/MuScriptor/muscriptor-small) revision `8c127f603b807520fa465c838e9bfee8a91ada4e`。三者均采用 CC BY-NC 4.0 并附额外合法使用条件；有效授权来自同一个 Hugging Face 账户在浏览器中逐项接受三个仓库的条款，随后登录 CLI。`hf auth login` 不能代替网页接受条款，源码安装、Colab 与自行部署 Space 也无法匿名全自动下载：

```bash
hf auth login
python download_muscriptor_model.py --size all
```

三档都是独立显式选择，不会互相静默替代；Large 质量优先，Medium 平衡速度与质量，Small 参数最少、速度最快。三档均固定使用官方 5 秒窗口、prelude forcing 和单次生成路径。Large 是约 1.3B 参数的 decoder-only Transformer，以 5 秒、16 kHz 单声道分片生成 onset、offset、pitch 和 36 组乐器事件。训练包含约 145 万 MIDI 合成预训练、17 万首/约 11,000 小时真实音乐微调，以及 300 首高质量转写的强化学习后训练。

作者在 372 首真实多乐器 `D_Test` 上报告：

| 模型 | Onset F1 | Frame F1 | Offset F1 | Drums F1 | Multi F1 |
|---|---:|---:|---:|---:|---:|
| YourMT3+ `YPTF.MoE+Multi (noPS)` | 32.5 | 45.5 | 17.8 | 41.4 | 21.9 |
| MuScriptor Large | **60.4** | **72.4** | **48.6** | **49.6** | **47.8** |

评价边界：这是作者自建留出集上的显著提升，不是所有公共 benchmark 的统一 SOTA 证明。论文的 8 个公共跨域数据集里，MuScriptor 的 Multi F1 高于 YourMT3+ 其中 6 个、低于 2 个；模型还不输出 velocity，乐器粒度固定为 36 组，权重仅限非商用。

Hub 仓库创建于 2026-06-30；论文和 Mirelo 文章发布于 2026-07-09；当前权重 revision 更新于 2026-07-10；官方源码 `v0.3.0` 发布于 2026-08-05。Mirelo Studio 另有一个“使用更多数据训练”的私有增强版，但官方未公布其权重、revision 或同协议分数，不能视作任一公开 MuScriptor checkpoint。

完整训练消融、8 个公共数据集逐项分数、Small/Medium/Large 规模对比、乐器条件收益与前沿观察见 [MuScriptor 模型研究、分数与项目定位](muscriptor-model.md)。

### MIROS

MIROS 是桌面版、Space 与 Colab 中 `SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT` 三个模式的可选固定版本多乐器后端，对齐项目锁定的 MusicFM / AI4Musician Challenge SOTA 路线。它通过本地上游源码和权重接入，不使用 PyPI 包；包装器先校验身份，再调用上游入口生成临时 MIDI，并转换为项目统一音符结构。

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

TransKun 默认 V2 是钢琴专用转写后端，适合纯钢琴或以钢琴为主的音频。项目通过 `src/core/transkun_transcriber.py` 调用 `transkun` PyPI 包随附的预训练资源：

```bash
python -m pip install "transkun==2.0.1"
```

可用性检查会确认 `transkun.transcribe`、`pretrained/2.0.pt` 和 `pretrained/2.0.conf` 是否存在。资源缺失对应的修复命令如下：

```bash
python -m pip install --force-reinstall "transkun==2.0.1"
```

### TransKun V2 Aug

`PIANO_TRANSKUN_V2_AUG` 是与默认 V2 并列的独立路线，使用官方 `checkpointTransformerAug.zip` 中的 `checkpointMSimplerAug/checkpoint.pt` 与 `model.conf`。下载器会校验固定资源；V2 Aug 不会静默替代默认 V2，默认 V2 也不会静默替代 V2 Aug。

```bash
python download_transkun_v2_aug_model.py
```

默认搜索模型位置包括：

```text
~/.cache/music_ai_models/transkun_v2_aug
models/transkun_v2_aug
```

### Aria-AMT

Aria-AMT 是另一条钢琴专用后端。项目通过 `src/core/aria_amt_transcriber.py` 调用 `amt.run transcribe`，默认 checkpoint 为：

```text
piano-medium-double-1.0.safetensors
```

依赖安装命令：

```bash
python -m pip install --no-deps --force-reinstall "aria-amt @ https://github.com/EleutherAI/aria-amt/archive/a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip"
```

模型下载命令：

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

依赖安装命令：

```bash
python -m pip install "piano-transcription-inference==0.0.6" "torchlibrosa>=0.1.0,<0.2" "matplotlib>=3.7.0,<4"
```

模型准备命令：

```bash
python download_bytedance_piano_model.py
```

默认搜索模型位置包括：

```text
~/.cache/music_ai_models/bytedance_piano
models/bytedance_piano
```

## 钢琴后端选择建议

四条钢琴路线都只面向钢琴，不负责完整混音的多乐器识别：

| 目标 | 推荐模式 | 说明 |
|------|----------|------|
| 使用项目默认 TransKun 路线 | `PIANO_TRANSKUN` | 使用 PyPI 包随附 V2 资源。 |
| 显式对比官方数据增强 checkpoint | `PIANO_TRANSKUN_V2_AUG` | 独立下载并固定校验；不会静默替代默认 V2。 |
| 使用另一种现代钢琴 AMT 后端 | `PIANO_ARIA_AMT` | 适合用同一批纯钢琴音频做 A/B。 |
| 需要踏板 CC64 | `PIANO_BYTEDANCE_PEDAL` | 保留 sustain pedal 控制事件；建议在目标运行环境实际验证。 |

## 模型与公开对比

本节按 2026-08-08 的公开资料与当前版本实际能力标注：当前入口同步开放 `SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT`、`PIANO_TRANSKUN`、`PIANO_TRANSKUN_V2_AUG`、`PIANO_ARIA_AMT` 与 `PIANO_BYTEDANCE_PEDAL` 七种模式；`SMART` 可选择 YourMT3+、MIROS 或 MuScriptor，分离后的 WAV 也可逐轨选择这三类多乐器路线。下列表格把“公开 benchmark”和“项目内入口状态”分开写，避免把研究指标误写成产品能力。

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
| MIROS (MusicFM) | 多乐器 | 本地 `ai4m-miros` 仓库 + 当前工程包装器 | 固定 checkpoint 质量 | 官方仓库标注为 Music Transcription Challenge winning model，可作为桌面版、Space 与 Colab 中 `SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT` 的显式可选后端 |

处理语义：

- 所有入口默认使用固定高质量处理策略。
- `MIROS` 当前为固定 checkpoint 推理，可用于与 YourMT3+ 做同任务 A/B。

### 当前人声分离模型：Leap XE vocals + PolarFormer accompaniment

`VOCAL_SPLIT` 的模型与输入输出契约对齐当前公开 TelkNet 工具：BS-RoFormer Leap XE 90-band 对原混音生成 vocals，BS PolarFormer public ONNX 也对原混音独立生成 accompaniment。两个规范 WAV 随后分别交给用户选择的 YourMT3+ 或 MIROS。

TelkNet 对齐依据是经授权核验的私有 `mason369/telknet` `dev` 提交 `52be6fec179be492f5229ba149545ac2833b284a`。当前工程只对齐其 YourMT3/MIROS“官方 writer 后只补 tempo、不做通用音符清理”的核心语义；本项目的两个分离主流程同样只交付 WAV，MIDI 由用户在逐轨工作台显式触发。没有证据证明该 `dev` 已部署线上，也不声称模式路由逐行一致、推理环境相同或输出文件位级一致。

| 项目 | 详情 |
|------|------|
| vocals 模型 | [BS-RoFormer Leap XE](https://huggingface.co/pcunwa/BS-Roformer-Leap)：`Xe/bs_leap_xe_voc.ckpt` + `Xe/leap_xe_config_voc.yaml` |
| accompaniment 模型 | [BS PolarFormer](https://huggingface.co/bgkb/bs_polarformer)：官方 `bs_polarformer_fp16.onnx` + `model_bs_polarformer_float16.yaml` |
| 运行方式 | Leap XE 使用 audio-separator 内的 BS-RoFormer 实现；PolarFormer 使用 ONNX Runtime |
| 模型准备 | `download_sota_models.py` 会准备并校验两组资源；也可分别运行 `download_vocal_model.py` 与 `download_accompaniment_model.py` |
| 打包行为 | release 工作流会把校验后的分离资源打进便携包；运行时缺模型或校验失败会明确报错 |
| 输出选项 | 分离阶段输出规范的 `vocals` 与 `accompaniment` WAV；逐轨 MIDI 仅在用户勾选路线并点击转换后生成，不自动合并 |

两条分离路线不会互相替代，也不会用单个输出静默补齐另一条路径；任一模型或所选转写后端失败都会让 `VOCAL_SPLIT` 显式失败。

#### 人声分离模型对比

> 注：本表只保留这次重新核验时能找到公开来源支撑的结论。若写明“未写入数值”，表示没有找到与当前 checkpoint 明确绑定、且口径足够清晰的公开数值。

| 模型/方向 | 来源 | 类型 | 状态 | 说明 |
|-----------|------|------|------|------|
| Leap XE vocals + PolarFormer accompaniment（当前） | [Leap XE 模型仓库](https://huggingface.co/pcunwa/BS-Roformer-Leap) / [PolarFormer 模型仓库](https://huggingface.co/bgkb/bs_polarformer) | 本地 PyTorch + ONNX 双模型 | 使用中 | 模型与输入输出契约对齐当前公开 TelkNet 工具，但不据此声称服务端源码或结果位级一致；两个模型目标不同，不拼接为一个“总 SDR”。 |
| BS-RoFormer ep317（公开可下载） | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) | 本地直替（audio-separator） | 可替换（权衡） | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` 公开可下载；ZFTurbo 表按 Multisong 写明 `SDR vocals = 10.87`。文件名中的 `12.9755` 是训练标签，不等同于表中 vocals SDR。 |
| MelBand-RoFormer (KimberleyJensen) | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) / [Hugging Face](https://huggingface.co/KimberleyJSN/melbandroformer) | 本地可用（vocals/other） | 可用（偏人声） | 公开权重 `MelBandRoformer.ckpt` 可核；ZFTurbo 表按 Multisong 写明 `SDR vocals = 10.98`。 |
| SCNet XL IHF（开源权重） | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) / [ZFTurbo Release v1.0.15](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.15) | 开源可下载（4-stem） | 需改造接入 | 公开权重是 4-stem 模型，不是本项目现有 2-stem 直替；ZFTurbo 表写明 MUSDB test avg 10.08、Multisong avg 9.92。 |
| Mel-RoFormer (ISMIR 2024) | [arXiv:2409.04702](https://arxiv.org/abs/2409.04702) / [ar5iv 表2](https://ar5iv.org/html/2409.04702v1) | 论文阶段（研究模型） | 论文已发表 | MUSDB18-HQ（论文表2，场景 b，含额外数据）仅报告 Vocals SDR；这是论文特定协议，不与 Multisong / MVSEP 数字混排。 |
| Mamba2 Meets Silence (v2, 2025) | [arXiv:2508.14556](https://arxiv.org/abs/2508.14556) | 论文阶段（研究模型） | 论文 | 摘要报告 cSDR 11.03 dB（作者称 best reported），强调稀疏人声段鲁棒性 |
| Windowed Sink Attention (2025) | [arXiv:2510.25745](https://arxiv.org/abs/2510.25745) | 论文阶段（效率优化方向） | 论文 + 开源代码 | 在微调设定下恢复原模型约 92% SDR，同时 FLOPs 降低约 44.5x（偏效率收益） |

结论（按口径）：

- 当前 README 不再把不同来源的人声分离分数混成排行榜。
- 若来源是 API/服务模型、没有公开 checkpoint 映射，文档只标注“非本地直替”，不写成可直接替换的本地模型。
- 若来源是论文特定协议，文档只说明协议，不与工程默认 checkpoint 的文件名分数横比。
> 口径：不同榜单、数据集和评测协议（Multisong、MUSDB、MVSEP、cSDR/uSDR）不可直接横比。

### 已恢复流程对比

下表覆盖已恢复到桌面版、Space 和 Colab 的额外流程。公开数据通常只覆盖“分离”或“钢琴 AMT”单项任务，不等于本项目端到端音频转 MIDI 的统一评分。

| 流程 | 当前仓库状态 | 上游模型/实现 | 可核验公开数据 | 与当前 `SMART` / `VOCAL_SPLIT` 的关系 |
|------|--------------|---------------|----------------|---------------------------------------|
| 六声部分离 + 逐轨显式转写 | `six_stem_split` 已在 pipeline、桌面 UI、Space 和 Colab 中开放 | `BS-Rofo-SW-Fixed.ckpt`（vocals, bass, drums, guitar, piano, other）+ 每条 WAV 独立选择 13 条转写路线 | MVSEP Algorithms #77 给出 6-stem SDR：vocals 11.30 / instrum 17.50 / bass 14.62 / drums 14.11 / guitar 9.05 / piano 7.83 / other 8.71 | 这些是音源分离 SDR，不是最终 MIDI 转写 F1；逐轨 AMT 的端到端质量没有公开统一 benchmark。 |
| 钢琴专用转写（TransKun 默认 V2） | `piano_transkun` 已在 pipeline、桌面 UI、Space 和 Colab 中开放 | `transkun==2.0.1`，使用该 wheel 随附并严格校验的资源 | 官方 model cards：TransKun V2 在 MAESTRO V3 上 Note Onset / Onset+Offset / Onset+Offset+Velocity F1 为 0.9832 / 0.9349 / 0.9296；pip 随包 No Ext checkpoint 为 0.9833 / 0.8149 / 0.8109 | 这是钢琴专精协议，适合纯钢琴；不能与 YourMT3+ 的 Slakh2100 多乐器 F1 直接横比。 |
| 钢琴专用转写（TransKun V2 Aug） | `piano_transkun_v2_aug` 已在 pipeline、桌面 UI、Space 和 Colab 中开放 | 官方 `checkpointTransformerAug.zip`，固定校验后加载 `checkpointMSimplerAug/checkpoint.pt` + `model.conf` | 不把其他 V2 checkpoint 的指标直接移植给 V2 Aug | 与默认 V2 并列，供同一音频显式 A/B，不是失败回退。 |
| 钢琴专用转写（Aria-AMT） | `piano_aria_amt` 已在 pipeline、桌面 UI、Space 和 Colab 中开放 | EleutherAI `aria-amt`，公开 preliminary piano v1 checkpoint `piano-medium-double-1.0.safetensors` | 官方 README 提供安装、checkpoint 下载和 CLI 用法；未给出与 TransKun 同口径的 MAESTRO/MAPS benchmark。本地打包资源中的 checkpoint 约 425.9 MiB。 | 已集成为钢琴转写 A/B 选项，但 README 不写入不存在的统一分数；比较时应使用同一批本地音频。 |

### 未来可关注的转写模型

下列对比按 2026-08-08 的公开资料更新。`MuScriptor D_Test Multi F1`、`Slakh2100 Multi (Onset-Offset) F1`、`MAESTRO onset F1` 与 2025 AMT Challenge 的 Multi Onset F1 不是同一协议，不能当成同一张排行榜直接横比。

#### 多乐器模型（公开可核实）

| 模型 | 公开来源 | Benchmark / 协议 | 公开结果 | 状态 | 说明 |
|------|----------|------------------|----------|------|------|
| [MuScriptor Large](https://huggingface.co/MuScriptor/muscriptor-large) | [论文](https://arxiv.org/abs/2607.08168) / [代码](https://github.com/muscriptor/muscriptor) | 作者 `D_Test`，372 首真实多乐器曲目；完整训练，CFG=2 | Onset / Frame / Offset / Drums / Multi F1 = **60.4 / 72.4 / 48.6 / 49.6 / 47.8**；同表 YourMT3+ Multi F1 = 21.9 | 已集成 | 很强的公开完整混音候选；公共跨域集 Multi F1 赢 6、输 2，不写成所有协议的绝对 SOTA |
| MuScriptor Small / Medium | [官方代码与三档权重](https://github.com/muscriptor/muscriptor#models) | `D_Real` only、CFG=2 规模消融 | Small Multi F1 38.2；Medium 39.7；Large 40.5 | 已集成 | 103M / 307M 固定权重已作为独立显式选择接入；三档不会互相静默替代，并按同一真实音频分别验收质量、速度和显存 |
| YPTF.MoE+Multi (noPS)（当前默认） | [官方 Space app.py](https://huggingface.co/spaces/mimbres/YourMT3/blob/main/app.py) / [Space noPS 结果文件](https://huggingface.co/spaces/mimbres/YourMT3/blob/main/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/result_mc13_full_plus_256_default_all_eval_final.json) | Slakh `multi_f` | **0.7398 / 73.98%** | 使用中 | 当前项目默认 YourMT3+ checkpoint；对齐官方 Hugging Face Space 默认项 |
| YPTF.MoE+Multi（论文表最终模型） | [YourMT3+ 论文](https://arxiv.org/abs/2407.04822) | Slakh2100 `Multi (Onset-Offset) F1` | **74.84**；同表 `MT3 = 62.0` | 论文公开结果 | 这是论文表中的最终模型口径，不把它写成当前 noPS 默认 checkpoint 的单独成绩 |
| [MT3](https://github.com/magenta/mt3) | [YourMT3+ 论文](https://arxiv.org/abs/2407.04822) / [Magenta 仓库](https://github.com/magenta/mt3) | Slakh2100 `Multi (Onset-Offset) F1` | **62.0** | 开源基线 | YourMT3+ 继承并扩展的 token-based 多乐器基线 |
| 2025 AMT Challenge 冠军 MIROS | [挑战论文](https://arxiv.org/abs/2603.27528) / [代码](https://github.com/amt-os/ai4m-miros) | 76 个受约束合成短片段；Multi Onset F1 | **0.5998**；YourMT3-YPTF-MoE-M 0.5938；MT3 0.3932 | 已集成 | MusicFM 编码器路线；挑战协议不能与 MuScriptor `D_Test` 或 Slakh 横比 |
| Mirelo Studio 改进版 | [Mirelo 官方文章](https://mirelo.ai/blog/turning-audio-to-midi) | 未公开 | 只说明“使用更多数据训练、更准确” | 私有服务观察项 | 没有公开权重、revision 或分数；不是当前 `muscriptor-large`，不能离线集成 |

#### 钢琴专精模型（公开可核实）

| 模型 | 公开来源 | Benchmark / 协议 | 公开结果 | 状态 | 说明 |
|------|----------|------------------|----------|------|------|
| [TransKun V2（论文 checkpoint）](https://github.com/Yujia-Yan/Transkun) | [TransKun 官方仓库 / model cards](https://github.com/Yujia-Yan/Transkun) | MAESTRO V3 `note onset F1 / onset+offset F1 / onset+offset+velocity F1` | **0.9832 / 0.9349 / 0.9296** | 开源 | 论文公开 checkpoint 的模型卡结果；项目默认入口使用 pip 随包资源 |
| [TransKun pip 随包 checkpoint（No Ext）](https://github.com/Yujia-Yan/Transkun) | [TransKun 官方仓库 / model cards](https://github.com/Yujia-Yan/Transkun) | MAESTRO V3 No Ext 同口径三项指标 | **0.9833 / 0.8149 / 0.8109** | 开源 | 对应项目默认 `PIANO_TRANSKUN`；上游说明为 `without pedal extension of notes` |
| TransKun V2 Aug | 官方数据增强 checkpoint | 与其他 V2 checkpoint 分开记录 | 未写入跨 checkpoint F1 | 对应 `PIANO_TRANSKUN_V2_AUG`；用同一批本地音频与默认 V2 显式 A/B。 |
| [Aria-AMT](https://github.com/EleutherAI/aria-amt) | [EleutherAI 官方仓库](https://github.com/EleutherAI/aria-amt) | 公开 checkpoint 发布 | 仓库公开 `piano-medium-double-1.0.safetensors`；但仓库页未给出与上表完全同口径的统一 MAESTRO/MAPS 榜单 | 开源 | 已集成为钢琴 A/B 选项；这里不伪造不存在的统一 benchmark 行 |
| [High-Resolution Piano Transcription with Pedals by Regressing Onset and Offset Times](https://arxiv.org/abs/2010.01815) | [论文](https://arxiv.org/abs/2010.01815) / [ByteDance 仓库](https://github.com/bytedance/piano_transcription) | MAESTRO `onset F1 / pedal onset F1` | **96.72% / 91.86%** | 论文 + 代码 | 代表性踏板感知钢琴论文；协议是钢琴专精口径，不应与多乐器 Slakh 分数混排 |

#### 论文阶段 / 协议不一致的研究方向

| 模型/方向 | 公开来源 | 公开协议 / 任务 | 可核实的公开信息 | 为什么不与上表混成同一分数榜 |
|-----------|----------|-----------------|------------------|------------------------------|
| 密集复音与乐器检测 | [2025 AMT Challenge 论文](https://arxiv.org/abs/2603.27528) | 1/2/3 乐器分组分析 | MIROS 从 1 种到 3 种乐器时 F-measure 从 0.7193 降到 0.4367；论文把 polyphony、相似音色和乐器泄漏列为主要失败模式 | 这是未来评测重点，不是可直接集成的新 checkpoint |
| [MR-MT3](https://arxiv.org/abs/2403.10024) | [论文](https://arxiv.org/abs/2403.10024) / [代码](https://github.com/gudgud96/MR-MT3) | Slakh2100；重点看 `onset F1`、`instrument leakage ratio`、`instrument detection F1` | 摘要明确写的是“improved onset F1 scores and reduced instrument leakage” | 它主打 leakage 抑制，并引入了新指标；不等于上面的 Slakh `Multi (Onset-Offset) F1` |
| [Jointist](https://arxiv.org/abs/2302.00286) | [论文](https://arxiv.org/abs/2302.00286) | 流行音乐联合转写 + 分离 | 摘要给出的公开结果是：转写提升 `>1 ppt`、分离提升 `+5 SDR`、downbeat `+1.8 ppt`、和弦/调性各 `+1.4 ppt` | 它是 joint transcription + separation 路线，公开协议与 Slakh / MAESTRO 完全不同 |
| MusicFM 编码器 + AMT 解码器 | [MusicFM 论文](https://arxiv.org/abs/2311.03318) / [仓库](https://github.com/minzwon/musicfm) / [HF 权重](https://huggingface.co/minzwon/MusicFM) | 预训练编码器迁移 | 公开的是基础编码器权重；通用可复现的完整 AMT decoder / 微调流水线并未作为现成后端发布 | 它更像 MIROS 这类路线背后的表示学习部件，不是拿来就能切换的通用后端 |
| [CountEM / Count The Notes](https://arxiv.org/abs/2511.14250) | [论文](https://arxiv.org/abs/2511.14250) / [项目页](https://yoni-yaffe.github.io/count-the-notes) / [代码](https://github.com/Yoni-Yaffe/count-the-notes) | 弱监督 AMT 训练方法 | 公开论文、代码和模型，核心贡献是“用音符直方图 + EM”替代精确对齐监督 | 这是训练范式创新，不是固定 checkpoint 的 turnkey 后端 |
| [PerceiverTF](https://arxiv.org/abs/2306.10785) | [论文](https://arxiv.org/abs/2306.10785) | 多乐器公开数据集（论文自有协议） | 摘要只明确说其在多个公开数据集上优于 MT3 / SpecTNT | 它更适合作为 YourMT3+ 的架构祖先来理解，不应和上表的统一数值行硬拼 |

补充说明：

- [Basic Pitch](https://github.com/spotify/basic-pitch) 依然是很有价值的轻量方案，但它不发布与上表同口径的 Slakh/MAESTRO 综合榜单。
- [Omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) 仍是有参考价值的多任务工具链，但其 GitHub latest release 仍为 `0.5.0`（2021-12-09），与当前多乐器/钢琴专精 SOTA 的公开比较协议并不一致。

截至 2026-08-08，多乐器 AMT 主要有三条路线：`MT3 / YourMT3+ / MR-MT3` 的 token-based 架构演进、MIROS 的 MusicFM 预训练编码器路线，以及 MuScriptor 使用大规模真实数据和 RL 后训练的 decoder-only 路线。完整验证范围包括密集复音、相似音色泄漏、稀有乐器、真实 jazz/pop 泛化、权重许可、速度和显存。各项 F1 的评测协议并不统一，钢琴 checkpoint 也需要单独记录协议。

## 默认处理策略

桌面版、Space、Colab 和独立 Web 前端不再提供可调质量入口。YourMT3+ 产品路线使用官方无重叠分段、固定 `bsz=8`、逐解码通道 detokenize/merge、`mix_notes` 和官方 MIDI writer；MIROS 直接保留官方 CLI writer 输出。MuScriptor 固定使用官方 v0.3.0 源码、权重、5 秒窗口和 MIDI writer，并提供两个显式分段链路：默认的“官方处理链路”保留上游原始分段状态；“分段边界连续性修复链路”作为显式可选项恢复经验证的跨分段连续音符，供同输入 A/B。该开关与速度方案独立，两条链路都不做项目级音符量化、过滤或 `NoteEvent` 重建。

七种模式和逐轨转写只使用 Beat This `final0`：先清除竞争拍点、补计漏拍位置，再以全局最小二乘拟合 BPM；下拍独立推断拍号，置信不足时不伪造 4/4。默认自动写入检测到的唯一 BPM；“跟随原曲速度变化”才写稳定的段落级 tempo map。手动 30–300 BPM 是明确的工程 tempo 覆盖，会保留检测 BPM 的音乐 tick；该流程不量化或清理音符，事件内容保持不变。

发布后的 Standard MIDI 会回读校验 tempo、拍号和全部非 tempo 事件；DAW 需启用 tempo map/速度图导入。MuseScore 3/4 可能把未量化的演奏型 MIDI 判定为 human performance，重新跟拍并覆盖乐谱页显示 BPM。该数值来自 MuseScore 的 MIDI 导入器；项目不会为了强制制谱软件显示文件 tempo 而量化或移动模型音符。

`SIX_STEM_SPLIT` 中，`BS-Rofo-SW-Fixed.ckpt` 生成六个真实 WAV stem；分离阶段不调用 MIDI 后端。每个 stem 都可选择 13 条路线之一并单独转换，不会自动批量转写或合并 MIDI。

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.11+，Windows 安装脚本优先使用 3.11-3.12 |
| PyTorch | NVIDIA 基线为 `torch/torchaudio==2.7.0`、`torchvision==0.22.0`；Intel XPU 基线为 `torch/torchaudio==2.11.0+xpu`、`torchvision==0.26.0+xpu` |
| Git | 源码安装必需；找不到 Git 时安装器会明确停止 |
| FFmpeg | 必需；用于可靠处理 MP3/M4A/FLAC/OGG 等格式 |
| GPU | Windows 完整七模式支持 NVIDIA CUDA 12.8 或 Intel XPU，两者使用独立环境；Linux/WSL2 仍只验收 NVIDIA，不会自动降级到 CPU/AMD |
| 磁盘 | 完整冷安装会同时保存 wheel、模型与下载缓存；实测工作集约 32.36 GB，建议开始前至少保留 40 GB 可用空间 |
| 系统 | Windows 10/11、Linux、WSL2 |

不同平台使用各自固定的兼容运行时；跨平台覆盖 NumPy/Torch 组合不属于支持范围：

| 平台 | Python / Torch | NumPy 与 GPU 运行时 | 发布边界 |
|------|----------------|---------------------|----------|
| Windows / NVIDIA 桌面与便携目标 | Python 3.11-3.12；Torch 2.7.0 / torchaudio 2.7.0 / torchvision 0.22.0 | NumPy 1.26.4；CUDA 12.8 wheel | 源码与便携发布均按此契约校验；`release.yml` 同时执行第三方许可闭集门禁、模型身份校验和成品烟测 |
| Windows / Intel XPU 桌面与本地便携目标 | Python 3.11-3.12；原生 Torch 2.11.0 XPU / torchaudio 2.11.0 XPU / torchvision 0.26.0 XPU | NumPy 1.26.4；`onnxruntime-openvino==1.24.1` + `openvino==2025.4.1`；启动门禁验证 FFT/STFT、BF16 与矩阵运算驻留 XPU，PolarFormer 固定使用 `OpenVINOExecutionProvider` 的 `GPU.0` | 最新完整三件套覆盖 PyTorch 官方矩阵中的 Arc B-Series（Battlemage）与 Core Ultra Series 3（Panther Lake）；Panther Lake 要求 Windows 11。运行环境为独立 `venv-xpu`；IPEX、CUDA ORT 混装和 CPU EP 回退会被门禁拒绝。官方 GitHub release 暂仍只构建 CUDA 包 |
| Linux / NVIDIA 源码运行 | Python 3.11+；Torch 2.7.0 / torchaudio 2.7.0 / torchvision 0.22.0 | NumPy 1.26.4；NVIDIA 驱动兼容 CUDA 12.8；仅 `cu128` | `install.sh` / `run.sh` 对完整七模式执行精确运行时校验；`build.yml` 只做源码、测试和打包契约检查 |
| Linux / AMD/ROCm | 不提供完整七模式兼容运行时 | PolarFormer 固定依赖 ONNX Runtime `CUDAExecutionProvider` | 当前不支持；安装脚本会明确停止，不静默改用 CPU |
| Hugging Face Space | Python 3.12.12；Torch 2.8.0 / torchaudio 2.8.0 / torchvision 0.23.0 | NumPy `>=2,<2.5`；ZeroGPU | 使用 `space/requirements.txt`；桌面 NumPy 1.26 不属于 Space 兼容组合 |
| Google Colab | Colab 当前预装 Python/Torch | 保留预装 Torch；只安装 pinned Web/runtime 依赖 | 避免替换 Torch 导致 CUDA 运行库冲突 |

Windows 源码版应放在本机磁盘的纯英文、无空格目录，例如：

```text
C:\MusicToMidi
D:\Projects\music-to-midi
```

含中文、空格或括号的路径可能导致 PyTorch DLL 加载失败。映射盘和 UNC/SMB 网络路径尚未通过源码虚拟环境身份验收，可能让 `venv` 记录路径与启动路径不一致；本机 NTFS 的纯英文无空格目录是当前已支持的源码运行位置。

## 快速开始

### MuScriptor gated 授权

MuScriptor Small / Medium / Large 都是 Hugging Face gated 模型，匿名用户无法下载。源码安装的授权条件是：同一个 Hugging Face 账户已在浏览器中分别接受 [Small](https://huggingface.co/MuScriptor/muscriptor-small)、[Medium](https://huggingface.co/MuScriptor/muscriptor-medium)、[Large](https://huggingface.co/MuScriptor/muscriptor-large) 的条款，并在账号获批后从项目隔离环境登录：

```powershell
# Windows；无 venv 时的最小授权环境仅包含官方 HF CLI
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install "huggingface_hub>=0.20,<2" "hf_xet==1.5.2"
.\venv\Scripts\hf.exe auth login
```

```bash
# Linux / WSL2
python3.11 -m venv venv
./venv/bin/python -m pip install "huggingface_hub>=0.20,<2" "hf_xet==1.5.2"
./venv/bin/hf auth login
```

登录使用具有三个 gated 仓库读取权限的个人 token。仓库、README、命令行参数和日志都不是受支持的 token 存放位置；无人值守环境的凭据入口是受保护的 `HF_TOKEN` secret。总下载脚本会在下载其它大型模型前依次做三个轻量权限预检，任一仓库未接受条款或未登录都会立即停止，不会改用其它模型。完整便携 release 已随包携带通过身份校验的权重，启动时不再从 Hub 下载；这不改变 CC BY-NC 4.0 与附加条款的适用范围。

### Windows

推荐：

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

或双击：

```text
run.bat
```

`run.ps1` 会检查虚拟环境、Beat This `final0`、五种 YourMT3+ 模式、MuScriptor Large / Medium / Small、BS-RoFormer SW Fixed、Leap XE、PolarFormer、TransKun V2 Aug、Aria-AMT、ByteDance Pedal、MIROS、SoundFont 与 FluidSynth；资源缺失或校验失败时会调用 `install.ps1`。

Intel GPU 使用独立的原生 XPU 环境：

```powershell
powershell -ExecutionPolicy Bypass -File .\install_xpu.ps1
powershell -ExecutionPolicy Bypass -File .\run_xpu.ps1
```

安装和每次启动都会做真实 `torch.xpu` 矩阵、FFT、STFT 与 BF16 卷积运算并拒绝 XPU→CPU 算子回落；随后用禁用 CPU EP 回退的 `OpenVINOExecutionProvider/GPU.0` 执行真实 ONNX MatMul 图。任一门禁失败都会停止，不改用 IPEX、DirectML、CUDA 或 CPU。

Leap XE 在 XPU 上保留官方完整约 20 秒音频窗口、全部 key/value、checkpoint 与后处理，只把 attention query 轴固定分为 128 行逐块求值并拼接。每个 query 仍注意完整上下文，所以这是推理期数值等价的显存有界实现，不是缩短窗口、降低模型或 CPU 回退；训练态误用会直接失败。该路径已在 16 GB Arc 140T 上完成真实两轨分离。

### Linux / WSL2

```bash
chmod +x run.sh
./run.sh
```

`run.sh` 会检查虚拟环境、核心依赖、Beat This `final0`、YourMT3+ 源码与五种模型模式、MuScriptor Large / Medium / Small、BS-RoFormer SW Fixed、Leap XE、PolarFormer、TransKun V2 Aug、Aria-AMT、ByteDance Pedal、MIROS、SoundFont 与 FluidSynth；资源缺失或校验失败时会调用 `install.sh`。

### 源码直接运行

受支持的源码运行环境是仓库内与加速器匹配的隔离环境：NVIDIA CUDA 使用 `venv`，Windows Intel XPU 使用 `venv-xpu`。入口会在导入 GUI 和模型依赖前严格校验解释器路径、`include-system-site-packages=false`、MuScriptor 包位置、版本和七个源码 SHA-256；全局 `python -m src.main` 会被拒绝。

```powershell
# Windows 桌面 App
.\venv\Scripts\python.exe -m src.main

# Windows Web 前后端；自动识别局域网 IPv4 并打开前端
.\venv\Scripts\python.exe -m src.web
```

```bash
# Linux / WSL2 桌面 App
./venv/bin/python -m src.main

# Linux / WSL2 Web 前后端；浏览器手动打开终端显示的地址
./venv/bin/python -m src.web --no-window
```

## 手动安装

### 1. 创建虚拟环境

Windows:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip wheel "setuptools==80.10.2"
```

Linux:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel "setuptools==80.10.2"
```

### 2. 安装 PyTorch

CUDA 12.8（完整七模式受支持且由启动器严格校验的运行时）:

```bash
pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

`cu118` / CUDA 11 不属于当前一键启动器和完整七模式验收契约；启动器不会把它静默当成已对齐环境。

Windows Intel XPU 的标准安装入口是 `install_xpu.ps1`。手动准备对应独立的 `venv-xpu` 和 `requirements-xpu.txt` 固定版本；在 `venv` 中覆盖 CUDA wheel 会破坏两套环境的隔离契约。项目采用“最新完整兼容组合”而不是单独追最高 Torch：当前 Torch XPU wheel 已高于 2.11，但最新匹配的 torchaudio XPU wheel 是 2.11，因此固定 `torch/torchaudio==2.11.0+xpu` 与 `torchvision==0.26.0+xpu`。[PyTorch 2.11 官方矩阵](https://docs.pytorch.org/docs/2.11/notes/get_start_xpu.html)已列出 Arc B-Series 与 Panther Lake / Core Ultra Series 3；PolarFormer 使用对齐 OpenVINO 2025.4.1 的 [ONNX Runtime OpenVINO 1.24.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.1)。

Intel XPU 没有可直接等同 NVIDIA `sm_XX` 的项目级兼容版本号。该栈通过 oneAPI/Level Zero 发现设备，常规 JIT 路径由 Intel Graphics Compiler 针对实际硬件生成代码；真实支持边界由固定 PyTorch 版本的官方硬件矩阵、系统/驱动要求和本项目启动时的真实算子门禁共同决定，不能把任意 Intel GPU 视为自动兼容。

MuScriptor Large 的 5.1 GiB checkpoint 在 XPU 上固定使用 `safetensors==0.8.0` 的官方 `pread` reader，按文件偏移顺序逐张量装入设备。XPU 路径先打开惰性的 `pread` 文件句柄，再建立约 5.50 GiB 的统一内存模型，避免模型先占用系统提交空间后才打开 checkpoint 所触发的 Windows `os error 1455`。该路径不创建完整 checkpoint 的 PyTorch 内存映射，权重、精度和官方 writer 均不改变，失败也不会回退到 mmap、扩大页面文件或自动重试。

Intel XPU 的本地 Web 后端会为每个 GPU 作业启动一个全新的处理进程，作业结束后由操作系统回收该进程的完整地址空间；常驻 HTTP 进程不跨作业保留 YourMT3、MIROS、MuScriptor 或分离模型。这样可避免统一内存模型在长会话中累积占用 Windows 系统提交额度。停止操作会终止并回收当前处理进程；非零退出、缺失结果清单或结果文件都会直接判为失败，不会重试、切 CPU 或伪造成功结果。

`torchaudio` 2.11 的 `load/save` 会委托给 TorchCodec，而 TorchCodec 的 Windows wheel 需要 full-shared FFmpeg DLL。项目不以 fallback 掩盖缺失 DLL：公开输入先由随包 FFmpeg 转成 WAV，再由固定的 libsndfile PCM 读取器生成 channels-first float32 张量；重采样与模型推理仍走已验证的 XPU 路线。

AMD/ROCm 当前不能完成七模式：当前固定的分离器执行契约只验收 NVIDIA `CUDAExecutionProvider` 或 Intel `OpenVINOExecutionProvider/GPU.0`，没有 AMD 对应的严格 GPU provider。安装脚本会明确停止，不会静默改用 CPU。

`release.yml` 只生成 CUDA 12.8 GPU 便携版，不生成 CPU 版。当前闭集清单包含 30 项第三方组件：26 项 `VERIFIED`、4 项附维护者具名责任与撤销联系记录的 `OWNER_ACCEPTED`、0 项 `BLOCKED`；工作流仍会在每次发布前重新校验清单、模型身份、SBOM、FFmpeg 构建信息和成品自检，任何一项不满足即停止。push / PR 的 `build.yml` 仅验证源码、测试与打包契约，不生成便携成品。本地源码开发如需 CPU-only PyTorch，应自行承担模型速度和依赖兼容性差异。

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
python -m pip install --no-deps "audio-separator==0.44.1"
python -m pip install --no-deps --force-reinstall "aria-amt @ https://github.com/EleutherAI/aria-amt/archive/a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip"
python -m pip install --no-deps --force-reinstall "muscriptor @ https://github.com/muscriptor/muscriptor/archive/d73147e75e5b9b0c0a79ebe154587db4fd603e0c.zip"
python -m src.utils.source_runtime
```

`requirements.txt` 有意避免 audio-separator 的 NumPy 2 元数据和 Aria-AMT 的旧 torchaudio 约束覆盖桌面兼容栈，因此这些包采用固定版本的 `--no-deps` 独立安装。MuScriptor 同样从官方 v0.3.0 精确提交安装且不解析依赖，随后统一校验虚拟环境、包路径、版本和源码哈希；`install.ps1` / `install.sh` 提供完整伴随依赖。

### 4. 准备 YourMT3+ 源码与模型

```bash
python download_sota_models.py
```

当前仓库已经包含受控且经过兼容补丁的 `YourMT3/amt/src`；可变上游 `master` 不满足这里的源码身份校验。`download_sota_models.py` 会准备 Beat This `final0`、五种官方 YourMT3+ checkpoint、固定 MIROS 源码与两组权重、MuScriptor Large / Medium / Small、`BS-Rofo-SW-Fixed.ckpt`、Leap XE、PolarFormer、TransKun V2 Aug、Aria-AMT、ByteDance、MuseScore General SoundFont、FluidSynth 与固定 MuseScore Studio 4.7.4，并严格校验默认 TransKun 2.0.1 包及其内置 V2 资源。

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
~/.music-to-midi/models/beat_this
~/.music-to-midi/models/audio-separator
~/.cache/music_ai_models/transkun_v2_aug
~/.cache/music_ai_models/aria_amt
~/.cache/music_ai_models/bytedance_piano
~/.cache/music_ai_models/fluidsynth/2.5.6
${HF_HOME:-~/.cache/huggingface}/hub  # MuScriptor 三档与 MuseScore SoundFont
external/ai4m-miros
```

Windows 中的 `~` 是 `%USERPROFILE%`。源码虚拟环境固定为仓库内 `venv`，桌面默认输出为仓库内 `MidiOutput\<音频文件名>`，日志在 `%USERPROFILE%\.music-to-midi\logs`。默认 TransKun V2 资源位于 `venv` 中的 `transkun==2.0.1` 包内；便携版只读取发布目录内的 `models`、`runtime` 与 `tools` 资源，不依赖上述源码缓存。

默认 TransKun V2 的模型资源随 `transkun==2.0.1` 安装；`PIANO_TRANSKUN` 资源或身份异常对应的修复命令是 `python -m pip install --force-reinstall "transkun==2.0.1"`。`PIANO_TRANSKUN_V2_AUG` 使用独立缓存，由 `python download_transkun_v2_aug_model.py` 准备。

### 6. 启动

```powershell
# Windows
.\venv\Scripts\python.exe -m src.main
```

```bash
# Linux / WSL2
./venv/bin/python -m src.main
```

## Google Colab

Colab 入口：

```text
colab_notebook.ipynb
```

使用步骤：

1. 打开笔记本。
2. 选择 GPU 运行时。
3. 若要使用 MuScriptor，先按上文逐项接受三个 gated 仓库条款，把 token 保存为 Colab 私有 secret `HF_TOKEN`，并在第 3 个代码单元勾选 `ENABLE_MUSCRIPTOR`；该单元会在启动前验证三档访问权限。
4. 依次运行其余单元格。
5. 最后一个单元格会启动 Gradio，并输出公开访问链接。

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

Space 版随部署包携带项目已验证并打过兼容补丁的 `YourMT3/amt/src` 源码树，与桌面版和 Colab 完全共用；不会在运行时切换到 Hugging Face Space 的可变源码。运行转换时只按所选模式检查或准备对应资源：所选 YourMT3+ 官方 checkpoint 或 MIROS、BS-RoFormer SW Fixed、Leap XE、PolarFormer、TransKun V2 Aug、Aria-AMT 或 ByteDance Pedal；缺失资源或身份校验失败会显式暴露。

ZeroGPU 部署只用于短片段试用，不承诺完整长歌端到端完成。[Hugging Face ZeroGPU 文档](https://huggingface.co/docs/hub/main/en/spaces-zerogpu) 当前公开配额为匿名用户每日 2 分钟、登录免费账户每日 5 分钟 GPU。当前保守的最小请求经 `large` GPU 平台倍率折算后已高于匿名额度，因此转换入口要求登录；Space 会按模式、后端和模型估算，再按固定的 `spaces==0.51.1` 平台倍率上界折算，超过登录免费账户 300 GPU 秒窗口的请求会在下载模型前明确拒绝。估算只是准入上限，不保证用户仍有足够当日配额或队列容量；Colab、桌面版或专用 GPU 更适合长歌。

当前公式下的最大输入时长是精确准入阈值，不是实测耗时承诺。下表适用于默认 `YPTF.MoE+Multi (noPS)` 与 MIROS；其它 YourMT3 checkpoint 使用各自系数：

| ZeroGPU 路线 | YourMT3 默认 noPS | MIROS |
|--------------|------------------:|------:|
| `SMART` | 2.00 秒 | 1.00 秒 |
| `VOCAL_SPLIT` | 0.53 秒 | 0.27 秒 |
| `SIX_STEM_SPLIT` | 0.22 秒 | 0.11 秒 |
| 任一钢琴专用模式 | 2.50 秒 | 不适用 |

Space 失败请求会立即删除请求目录；成功结果保留给 Gradio 下载，默认在 24 小时后进入过期清理，随后在清理轮次或实例正常退出时删除。Colab 同样立即清理失败目录，并把成功结果保留到当前运行时结束；Gradio 缓存在 24 小时后进入清理。

## 便携版打包

当前 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) 的 30 项闭集清单为 26 项 `VERIFIED`、4 项 `OWNER_ACCEPTED`、0 项 `BLOCKED`。`OWNER_ACCEPTED` 表示上游未声明许可时由维护者具名承担再分发决定，并不等同于获得上游授权；任一项目重新变为未解决状态时，官方 release 会在构建前显式阻断。

Windows CUDA 桌面 App、Web 后端、Web 前端三包：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_web_executables.ps1
```

Intel XPU 的构建环境是独立且已验收的 `venv-xpu`：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_portable_xpu.ps1
```

完整 XPU 成品超过 20 GB。仓库所在卷空间不足时，可把临时目录和成品目录指定到空间充足的 NTFS 卷；删减模型或保留半成品会使成品校验失败：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_portable_xpu.ps1 `
  -BuildRoot C:\MusicToMidi-XPU-build `
  -DistRoot C:\MusicToMidi-XPU-dist
```

XPU wrapper 只在源与 staging 同卷时建立 NTFS 硬链接，跨卷资源正常复制；两者都继续执行 staged 与 packaged SHA-256/manifest 校验。硬链接失败会直接停止，不会静默改回复制。有效的两个根目录互不嵌套；盘根和项目根会被参数校验拒绝。

自定义 Python 或 FFmpeg 路径：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_web_executables.ps1 `
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

构建结果包含以下三个完整目录：

```text
dist/MusicToMidi-App/
dist/MusicToMidi-WebBackend/
dist/MusicToMidi-WebFrontend/
```

发布时以对应的完整目录为单位：桌面用户使用 App 目录，Web 用户使用 WebBackend 与 WebFrontend 两个目录。

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
download_sota_models.py      # Beat This + 五种 YourMT3 + MIROS + 三档 MuScriptor + 三种分离 + 四条钢琴 + 播放资产
download_vocal_model.py      # Leap XE vocals 模型下载
download_accompaniment_model.py # PolarFormer accompaniment 下载入口
download_multistem_model.py  # BS-RoFormer SW Fixed 六声部分离模型下载
download_transkun_v2_aug_model.py # TransKun V2 Aug 下载与校验
download_aria_amt_model.py   # Aria-AMT 模型下载
download_bytedance_piano_model.py # ByteDance Pedal 模型下载
download_vocal_harmony_model.py # PolarFormer accompaniment 历史兼容入口
MusicToMidi.spec             # PyInstaller 配置
```

## 开发命令

以下命令同样在仓库内 `venv` 激活后执行。

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

源码目录检查：

```text
YourMT3/amt/src
```

模型检查：

```bash
python -c "from src.utils.yourmt3_downloader import get_model_path; print(get_model_path())"
```

缺失时：

```bash
python download_sota_models.py
```

如果受控的 `YourMT3/amt/src` 缺失，恢复来源是当前项目版本中的对应目录；可变上游 `master` 无法通过三端源码一致性与便携 manifest 校验。

### 人声分离不可用

依赖和模型检查：

Windows / Linux NVIDIA CUDA 环境：

```bash
python -m pip install --no-deps "audio-separator==0.44.1" "onnxruntime-gpu==1.23.2"
python download_vocal_model.py
python download_accompaniment_model.py
```

macOS 或明确的 CPU 环境把 `onnxruntime-gpu==1.23.2` 换成 `onnxruntime==1.23.2`；AMD/ROCm 不能提供 PolarFormer 所需的 `CUDAExecutionProvider`，当前不支持完整七模式。

### 六声部分离不可用

该路线依赖 `audio-separator==0.44.1` 和 BS-RoFormer SW Fixed 资源：

```bash
python download_multistem_model.py
```

### 钢琴专用转写不可用

默认 TransKun V2 模式需要 `transkun` 包和其随包预训练资源：

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

ByteDance Pedal 模式需要 `piano-transcription-inference`、`torchlibrosa` 和 ByteDance Piano checkpoint：

```bash
python -m pip install "piano-transcription-inference==0.0.6" "torchlibrosa>=0.1.0,<0.2"
python download_bytedance_piano_model.py
```

### MIROS 不可用

本地仓库位置与文件完整性检查：

```text
ai4m-miros/main.py
ai4m-miros/transcribe.py
```

若提示缺少 Python 模块，缺失依赖以 MIROS 上游仓库的运行清单为准。

## 许可证

本项目使用 MIT License。第三方模型、数据和上游仓库遵循各自许可证与使用条款；改编代码声明与完整许可证见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
