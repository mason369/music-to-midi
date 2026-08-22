---
title: Music to MIDI
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.17.3"
python_version: "3.12.12"
app_file: app.py
pinned: false
license: mit
suggested_hardware: zero-a10g
short_description: MuScriptor、YourMT3+、MIROS 与钢琴模型驱动的音频转 MIDI
models:
  - MuScriptor/muscriptor-small
  - MuScriptor/muscriptor-medium
  - MuScriptor/muscriptor-large
  - MuScriptor/assets
  - mimbres/YourMT3
  - minzwon/MusicFM
  - pcunwa/BS-Roformer-Leap
  - bgkb/bs_polarformer
  - noblebarkrr/mvsepless_resources
datasets:
  - loubb/aria-midi
tags:
  - audio-to-midi
  - midi
  - music-transcription
  - ai-music
  - muscriptor
  - yourmt3
  - zerogpu
  - stem-separation
  - piano-transcription
---

# Music to MIDI

将 `MP3`、`WAV`、`FLAC`、`OGG`、`M4A` 音频转换为可编辑 MIDI 或分离后的 WAV 音轨。Space、桌面版与 Colab 使用相同的七个模式：`SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT`、`PIANO_TRANSKUN`、`PIANO_TRANSKUN_V2_AUG`、`PIANO_ARIA_AMT`、`PIANO_BYTEDANCE_PEDAL`。

五个直接转写模式一次点击生成 MIDI；两个分离模式生成 WAV 后进入多轨工作台。复选框和模型用于设置音轨，每条音轨都有十三种转写路线，并通过各自的“开始转换”按钮单独处理。

> ZeroGPU 适合短片段试用。[Hugging Face ZeroGPU 文档](https://huggingface.co/docs/hub/main/en/spaces-zerogpu) 当前列出的配额是匿名用户每日 2 分钟、登录免费账户每日 5 分钟 GPU。最小转换请求经 `large` GPU 平台倍率折算后已经超过匿名额度，因此转换入口要求登录。应用在下载模型前估算用量，超过登录免费账户 300 GPU 秒窗口的任务会被拒绝；Colab、桌面版或专用 GPU 更适合长音频。准入估算不包含账户剩余额度和实时排队情况。

以下阈值只用于准入估算，不代表实测耗时。逐轨 MIDI 会按该音轨选择的模型单独估算：

| ZeroGPU 任务 | 准入音频长度上限 |
|---|---:|
| `SMART` + 默认 `YPTF.MoE+Multi (noPS)` | 2.00 秒 |
| `SMART` + MIROS | 1.00 秒 |
| `SMART` + MuScriptor Large | ≤ 5/6 秒（约 0.833 秒） |
| `SMART` + MuScriptor Medium | ≤ 25/12 秒（约 2.083 秒） |
| `SMART` + MuScriptor Small | ≤ 25/7 秒（约 3.571 秒） |
| `VOCAL_SPLIT`（仅分离） | ≤ 2/3 秒（约 0.666 秒） |
| `SIX_STEM_SPLIT`（仅分离） | ≤ 5/18 秒（约 0.277 秒） |
| 任一钢琴专用直接转写 | 2.50 秒 |

## MuScriptor gated 授权

MuScriptor [Small](https://huggingface.co/MuScriptor/muscriptor-small)、[Medium](https://huggingface.co/MuScriptor/muscriptor-medium)、[Large](https://huggingface.co/MuScriptor/muscriptor-large) 是三个独立的 gated 仓库，不能匿名全自动下载。有效部署条件包括：同一个 Hugging Face 账户已逐项接受条款，并在网页中取得三个仓库各自的读取权限。仅登录 CLI、复制 Space 或同步源码都不会取得这些网页授权。

获批后的凭据位置是 Space `Settings -> Variables and secrets` 中的私有 secret `HF_TOKEN`。公开 variable、仓库和日志都不是受支持的 token 存放位置，并会造成凭据暴露。访客使用部署者已授权的运行实例时不会看到该 secret；复制或重新部署 Space 时，新的部署环境对应新的账户授权和 `HF_TOKEN`。缺少任一授权时，对应 MuScriptor 路线会明确失败，不会改用其它模型。

## 本 Space 使用的模型、固定来源与许可

Hugging Face 卡片顶部的 `models` / `datasets` 元数据列出这个 Space 读取的 Hub 仓库。顶部 `license: mit` **只表示本 Space 自有应用代码使用 MIT**；第三方源码、checkpoint、SoundFont 和数据集仍按各自许可使用。

| 功能 | 实际来源与固定版本 | 在本 Space 中的用途 | 许可/使用边界 |
|---|---|---|---|
| MuScriptor Large / Medium / Small | [`MuScriptor/muscriptor-large`](https://huggingface.co/MuScriptor/muscriptor-large)、[`muscriptor-medium`](https://huggingface.co/MuScriptor/muscriptor-medium)、[`muscriptor-small`](https://huggingface.co/MuScriptor/muscriptor-small) 的固定 revision；推理源码固定为官方 [`v0.3.0`](https://github.com/muscriptor/muscriptor/tree/d73147e75e5b9b0c0a79ebe154587db4fd603e0c) | `SMART` 和逐轨多乐器转写；使用官方 5 秒窗口与 prelude forcing；Beat This 网格驱动 onset 校正、tempo 和小节相位，tempo 复制到每个音符轨；空选时自动识别，非空选择成为生成约束 | 三个权重仓库均 gated、CC BY-NC 4.0；逐项网页授权和具备读取权限的 `HF_TOKEN` 是运行条件。授权、节拍信息或运行时身份校验失败时任务终止 |
| MuScriptor 播放资源 | [`MuScriptor/assets`](https://huggingface.co/MuScriptor/assets/tree/7755beb2da7cb1d3c663ff4a9ad0d0e99437f78f)；FluidSynth `2.5.6` | 把 MIDI 及其乐器分轨合成为可试听预览 | 资源仓库声明 MIT；FluidSynth 为 LGPL-2.1-or-later。FluidSynth 合成失败时返回错误 |
| YourMT3+ 五个 checkpoint | [`mimbres/YourMT3`](https://huggingface.co/mimbres/YourMT3/tree/5e66c1ea173a8186e0d20432b841d3180cc015b5) @ `5e66c1ea173a8186e0d20432b841d3180cc015b5` | 默认多乐器路线及五个逐轨选择项 | 固定 Space revision 声明 Apache-2.0；启动时使用该 revision 与项目的受控兼容补丁 |
| MIROS + MusicFM | [`amt-os/ai4m-miros`](https://github.com/amt-os/ai4m-miros/tree/668a0aa6357bb3f09e767c9ece378956c2ffd182)；[`minzwon/MusicFM`](https://huggingface.co/minzwon/MusicFM/tree/546287d5e3e9ea5b42a4135d1dbca96ac12a0a9c) | 2025 AMT Challenge 路线的完整混音/逐轨多乐器转写 | MusicFM 声明 MIT；MIROS 源码与 fine-tuned checkpoint 上游未声明许可，项目仅保留完整归属与维护者责任记录，不声称额外许可 |
| Leap XE vocals | [`pcunwa/BS-Roformer-Leap`](https://huggingface.co/pcunwa/BS-Roformer-Leap/tree/4e47d6662ae82eaa8b4ac4329fe66099a843b48e) | `VOCAL_SPLIT` 的 vocals WAV | 上游未声明许可；完整边界见第三方声明 |
| PolarFormer accompaniment | [`bgkb/bs_polarformer`](https://huggingface.co/bgkb/bs_polarformer/tree/9158719ee2173edd480a735764627526506fe4af) | `VOCAL_SPLIT` 的 accompaniment WAV | 上游模型卡声明 MIT |
| BS-RoFormer SW Fixed | [`noblebarkrr/mvsepless_resources`](https://huggingface.co/noblebarkrr/mvsepless_resources/tree/370198fbb6997e3f5774778254698794e7b1267d) | `SIX_STEM_SPLIT` 的六条 WAV | 上游未声明许可；分离质量使用 SDR 类指标，与 MIDI F1 分开记录 |
| TransKun V2 / V2 Aug | [`Yujia-Yan/Transkun`](https://github.com/Yujia-Yan/Transkun)；`transkun==2.0.1` 与官方 V2 Aug 文件 | 两条独立钢琴路线 | 包内 V2 资源随 MIT 包发布；V2 Aug 按官方项目发布记录单独固定，两条路线分别加载各自 checkpoint |
| Aria-AMT | [`EleutherAI/aria-amt`](https://github.com/EleutherAI/aria-amt/tree/a1ab73fc901d1759ec3bc173c146b3c6a3040261)；[`loubb/aria-midi`](https://huggingface.co/datasets/loubb/aria-midi/tree/8cc4cf5c83b47f2689ac256a947b2a57c17a4c8b) | 钢琴专用逐轨/直接转写 | 源码 Apache-2.0；checkpoint 为 CC BY-NC-SA 4.0，保持非商用和原许可 |
| ByteDance Pedal | [`piano-transcription-inference==0.0.6`](https://pypi.org/project/piano-transcription-inference/0.0.6/)；[官方 checkpoint](https://doi.org/10.5281/zenodo.4034264) | 保留 sustain pedal `CC64` 的钢琴转写 | 运行包 MIT；checkpoint CC BY 4.0 |

完整归属、文件身份与撤销联系人记录见 [`THIRD_PARTY_NOTICES.md`](https://github.com/mason369/music-to-midi/blob/master/THIRD_PARTY_NOTICES.md)。访问本 Space 不会向访客分发部署者的 `HF_TOKEN`；复制 Space 后的独立部署使用复制者自己的 gated 模型授权和 secret。

## 功能与交互

| 功能 | 行为 |
|------|------|
| 多乐器转写 | `SMART` 可选择 YourMT3+、MIROS 或 MuScriptor Large / Medium / Small；YourMT3+ 默认 `YPTF.MoE+Multi (noPS)`，并提供五种官方 checkpoint。 |
| MuScriptor 乐器选择 | 空选时由模型检测；非空选择在生成阶段屏蔽未选乐器 token，并校验事件流和最终 MIDI。 |
| 音源分离 | Leap XE 90-band 与 PolarFormer 生成 vocals、accompaniment 两条 WAV；`BS-Rofo-SW-Fixed.ckpt` 生成 bass、drums、guitar、piano、vocals、other 六条 WAV。 |
| 逐轨 MIDI | 每条波形音轨都可试听、下载或添加本地音频。勾选“转 MIDI”、从 13 个模型中选择一个，再点击该行“开始转换”；一次只处理该音轨。 |
| 钢琴模式 | 四个钢琴模式直接生成一个 MIDI；ByteDance Pedal 保留延音踏板 CC64。 |
| BPM 与试听 | 默认自动检测唯一 BPM；也可跟随原曲速度变化，或手动设置 30–300 BPM。模型事件先按检测 BPM 映射到音乐 tick，再按工程 BPM 播放，不做量化、过滤或音符重建。 |
| 任务边界 | 每个请求的文件位于独立目录；过期、越界或空文件会使任务失败。主任务与逐轨任务按提交顺序使用同一 GPU 处理通道。 |

## 与 TelkNet 公开工具的对齐边界

`VOCAL_SPLIT` 采用 TelkNet 公开网站展示的模型与输入输出契约：Leap XE 90-band 和 PolarFormer 都读取原混音，分别生成 vocals 与 accompaniment。核验时的公开网站还展示 YourMT3+ / MIROS、六声部逐 stem MIDI 和 TransKun V2 Aug，而网站链接的公开 GitHub `master` 尚未包含这些完整契约。核验范围不含 TelkNet 服务端私有源码，因此“对齐”只指公开的模型与输入输出行为，不覆盖代码、运行环境或结果文件的逐位一致性。

## MuScriptor Large 公开评价

[官方模型卡](https://huggingface.co/MuScriptor/muscriptor-large)在作者自建的 372 首多乐器 `D_Test` 上报告 Onset / Frame / Offset / Drums / Multi F1 为 **60.4 / 72.4 / 48.6 / 49.6 / 47.8**；同表 YourMT3+ Multi F1 为 21.9。这些分数并非 Space 本地实测，也不能外推到所有数据集：论文的 8 个公共跨域集里 Multi F1 高于 YourMT3+ 的有 6 个，另 2 个较低。模型不输出 velocity，权重采用 CC BY-NC 4.0 非商用许可。

Mirelo Studio 另有一个“使用更多数据训练”的私有版本；截至 2026-08-08 没有公开权重、revision 或同协议分数。本 Space 使用可固定和校验的公开 Large / Medium / Small 权重。完整分数与后续候选见仓库的 [`docs/muscriptor-model.md`](https://github.com/mason369/music-to-midi/blob/master/docs/muscriptor-model.md)。

MuScriptor Small / Medium / Large 是三个独立选项。所选档位失败、未授权或额度不足时任务终止。Small 与 Medium 的质量、速度、显存和首段延迟以部署 GPU 上的分别测试为准，参数量不能代替实测结果。

## 输出行为

- `SMART` 输出一个所选后端生成的 MIDI。
- `VOCAL_SPLIT` 生成两条经过校验的 WAV；每条音轨的 MIDI 转换单独进行。
- `SIX_STEM_SPLIT` 生成六条经过校验的 WAV；每条音轨的 MIDI 转换单独进行。
- 分离完成后，用户可以对任意一条分离或新增音轨选择十三条路线之一，并点击该行“开始转换”生成一个独立 MIDI。未勾选、未选择模型或仅改变选项都不会转换。
- 四个钢琴模式各直接输出一个固定后端 MIDI，不创建音轨面板。

## 输出文件生命周期

- 主转换或分离失败时立即删除该请求的输出目录，结果页不提供半成品。
- 逐轨转换失败会显示错误，并保留已经验证成功的分离 WAV；失败轨道没有 MIDI 下载项。
- 成功结果默认在 24 小时后进入过期清理。Space 实例正常退出时删除本实例目录，Gradio 文件缓存也配置为相同清理周期。

## 使用方法

1. 上传一个音频文件并选择模式。
2. 对 `SMART` 或任一钢琴模式，点击“开始转换”并下载单个 MIDI。
3. 对人声/伴奏或六声部分离模式，点击“开始分离”并等待 WAV 波形音轨出现。
4. 如需 MIDI，在目标音轨勾选“转 MIDI”、选择模型，再点击该行“开始转换”。每次只处理这一条音轨。
5. 如需加入其它音频，使用音轨面板的“添加音轨”；它同样不会自动转 MIDI。

## 技术栈

| 层 | 版本或组件 |
|----|------------|
| Space 运行时 | Python 3.12.12、Torch 2.8.0、torchaudio 2.8.0、torchvision 0.23.0、NumPy `>=2,<2.5` |
| ZeroGPU / Web | `spaces==0.51.1`、Gradio 6.17.3、Pydantic 2.10.6 |
| 分离运行时 | `audio-separator==0.44.1`、`onnxruntime-gpu==1.23.2` |
| 转写引擎 | YourMT3+、MIROS、MuScriptor Large / Medium / Small、TransKun V2 / V2 Aug、Aria-AMT、ByteDance Pedal |
| 分离模型 | Leap XE vocals、PolarFormer accompaniment、BS-RoFormer SW Fixed |
| 节拍检测 | Beat This `final0`；所有模式使用同一固定 checkpoint |

Space 的 Torch 2.8 / NumPy 2 环境是独立部署契约，不能用桌面版 Torch 2.7 / NumPy 1.26 依赖覆盖。PolarFormer 依赖 ONNX Runtime `CUDAExecutionProvider`，因此 AMD/ROCm 当前不支持完整七模式；CUDA provider 不可用时启动检查会失败。

Space 源码同步不等于取得第三方模型的额外授权。当前 portable release 的 29 项闭集清单为 25 项 `VERIFIED`、4 项附维护者具名责任记录的 `OWNER_ACCEPTED`、0 项 `BLOCKED`；发布仍会逐次 fail-closed 校验，任何项目未满足就停止。Space 运行时下载的公开制品继续受各上游许可与平台条款约束；MuScriptor Small / Medium / Large 的访问条件包含各自的 Hugging Face 模型条款授权。

## 链接

- [GitHub 仓库](https://github.com/mason369/music-to-midi)
- [桌面版](https://github.com/mason369/music-to-midi/releases)
- [第三方代码与许可证声明](https://github.com/mason369/music-to-midi/blob/master/THIRD_PARTY_NOTICES.md)
