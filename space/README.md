---
title: Music to MIDI
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.12.12"
app_file: app.py
pinned: false
license: mit
suggested_hardware: zero-a10g
short_description: AI 音频转 MIDI，支持多乐器、人声分离、六声部分离和钢琴转写
tags:
  - audio-to-midi
  - midi
  - music-transcription
  - ai-music
  - stem-separation
  - piano-transcription
---

# Music to MIDI - AI Audio to MIDI

将 `MP3`、`WAV`、`FLAC`、`OGG`、`M4A` 音频转换为可编辑 MIDI。Space、桌面版与 Colab 使用相同的七个模式：`SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT`、`PIANO_TRANSKUN`、`PIANO_TRANSKUN_V2_AUG`、`PIANO_ARIA_AMT`、`PIANO_BYTEDANCE_PEDAL`。

五个直接转写模式保持一次点击直接生成 MIDI；两个分离模式只先生成 WAV，并在真实波形音轨面板中让用户逐轨决定是否转 MIDI。选择复选框或模型不会开始推理，必须点击该音轨自己的“开始转换”。

> **ZeroGPU 使用边界**：此部署只承诺短片段试用，不承诺完整长歌端到端完成。[Hugging Face ZeroGPU 文档](https://huggingface.co/docs/hub/main/en/spaces-zerogpu) 当前公开配额为匿名用户每日 2 分钟、登录免费账户每日 5 分钟 GPU。当前保守的最小请求经 `large` GPU 平台倍率折算后已高于匿名额度，因此转换必须先登录。应用会在下载模型前严格估算；超过登录免费账户 300 GPU 秒窗口的任务会明确拒绝，不会静默切换 CPU。长歌请使用 Colab、桌面版或专用 GPU。估算通过也不代表用户仍有足够当日配额或队列容量。

当前公式的精确准入阈值如下。它们不是实测耗时承诺；逐轨手动 MIDI 会按该音轨实际选择的路线重新独立估算：

| ZeroGPU 任务 | 准入音频长度上限 |
|---|---:|
| `SMART` + 默认 `YPTF.MoE+Multi (noPS)` | 2.00 秒 |
| `SMART` + MIROS | 1.00 秒 |
| `VOCAL_SPLIT`（仅分离） | ≤ 2/3 秒（约 0.666 秒） |
| `SIX_STEM_SPLIT`（仅分离） | ≤ 5/18 秒（约 0.277 秒） |
| 任一钢琴专用直接转写 | 2.50 秒 |

## 功能与交互

- **多乐器 MIDI 直接转写**：`SMART` 可选择 YourMT3+ 或 MIROS；YourMT3+ 可选择五种官方 checkpoint，默认 `YPTF.MoE+Multi (noPS)`。
- **人声/伴奏 WAV 分离**：Leap XE 90-band 与 PolarFormer 分别读取原混音，只输出 vocals、accompaniment 两条 WAV，不自动生成 MIDI。
- **六声部 WAV 分离**：`BS-Rofo-SW-Fixed.ckpt` 输出 bass、drums、guitar、piano、vocals、other 六条 WAV，不自动生成 MIDI。
- **真实波形音轨面板**：分离完成后，每条音轨用 `gr.Audio` 显示对应波形，可试听、下载，也可添加本地音频。新增音频会复制到当前请求专属目录。
- **逐轨显式 MIDI**：每条音轨都有“转 MIDI”复选框、十个明确路线的下拉框和独立“开始转换”按钮。十条路线为五个 YourMT3+ checkpoint、MIROS，以及 TransKun V2、TransKun V2 Aug、Aria-AMT、ByteDance Pedal 四个钢琴模型。一次点击只转换该音轨。
- **钢琴专用直接转写**：四个钢琴模式直接生成一个 MIDI，不显示分离音轨面板；ByteDance Pedal 保留延音踏板 CC64。
- **严格任务隔离**：分离音频、添加音频和逐轨 MIDI 的所有可见路径都必须位于当前请求目录；过期、越界或空文件会明确失败。
- **串行 GPU 调度**：主转换、WAV 分离和逐轨 MIDI 共用同一个 GPU 并发队列，防止多个模型任务争抢显存。

## 与 TelkNet 公开工具的对齐边界

`VOCAL_SPLIT` 的模型与输入输出契约对齐 TelkNet 当前公开网站：Leap XE 90-band 对原混音生成 vocals，PolarFormer 也对原混音独立生成 accompaniment。本轮核验时，该网站也已展示 YourMT3+ / MIROS、六声部逐 stem MIDI 与 TransKun V2 Aug 路线，但其链接的公开 GitHub `master` 仍落后于网站契约。这里的“对齐”不表示本项目拥有 TelkNet 服务端私有实现，也不声称与公开仓库或服务端源码逐行一致、推理环境完全相同或结果文件位级一致。

## 输出规则

- `SMART` 输出一个所选后端生成的 MIDI。
- `VOCAL_SPLIT` 主按钮只输出两条经过校验的 WAV；不会自动生成 vocal、accompaniment 或 merged MIDI。
- `SIX_STEM_SPLIT` 主按钮只输出六条经过校验的 WAV；不会自动生成六个 stem MIDI 或 merged MIDI。
- 分离完成后，用户可以对任意一条分离或新增音轨选择十种路线之一，并点击该行“开始转换”生成一个独立 MIDI。未勾选、未选择模型或仅改变选项都不会转换。
- 四个钢琴模式各直接输出一个固定后端 MIDI，不创建音轨面板。

## 输出文件生命周期

- 主转换或分离失败时立即删除该请求的输出目录，不把半成品作为成功结果发布。
- 逐轨转换失败会明确报错，并保留已经验证成功的分离 WAV；不会伪造 MIDI 或标记成功。
- 成功结果默认在 24 小时后进入过期清理。Space 实例正常退出时删除本实例目录，Gradio 文件缓存也配置为相同清理周期。

## 使用方法

1. 上传一个音频文件并选择模式。
2. 对 `SMART` 或任一钢琴模式，点击“开始转换”并下载单个 MIDI。
3. 对人声/伴奏或六声部分离模式，点击“开始分离”并等待 WAV 波形音轨出现。
4. 如需 MIDI，在目标音轨勾选“转 MIDI”、选择模型，再点击该行“开始转换”。每次只处理这一条音轨。
5. 如需加入其它音频，使用音轨面板的“添加音轨”；它同样不会自动转 MIDI。

## 技术栈

- **Space 运行时**：Python 3.12.12、Torch 2.8.0、torchaudio 2.8.0、torchvision 0.23.0、NumPy `>=2,<2.5`
- **ZeroGPU / Web**：`spaces==0.51.0`、Gradio 4.44.1
- **分离运行时**：`audio-separator==0.44.1`、`onnxruntime-gpu==1.23.2`
- **转写引擎**：YourMT3+、MIROS、TransKun V2 / V2 Aug、Aria-AMT、ByteDance Pedal
- **分离模型**：Leap XE vocals、PolarFormer accompaniment、BS-RoFormer SW Fixed
- **节拍检测**：librosa

Space 的 Torch 2.8 / NumPy 2 环境是独立部署契约，不能用桌面版 Torch 2.7 / NumPy 1.26 依赖覆盖。PolarFormer 依赖 ONNX Runtime `CUDAExecutionProvider`，因此 AMD/ROCm 当前不支持完整七模式，也不会静默切换到 CPU 假装成功。

Space 源码同步不代表第三方模型或桌面便携包已取得再分发许可；当前 portable release 仍会按 `THIRD_PARTY_NOTICES.md` 的闭集清单 fail-closed 阻断。Space 运行时按所选路线下载公开制品时，也必须遵守各上游许可与平台条款。

## 链接

- [GitHub 仓库](https://github.com/mason369/music-to-midi)
- [桌面版](https://github.com/mason369/music-to-midi/releases)
- [第三方代码与许可证声明](https://github.com/mason369/music-to-midi/blob/master/THIRD_PARTY_NOTICES.md)