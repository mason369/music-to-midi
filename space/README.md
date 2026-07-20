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
short_description: MuScriptor、YourMT3+、MIROS 与钢琴模型驱动的音频转 MIDI
models:
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

# Music to MIDI - AI Audio to MIDI

将 `MP3`、`WAV`、`FLAC`、`OGG`、`M4A` 音频转换为可编辑 MIDI。Space、桌面版与 Colab 使用相同的七个模式：`SMART`、`VOCAL_SPLIT`、`SIX_STEM_SPLIT`、`PIANO_TRANSKUN`、`PIANO_TRANSKUN_V2_AUG`、`PIANO_ARIA_AMT`、`PIANO_BYTEDANCE_PEDAL`。

五个直接转写模式保持一次点击直接生成 MIDI；两个分离模式只先生成 WAV，并在真实波形音轨面板中让用户逐轨决定是否转 MIDI。选择复选框或模型不会开始推理，必须点击该音轨自己的“开始转换”。

> **ZeroGPU 使用边界**：此部署只承诺短片段试用，不承诺完整长歌端到端完成。[Hugging Face ZeroGPU 文档](https://huggingface.co/docs/hub/main/en/spaces-zerogpu) 当前公开配额为匿名用户每日 2 分钟、登录免费账户每日 5 分钟 GPU。当前保守的最小请求经 `large` GPU 平台倍率折算后已高于匿名额度，因此转换必须先登录。应用会在下载模型前严格估算；超过登录免费账户 300 GPU 秒窗口的任务会明确拒绝，不会静默切换 CPU。长歌请使用 Colab、桌面版或专用 GPU。估算通过也不代表用户仍有足够当日配额或队列容量。

当前公式的精确准入阈值如下。它们不是实测耗时承诺；逐轨手动 MIDI 会按该音轨实际选择的路线重新独立估算：

| ZeroGPU 任务 | 准入音频长度上限 |
|---|---:|
| `SMART` + 默认 `YPTF.MoE+Multi (noPS)` | 2.00 秒 |
| `SMART` + MIROS | 1.00 秒 |
| `SMART` + MuScriptor Large | ≤ 5/6 秒（约 0.833 秒） |
| `VOCAL_SPLIT`（仅分离） | ≤ 2/3 秒（约 0.666 秒） |
| `SIX_STEM_SPLIT`（仅分离） | ≤ 5/18 秒（约 0.277 秒） |
| 任一钢琴专用直接转写 | 2.50 秒 |

## 本 Space 使用的模型、固定来源与许可

Hugging Face 卡片顶部的 `models` / `datasets` 元数据列出了这个 Space 实际读取的 Hub 仓库，便于平台显示来源关系。顶部 `license: mit` **只表示本 Space 自有应用代码使用 MIT**，不会把第三方源码、checkpoint、SoundFont 或数据集重新授权为 MIT。

| 功能 | 实际来源与固定版本 | 在本 Space 中的用途 | 许可/使用边界 |
|---|---|---|---|
| MuScriptor Large | [`MuScriptor/muscriptor-large`](https://huggingface.co/MuScriptor/muscriptor-large) @ `8809fdfbed2affa7ade94a7059e746e3880720e7`；推理源码 [`muscriptor/muscriptor`](https://github.com/muscriptor/muscriptor/tree/302343e8992bdfc619f77f1988168374ed5d675d) | `SMART` 和逐轨多乐器转写；空选自动识别，非空多选成为真实生成约束 | 权重仓库 gated，CC BY-NC 4.0；自行复制/部署前必须在模型页接受条款并为 Space 配置有权限的 `HF_TOKEN`。未授权时该路线明确失败，不改用其他模型 |
| MuScriptor 播放资源 | [`MuScriptor/assets`](https://huggingface.co/MuScriptor/assets/tree/7755beb2da7cb1d3c663ff4a9ad0d0e99437f78f)；FluidSynth `2.5.6` | 把 MIDI 及其乐器分轨合成为可试听预览 | 资源仓库声明 MIT；FluidSynth 为 LGPL-2.1-or-later。只用于真实合成，不用静音或假音频冒充 MIDI 预览 |
| YourMT3+ 五个 checkpoint | [`mimbres/YourMT3`](https://huggingface.co/mimbres/YourMT3/tree/5e66c1ea173a8186e0d20432b841d3180cc015b5) @ `5e66c1ea173a8186e0d20432b841d3180cc015b5` | 默认多乐器路线及五个逐轨选择项 | 固定 Space revision 声明 Apache-2.0；本项目携带受控兼容补丁，不运行时切换可变上游源码 |
| MIROS + MusicFM | [`amt-os/ai4m-miros`](https://github.com/amt-os/ai4m-miros/tree/668a0aa6357bb3f09e767c9ece378956c2ffd182)；[`minzwon/MusicFM`](https://huggingface.co/minzwon/MusicFM/tree/546287d5e3e9ea5b42a4135d1dbca96ac12a0a9c) | 2025 AMT Challenge 路线的完整混音/逐轨多乐器转写 | MusicFM 声明 MIT；MIROS 源码与 fine-tuned checkpoint 上游未声明许可，项目仅保留完整归属与维护者责任记录，不声称额外许可 |
| Leap XE vocals | [`pcunwa/BS-Roformer-Leap`](https://huggingface.co/pcunwa/BS-Roformer-Leap/tree/4e47d6662ae82eaa8b4ac4329fe66099a843b48e) | `VOCAL_SPLIT` 的 vocals WAV | 上游未声明许可；完整边界见第三方声明 |
| PolarFormer accompaniment | [`bgkb/bs_polarformer`](https://huggingface.co/bgkb/bs_polarformer/tree/9158719ee2173edd480a735764627526506fe4af) | `VOCAL_SPLIT` 的 accompaniment WAV | 上游模型卡声明 MIT |
| BS-RoFormer SW Fixed | [`noblebarkrr/mvsepless_resources`](https://huggingface.co/noblebarkrr/mvsepless_resources/tree/370198fbb6997e3f5774778254698794e7b1267d) | `SIX_STEM_SPLIT` 的六条真实 WAV | 上游未声明许可；不会把分离 SDR 写成 MIDI F1 |
| TransKun V2 / V2 Aug | [`Yujia-Yan/Transkun`](https://github.com/Yujia-Yan/Transkun)；`transkun==2.0.1` 与官方 V2 Aug 文件 | 两条独立钢琴路线 | 包内 V2 资源随 MIT 包发布；V2 Aug 按官方项目发布记录单独固定，不互相静默替代 |
| Aria-AMT | [`EleutherAI/aria-amt`](https://github.com/EleutherAI/aria-amt/tree/a1ab73fc901d1759ec3bc173c146b3c6a3040261)；[`loubb/aria-midi`](https://huggingface.co/datasets/loubb/aria-midi/tree/8cc4cf5c83b47f2689ac256a947b2a57c17a4c8b) | 钢琴专用逐轨/直接转写 | 源码 Apache-2.0；checkpoint 为 CC BY-NC-SA 4.0，保持非商用和原许可 |
| ByteDance Pedal | [`piano-transcription-inference==0.0.6`](https://pypi.org/project/piano-transcription-inference/0.0.6/)；[官方 checkpoint](https://doi.org/10.5281/zenodo.4034264) | 保留 sustain pedal `CC64` 的钢琴转写 | 运行包 MIT；checkpoint CC BY 4.0 |

完整归属、文件身份与撤销联系人记录见 [`THIRD_PARTY_NOTICES.md`](https://github.com/mason369/music-to-midi/blob/master/THIRD_PARTY_NOTICES.md)。访问本 Space 不会向访客分发部署者的 `HF_TOKEN`；复制 Space 的用户需要自行接受 gated 模型条款并配置自己的 secret。

## 功能与交互

- **多乐器 MIDI 直接转写**：`SMART` 可选择 YourMT3+、MIROS 或 MuScriptor Large；YourMT3+ 可选择五种官方 checkpoint，默认 `YPTF.MoE+Multi (noPS)`。
- **MuScriptor 真实约束**：空选乐器时自动检测；非空多选会在生成阶段屏蔽未选乐器 token，并校验事件流和最终 MIDI，不是只隐藏界面轨道。
- **人声/伴奏 WAV 分离**：Leap XE 90-band 与 PolarFormer 分别读取原混音，只输出 vocals、accompaniment 两条 WAV，不自动生成 MIDI。
- **六声部 WAV 分离**：`BS-Rofo-SW-Fixed.ckpt` 输出 bass、drums、guitar、piano、vocals、other 六条 WAV，不自动生成 MIDI。
- **真实波形音轨面板**：分离完成后，每条音轨用 `gr.Audio` 显示对应波形，可试听、下载，也可添加本地音频。新增音频会复制到当前请求专属目录。
- **逐轨显式 MIDI**：每条音轨都有“转 MIDI”复选框、十一个明确路线的下拉框和独立“开始转换”按钮。十一路线为五个 YourMT3+ checkpoint、MIROS、MuScriptor Large，以及 TransKun V2、TransKun V2 Aug、Aria-AMT、ByteDance Pedal 四个钢琴模型。一次点击只转换该音轨。
- **钢琴专用直接转写**：四个钢琴模式直接生成一个 MIDI，不显示分离音轨面板；ByteDance Pedal 保留延音踏板 CC64。
- **严格任务隔离**：分离音频、添加音频和逐轨 MIDI 的所有可见路径都必须位于当前请求目录；过期、越界或空文件会明确失败。
- **串行 GPU 调度**：主转换、WAV 分离和逐轨 MIDI 共用同一个 GPU 并发队列，防止多个模型任务争抢显存。

## 与 TelkNet 公开工具的对齐边界

`VOCAL_SPLIT` 的模型与输入输出契约对齐 TelkNet 当前公开网站：Leap XE 90-band 对原混音生成 vocals，PolarFormer 也对原混音独立生成 accompaniment。本轮核验时，该网站也已展示 YourMT3+ / MIROS、六声部逐 stem MIDI 与 TransKun V2 Aug 路线，但其链接的公开 GitHub `master` 仍落后于网站契约。这里的“对齐”不表示本项目拥有 TelkNet 服务端私有实现，也不声称与公开仓库或服务端源码逐行一致、推理环境完全相同或结果文件位级一致。

## MuScriptor Large 公开评价

[官方模型卡](https://huggingface.co/MuScriptor/muscriptor-large)在作者自建的 372 首真实多乐器 `D_Test` 上报告 Onset / Frame / Offset / Drums / Multi F1 为 **60.4 / 72.4 / 48.6 / 49.6 / 47.8**；同表 YourMT3+ Multi F1 为 21.9。它是很强的公开完整混音候选，但这些分数不是 Space 本地实测，也不是跨所有数据集的统一 SOTA：论文的 8 个公共跨域集里 Multi F1 赢 6、输 2，模型还不输出 velocity，并受 CC BY-NC 4.0 非商用许可约束。

Mirelo Studio 另有一个“使用更多数据训练”的私有增强版；截至 2026-07-19 没有公开权重、revision 或同协议分数，所以本 Space 使用的是可固定、可校验的公开 Large 权重，不把私有服务版本冒充成本地模型。完整分数与前沿观察见仓库的 [`docs/muscriptor-model.md`](https://github.com/mason369/music-to-midi/blob/master/docs/muscriptor-model.md)。

MuScriptor Small / Medium 虽已有公开权重，但当前 Space **没有**把它们显示成可选后端，也不会在 Large 失败或额度不足时静默替代。后续只有完成同音频质量、速度、显存、首段延迟和三端一致性验证后，才会作为独立可见路线加入。

## 输出规则

- `SMART` 输出一个所选后端生成的 MIDI。
- `VOCAL_SPLIT` 主按钮只输出两条经过校验的 WAV；不会自动生成 vocal、accompaniment 或 merged MIDI。
- `SIX_STEM_SPLIT` 主按钮只输出六条经过校验的 WAV；不会自动生成六个 stem MIDI 或 merged MIDI。
- 分离完成后，用户可以对任意一条分离或新增音轨选择十一路线之一，并点击该行“开始转换”生成一个独立 MIDI。未勾选、未选择模型或仅改变选项都不会转换。
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
- **转写引擎**：YourMT3+、MIROS、MuScriptor Large、TransKun V2 / V2 Aug、Aria-AMT、ByteDance Pedal
- **分离模型**：Leap XE vocals、PolarFormer accompaniment、BS-RoFormer SW Fixed
- **节拍检测**：librosa

Space 的 Torch 2.8 / NumPy 2 环境是独立部署契约，不能用桌面版 Torch 2.7 / NumPy 1.26 依赖覆盖。PolarFormer 依赖 ONNX Runtime `CUDAExecutionProvider`，因此 AMD/ROCm 当前不支持完整七模式，也不会静默切换到 CPU 假装成功。

Space 源码同步不代表第三方模型或桌面便携包已取得再分发许可；当前 portable release 仍会按 `THIRD_PARTY_NOTICES.md` 的闭集清单 fail-closed 阻断。Space 运行时按所选路线下载公开制品时，也必须遵守各上游许可与平台条款。

## 链接

- [GitHub 仓库](https://github.com/mason369/music-to-midi)
- [桌面版](https://github.com/mason369/music-to-midi/releases)
- [第三方代码与许可证声明](https://github.com/mason369/music-to-midi/blob/master/THIRD_PARTY_NOTICES.md)
