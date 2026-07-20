# MuScriptor 模型研究、分数与项目定位

> 核验日期：2026-07-19。本文只记录可追溯到官方模型卡、论文、代码仓库、Mirelo 官方文章或公开挑战论文的结论。不同数据集和评测协议的分数不混排。

## 1. 结论

`MuScriptor/muscriptor-large` 是当前很有竞争力的**开放权重、完整混音、多乐器音频转 MIDI**模型。本项目把它作为 `SMART` 和分离后逐 WAV 转 MIDI 的高质量候选，而不是宣称它在所有 AMT 数据集、所有乐器和所有指标上都是绝对 SOTA。

这个判断基于三点：

- 在作者自建的 372 首真实多乐器 `D_Test` 上，公开 Large 权重相对论文采用的 YourMT3+ 基线大幅提升，Multi F1 从 21.9 提升到 47.8。
- 在论文列出的 8 个公开跨域数据集上，它的 Multi F1 高于 YourMT3+ 其中 6 个，但在 RWC-C 和 RWC-R 上较低；Onset/Offset 也并非处处领先。
- 它公开了推理代码和三档权重，但仍有不输出 velocity、36 组乐器分类、稀有乐器长尾、较高显存/算力需求和 CC BY-NC 4.0 非商用许可等边界。

因此，准确表述应是：**MuScriptor Large 是截至核验日最值得优先试用的公开完整混音 AMT 模型之一，也是作者数据上明显强于 YourMT3+ 基线的新模型；目前没有足够独立、同协议证据把它写成跨所有 benchmark 的无条件 SOTA。**

## 2. 模型身份与发布时间

| 项目 | 已核实信息 |
|---|---|
| 开发者 | Kyutai、Mirelo AI、IRCAM 研究人员 |
| 论文 | [MuScriptor: An Open Model for Multi-Instrument Music Transcription](https://arxiv.org/abs/2607.08168)，arXiv v1：2026-07-09 |
| 官方代码 | [muscriptor/muscriptor](https://github.com/muscriptor/muscriptor)，MIT；公开 release `v0.2.1`：2026-07-10 |
| 官方权重 | [MuScriptor/muscriptor-large](https://huggingface.co/MuScriptor/muscriptor-large)，gated、CC BY-NC 4.0，并附额外合法使用条件 |
| Hub 时间 | [Hugging Face API](https://huggingface.co/api/models/MuScriptor/muscriptor-large) 记录仓库创建于 2026-06-30、最后更新于 2026-07-10 |
| 本项目固定代码 | commit `302343e8992bdfc619f77f1988168374ed5d675d`，包版本 `0.2.2a1` |
| 本项目固定权重 | revision `8809fdfbed2affa7ade94a7059e746e3880720e7`，`model.safetensors` 5,465,642,136 bytes |

这解释了“模型页面似乎更早、代码和正式资料随后才出现”的现象：Hub 仓库在正式发布前已于 6 月 30 日建立；论文和 Mirelo 文章在 7 月 9 日发布，GitHub release 与当前权重 revision 在 7 月 10 日更新。Hub 的 `createdAt` / `lastModified` 是仓库元数据时间，不是训练完成时间，也不表示更早已有同一套公开代码、文档和最终 revision。

## 3. 架构、训练数据与输出能力

- decoder-only Transformer；模型卡把 Large 写作约 1.3B 参数，当前代码 README 四舍五入写作 1.4B，结构为 48 层、隐藏维度 1536、24 个注意力头。
- 输入为 16 kHz 单声道音频的 5 秒分片，使用 512-bin mel-spectrogram，100 Hz 帧率。
- 输出为 MT3 风格事件序列，包含 onset、offset、pitch 和 instrument；128 个 MIDI program 映射到 `MT3_FULL_PLUS` 的 36 个乐器组。
- 训练链路为约 145 万（1.45M）MIDI 的合成预训练、17 万首（170k）/约 11,000 小时真实音乐微调，再以 300 首高质量转写做类 GRPO 的强化学习后训练。
- 支持乐器条件输入。官方接口会在生成阶段屏蔽未选乐器 token；本项目还会对流式事件和最终 MIDI 再校验一次，发现越界就拒绝发布。
- 不生成 velocity；鼓只评 onset；同一乐器、同一音高的重叠同音不能由当前 tokenizer 完整表示。

## 4. `D_Test` 主要分数

以下是[官方模型卡](https://huggingface.co/MuScriptor/muscriptor-large)给出的 headline 结果：372 首作者保留的真实多乐器测试曲目，MuScriptor Large 使用完整 `D_Synth + D_Real + D_RL` 训练链路与 CFG=2。分数均为 instrument-agnostic `mir_eval` F1；Multi F1 还要求乐器预测正确。

| 模型 | Onset F1 | Frame F1 | Offset F1 | Drums F1 | Multi F1 |
|---|---:|---:|---:|---:|---:|
| YourMT3+ `YPTF.MoE+Multi (noPS)` | 32.5 | 45.5 | 17.8 | 41.4 | 21.9 |
| MuScriptor Large | **60.4** | **72.4** | **48.6** | **49.6** | **47.8** |
| 绝对提升 | **+27.9** | **+26.9** | **+30.8** | **+8.2** | **+25.9** |

`D_Test` 来自作者内部真实音乐数据的严格留出集。论文没有提供这个测试集的公开下载入口，所以这些数字适合说明论文内部对比，不能当作外部团队已经独立复现的公共排行榜。

### 训练阶段与 CFG 消融

论文 Table 1 的 1.3B 模型结果如下：

| 训练数据 | CFG | Onset | Frame | Offset | Drums | Multi |
|---|---:|---:|---:|---:|---:|---:|
| `D_Synth` | 1 | 26.1 | 51.3 | 14.2 | 23.1 | 15.2 |
| `D_Synth` | 2 | 34.5 | 48.9 | 16.1 | 21.0 | 16.2 |
| `D_Synth + D_Real` | 1 | 52.5 | 69.4 | 42.0 | 44.7 | 41.7 |
| `D_Synth + D_Real` | 2 | 54.4 | 69.3 | 42.3 | 43.3 | 41.6 |
| `D_Synth + D_Real + D_RL` | 1 | **60.4** | **73.3** | **49.0** | **50.2** | **48.2** |
| `D_Synth + D_Real + D_RL` | 2 | 60.4 | 72.4 | 48.6 | 49.6 | 47.8 |

官方当前代码说明已发布的 post-RL 权重应保持 `cfg_coef=1`；模型卡 headline 表仍列 CFG=2。项目调用固定上游接口的默认 CFG=1，不把 CFG=2 的模型卡数字冒充为本地实测结果。

## 5. 公共跨域数据集对比

以下数字来自[论文 Table 2](https://arxiv.org/html/2607.08168)，比较 MuScriptor Large 与 YourMT3+。每格为 `Onset / Frame / Offset / Drums / Multi` F1，`–` 表示数据集没有鼓指标。

| 数据集 | YourMT3+ | MuScriptor Large | Multi F1 判断 |
|---|---|---|---|
| Bach10 | 59.8 / 66.0 / 48.0 / – / 26.4 | 43.1 / 85.0 / 36.0 / – / **34.7** | MuScriptor 较高 |
| Dagstuhl ChoirSet | 22.3 / 51.0 / 10.8 / – / 2.6 | 14.4 / 80.7 / 11.5 / – / **11.5** | MuScriptor 较高 |
| PHENICX-Anechoic | 56.7 / 58.9 / 18.7 / – / 12.2 | 56.1 / 74.6 / 32.6 / – / **25.7** | MuScriptor 较高 |
| RWC-P | 36.1 / 51.6 / 20.6 / 36.5 / 19.1 | 46.1 / 61.2 / 25.6 / 42.1 / **25.6** | MuScriptor 较高 |
| RWC-C | 71.7 / 71.3 / 44.3 / 8.7 / **40.5** | 67.7 / 70.5 / 36.9 / 23.7 / 36.0 | YourMT3+ 较高 |
| RWC-G | 36.9 / 49.4 / 20.5 / 25.6 / 17.2 | 44.7 / 58.7 / 24.4 / 29.2 / **23.7** | MuScriptor 较高 |
| RWC-J | 52.9 / 57.2 / 31.1 / 30.6 / 26.4 | 59.4 / 62.7 / 33.9 / 31.3 / **31.8** | MuScriptor 较高 |
| RWC-R | 46.1 / 61.5 / 28.6 / 51.3 / **23.1** | 47.1 / 68.6 / 24.8 / 36.5 / 20.3 | YourMT3+ 较高 |

这张表说明 MuScriptor 的真实音乐训练显著改善了多数跨域场景，但也说明它并不是在每个数据集、每种时间精度指标上都优于 YourMT3+。

## 6. 模型规模与乐器条件

论文的规模消融只使用 `D_Real`、CFG=2，因此不能与完整训练链路的 headline 分数混用：

| 规模 | 参数量 | Onset | Frame | Offset | Drums | Multi | 是否公开权重 |
|---|---:|---:|---:|---:|---:|---:|---|
| 60M | 60M | 47.7 | 65.7 | 35.3 | 39.8 | 35.2 | 否，论文消融 |
| Small | 约 100M | 51.2 | 67.2 | 38.7 | 41.5 | 38.2 | 是 |
| Medium | 约 300M | 52.4 | 68.0 | 40.3 | 42.0 | 39.7 | 是 |
| Large | 约 1.3B | **53.2** | **68.7** | **41.0** | **42.5** | **40.5** | 是，本项目当前使用 |

同一消融设置下，提供正确乐器列表会把 Onset / Frame / Offset / Drums / Multi 从 `51.6 / 66.5 / 40.1 / 40.6 / 38.7` 提升到 `53.2 / 68.7 / 41.0 / 42.5 / 40.5`。这也是本项目把乐器多选实现为真实生成约束、而不是显示过滤器的依据。

## 7. Mirelo Studio 的“改进版本”是什么

[Mirelo 官方文章](https://mirelo.ai/blog/turning-audio-to-midi)明确区分了两件事：共同开源的模型，以及“使用更多数据训练、部署在 Mirelo Studio 的改进版本”。截至核验日，官方没有发布这个 Studio 版本的：

- Hugging Face 权重仓库或 revision；
- 参数量、训练数据明细或 checkpoint 名称；
- 与公开 Large 同协议的分数；
- 可离线调用的代码/模型映射。

所以它**不能被认定为** `MuScriptor/muscriptor-large` 的同一权重，也不能在本项目里通过重命名或切换 revision 获得。当前项目只集成可下载并可做哈希校验的公开 Large 权重。

## 8. 前沿与未来候选

| 方向 | 当前证据 | 项目判断 |
|---|---|---|
| MuScriptor Small / Medium | 官方公开 103M / 307M 权重，与 Large 共用接口；官方代码把 Medium 作为速度/质量折中、Small 作为 CPU 实用选项 | 最接近可落地的后续候选。接入前应在同一批本地音频上测质量、速度、首段延迟和显存，不因体积小就静默替代 Large。 |
| Mirelo Studio 改进版 | 官方只确认“更多数据、更准确”，没有公开权重或同协议分数 | 仅列观察项；在公开可下载、许可明确、可复现前不能集成。 |
| MIROS / MusicFM 路线 | [2025 AMT Challenge 论文](https://arxiv.org/abs/2603.27528)给出 MIROS F1 0.5998、YourMT3-YPTF-MoE-M 0.5938、MT3 0.3932；测试集是 76 个受约束合成短片段 | 已作为独立后端集成，但挑战分数不能与 MuScriptor `D_Test` 或 Slakh 分数横比。 |
| 更强乐器检测与抗泄漏 | 挑战论文指出密集复音、相似音色、乐器 hallucination/leakage 仍是主要失败模式，并计划扩大 jazz/pop、稀有乐器与乐器检测评测 | 新模型优先看 instrument-aware F1、泄漏率和三乐器以上退化，而不只看单一 note F1。 |
| 多模态 AMT | 2026 年研究开始联合音频与 MusicXML/图像乐谱 | 这类系统通常要求额外乐谱输入，不是本项目“仅音频转 MIDI”的直接替换；只有公开纯音频推理路径后再评估。 |

未来模型进入本项目至少需要满足：公开且可固定的权重、明确许可、可复现推理代码、真实 MIDI writer、相同输入上的本地 A/B、长音频边界、流式事件契约、显存/速度数据，以及桌面、Space、Colab 三端一致性。只发表论文、只提供网页服务或只给不同协议高分，不等于已经成为可部署后端。

## 9. 主要来源

- [官方模型卡与分数](https://huggingface.co/MuScriptor/muscriptor-large)
- [MuScriptor 论文](https://arxiv.org/abs/2607.08168)
- [官方推理代码](https://github.com/muscriptor/muscriptor)
- [Mirelo：Turning audio to MIDI](https://mirelo.ai/blog/turning-audio-to-midi)
- [2025 AMT Challenge 结果与未来方向](https://arxiv.org/abs/2603.27528)
