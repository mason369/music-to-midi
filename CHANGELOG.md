# 更新说明 / Changelog

## v1.8.0 — 2026-09-23

**2026-09-23 发布修复 / Release fix**

- Space 镜像构建会额外安装 `gradio[mcp,oauth]==6.17.3`；其 MCP 扩展要求 Pydantic 2.11.10–2.12.5。将 Space 专用依赖调整为 `pydantic==2.12.5` 与对应的 `pydantic-core==2.41.5`，解除此前 2.10.6 导致的解析冲突；Colab 独立运行环境不受影响。
- Space 源码同步补齐模型就绪探针模块及对应变更触发路径，修复镜像构建后因缺少 `src.model_profile_runtime_probe` 而启动失败的问题。
- Space 同步包增加页面启动使用的品牌图标；同步工作流自身变化也会触发重新部署，避免缺文件修复无法自动生效。
- ZeroGPU 平台将 `spaces==0.51.3` 作为构建依赖注入；Space 自身改为接受兼容版本范围，避免与平台注入版本冲突。
- The Space image installs `gradio[mcp,oauth]==6.17.3`, whose MCP extra requires Pydantic 2.11.10–2.12.5. Updated the Space-specific pins to `pydantic==2.12.5` and its matching `pydantic-core==2.41.5`, resolving the dependency conflict from 2.10.6. Colab uses a separate environment and is unchanged.
- Included the model-readiness probe module and its change trigger in the Space source sync, fixing startup failure after image build when `src.model_profile_runtime_probe` was absent.
- Included the brand icon required at Space startup, and made changes to the sync workflow itself trigger redeployment.
- The ZeroGPU builder injects `spaces==0.51.3`; the Space dependency now accepts the compatible range rather than conflicting with the platform-selected version.

**2026-09-21 更新 / Update**

### 中文

#### 和弦显示与短片段

- 桌面和弦栏改为按根音区分的 12 色低饱和圆角区块，等音根音保持同色；当前和弦使用更亮的填色与顶边，悬停突出边框，播放头统一为青色。
- 完整保留 ChordMini 的 170 类和弦含义，以紧凑记谱显示升降号、大小调、七和弦及其他扩展类型；转位标签将低音级数转换为正确拼写的低音音名，悬停仍可查看模型原始标签和起止时间。
- 修复短和弦被压成省略号或无字细条的问题：空间不足时在区块上方自动错行显示完整名称，并用引线连接对应片段；标签可点击试听。保持片段真实起止边界，不拉宽短片段覆盖后续和弦。
- 缩放、窗口宽度变化与横向滚动时重新安排标签并避让；滚动中保留行高，减少钢琴卷帘上下跳动。无和弦和未知标签使用中性色且不发声，保留小节切分、工程 BPM 与源音频时间偏移对齐。
- 即使关闭播放跟随，当前和弦高亮仍随播放位置更新；通过时间索引与可见区域裁剪减少绘制开销。

#### MIDI 音符反馈

- 桌面、Space 与 Colab 结果编辑器增加当前位置的 MIDI 音符高亮，随播放、暂停和跳转更新，并遵守静音及独奏后的可听声部状态。
- 音符悬停时突出显示，并提供音高、MIDI 编号、声部、力度、起止时间和时长。浏览器提示自动保持在视口内，支持窄屏换行；反馈层不拦截原有选区、拖动与编辑操作。
- 桌面使用音符时间索引及局部重绘，保留既有缓存瓦片；Space/Colab 共用透明反馈画布，只绘制可见音符的反馈。显示与悬停不修改音符、时值、选区或导出数据。

#### 国际化、文档与验证

- 同步 `zh_CN` 与 `en_US` 的和弦提示及音符详情，检查全部 619 个翻译键、键集合、占位符和新增文案回退；更新两种语言的 README、界面说明与和弦/音符反馈架构图。
- 新增 22 个和弦及桌面反馈用例、2 个浏览器运行时反馈用例，覆盖完整标签与引线命中、170 类记谱、等音配色、时间边界、偏移、播放状态、静音、悬停及数据不变性。连同既有编辑、响应式、跨平台、API、Docker、便携包和启动入口回归，共 519 个不同测试通过。
- 使用真实歌曲、7357 个 MIDI 音符和 232 个原音和弦区间完成桌面实际播放、暂停、跳转、滚动、缩放和有声试听验收；在 0.5×/1×/4× 缩放与 100/125/150 BPM 组合下，本机测得约 60.5–62.2 帧/秒。短片段标签完整、互不重叠且可命中正确片段。
- 本地 Gradio 加载 Space/Colab 正式共享运行时，通过真实浏览器验证中英文悬停、播放到结束、暂停及跳转高亮和 320px 窄视口边界，未出现浏览器错误或警告。
- 桌面、便携 App 与可执行 App 复用 Qt 组件；Space/Colab 同步 MIDI 反馈，和弦栏仍沿用桌面范围。独立 Web/API、Docker 与便携 Web 没有该钢琴卷帘，按 API、部署和打包契约验证。本轮没有生成新发行包或部署远端 Space/Colab。

### English

#### Chord display and short segments

- Restyled the desktop chord lane with rounded blocks in 12 muted root-note colors; enharmonic roots share a color. The current chord has a brighter fill and top edge, hovering emphasizes its border, and the playhead is cyan.
- Preserved the meaning of all 170 ChordMini classes while displaying compact accidentals, major/minor qualities, seventh chords and other extensions. Inversion labels convert bass degrees to correctly spelled bass-note names; tooltips retain the original model label and time interval.
- Fixed short chords collapsing into ellipses or unlabeled slivers. When a name does not fit, its complete label is placed in staggered rows above the blocks, with a leader line to the corresponding segment; the label also supports click-to-audition. Native time boundaries remain intact without widening a short segment over its neighbor.
- Labels are rearranged to avoid collisions during zoom, resizing and horizontal scrolling. Row height is retained while scrolling to reduce vertical piano-roll movement. No-chord and unknown labels remain neutral and silent, preserving bar splits, project-BPM alignment and source-audio offsets.
- Current-chord highlighting now follows playback position even with playback follow disabled. Time indexes and visible-region clipping reduce rendering work.

#### MIDI note feedback

- Desktop, Space and Colab result editors now highlight MIDI notes at the current position during playback, pause and seeking, respecting audible parts after mute and solo selections.
- Hovering emphasizes a note and shows its pitch, MIDI number, part, velocity, start/end times and duration. Browser tooltips stay within the viewport and wrap on narrow screens; the feedback layer leaves existing selection, dragging and editing interactions available.
- Desktop uses note-time indexes and localized repaints while retaining cached tiles. Space/Colab share a transparent feedback canvas that renders feedback only for visible notes. Display and hover feedback do not modify notes, timing, selections or exported data.

#### Localization, documentation and validation

- Updated chord hints and note details for `zh_CN` and `en_US`, checking all 619 translation keys, key completeness, placeholders and new-text fallbacks. Updated READMEs, interface documentation and the chord/note feedback architecture diagram in both languages.
- Added 22 chord/desktop feedback cases and 2 browser-runtime feedback cases covering complete labels and leader-line targets, all 170 notations, enharmonic colors, time boundaries, offsets, playback state, mute, hover and unchanged data. Together with existing editor, responsive-layout, platform, API, Docker, portable-package and launcher regressions, 519 distinct tests passed.
- Used an actual song with 7357 MIDI notes and 232 source-audio chord intervals for desktop playback, pause, seeking, scrolling, zoom and audible chord audition. Local measurements reached approximately 60.5–62.2 frames per second across the tested 0.5×/1×/4× zoom and 100/125/150 BPM combinations. Short-segment labels were complete, non-overlapping and mapped to the correct segments.
- Loaded the production Space/Colab shared runtime in local Gradio and used a real browser to verify Chinese/English hover details, playback to the end, pause/seek highlighting and tooltip bounds at a 320px viewport, with no browser errors or warnings.
- Desktop, portable App and executable App share the Qt components. Space/Colab receive MIDI feedback; the chord lane retains its desktop scope. Standalone Web/API, Docker and portable Web have no such piano roll and were checked through API, deployment and packaging contracts. This change did not produce new release packages or deploy remote Space/Colab instances.

**2026-09-20 更新 / Update**

### 中文

- 修复本地桌面和弦栏在跟随播放时逐段跳动：卷帘每帧的小数像素位移直接触发和弦重绘，不再不断重启 80 毫秒内容刷新计时器，也不再依赖 12 像素滚动条步进才更新。
- 保留和弦与小节内容变化的延迟重算，持续滚动时新的小节布局仍能生效；和弦识别、原生时间边界和试听行为保持原有契约。
- 新增 7 个回归用例，覆盖中英文、0.5×/1×/4× 缩放下滚动条未变化时的实际重绘，以及连续滚动中的小节布局更新；旧实现全部失败，修复后通过。
- 使用反馈中的 7357 音符、约 243.5 秒结果测量真实桌面绘制：默认缩放下和弦刷新由约 7.7 帧/秒提高到约 62.5 帧/秒，典型帧间隔由约 128 毫秒降至约 16 毫秒。该数据为本机测量，不代表所有设备的固定帧率。
- 实际声卡播放在 0.5×/1×/4× 视图缩放、100/125/150 BPM 下测得约 59–62 帧/秒；暂停、手动滚动、缩放、重新跟随、前后跳转、点击和弦试听及停止归零均通过。结果编辑、和弦、响应式布局、跨平台与打包契约、国际化共 476 项相关测试通过，中英文均完成实际界面检查。
- 桌面与便携 App 共用此组件；Space/Colab 没有桌面和弦栏，独立 Web/API、Docker 与便携 Web 没有该钢琴卷帘，因而不涉及这条 Qt 刷新链路。同步两种语言的 README 与和弦架构文档。

### English

- Fixed the desktop chord lane jumping during playback follow. Each frame of fractional piano-roll motion now requests a chord repaint directly, instead of repeatedly restarting the 80 ms content refresh timer or waiting for a 12-pixel scrollbar step.
- Retained delayed rebuilds for chord and bar content changes. New bar layouts can take effect during continuous scrolling; chord recognition, native time boundaries and audition behavior retain their existing contracts.
- Added 7 regression cases covering actual repaints without scrollbar changes in both languages at 0.5×/1×/4× zoom, plus bar-layout updates during continuous scrolling. All fail on the old implementation and pass with the fix.
- Measured the actual desktop renderer using the reported 7357-note result, approximately 243.5 seconds long. At default zoom, chord updates increased from about 7.7 to 62.5 frames per second, with the typical frame interval dropping from about 128 ms to 16 ms. These are local measurements, not a fixed frame-rate guarantee for every device.
- Actual audio-device playback measured approximately 59–62 frames per second at 0.5×/1×/4× view zoom and 100/125/150 BPM. Pause, manual scrolling, zoom, follow re-entry, forward/backward seeks, chord audition and stop/rewind passed. All 476 related result-editor, chord, responsive-layout, platform, packaging-contract and localization tests passed, with actual interface checks in both languages.
- Desktop and portable App share this component. Space/Colab have no desktop chord lane; standalone Web/API, Docker and portable Web have no such piano roll, so this Qt refresh path does not apply to them. Updated READMEs and chord architecture documentation in both supported languages.

**2026-09-19 更新 / Update**

### 中文

#### MIDI 结果编辑与导出

- 修复结果生成后声部默认静音、音符不可见的问题。新结果默认显示并启用全部声部，清除上一份结果遗留的静音与独奏状态。
- 桌面、Space 和 Colab 的结果编辑器支持同时独奏多个声部；可将电贝司、原声贝司等多个识别声部一起试听并导出。手动静音与独奏选择分别保存，取消独奏后恢复相应状态。
- 新增一键导出所有声部的 MIDI ZIP，每个声部一个文件；保留可听声部组合导出。桌面与浏览器编辑器导出反映当前编辑结果与工程 BPM，批量导出不必逐个切换独奏。
- 独立 Web/API 的任务结果与已保存项目结果增加分轨 ZIP 下载。分轨处理保留音符配对，以及 tempo、拍号、控制器等其他事件的原始绝对 tick，不改写源 MIDI。
- 桌面和共享浏览器结果编辑器支持 Space 播放/暂停、Shift+Space 停止并回到开头；输入控件中的空格输入不触发播放。

#### 本地桌面和弦

- 在钢琴卷帘上方增加原音和弦栏，默认使用已核对的 TelkNet `chordmini_btc` 路线。采用官方 ChordMini BTC CL (full)，不使用旧 Omnizart、teacher checkpoint 或 MIDI 音符模板估计。
- 固定 ChordMini 源码提交 `aa6e3a8d7b017f082fd2aaff9329d5c26af49c03`，权重为 `btc_model_best.pth`，SHA-256 为 `e0a12ca6d881f81e01dfb459b4836ec7dac8d8f87b0170a412b5807256b3a1fc`。源码清单与权重按真实文件字节严格校验。
- 使用原音 PCM32 解码、官方特征与推理实现、170 类词表、50% 重叠、logit 投票及 9 帧 Gaussian 平滑；固定 seed 42 并关闭 TF32，与核验的 TelkNet 默认参数一致。
- 和弦栏保留模型原始起止边界，显示时按小节切分，并随工程 BPM、横向滚动、缩放及 MuScriptor 前导小节偏移保持对齐。点击和弦通过 FluidSynth 试听；无和弦或未知标签不发声。
- 识别在独立后台进程中执行并共享 GPU 锁；结果按原音 SHA-256 和识别参数缓存。失败显示原因并支持点击重试，关闭窗口可取消等待或运行中的任务，试听结束和关闭时释放音频资源。

#### 安装、打包与文档

- 新增 `download_chordmini_model.py`，并接入 `download_sota_models.py`。便携 App 的准备、暂存、打包和成品检查均包含固定 ChordMini 资源；内部 worker 支持源码与可执行入口，隔离官方 `src` 命名空间。
- 同步 PyInstaller、便携构建、Docker 资源清单、HF 同步与发布契约，并补充 ChordMini 的第三方来源与许可记录。
- 明确分卷压缩包的使用方式：所有卷放在同一目录，从第一卷开始解压，由支持该格式的解压器连续读取其余分卷，不必分别解压 13 次。
- 同步中英文界面、README、平台说明和和弦架构文档，覆盖 `zh_CN` 与 `en_US` 两种支持语言。

#### 验证记录与范围

- Windows / Python 3.11.15 / PyTorch 2.7.0+cu128 / RTX 4070 Ti SUPER：13 条公开 MIDI 路线直接转写、两种分离模式，以及两种分离后各选一个真实 WAV 跑遍 13 条路线，共 39 次 MIDI 推理与 2 次分离全部通过。
- 13 个真实模型输出逐一完成 PyQt 界面验收；共享 Gradio 编辑器在 Chromium 中完成 13 模型 × 2 语言的 26 组操作，校验 26 个分轨 ZIP 与 26 个组合 MIDI；API 的任务/项目两种路由 × 两种语言共 52 次请求通过。
- 在同一本机环境中，与核验的 TelkNet 原始 BTC 函数比较，同一测试音频的 10 段和弦标签、起止时间与帧间隔完全一致；应用内部 worker 入口也通过。和弦试听、失败重试和关闭取消经过实际验证。
- 完整测试套件为 1627 项通过、1 项默认跳过；显式启用 Windows 音频设备测试后，对应文件 6 项全部通过，包含该跳过项。后续受影响部分的 311 项及 276 项定向回归均通过，不累加重复用例。
- 两种语言的 JSON、键集合和占位符一致性、实际界面渲染，以及 Black、isort、关键 Python 静态检查、JavaScript 语法与 Git 空白检查通过。
- 本轮为源码与本机验收，未生成新的便携包、EXE 或 Docker 镜像，未在远端 Space / Colab GPU 重新逐模型验收。桌面和弦栏仅适用于本地桌面；其他交付面按其共享运行时、API 与打包契约验证。本机未安装 360 压缩，未声称覆盖其各个版本；短音频功能验收不代表任意歌曲的识别精度零误差。

### English

#### MIDI result editing and export

- Fixed newly generated results starting with muted or hidden parts. New results show and enable every part and clear mute/solo selections inherited from the previous result.
- Desktop, Space and Colab result editors now allow multiple soloed parts. Parts such as electric and acoustic bass can be auditioned and exported together. Manual mute and solo selections are stored separately so leaving solo restores the corresponding state.
- Added one-click export of every MIDI part as a ZIP containing one file per part, while retaining combined export of audible parts. Desktop and browser editor exports reflect current edits and the project BPM; batch export no longer requires soloing each part separately.
- Standalone Web/API job results and saved project results now offer a MIDI parts ZIP. Splitting preserves note pairing and the original absolute ticks of tempo, time signatures, controllers and other events, without modifying the source MIDI.
- Desktop and shared browser result editors support Space to play/pause and Shift+Space to stop and return to the beginning. Typing spaces in input controls does not start playback.

#### Local desktop chords

- Added an original-audio chord lane above the piano roll, using the verified TelkNet `chordmini_btc` route by default. It runs official ChordMini BTC CL (full), without legacy Omnizart, the teacher checkpoint or MIDI note-template estimation.
- Pinned ChordMini source commit `aa6e3a8d7b017f082fd2aaff9329d5c26af49c03` and checkpoint `btc_model_best.pth`, SHA-256 `e0a12ca6d881f81e01dfb459b4836ec7dac8d8f87b0170a412b5807256b3a1fc`. Source manifest entries and checkpoint contents are strictly verified against actual file bytes.
- Uses PCM32 decoding of the original audio, official features and inference, a 170-class vocabulary, 50% overlap, logit voting and 9-frame Gaussian smoothing. Seed 42 and disabled TF32 match the verified TelkNet defaults.
- The chord lane preserves native prediction boundaries, divides display blocks at bar lines, and stays aligned with project BPM, horizontal scrolling, zoom and MuScriptor's leading-bar offset. Clicking a chord auditions it through FluidSynth; no-chord and unknown labels remain silent.
- Recognition runs in a separate background process using the shared GPU lock. Results are cached by source-audio SHA-256 and inference recipe. Failures show their cause and support click-to-retry; closing the window cancels waiting or active work and releases audition audio resources.

#### Setup, packaging and documentation

- Added `download_chordmini_model.py` and integrated it into `download_sota_models.py`. Portable App preparation, staging, packaging and final checks include pinned ChordMini resources. The internal worker supports source and executable entry points while isolating the official `src` namespace.
- Updated PyInstaller, portable builds, Docker resource manifests, HF synchronization and release contracts, and documented ChordMini provenance and licensing.
- Clarified split-archive extraction: keep every volume in one directory and start extraction from the first volume. An extractor supporting the archive format reads subsequent volumes automatically; extracting all 13 separately is unnecessary.
- Updated interface translations, READMEs, platform instructions and chord architecture documentation for both supported languages, `zh_CN` and `en_US`.

#### Validation and scope

- On Windows / Python 3.11.15 / PyTorch 2.7.0+cu128 / RTX 4070 Ti SUPER, all 13 public MIDI routes ran directly, both separation modes ran, and one actual separated WAV from each mode ran through all 13 routes: 39 MIDI inference runs and 2 separation runs passed.
- Each of the 13 actual model outputs passed PyQt interface acceptance. Chromium exercised the shared Gradio editor across 13 models × 2 languages, validating 26 part ZIPs and 26 combined MIDI downloads. API job/project routes across both languages passed 52 requests.
- In the same local environment, the adapter matched the verified original TelkNet BTC function on all 10 chord labels, start/end times and frame duration for the same test audio. The application's internal worker entry point also passed. Chord audition, failure retry and cancellation on close were exercised.
- The full suite reported 1627 passed and 1 skipped by default. Explicitly enabling Windows audio-device tests produced 6 passes in the relevant file, including that skipped test. Subsequent focused runs of 311 and 276 affected tests passed; repeated tests are not added to the total.
- Both locales passed JSON, key-set and placeholder checks and actual rendered-interface validation. Black, isort, critical Python static checks, JavaScript syntax checks and Git whitespace checks passed.
- This validation covers source code and local execution. No new portable package, EXE or Docker image was built, and models were not rerun on remote Space / Colab GPUs. The chord lane is a local desktop feature; other delivery surfaces were checked through their shared runtimes, APIs and packaging contracts. 360 Compression was not installed locally, so its individual versions were not tested. Short-audio functional acceptance does not imply error-free recognition of every song.
