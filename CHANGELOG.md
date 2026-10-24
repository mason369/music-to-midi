# 更新说明 / Changelog

## v1.10.0 — 2026-10-24

### 统一发布 / Unified release

- 发布标签先核对工程版本、运行时版本和精确源码提交。Actions 统一运行源码与打包契约、Windows/Linux 完整 GPU 包、容器发行包和 Space 同步，全部成功后由唯一任务公开 Release；容器工作流不再抢先创建同名 Release。
- Windows/Linux 分卷必须连续完整，每份包记录来源提交、大小与 SHA-256。发布前复核构建清单，上传保持草稿，远端附件集合、大小、状态和摘要一致后才公开。上传错误直接失败，不自动重试或覆盖已公开版本。
- Release 仅附软件包和统一下载校验值，不单独附测试报告、截图、日志、SBOM 或构建清单。Docker 配置、镜像 digest、依赖清单和必要许可仍在包内；对外说明改为完整中英文功能与使用说明，原始验收记录留本地。
- Space 同步后验证精确 Hub 提交、`RUNNING` 状态、Gradio 版本及实际组件/事件图，避免源码推送成功掩盖应用启动失败。构建、运行、HTTP 和超时错误均阻断 Release，不自动重启、重装或重试。
- The release tag must match project/runtime versions and the exact source commit. One Actions workflow coordinates source/contracts, complete Windows/Linux GPU bundles, Docker distribution and Space sync; one final job publishes the Release only after every dependency succeeds. Container builds no longer create a competing Release.
- Portable volumes must be complete and consecutive. Build manifests bind file sizes and SHA-256 to the source commit. Uploads stay in a draft until the remote asset set, state, sizes and digests match. Upload errors fail immediately, without automatic retries or overwriting a public version.
- Release assets contain software packages and one download-checksum file. Test reports, screenshots, logs, standalone SBOMs and build manifests are excluded. Docker configuration, image digests, dependency inventories and required notices remain inside packages. Bilingual release notes describe features and usage; raw acceptance evidence remains local.
- Space sync verifies the exact Hub revision, `RUNNING` status, Gradio version and live component/event graph. A successful push cannot hide an application startup failure. Build, runtime, HTTP and timeout errors block the Release without automatic restarts, dependency reinstalls or retries.

以下保留各阶段的实现与验证记录，其中“候选、未发布”描述对应当时的阶段。
The implementation and validation history below retains each stage's original candidate/publication status.

### 乐谱、和弦追加复核 / Score and chord follow-up

- 修复 GUI 诊断在完成信号已到、截止计时器先于下一轮询触发时误报超时：到期时核对真实完成状态，仍保留真正超时和谓词异常；真实 Qt 事件复现修复前失败、修复后通过，不延长等待或重试。
- Fixed false GUI diagnostic timeouts when completion precedes the deadline but no further polling tick runs. The deadline checks actual completion while retaining genuine timeouts and predicate errors. Real Qt events reproduce the failure before the fix and pass after it, without longer waits or retries.

- 修复真实歌曲在“75% 强度＋自动网格＋三连音”下因中途谱号变化而排谱失败：谱号保持自己的精确位置，不再将该位置作为音符切分点。346 音符的实曲片段保留全部音符，记谱起点／结束时间最大误差分别约 0.50／1.25 ms；源 MIDI 不变，不新增转换或重试。
- Fixed real-song engraving with 75% strength, automatic grid and triplets: mid-note clef changes keep their own positions instead of introducing unnotatable note fragments. All 346 excerpt notes remain; maximum notation start/end errors are approximately 0.50/1.25 ms. Source MIDI remains unchanged, with no added conversion or retry.

- 修复排谱导入再次按自由演奏重建节奏，导致已量化音符移位的问题；保持既有音乐网格，显式生成三连音及跨小节延音线，覆盖 4/4、6/8、9/8、5/8、7/16、7/32、11/32 和五档网格。按音乐时间恢复拍号，兼容重复声明及等长的 3/4↔6/8 切换。补回 MuseScore MIDI 导入遗漏的不在音符起点上的速度事件，所有源速度事件进入 MusicXML。共享乐谱缓存升级到版本 2，桌面、Space/Colab、独立 Web/API、Docker 与便携入口使用相同规则。
- 吉他／贝斯指法采用独立弦位匹配，检查音域、同时发音、持续音和延音线，修复贪心分配造成的同弦冲突。无法完整分配的声部保留全部五线谱与 MIDI，ZIP 内提供中英文说明和逐小节冲突报告；不删除、移调或截断音符来强行输出 Tab。
- 修复 GUI 验收中 MuseScore 反复弹出 Qt 平台插件错误：外部排谱进程不再继承父程序的 PyQt 插件／QML 路径和平台选择，Windows 使用官方包自带的 `windows`，Linux 使用其 `offscreen`。不修改父进程或系统配置。另修正乐谱界面测试的 QApplication 生命周期，消除其破坏后续 Linux 和弦音量动画的问题；产品音量渐变及原有断言不变。
- 关闭 MIDI 导入的时值简化，并恢复上游导入器截断的同音高重叠：先按乐器、音高和起点核对音符，再以独立声部及必要的附加谱表保留实际结束时间。指定整曲恢复 31 处重叠延音，最终 MusicXML 的全部 7,286 个音符音高、量化后起点和结束时间均与私有 MIDI 副本一致；8 个 PDF、122 页完成渲染和逐页检查。两条吉他轨包含超过六弦容量的和弦，明确报告并保留完整五线谱；贝斯 Tab 通过弦位校验。
- 新运行官方 ChordMini CUDA 得到 232 个原生时间区间，与既有成功结果逐项一致，未知区间监听静音，峰值无削波；没有人工标注，不宣称识别准确率。
- Windows 与 Linux 固定采用官方 stable 通道 MuseScore Studio 4.7.6（261022093，源码 `96480d58445e3b24b230a3fe7d4725d639aa2eb3`），同步校验值、许可证、Docker、发布工作流和打包路径。4.7.5 验收仍捕获退出崩溃：音频线程在清理期间重新初始化，留下失效订阅。采用上游 [关闭顺序修复 #34781](https://github.com/musescore/MuseScore/pull/34781)，拒绝缺少修复的旧运行时；不吞非零退出码、不重跑转换、不禁用音源。2026-10-24 核验时，4.7.6 的 GitHub 正式发行流程仍在等待，本项目仅做本地候选验收。
- Fixed a second human-performance reconstruction during score import that shifted already quantized notes. Engraving keeps the existing grid, writes explicit triplets and barline ties across five grids and 4/4, 6/8, 9/8, 5/8, 7/16, 7/32 and 11/32, and restores every source tempo event, including off-note changes omitted by MuseScore MIDI import. Meters are restored by musical position, including repeats and equal-length 3/4↔6/8 changes. Shared score caches are now version 2 across desktop, Space/Colab, Web/API, Docker and portable entry points.
- Guitar/bass fingering uses independent string matching with pitch-range, polyphony, sustain and tie checks. Unassignable parts retain complete standard notation/MIDI and receive bilingual instructions plus measure-level conflict reports instead of conflicting Tab. No notes are removed, transposed or shortened to force an assignment.
- Fixed repeated MuseScore Qt platform-plugin dialogs by isolating the child from the parent's PyQt plugin/QML paths and platform selection: the official Windows runtime uses `windows`, Linux uses `offscreen`. Parent/system settings are untouched. The score UI test now uses the shared QApplication lifecycle so it cannot break later Linux chord-volume animations; production gain ramps and existing assertions remain unchanged.
- Disabled MIDI-import duration simplification and restored same-pitch overlaps shortened by the upstream importer. Instrument/pitch/onset matching precedes restoration through independent voices and additional staves when needed. The requested song restores 31 overlapping durations; all 7,286 pitches, quantized starts and ends match the private MIDI copy. All 122 pages across eight PDFs were rendered and inspected. Two guitar parts exceed six-string capacity and are explicitly reported while retaining full standard notation; bass Tab passes string validation.
- A fresh official ChordMini CUDA run produced 232 native intervals identical to the earlier successful result, with silent unknown intervals and unclipped monitoring. No annotated reference supports an accuracy claim.

Both platforms pin the official stable-channel MuseScore Studio 4.7.6 build 261022093 at source `96480d58445e3b24b230a3fe7d4725d639aa2eb3`, with matching hashes, licensing and packaging. Acceptance still caught a native shutdown crash in 4.7.5: audio initialization could run during teardown and leave dangling subscriptions. Upstream [shutdown-order fix #34781](https://github.com/musescore/MuseScore/pull/34781) addresses this race; older runtimes are rejected. Nonzero exits remain errors, without conversion retries or disabled audio. As checked on 2026-10-24, upstream GitHub publication was still pending; this project remains a local candidate.

### 官方 0.4.0 融合与发版复核 / Official 0.4.0 integration and release preparation

- 实际 MuScriptor 推理包升级为未修改的官方 0.4.0，固定 `8a1ef88672cc23c26619adaff37ec48d82c0b906`，同步七文件身份校验、接口检查、安装器、Space、Docker、CI、打包与第三方清单。保留三档权重、精度、五秒窗口和官方解码；既有后端接受项目唯一一次 Beat This 分析，时间轴由共享层统一处理，不新增版本转换路线或重复推理。
- 根治本次复现的 Windows 项目清单 WinError 5：存在性探针进入原有状态互斥，构造器只解析稳定父目录，消除清单元数据句柄与原子替换的冲突。两组真实 Win32 句柄／事件屏障测试在旧实现确定性失败、修复后通过；没有重试、等待补丁、吞异常或备用清单。
- 修复独立 Web 项目面板切换语言时默认名称、状态与可访问标签残留旧语言，保留自定义名称及当前选择；中英文真实浏览器验证。修复音色控件从不同端口加载脚本被 CSP 拦截的问题，脚本／样式同源交付并复用共享实现，覆盖独立前后端、Docker 与 Gradio，不放宽安全策略。
- 项目结果在切换语言等重绘中继续隐藏独立任务专用的删除按钮；Linux 项目包可导入 Windows 后直接恢复 MIDI 和乐谱，不重新推理。
- Windows 全量 1,865 通过、270 子测试通过；本机 WSL Linux CUDA 全量 1,829 通过、36 项 Windows 专用跳过、270 子测试通过。Linux 原生包移目录后完成全部 13 条 MIDI 路线和两种 WAV 分离模式的真实 GPU 片段验收，以及浏览器上传、转换和 MuseScore 乐谱下载；相关后续同源修复另有 100 项 Windows 回归通过，不把重叠批次累计为总数。

- 修复 Type 0/1 MIDI 首拍纠错按轨读取原音偏移的问题：conductor 轨中唯一的偏移现在供所有同步轨道共用，保留正／负偏移，与 BPM、倍频及空小节裁切组合时仍保持原音相对时间；Type 2 序列分别读取自己的偏移。新增六组真实多轨、变速 MIDI 回归，修改前六组均能复现问题。没有增加转写调用、音符过滤或精度降级。
- 修复结果页用外框宽度计算流式布局导致的英文窄窗口控件裁切，改用实际内容宽度；测试同时核对布局所需高度与控件边界。
- 修正 FLAC 跨平台验收假设：Windows 的 FFmpeg 可拒绝尾部标签，Linux 的版本可完整解码；拒绝时核对原错误，成功时逐样本核对 PCM。生产严格解码、完整帧数／样本 MD5 校验及坏帧拒绝保持不变。真实音频设备测试同时检查目标音量即时更新与下一缓冲区实际应用，不再用短于设备回调间隔的固定等待断言。
- 版本、Docker 默认标签与安装契约准备为 1.10.0；发布工作流在资源准备后执行受影响回归。增加 `[no publish]` 提交标记及契约测试：保持 CI 与镜像构建／冒烟，阻止本次推送同步 Hugging Face 和发布 GHCR；不创建生产标签、Release 或部署。
- 指定 243 秒歌曲使用官方 0.4.0 Large 完成 49/49 窗口、7,286 音符、6 声部／7 轨，耗时 413.47 秒，旧基线 406.17 秒如实保留。同片段旧→新→新→旧对照的 697 个事件完全一致，平均生成 15.5720→15.5411 秒，未观察到速度回归。新旧整曲差异符合官方 tick 取整和鼓时值由 10 ms 改为 1/32 拍的语义。真实 FluidSynth/MuseScore General 输出 48 kHz/24-bit WAV、320 kbps MP3，无削波；纠错保留全部音符载荷、原音相对误差小于 0.63 ms，不重新推理。没有标注参考，不将一致性当作准确率评分；详见[候选说明](docs/releases/v1.10.0.md)。
- Upgraded the actual inference package to unmodified official MuScriptor 0.4.0 at `8a1ef88672cc23c26619adaff37ec48d82c0b906`, aligning seven-file identity/API checks, installers, Space, Docker, CI, packages and notices. Weights, precision, five-second windows and official decoding remain; the existing backend consumes one project Beat This analysis and shared timing, with no version-labelled route or extra inference.
- Fixed the reproduced Windows manifest WinError 5 by protecting existence probes with the existing state mutex and resolving only the stable parent directory during construction. Two real Win32-handle/barrier regressions fail deterministically before repair and pass afterward. No retries, delay patches, swallowed errors or alternate manifests were added.
- Web project language changes now update untouched defaults, statuses and accessible labels while retaining custom names/selection, verified in both locales. SoundFont scripts/styles are served from the page origin through the shared implementation, fixing cross-port script blocking in standalone frontend/backend, Docker and Gradio without weakening CSP.
- Project-result redraws, including language changes, keep standalone-job deletion controls hidden. A Linux project archive imports into Windows and restores MIDI/scores without repeating inference.
- Full Windows: 1,865 passed plus 270 subtests. Local WSL CUDA: 1,829 passed, 36 Windows-only skips plus 270 subtests. Relocated native Linux packages exercised all 13 MIDI routes and two WAV-separation modes on actual GPU audio, plus browser upload/conversion and MuseScore download. The later same-origin change passed 100 affected Windows checks; overlapping batches are not summed.
- Fixed shared source origins in synchronous Type 0/1 MIDI: a conductor-only offset now applies to every track during downbeat correction and trimming, including positive/negative offsets and BPM/beat-scale combinations. Type 2 sequences retain independent origins. All six new real multitrack regressions reproduced the issue before the fix. No extra transcription, note filtering or precision change was introduced.
- Fixed narrow English editor controls clipping by measuring the frame's actual content width; regression checks cover both required layout height and control bounds. Cross-platform FLAC checks now require exact PCM when FFmpeg accepts trailing tags and the original diagnostic when it rejects them. Strict production validation and corrupted-frame rejection are unchanged. Device checks verify both immediate gain targets and actual buffer consumption instead of assuming a fixed short callback interval.
- Prepared 1.10.0 application/Docker defaults and resource-backed release regressions. A tested `[no publish]` commit gate preserves CI/container build-smoke checks but prevents this push from publishing to Hugging Face/GHCR. No production tag, Release or deployment is created.
- Official 0.4.0 Large completed all 49 chunks of the requested 243-second song: 7,286 notes, six voices/seven tracks, 413.47 seconds versus the honestly retained 406.17-second baseline. Controlled old/new/new/old excerpt runs yielded identical 697-event sequences and mean generation 15.5720→15.5411 seconds, with no observed slowdown. Whole-song MIDI differences match official tick rounding and drums changing from 10 ms to 1/32 beat. FluidSynth/MuseScore General produced unclipped 48 kHz/24-bit WAV and 320 kbps MP3; corrections retain payloads within 0.63 ms source-relative error without inference. No annotated reference supports an accuracy claim; details are in the candidate notes.

### MIDI 时间轴、纠错与导出增强 / MIDI timing, correction and export

以下保留第一阶段实现与验证记录；其中 0.3.0 包及尚未打包的状态已由上面的 0.4.0 融合和 Windows/WSL 验收替代。

The following preserves first-stage implementation and validation history; its 0.3.0 package and pre-packaging status are superseded by the integration and Windows/WSL acceptance above.

- 修通自适应速度图在桌面、Space/Colab 共用编辑器、工程重开、原音/MIDI 联动试听、音符编辑、量化及 MIDI/WAV/分轨/乐谱导出中的传递。工程变速缩放整张速度图；保留踏板、控制器、pitch bend、SysEx、音符释放力度及其他保留事件。新增工程倍率元数据，裁切后只剩一个速度段时仍能恢复参考时间轴，避免二次变速。
- 新增显式、不改变原音相对演奏时间的 BPM、拍号、首个强拍和 0.25×／0.5×／1×／2×／4× 拍点倍频纠错，与真实工程变速分开并保留既有倍率。沿用 4–400 BPM 范围，越界先报错并保留当前结果；尚未失焦的 BPM 数字输入也会在应用时提交。
- 保留五档网格、默认 1/32、全部轨道及所选音符范围，增加 0–100% 强度、三连音和自动网格。0% 不产生编辑；三连音使用分数 tick 步长，最后才取整，避免累计偏移。变速段按音乐拍位置吸附；键盘移动跨速度边界时使用当地网格，低于 10 毫秒的细网格音符不再被强行拉长。
- 新增默认关闭的短休止填补、开头空小节裁切及速度图平滑／简化。休止从音符释放处寻找同乐器下一起音，支持重叠和声；裁切同步处理全部事件、速度图与原音的有符号偏移。平滑同时验证 25 毫秒与 1/16 拍界限，无法满足时明确失败。原始 MIDI、音频保持完整。
- 同步独立 Web/API、主任务与逐轨 CLI、已保存项目；Docker、便携 Web 与桌面可执行入口复用共享实现。PDF/MusicXML/分谱及适用 Tab 沿用当前量化设置，在私有副本排版，各组设置独立缓存。补齐两种语言全部新文案、README、操作说明和架构图。保留已验证的官方 MuScriptor 0.3.0 推理、固定权重及 Torch 2.7 / NumPy 1.26 桌面基线，本轮移植有价值的 0.4.0 工作流能力，不升级整包或修改官方推理精度／解码。
- 整合版本全量 1,851 项通过、5 项设备验收跳过；最后的浏览器 BPM 输入提交修正后，相关 126 项再次通过。真实 Chrome 验收覆盖中英文共享结果与独立 Web、320 像素窄屏、MIDI 下载完成事件、FluidSynth 试听和 MuseScore 4.7.4 五文件乐谱包；桌面手动变速后纠错经真实 SoundFont 合成回归。两种语言各 660 个桌面／共享键、283 个独立 Web 键通过完整性与占位符检查。Space/Colab、Docker、便携／可执行入口通过对应源码及交付契约检查；本轮未重新部署、打包或执行七模式模型／物理音频设备全验收。详见 [时间轴与导出说明](docs/midi-timing-export.md)。
- Adaptive tempo maps now survive playback, editing, quantization, reopened projects and MIDI/WAV/stem/sheet exports. Project speed scales the whole map while retaining controllers, sustain, pitch bend, SysEx and note-release velocities. Explicit playback-rate metadata prevents double scaling when trimming leaves a single tempo segment.
- Added explicit grid BPM, meter, source-relative downbeat and power-of-two beat corrections while retaining performance timing and the current project-speed ratio. The existing 4–400 BPM range is validated before replacing a result, and pending numeric BPM input is committed first. The five grids, default 1/32 and all-track/selected-note scopes remain; strength, triplets and automatic selection are new. Zero strength is a no-op; fractional tick steps avoid cumulative rounding, tempo-boundary nudges follow musical time and sub-10 ms notes retain their length during edits.
- Optional gap filling, leading-bar trimming and tempo smoothing are off by default. Chord releases use the next same-instrument onset; trimming moves the full map/events and original-audio offset together. Smoothing enforces 25 ms and 1/16-beat limits and fails explicitly if they cannot be met. Sources remain intact. Desktop, shared Space/Colab editors, standalone Web/API, primary/manual CLI and saved-project sheet endpoints use the options; Docker and portable/executable entries reuse the implementation. PDF/MusicXML/part/Tab exports operate on private copies and cache different settings separately.
- Official MuScriptor 0.3.0 inference, pinned weights and the desktop Torch 2.7 / NumPy 1.26 baseline remain unchanged. Both supported locales, READMEs, instructions and architecture were updated. The integrated version passed 1,851 tests with five device-acceptance skips; 126 relevant tests passed again after the final browser input fix. Real bilingual Chrome, 320 px layouts, MIDI downloads, FluidSynth and a five-file MuseScore score package were exercised, along with desktop manual-speed correction. Each locale passed 660 shared/desktop and 283 standalone-Web key/placeholder checks. Cloud, Docker and portable/executable delivery contracts were checked without rebuilding, redeploying or claiming seven-model/physical-device acceptance in this stage.

### 和弦 FLAC 解码 / Chord FLAC decoding

- 修复桌面和弦轨在音频完整、但 FLAC 附带尾部标签时出现 `invalid sync code / invalid frame header` 并显示识别失败的问题。已声明帧数的原生 FLAC 通过现有 libsndfile 无损解码为原采样率、原声道 PCM32，核对完整帧数和文件提供的样本 MD5；支持尾部 ID3v1 / APEv2 和前置 ID3v2（含页脚），原文件保持不变。其他格式与未声明长度的 FLAC 沿用严格 FFmpeg。
- 真正的坏帧、截断、流信息错误和样本校验不一致仍会失败，不丢弃片段、不补静音、不调用模型或写入成功缓存；新增错误完整覆盖 `zh_CN`、`en_US`。保留固定 ChordMini 模型、官方前端、推理参数、原生时间边界和成功缓存身份。桌面源码、便携 App / 可执行 App 共用修复，Space/Colab、独立 Web/API、Docker 与便携 Web 不提供桌面和弦栏。
- 本轮 455 个不同用例通过：288 项解码、和弦、同步播放与国际化测试，165 项交付面契约测试，以及两种语言的真实 Windows Qt / CUDA ChordMini / FluidSynth 失败重试验收。新增 28 项解码回归；`zh_CN`、`en_US` 两种语言的全目录键集、解析、占位符及本次四条新增文案的无回退检查通过。带尾部标签 FLAC 的真实模型输出与相同 PCM 的 WAV 对照完全一致。反馈者仅提供截图，本轮以能复现相同报错的样本验收，未声称逐文件确认其原音。尚未重新打包或独立发布。
- 设备回归与依赖真实鼠标事件的测试分开执行：和弦原生设备验收通过，既有悬停用例在其原有 offscreen 环境通过。
- Fixed desktop chord analysis failing with `invalid sync code / invalid frame header` on complete FLAC audio followed by trailing tags. Native FLAC with a declared frame count now uses the existing libsndfile decoder for lossless, native-rate/channel PCM32, with full frame-count and available sample-MD5 verification. Trailing ID3v1 / APEv2 and leading ID3v2 tags, including footers, are supported without changing the source. Other formats and unknown-length FLAC retain strict FFmpeg decoding.
- Actual frame corruption, truncation, invalid stream information and sample-checksum mismatches still fail before inference or success-cache publication, with no skipped audio or inserted silence. New errors cover both `zh_CN` and `en_US`. Pinned ChordMini resources, the official frontend, inference settings, native boundaries and cache identities are preserved. Source, portable and executable desktop Apps share the fix; browser/API delivery surfaces do not expose this chord lane.
- Validation passed 455 distinct cases: 288 decoding, chord, synchronized playback and localization checks, 165 delivery contracts, and real Windows Qt / CUDA ChordMini / FluidSynth failure-and-retry acceptance in both locales. There are 28 new decoding regressions. Both supported locales, `zh_CN` and `en_US`, passed full-catalog parsing, key-set and placeholder checks, plus fallback checks on all four strings added in this change. Real model segments for tagged FLAC exactly match an equivalent-PCM WAV reference. Only a screenshot was supplied; acceptance uses samples reproducing the same error and does not claim verification of the reporter's original file. No release package was rebuilt or independently published in this stage.
- Physical-device checks ran separately from mouse-event tests: native chord-device acceptance passed, and existing hover checks passed under their original offscreen platform.

## v1.9.0 — 2026-10-12

- Space 说明中的模型研究链接改为 GitHub 绝对地址，避免同步到 Hugging Face 后指向不存在的 docs 目录；补齐本次乐器条件、时间轴说明的英文内容，并将 Space 与英文 README 的组件数量/状态纠正为既有清单的 31 项、26 VERIFIED、5 OWNER_ACCEPTED、0 BLOCKED。未修改任何许可证或维护者接受记录。
- Space model-investigation links now use absolute GitHub URLs so the synchronized card does not reference a missing docs directory. English conditioning and timing explanations are included, and Space/English README inventory wording now matches the existing 31 components: 26 VERIFIED, five OWNER_ACCEPTED, none BLOCKED. License and maintainer acceptance records are unchanged.
- 运行时与项目元数据统一为 1.9.0，Docker 自托管/生产 Compose、环境模板和生产安装脚本同步使用 `v1.9.0` 镜像。新增完整中英文发布说明，包含自 v1.8.0 以来的功能、修复、下载方式、实测结果与保留限制。
- Runtime and project metadata are aligned at 1.9.0; self-host/production Compose files, environment examples and production setup select `v1.9.0` images. Full Chinese/English release notes cover features and fixes since v1.8.0, downloads, verification and remaining limitations.
- MuScriptor 导入音频和新增分轨默认保留空乐器条件，不再由文件名推断曲目编制；桌面、Space、Colab 与独立 Web 同步，手动及项目中保存的选择继续生效。中英文说明明确全选不等于自动识别。官方 35 类条件生成退化未改动模型或解码处理。
- 修正 MuScriptor 在官方 writer 写出前误报“禁用 onset 校正、零小节偏移”的日志。现在观测官方最终 BeatGrid 的原始返回值，文件成功发布后报告 onset delay、小节偏移和净偏移；写出前保持待计算，复用实例清空旧值，失败不发送完成信息。保持官方 MIDI 字节一致，没有增加推理、重试或音符分析。
- 修正稀疏钢琴 BPM 对短乐句的误判：汇总真实乐句内拍间隔，不把长空白当作连续拍点；保留节拍可信度与变速证据，不新增节拍检测或转写。
- Aria 长短音频预处理对齐官方分支，并让首窗读取与模型加载并行。复用官方已有重叠窗口恢复非法序列，未恢复区间明确报错，禁止将缺失时段的 MIDI 当成完整结果。
- Aria MIDI 依赖固定到官方 `ariautils` 提交 `7c815755c6889a784b6a1b3343e9e8daa3e7d128`，使用上游对同起点和零时值音符的修正；根目录、Space 与 Colab 安装约束同步，Space 修复安装同时覆盖两个 Aria 包，PyInstaller 收集依赖来源元数据。未加入自定义 token 约束、项目级音符修补或模型精度变更。新增错误文案覆盖全部语言 `zh_CN`、`en_US`。
- 发布前全仓后台回归 **1,783 项通过、3 项真实音频设备验收跳过**，覆盖桌面、Space/Colab、Web/API、Docker、便携/可执行与发布入口。真实 Large 五乐器 20 秒样本输出 405 音符、无 token 上限警告，MIDI 与官方 writer 对照字节一致；此前 20 秒/317 秒真实事件重放亦一致。当前 Aria 最终 GPU 与官方对照证据、性能及边界见[钢琴分析](docs/piano-transcription.md)，时间轴数值和验收范围见[MuScriptor 分析](docs/muscriptor-model.md#2026-10-12-时间轴日志修正与发布前核验)。
- 复核现有 31 项组件清单（26 VERIFIED、5 OWNER_ACCEPTED）并纠正 AGENTS 中过期描述；许可证与维护者接受记录未修改。当前固定运行时身份检查通过；`pip check` 仍有安装脚本已记录的 Aria/torchaudio 与 audio-separator/NumPy 声明冲突。以上本机核验不代替安装包、镜像或云端应用的构建与部署验收，发布结果以对应 Actions 和 Release 为准；官方已知生成问题与未完成的设备验收不作解决声明。
- MuScriptor imports and new stems now leave instrument conditioning empty by default across desktop, Space, Colab and standalone Web. Filenames no longer impose conditions; explicit and saved selections remain effective. Both locales explain that selecting all instruments is not automatic detection. The upstream 35-condition generation issue is unchanged.
- MuScriptor timing telemetry now observes the unchanged official writer's final BeatGrid and reports onset delay, bar offset and net shift after successful publication. Pending values are unknown, reused instances reset their timing, and failed writing emits no completion. MIDI bytes are preserved with no extra inference, retries or note-analysis pass.
- Sparse piano tempo fitting now pools real within-phrase intervals and retains uncertainty and variable-tempo evidence without inventing beats across silence or adding beat detection or transcription passes.
- Aria preprocessing follows its official short/long audio branches; first-window reading overlaps model loading. Existing official overlapping windows recover invalid sequences when possible; unrecovered intervals fail explicitly instead of publishing incomplete output.
- Aria's MIDI dependency is pinned to unmodified official ariautils `7c815755c6889a784b6a1b3343e9e8daa3e7d128`, including upstream fixes for same-start and zero-duration notes. Root, Space and Colab constraints agree; Space repairs both Aria packages and PyInstaller includes source metadata. No custom token constraints, project-level note repairs or precision changes were introduced. New errors cover all supported locales, `zh_CN` and `en_US`.
- Full background regression: **1,783 passed, 3 physical audio-device checks skipped**, covering desktop, Space/Colab, Web/API, Docker, portable/executable and release contracts. A fresh 20-second Large/five-instrument GPU run produced 405 notes without token-limit warnings and byte-identical output to the official writer. Recorded 20-second and 317-second event replays also matched. Detailed Aria and MuScriptor evidence and performance boundaries are linked above.
- The existing 31-component inventory (26 VERIFIED, 5 OWNER_ACCEPTED) and pinned local runtime identities passed review; stale AGENTS wording was corrected without changing license or acceptance records. Two documented pip metadata conflicts remain in the intentional desktop runtime. Local source checks do not replace package, image or cloud deployment acceptance; publication status belongs to the corresponding Actions and Release. Known upstream generation limits and unverified hardware cases remain unresolved.

- 补充完整流行歌曲实测：240.9836 秒原曲经 MuScriptor Large 官方自动识别输出 8 轨、6,560 个音符，无 token 上限警告；RTX 4070 Ti SUPER 上含加载与节拍分析耗时 337.439 秒。三首歌曲各取 5 秒，35 类条件下均未生成可用音符，其中两次达到 2,000 token 上限；不宣称所有歌曲必现，也不将五分钟当成安全阈值。上限按每个 5 秒窗口计算，317 秒循环样本的既有验证不代表最长支持时长。
- 在独立 Torch/torchaudio 2.5.0+cu124 诊断环境运行完整官方 Aria 流程，同一 240.9836 秒钢琴音频在第 8 窗仍产生非法序列，下一窗恢复后输出 6,213 条有效音符。应用现有 2.7.0+cu128 环境与模型保持原样；其已验收结果为 6,598 条有效音符。不同环境的音符数量不是质量评分，官方非法序列不能声称根治。依赖声明冲突按维护者决定保留。
- Additional full-song test: official automatic conditioning with MuScriptor Large completed a 240.9836-second pop recording with 8 tracks and 6,560 notes, without token-limit warnings. Total time including loading and beat analysis was 337.439 seconds on an RTX 4070 Ti SUPER. Three distinct five-second excerpts produced no usable notes with all 35 conditions; two exhausted the 2,000-token limit. The limit applies per five-second window, not per song. Neither universal failure nor a five-minute safe threshold is claimed; the earlier 317-second loop is not a maximum-duration benchmark.
- A separate Torch/torchaudio 2.5.0+cu124 diagnostic environment reproduced an invalid Aria sequence at window 8 through the complete official pipeline. Its next window recovered, yielding 6,213 valid notes for the 240.9836-second piano recording. The application's 2.7.0+cu128 environment and models remain unchanged, with 6,598 valid notes in its accepted run. Note-count differences are not quality scores; upstream invalid sequences are not claimed fixed. Dependency metadata conflicts remain by maintainer decision.

### 2026-10-11 — 和弦监听 / Chord playback


- 桌面和弦栏新增“和弦随曲播放”和 0～200% 独立音量。首次默认关闭随播、音量 100%，两项偏好持久保存；关闭随播仍支持点击和弦单独试听，0% 静音两种试听，静音时点击不会暂停歌曲。窄窗口自动换行，中英文文案、键盘操作、准备状态与失败重试同步适配。
- 和弦随播按 ChordMini 原始区间与 MuScriptor 开头偏移合成 SoundFont PCM，接入原音/MIDI 的单一音频时钟，统一暂停、跳转、工程 BPM 和倍速。长和弦不因小节显示切分重复发声，N/X 区间保持静音。音量无需重复合成，200% 实际增强输出；响度校准、最长 20 ms 增益过渡与双声道联动前视峰值保护用于试听，原 MIDI、编辑历史及导出保持原样。
- 后台合成支持取消、缓存与代次校验。关闭随播、静音、切换音色和关闭结果页会阻止过期音频接入；错误显示真实原因并允许明确重试。便携 App / 可执行 App 共用桌面实现；Space、Colab、独立 Web/API、Docker、便携 Web 没有和弦栏，保留各自的交付契约。
- 本轮共 674 个不同测试通过，覆盖和弦音频、同步走带、结果编辑器、下载导出、响应式布局、国际化，以及 Space/Colab、Web/API、Docker、部署和便携/可执行入口契约。中英文两套语言各 633 个键完成解析、完整性和占位符一致性检查；两种语言均在真实 Windows Qt 窗口与本机 FluidSynth / 音频设备上验证随播、跳转、变速、静音及点击试听。本次未重新打包或发布。
- Added desktop **Chords follow song** and independent **0–200% chord volume**, initially off/100% with persistent preferences. Follow off preserves click auditions; 0% mutes both without pausing the song on a muted click. Controls wrap in narrow windows and support keyboard input, both locales, preparation states and explicit failure retry.
- Chord accompaniment renders native ChordMini intervals and the MuScriptor lead-in offset onto the existing original/MIDI audio clock. Pause, seek, project BPM and speed stay aligned; visual bar splits do not retrigger sustained chords and N/X stays silent. Volume changes reuse audio, with real 200% boost, loudness calibration, gain ramps of at most 20 ms and stereo-linked lookahead peak protection. Monitoring does not alter MIDI, edit history or exports.
- Background synthesis supports cancellation, caching and generation checks. Disabling follow, muting, changing SoundFonts and closing results reject stale audio. Errors retain their cause and can be retried. Portable/executable desktop Apps share the implementation; Space, Colab, standalone Web/API, Docker and portable Web retain their separate delivery contracts without a chord lane.
- Validation passed 674 distinct tests covering chord audio, synchronized transport, the result editor, downloads and exports, responsive layouts, localization, and the Space/Colab, Web/API, Docker, deployment and portable/executable entry contracts. Both supported locales passed parsing, completeness and placeholder checks across all 633 keys. Both were also exercised in real Windows Qt windows with local FluidSynth and an audio output device for follow playback, seek, speed, mute and click auditions. No release package was rebuilt or published.

### 2026-10-02 — 导出错误处理 / Export error handling

- 修复桌面结果页下载 MIDI 时，导出异常被重新抛回 Qt 动作回调而导致程序闪退的问题（#13）。下载入口统一覆盖 BPM/资源检查、保存对话框和 MIDI/立体声 WAV 写入异常；失败显示既有本地化错误提示，并将完整调用栈写入日志，保留当前结果供修正问题后重试。取消保存及成功导出的 MIDI 音符、工程 BPM、事件和原始文件处理保持原有行为。
- 增加独立 Qt 子进程回归，覆盖目录冲突、源 MIDI 缺失、参考 BPM 缺失、保存对话框错误、立体声源文件缺失和资源未就绪；中英文均验证失败后进程存活、现有文件不被破坏、日志含真实原因，以及随后成功重试，另验证取消保存不写文件或报告错误。
- 本轮共 356 个不同测试通过：118 个 MuScriptor/结果工作台测试、28 个编辑器动作测试、14 个中英文下载进程回归，以及 196 个国际化、Web/API、Space/Colab、Docker、部署与便携/可执行入口契约测试。
- 同步中英文 README、结果编辑器说明与下载异常处理架构图。桌面源码、便携 App 与可执行 App 共享修复；Space/Colab 和独立 Web/API、Docker、便携 Web 不调用此 Qt 下载入口，按各自的下载、部署与打包契约回归检查。本次不发布新的发行包；MuScriptor 多乐器中途闪退尚待反馈者的版本、配置与日志，未将本修复声称为其解决方案。
- Fixed desktop MIDI-download errors escaping back into a Qt action and terminating the application (#13). The action now handles BPM/asset checks, file-dialog failures and MIDI/stereo WAV writes, displays the existing localized error, logs the full traceback and retains the result for retry. Cancellation and successful exports retain their existing note, project-tempo, event and source-file behavior.
- Added isolated Qt-process regressions for destination-directory conflicts, missing source MIDI, missing reference BPM, file-dialog errors, missing stereo audio and unavailable assets. Both supported languages verify process survival, preservation of existing files, complete failure diagnostics and a successful retry, plus cancellation without writes or errors.
- Passed 356 distinct tests: 118 MuScriptor/result-workbench tests, 28 editor-action tests, 14 bilingual download-process regressions and 196 i18n, Web/API, Space/Colab, Docker, deployment and portable/executable-entry contract tests.
- Updated both language READMEs, the result-editor guide and the download-error architecture diagram. Source, portable and executable desktop entries share the fix; Space/Colab and standalone Web/API, Docker and portable Web use separate download paths and are checked through their delivery contracts. No new release bundle is published here. The reported MuScriptor multi-instrument crash still requires the reporter's version, settings and logs.

## v1.8.0 — 2026-09-23

**2026-09-23 发布修复 / Release fix**

- Space 镜像构建会额外安装 `gradio[mcp,oauth]==6.17.3`；其 MCP 扩展要求 Pydantic 2.11.10–2.12.5。将 Space 专用依赖调整为 `pydantic==2.12.5` 与对应的 `pydantic-core==2.41.5`，解除此前 2.10.6 导致的解析冲突；Colab 独立运行环境不受影响。
- Space 源码同步补齐模型就绪探针模块及对应变更触发路径，修复镜像构建后因缺少 `src.model_profile_runtime_probe` 而启动失败的问题。
- Space 同步包增加页面启动使用的品牌图标；同步工作流自身变化也会触发重新部署，避免缺文件修复无法自动生效。
- ZeroGPU 平台将 `spaces==0.51.3` 作为构建依赖注入；Space 自身改为接受兼容版本范围，避免与平台注入版本冲突。
- 修复 Linux PyInstaller 构建时新增的 ChordMini 校验导入找不到项目 `src` 包；spec 现在先加入仓库根路径，并使用隔离进程回归验证。
- The Space image installs `gradio[mcp,oauth]==6.17.3`, whose MCP extra requires Pydantic 2.11.10–2.12.5. Updated the Space-specific pins to `pydantic==2.12.5` and its matching `pydantic-core==2.41.5`, resolving the dependency conflict from 2.10.6. Colab uses a separate environment and is unchanged.
- Included the model-readiness probe module and its change trigger in the Space source sync, fixing startup failure after image build when `src.model_profile_runtime_probe` was absent.
- Included the brand icon required at Space startup, and made changes to the sync workflow itself trigger redeployment.
- The ZeroGPU builder injects `spaces==0.51.3`; the Space dependency now accepts the compatible range rather than conflicting with the platform-selected version.
- Fixed a Linux PyInstaller failure where the new ChordMini validation import could not locate the project `src` package. The spec now adds the repository root first, with an isolated-process regression check.

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
