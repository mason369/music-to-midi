# 音色库与乐谱导出 / SoundFonts and sheet music

## MIDI 声部、和弦与走带

- 新结果默认所有声部可见、可听；重新打开结果不继承上一结果的静音或独奏。S 可多选，两条贝斯可一起试听、一起导出；手动静音优先于独奏，退出独奏后恢复原静音设置。
- “全部声部合并导出 MIDI”导出当前完整编辑；“当前可听声部合并导出 MIDI”应用静音和多选独奏；“所有声部分别导出 MIDI（ZIP）”不受试听选择影响，每个乐器一个带编号、名称和 BPM 的文件。同一乐器按编辑器声部归组，不按文件中的空元数据轨生成多余 MIDI。
- 编辑器导出沿用 `midi_editor.export_edited_midi` 的音符、tempo、拍号、CC64、pitch bend 和其他事件校验。ZIP 完整校验后才替换目标文件，原 MIDI 保持不变；没有可听声部时明确提示，不生成空文件。
- 桌面 MIDI/立体声 WAV 下载在 Qt 动作边界处理异常：资源检查、保存对话框和写文件失败均记录完整调用栈并显示现有中英文错误提示，保留结果以便修正问题后重试。取消保存不写文件；导出失败不会通过 Qt 回调继续抛出异常。便携 App 和可执行 App 复用此组件；Space/Colab 的浏览器下载与独立 Web/API 的文件响应不经过 Qt 回调。
- 空格播放/暂停，Shift+空格或停止按钮停止并回到零点，同时重置进度条、播放头和水平视图；文本、数值输入和下拉框保留键盘原行为。快捷键仅作用于当前获得焦点的结果编辑器。
- 本地桌面和弦轨使用与 TelkNet 默认工具相同的 **ChordMini BTC**，固定官方源码 `aa6e3a8d7b017f082fd2aaff9329d5c26af49c03` 和 `btc_model_best.pth`。原音按原采样率、原声道解码为 PCM32，再交给官方特征提取与推理；170 类完整标签涵盖 14 种和弦性质及 N/X，无和弦/不确定区间不试听。保留原生边界，只在小节处分割显示，MuScriptor 开头补齐偏移同步到和弦。小节、和弦、音符共用参考 BPM 和拍号；工程变速不会导致三者漂移，编辑 MIDI 不会改写原音和弦。后台分析可取消并与转写共用 GPU 锁；失败显示原因、点击重试，不替换为 MIDI 估计或 Omnizart。
- 点击和弦暂停整曲并通过当前 SoundFont 单独试听。取消、快速连续点击、停止和关闭编辑器会取消旧请求，防止较慢的旧试听覆盖新选择。
- FLAC 兼容处理：对 STREAMINFO 声明帧数的原生 FLAC，使用现有 libsndfile 无损读取整数样本，保持原采样率、声道、精度和完整时长，核对声明帧数及可用的样本 MD5，再送入官方前端。支持尾部 ID3v1 / APEv2 标签和前置 ID3v2（含页脚）的文件，不修改原文件。无损数据完整时，尾部标签不再被严格 FFmpeg 误当作坏音频帧；真正的坏帧、截断和校验不一致必须失败，不运行模型、不写成功缓存。MD5 全零表示文件未提供该可选校验，仍检查完整帧数；未声明帧数的 FLAC 和其他格式沿用严格 FFmpeg 解码。校验定义参照 [FLAC RFC 9639 §8.2](https://www.rfc-editor.org/rfc/rfc9639.html#section-8.2)，读取方式参照 [python-soundfile 文档](https://python-soundfile.readthedocs.io/en/latest/#soundfile.SoundFile)。
- 和弦栏上方的“和弦随曲播放”首次默认关闭，0～200% 独立音量默认 100%，并以固定应用名的用户级 INI 记住两项设置。关闭随播仍可点击试听；0% 静音两种试听，静音时点击和弦不会暂停歌曲。200% 提供实际音频增益，点击试听与随播采用相同的响度校准和音量标尺。控制条在窄窗口自动换行，支持键盘调整、中英文切换和明确的准备/失败/重试状态。
- `chord_audio` 从原始和弦区间和 MuScriptor 开头偏移生成一次完整 SoundFont 音轨，保留长和弦和转位，N/X 区间不发声；显示用小节分块不进入发声调度。`ChordMonitor` 在用户启用时后台准备音轨，用代次检查丢弃过期结果；静音/关闭随播取消未完成请求，换音色清除旧音轨，失败需要明确重试。已完成音轨在当前结果内复用，切换音量无需重新合成。
- 和弦 PCM 接入 `SynchronizedPcmPlayer`，与原音、MIDI 和声部混音共用帧坐标及一个音频输出。暂停、跳转、BPM、倍速和从中途播放因此统一生效。音量最多用 20 ms 平滑过渡；叠加时采用双声道联动的 10 ms 前视峰值保护（PCM 已在内存中，无需额外播放延迟），避免高音量硬削波。关闭和弦且过渡结束后，原音/MIDI 混音路径保持原始样本。此设置只控制监听，不更改识别和弦、转写 MIDI、编辑历史或任何导出。
- 和弦栏直接响应卷帘每一帧的横向位移与缩放，包含两次滚动条步进之间的小数像素位移；仅和弦、小节等内容变化使用延迟重算。持续播放不会推迟和弦绘制或阻止新的小节布局生效。
- 和弦按根音保持一致配色，名称使用紧凑音乐记法（如 `C♯m7`），原始模型标签保留在悬停详情中。当前片段高亮，悬停加亮边框；短片段不截断名称，而是用错开的完整标签与引线指向真实时间块，标签也可点击试听。N/X 使用中性色且不试听。标签避让只影响绘制，播放期间保留标签行高度，避免卷帘上下跳动。
- 桌面、Space 和 Colab 的音符在播放头所在时间区间高亮；暂停和跳转后仍显示当前位置，区间采用起点包含、终点不包含。静音/非独奏声部不显示播放高亮。悬停显示音高、MIDI 编号、声部、力度、起止时间和时长，不改变选择或音符数据。桌面在原缓存瓦片上叠加反馈，仅重绘进入/离开高亮的音符；浏览器用独立透明 Canvas 绘制可见音符反馈，复用现有编辑与播放流程。

| 交付面 | 多选独奏、可听组导出、批量 MIDI、快捷键 | 和弦轨 |
|---|---|---|
| PyQt 桌面 / 便携 App / 可执行 App | 同一 `MuscriptorResultWidget` | 原音 ChordMini BTC 识别与 SoundFont 试听 |
| Space / Colab | 同一 `muscriptor_result_runtime` 浏览器实现 | 本次按反馈限定本地窗口 |
| 独立 Web/API / Docker / 便携 Web | 结果与恢复项目都支持批量 MIDI ZIP；没有 MIDI 编辑器，不适用 MIDI 独奏或编辑器快捷键 | 没有钢琴卷帘，不适用 |

```mermaid
flowchart LR
    MIDI[模型 MIDI 与当前编辑音符] --> Editor[桌面 / Space / Colab 结果编辑器]
    Editor --> Mix[手动静音 + 多选独奏 → 试听与可听组导出]
    Editor --> Export[midi_editor 校验 → 合并 MIDI / 声部 ZIP]
    Export --> Download[桌面 Qt 下载动作 / Desktop Qt download action]
    Download --> Failure[失败记录调用栈 + 显示原因 + 保留结果以便重试 / Log failure, report cause and allow retry]
    Audio[原始音频 / 所选 stem WAV] --> Decode[FLAC 无损 PCM32 + 帧数 / MD5 校验<br/>FLAC lossless PCM32 + frame count / MD5 verification<br/>其他格式严格 FFmpeg / Strict FFmpeg for other formats]
    Decode --> Worker[GPU 锁 → 可取消 ChordMini BTC 子进程]
    Worker --> Native[原音秒坐标和弦 + 固定模型身份缓存]
    Native --> Chords[时间偏移与小节显示 → 彩色和弦与完整避让标签]
    Native --> Monitor[原始区间 + 时间偏移 → 可取消 FluidSynth 音轨 → 响度校准]
    Controls[随播开关 / 0～200% 音量 / 偏好记忆] --> Monitor
    Monitor --> Clock[单一 PCM 时钟 + 原音 / MIDI → 平滑增益与峰值保护]
    Chords --> Audition[点击 → 暂停歌曲 → 同音量标尺单独试听]
    Controls --> Audition
    Editor --> View[逐帧位移与缩放 / Per-frame view motion]
    View --> Chords
    View --> Feedback[播放位置 / 悬停 → 音符反馈层与详情]
    Editor --> Feedback
    MIDI --> API[midi_exports → 保留源 tempo map 与事件 tick → Web/API ZIP]
```

## MIDI instruments, chords and transport

Desktop MIDI and stereo WAV downloads handle failures at the Qt action boundary. Asset checks, file dialogs and writes log full tracebacks and display the existing Chinese or English error message, leaving the result available for retry. Cancelling the dialog writes nothing; errors do not escape back into Qt. Portable desktop and executable App entries share this widget. Space/Colab browser downloads and standalone Web/API file responses do not use the Qt callback.

New results start with all instruments audible and visible. Solo is additive; manual mutes remain independent and take priority. Export all current notes, only the audible group, or all instruments as separate numbered MIDI files in one ZIP. Batch export ignores monitor selection and preserves the current project BPM and edits. Empty selections report an error instead of replacing a file.

Space plays/pauses; Shift+Space or Stop and rewind returns transport and horizontal view to zero. Text, numeric and select inputs retain normal keyboard behavior. Only the focused result editor receives transport shortcuts. The desktop uses TelkNet’s default ChordMini BTC recipe on source audio, with the exact pinned source and BTC checkpoint. Native-rate, original-channel PCM32 decoding precedes the official feature frontend. All 170 classes, including seventh/suspended qualities and N/X, retain their original boundaries. Bar splits affect display only; MuScriptor lead-in offsets and project tempo changes remain aligned. MIDI edits do not alter source-audio chords. Background inference shares the accelerator lock, supports cancellation and reports failures with click-to-retry; it never falls back to MIDI templates or Omnizart. Chord recognition remains a model prediction, not a guarantee of score accuracy.

Space and Colab share the browser mixer/export/transport implementation. Chord display is desktop-only as requested. Standalone Web/API, Docker and portable Web offer instrument ZIP downloads for both jobs and restored project results, preserving the source tempo map and event ticks. They have no MIDI editor/solo state, so editor-specific shortcuts and audible-group exports do not apply. Those browser surfaces retain their existing SoundFont/FluidSynth dependencies. Only desktop chord analysis requires the additional pinned ChordMini BTC resources.

For native FLAC with a declared STREAMINFO frame count, the existing libsndfile decoder reads lossless integer samples at the original rate, channel count and precision, then verifies the full frame count and available sample MD5 before the official frontend runs. Trailing ID3v1 / APEv2 tags and leading ID3v2 tags, including footers, are supported without modifying the source. Complete samples no longer fail because strict FFmpeg interprets a trailing tag as an audio frame. Actual frame damage, truncation and checksum mismatches fail before model inference or success-cache publication. An all-zero MD5 denotes an omitted optional checksum; frame-count validation still applies. FLAC with an unknown declared length and other formats retain strict FFmpeg decoding. See [FLAC RFC 9639 §8.2](https://www.rfc-editor.org/rfc/rfc9639.html#section-8.2) and the [python-soundfile API](https://python-soundfile.readthedocs.io/en/latest/#soundfile.SoundFile).

Desktop chord controls provide **Chords follow song** (initially off) and **0–200% chord volume** (initially 100%), persisted in a user INI under a stable application name. Follow off still allows click auditions; 0% mutes both, and clicking while muted does not pause the song. 200% applies real audio gain, using the same loudness calibration and volume scale for accompaniment and click auditions. The controls wrap in narrow windows, support keyboard input and both locales, and report preparation failures with an explicit retry.

`chord_audio` renders native chord intervals and the MuScriptor lead-in offset into SoundFont PCM; visual bar splits never retrigger notes and N/X remains silent. `ChordMonitor` prepares audio on demand in a cancellable worker, rejects stale generations, caches the current result and invalidates it when the SoundFont changes. Muting or disabling follow cancels pending work. The PCM joins the existing `SynchronizedPcmPlayer` frame axis and single output, so pause, seek, project BPM and speed apply equally. Gain ramps take at most 20 ms. Stereo-linked 10 ms lookahead peak protection uses the already loaded PCM without adding transport latency. Once disabled and the ramp ends, the original/MIDI mix is sample-identical. Preferences affect monitoring only, not chord recognition, MIDI, edit history or exports. Portable and executable desktop Apps use these same modules; the browser/API delivery surfaces do not expose the desktop chord lane.

The chord lane repaints with each frame of horizontal motion or zoom, including fractional pixel movement between scrollbar steps. Only chord and bar content changes use a delayed rebuild. Continuous playback therefore neither delays chord drawing nor prevents a new bar layout from taking effect.

Chord colors remain stable by root. Compact notation such as `C♯m7` preserves the full model label in hover details. The current interval and hovered blocks receive separate highlights. Short intervals use full, staggered labels with leaders to their true timing; labels also support audition. N/X remain neutral and silent. Label placement never alters model boundaries, and reserved row height prevents vertical jumps while scrolling.

Desktop, Space and Colab highlight MIDI notes at the current playhead position, including after pause or seek, using half-open time intervals. Muted or excluded solo instruments do not receive playback highlights. Hover details show pitch, MIDI number, instrument, velocity, start, end and duration without modifying selection or note data. Desktop feedback overlays cached tiles and invalidates changed note regions; browsers use a separate transparent canvas over visible notes. Existing editing and playback controls retain their behavior.

## 使用音色库

桌面、Space 和 Colab 的 MIDI 结果页提供“音色库与乐器音色”。导入 `.sf2` 或 `.sf3` 文件，在“来源乐器”中逐项选择音色预设，再点击“应用音色”。选择下拉框本身不会改变试听。切回“默认音色库”并点击“应用音色”，可恢复默认声音。

应用后，主试听、独奏/静音、立体声对照、WAV 导出和分乐器 WAV 导出使用同一音色选择。WAV 保留现有两档规格：24-bit / 48 kHz 和 16-bit / 44.1 kHz。

独立 Web 的每个 MIDI 下载结果提供“音色试听”，分离后的逐轨 MIDI 结果也有此入口。应用音色后，可播放和下载 24-bit / 48 kHz WAV。API 接受两档音频规格。

- 单个音色库上限为 1 GiB。预设列表读取文件自身的名称、bank 和 program；编号显示为从 0 开始的 `bank:program`。
- 可导入多个库，并为当前结果选择一个活动库。各来源乐器可以选择活动库中的不同预设。
- 鼓轨选择 bank 128 的鼓组；旋律轨选择其他 bank 的旋律预设。缺少原乐器对应预设时，必须明确选择可用预设。
- 导入保留独立副本和 SHA-256 身份，渲染检查文件是否变化。无效文件、缺失预设和合成失败会报错。
- 自定义音色渲染不支持异步 Type 2 MIDI 或带 SysEx 音源控制的 MIDI，遇到这些情况会停止并说明原因。
- 桌面的导入库随当前结果保留；Space/Colab 随请求目录保留。独立 API 的音色库随对应任务保存和清理。

**音色库改变播放声音，不扩展 AI 模型的识别类别。** MIDI、MusicXML 和 PDF 保留转录/编辑后的原乐器编号，文件不嵌入用户音色库。MuScriptor 的官方模型及权重不作修改。

### 独立 API

以下接口使用已成功完成任务的 `job_id` 和 MIDI `artifact_id`。音色库 ID 仅在所属结果内有效。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/v1/jobs/{job_id}/soundfonts/{artifact_id}` | 返回来源乐器和已导入库的预设目录 |
| POST | `/api/v1/jobs/{job_id}/soundfonts/{artifact_id}` | multipart 上传 `file`，返回音色库 ID 和真实预设 |
| POST | `/api/v1/jobs/{job_id}/soundfonts/{artifact_id}/render` | 按以下 JSON 返回 WAV |

```json
{
  "preset": "pcm24_48000",
  "soundfont": {
    "id": "上传响应中的 SHA-256 ID",
    "assignments": [
      {"source_program": 0, "is_drum": false, "bank": 8, "program": 80}
    ]
  }
}
```

`preset` 也支持 `pcm16_44100`。`soundfont: null` 使用默认音色库。以上预设示例必须存在于实际导入的库中；服务端不会替换不存在的预设。

## 乐谱导出

功能对照 MuScriptor 官方 [`sheets.py`](https://github.com/muscriptor/muscriptor/blob/e34b397bf0584e67bfd81dc591c390e6dcb03350/muscriptor/utils/sheets.py)，使用 MuseScore 4 排版。

桌面结果页选择“下载 → 乐谱包（MusicXML / PDF）”；Space/Colab 点击同名按钮；独立 Web 点击 MIDI 结果旁的“生成乐谱”。ZIP 包含：

- `score.mid`：用于排版的 MIDI 副本；
- `score.musicxml`：可继续编辑的乐谱；
- `full_score.pdf`：总谱；
- 按乐器编号命名的 PDF 分谱；
- 适用吉他/贝斯的 `_tab.pdf` 指法谱；
- `sheet_music_manifest.json`：源文件校验值、排版网格、音符数和 MuseScore 版本。

排版前仅量化私有 MIDI 副本。结果编辑器与独立 Web 使用当前网格，默认 `1/32`；量化强度、三连音和自动网格设置同步用于乐谱副本，HTTP 入口可传 `quantize_grid`、`quantize_strength`、`quantize_triplet`、`quantize_auto`，各组设置独立缓存。源 MIDI 不被覆盖。MuseScore 不可用、没有真实输出、分谱失败或任务取消时不会发布成功结果。速度图、无变速纠错、休止填补与空小节裁切的操作和架构见 [MIDI 时间轴与导出增强](midi-timing-export.md)。

乐谱是对当前 MIDI 的自动排版。它不保证修正模型识别错误，也不能保证所有自由节奏、复杂复调或指法选择都符合人工制谱要求；可在 MuseScore 中继续编辑 MusicXML。

## English

Import SF2/SF3 files in the MIDI result's **SoundFont and instrument sounds** panel. Select a preset for each source instrument and click **Apply sounds**. Selection alone does not render. Desktop, Space and Colab apply the selection to preview and WAV/stem exports. The standalone Web client provides a **SoundFont preview** action for each MIDI result, with playback and a 24-bit / 48 kHz WAV download. Its API also supports 16-bit / 44.1 kHz.

Each file is limited to 1 GiB. One imported library is active per result, with individual instrument assignments. Drum presets use bank 128. Missing presets, changed files, malformed banks and synthesis failures are explicit errors. Asynchronous Type 2 MIDI and SysEx sound-source control are unsupported for custom rendering. SoundFonts do not extend model recognition, alter the exported MIDI instruments, or become embedded in MIDI or sheet music.

**Sheet music (MusicXML / PDF)** exports a ZIP containing a MIDI engraving copy, MusicXML, full-score PDF, part PDFs, applicable guitar/bass tablature, and an integrity manifest. The implementation follows the official MuScriptor MuseScore workflow. Only a private copy is quantized for engraving; the original MIDI is preserved. Automatic engraving may still require human review for transcription accuracy, complex rhythm, voicing and fingering.

The current grid, strength, triplet and automatic-grid settings also apply to the engraving copy, with separate cache identities. HTTP job and project-result endpoints accept all four settings. See [MIDI timing and export enhancements](midi-timing-export.md) for adaptive maps, corrections that retain performance speed, gap filling, leading-bar trimming and the shared architecture.

For deployment, the same shared modules serve desktop, Space, Colab, standalone API, Docker and portable launchers. Existing FluidSynth and MuseScore runtime requirements remain in effect. Desktop chord analysis additionally requires the pinned ChordMini BTC resources; run `python download_chordmini_model.py` or `python download_sota_models.py`. Full portable App builds include and verify these resources. Space/Colab and standalone Web/API/Docker do not expose the desktop chord lane or run its model; their existing shared exports and playback paths remain covered separately.
