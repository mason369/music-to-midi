# 项目、工作流与断点续跑

项目保存歌曲、模型选择、处理状态和完成产物。`workflow.json` 只保存可重复使用的工作流程；`music-to-midi-project.json` 记录当前项目；`.mtmproject` 项目包包含原音、分轨、MIDI 和清单，可以在另一台机器或另一种界面导入。

## 桌面与独立 Web

1. 多选或拖入多首音频。桌面普通文件入口默认保存在输出目录的 `Projects/Library`；也可通过“新建项目”使用独立目录。独立 Web 将项目保存在服务数据目录的 `projects` 子目录。
2. 选择“六声部分离”，勾选之后需要转 MIDI 的声部，并为每个声部选择模型。没有勾选声部时只生成六条 WAV。
3. 选择设置应用于“当前歌曲”或“全部歌曲”，点击“应用当前设置”。导入工作流同样只保存配置。
4. 点击“继续整个项目”，按照每首歌曲保存的配置依次处理。主界面的“开始分离”只执行当前歌曲的分离；逐轨“开始转换”只处理该轨。
5. 下次打开同一项目或再次加入相同内容的原音，继续已有歌曲。已完成的分离和符合当前模型参数的 MIDI 会先校验，再复用。

桌面和独立 Web 的逐轨模型、勾选、静音、独奏、音量与偏移会保存到项目中。切换歌曲后通过“打开歌曲结果”恢复结果工作台。手动添加的外部音轨也可保存并继续转写；桌面在首次转换该外部音轨时将它导入当前项目。

外部音轨的配置属于各自歌曲。把工作流应用于全部歌曲时，每首歌保留自己的外部音轨计划；可复用的工作流 JSON 不包含这些音轨 ID。切换为直接转写模式，或在 CLI 使用 `--clear-stems`，会清空相应逐轨计划，已有产物仍保留。

## Space 与 Colab

使用界面顶部“项目与批处理”：新建项目、添加多个文件、保存工作模式和逐声部模型后，再点击“继续整个项目”。“打开项目”恢复本浏览器保存的项目；也可导入 `.mtmproject`。完成的 MIDI 可以在共享编辑器中试听、编辑、量化和导出。混音修改需点击“保存混音设置”。

Space/Colab 的运行磁盘可能在会话销毁时被回收。浏览器只保存项目标识，不会保存音频本体；离开运行环境前导出项目包，下一次导入后继续。项目功能不延长 ZeroGPU 配额，也不绕过单次音频长度限制。

原有单曲快速转换入口仍可使用；需要跨会话恢复或批处理时使用项目入口。

## 命令行

```powershell
python -m src.cli project create D:\MusicProject song1.wav song2.wav --mode six_stem_split --stem piano=piano_transkun --stem bass=miros
python -m src.cli project run D:\MusicProject
python -m src.cli project status D:\MusicProject --json
python -m src.cli project configure D:\MusicProject --stem piano=piano_aria_amt --save-profile D:\workflow.json
python -m src.cli project run D:\MusicProject
python -m src.cli project export D:\MusicProject D:\MusicProject.mtmproject
python -m src.cli project import D:\RestoredProject D:\MusicProject.mtmproject
```

`--profile` 读取工作流；`--song` 指定歌曲 ID；`--fail-fast` 在首次失败时停止队列。默认会记录失败并继续其他歌曲，最终返回失败状态和非零退出码。Ctrl+C 保留已完成步骤。现有 `batch --resume` 命令继续兼容。

## 恢复与校验边界

- 恢复粒度是“单次分离”和“单个声部的一次 MIDI 转写”。已完成步骤不会重算；被中止的模型步骤从头执行，不会从神经网络推理的中间帧恢复。
- 同一项目内按原音 SHA-256 去重。更换模型或影响 MIDI 的配置只使相应转写步骤重新待处理，保留历史产物。单纯更换后续 MIDI 模型不使分离失效。
- 原音和成功产物都有大小与 SHA-256 校验。缺失或被外部修改会明确失败，不把损坏结果当成功，也不静默重新计算覆盖它。
- JSON 原子提交，操作系统锁隔离读写与推理；进程退出后锁释放。多个窗口不能同时修改正在执行的项目。
- 项目包只接收清单内的文件，拒绝路径穿越、符号链接、清单不一致及校验失败。移动整个项目目录或使用项目包，勿只移动 JSON。
- Docker 应持久化已有 API 数据卷；便携包使用相同项目格式。更新程序前保留项目及输出目录。

## 验收范围

代码与契约检查覆盖桌面、CLI、独立 Web/API、Space/Colab 共享组件、Docker 源码复制及便携入口。实际云端 ZeroGPU/Colab、Linux GPU、Intel XPU 和新构建便携包的验收必须分别执行，不能从本机 NVIDIA 或契约测试推断已经通过。
