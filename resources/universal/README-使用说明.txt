MusicToMidi Windows Universal 便携包

一、解压完整包

- 必须把所有 .wim.001、.002……分卷放在同一目录。
- 使用 7-Zip 26.02 或兼容版本打开 .001，只对 .001 执行解压；不要用 Windows 资源管理器直接打开分卷。
- 建议解压到 NTFS 磁盘。WIM 会恢复 CUDA/XPU 以及 App/WebBackend 之间的硬链接，完整包实际占用约 28.2 GiB，但逻辑文件总量约 88.3 GiB。
- 如果要把 App 或 WebBackend 单独移到另一台电脑，最省空间的做法是把分卷复制过去，再用 7-Zip 只解压对应顶层目录。普通文件复制依然可用，但可能把硬链接展开成多份实体文件，占用更多空间。

二、目录与 GPU 选择

解压后有三个顶层程序目录：

- MusicToMidi-App：桌面 App。
- MusicToMidi-WebBackend：Web 推理后端。
- MusicToMidi-WebFrontend：可独立复制到局域网其他电脑的 Web 前端。

App、CLI 与 WebBackend 的顶层 EXE 会自动选择运行环境：

- 检测到 NVIDIA GPU 时选择 CUDA。
- 未检测到 NVIDIA、但检测到 Intel Arc GPU 时选择 XPU。
- 同时存在 NVIDIA 与 Intel Arc 时固定优先 CUDA。
- 可以用 MUSIC_TO_MIDI_ACCELERATOR=cuda 或 xpu 显式指定。
- 选择后的运行环境若启动失败，会原样返回失败；任务停止。

CUDA 与 XPU 位于各自的 runtimes 子目录，原生 DLL 不会混装。模型、固定源码、SoundFont、FluidSynth 与 FFmpeg 在 NTFS/WIM 中通过硬链接复用，因此只占一份物理空间；单独复制 App 或 WebBackend 目录时仍会得到该角色所需的完整文件。

三、命令行与批处理

- App 或 WebBackend 目录都提供 MusicToMidiCLI.exe。
- 最简单用法：MusicToMidiCLI.exe D:\Audio\song.wav
- 目录批处理：MusicToMidiCLI.exe batch D:\Audio --recursive -o D:\MidiOutput
- 查看全部模式和参数：MusicToMidiCLI.exe convert --help
- 查看 13 条逐轨路线：MusicToMidiCLI.exe routes
- 分离模式只先输出 WAV；对选中的音轨显式运行 track-to-midi，不会自动转换全部 stem。
- 每个输入使用独立输出目录和 SHA-256 清单；成功项只有在源文件、参数和全部产物校验一致时才断点跳过。任何失败都会显示根因并返回非零退出码。

四、局域网分开部署

- 在有 NVIDIA 或受验证 Intel Arc GPU 的电脑运行 MusicToMidi-WebBackend\MusicToMidiBackend.exe。
- 其他电脑只需解压独立的 WebFrontend ZIP，运行 MusicToMidi-WebFrontend\MusicToMidiFrontend.exe，并把 backend_url 设置为 GPU 电脑的实际局域网地址。
- 后端 allowed_origins 必须包含前端访问地址。本项目的 Web 模式面向可信局域网，不包含认证、授权或 TLS，禁止直接暴露到互联网。
