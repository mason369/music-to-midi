# Windows 便携版打包

目录式 `exe` 便携版随包提供 Python 环境，解压后即可启动。

当前源码的伴奏模型为 Leap Instrumental。[第三方清单](../THIRD_PARTY_NOTICES.md) 已记录该固定权重及原始配置的独立 `OWNER_ACCEPTED` 分发决定；完整包仍须通过组件清单、资源身份与成品自检，产物验收状态以对应版本 Actions 结果为准。

## 一键打包

在已安装好开发环境、且模型已下载完成的构建机上执行。该命令一次生成桌面 App、Web 后端、Web 前端三个独立目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_web_executables.ps1
```

Intel XPU 使用独立环境与入口：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_portable_xpu.ps1
```

自定义 Python 或 `ffmpeg` 路径：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_web_executables.ps1 `
  -PythonExe .\venv\Scripts\python.exe `
  -FfmpegDir C:\ffmpeg\bin
```

完整 XPU 包超过 20 GB。需要把构建临时目录和成品放到其他 NTFS 目录时：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_portable_xpu.ps1 `
  -BuildRoot C:\MusicToMidi-XPU-build `
  -DistRoot C:\MusicToMidi-XPU-dist
```

XPU 构建默认对同卷的暂存资源使用 NTFS 硬链接，跨卷资源逐文件复制；暂存资源与成品资源都会进行身份校验。硬链接失败会停止构建，不会自动改为复制。

## 便携包内容

- Python 解释器与依赖库
- YourMT3+、MIROS、MuScriptor 的固定源码与权重，包括五种 YourMT3+ checkpoint 和 MuScriptor Small / Medium / Large
- Leap XE、Leap Instrumental 与 BS-RoFormer SW Fixed 分离模型
- TransKun V2 / V2 Aug、Aria-AMT 与 ByteDance Pedal 钢琴模型
- Beat This `final0` 节拍模型
- MuseScore General SoundFont、FluidSynth 与 MuseScore Studio 乐谱运行环境
- `ffmpeg.exe` 与 `ffprobe.exe`

完整 App 和 Web 后端包需要以上全部资源；缺少资源或身份校验失败时，构建会停止。Web 前端目录只提供浏览器界面，通过网络连接推理后端。

## 目标机器依赖

- CUDA 包需要兼容的 NVIDIA 显卡与驱动；驱动需在目标机器上单独安装
- Intel GPU 使用 XPU 包，并安装兼容的 Intel 显卡驱动；新增 Leap Instrumental 伴奏路线尚未完成 XPU 实机验收
- 不提供 Vulkan 运行环境

## 非 NVIDIA 显卡说明

- Windows Intel 使用独立的 PyTorch XPU + OpenVINO GPU 便携包；启动时检查 XPU FFT/STFT/BF16 与 OpenVINO `GPU.0` 运算。
- XPU 环境不兼容 IPEX、DirectML 或 CUDA ORT 混装；启动检查失败时停止。
- AMD/ROCm 当前缺少完整七模式所需的 GPU 运行环境，因此不在便携版支持范围内；启动检查失败时会停止。

## 发布目录

- 桌面 App 的完整目录为 `dist\MusicToMidi-App\`
- Web 版的两个完整目录为 `dist\MusicToMidi-WebBackend\` 与 `dist\MusicToMidi-WebFrontend\`
- XPU 对应目录为 `dist\MusicToMidi-XPU-App\` 与 `dist\MusicToMidi-XPU-WebBackend\`
- 每个程序都需要 EXE 和同目录 `_internal`；复制整个目录即可保留完整运行文件
- 首次运行时程序会优先读取 exe 邻近的 `models/` 和 `tools/ffmpeg/`
- 日志和运行时缓存优先写入 EXE 邻近的 `runtime/`；该目录不可写时改用用户目录
