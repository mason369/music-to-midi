# 独立 Web 前端与推理后端

前端与推理后端是两个独立进程。前端提供静态界面、PCM 波形、Web Audio 播放与 HTTP/SSE 客户端；后端运行模型、管理任务队列与取消状态，并保存任务进度和结果文件。API 版本不是 2.0 时连接会停止；浏览器每 5 秒检查一次服务状态，断线后不会继续显示“在线”。

“量化音符”默认关闭；显式勾选后，所选 `1/4`～`1/64` 网格会对输出 MIDI 的全部轨道执行一次起点与时值量化，默认网格为 `1/32`。独立 Web/API 不提供只处理当前选区的范围，因为建单界面没有钢琴卷帘选区。

每个成功 MIDI 制品旁都有“生成并下载乐谱”按钮，分离后的逐轨 MIDI 也各自独立提供。后端通过 `POST /api/v1/jobs/<job-id>/sheet-music` 对该 MIDI 的私有副本按 `1/32` 量化，发布包含量化 MIDI、MusicXML、总谱/分谱 PDF 和适用 Tab PDF 的 ZIP；原 MIDI 不会被改写。后端必须具备 MuseScore Studio 4，缺失或制谱失败会返回明确错误。

此版本面向受信任局域网，因此不内置认证、授权或 TLS。把前端 `5173` 或后端 `8765` 端口映射到互联网会直接暴露无认证服务。前端宿主发送 CSP、Permissions-Policy、nosniff 与 no-referrer 响应头，但这些响应头不会增加互联网访问所需的身份保护。

## Windows 发布包

运行 `build_web_executables.ps1` 后会得到三个职责独立的目录：

| 目录 | 启动文件 | 说明 |
|------|----------|------|
| `dist/MusicToMidi-App` | `MusicToMidi.exe` | 桌面 App；不启动 Web 服务 |
| `dist/MusicToMidi-WebBackend` | `MusicToMidiBackend.exe` | 完整推理后端；包含模型运行依赖和独立 `_internal` |
| `dist/MusicToMidi-WebFrontend` | `MusicToMidiFrontend.exe` | 小型前端宿主；不包含推理模型和 Torch |

每个目录都包含相应用途所需的完整文件，可以按目录单独复制。桌面版运行 App；Web 版运行 WebBackend 和 WebFrontend；访问端电脑只需要浏览器。

XPU 构建使用 `MusicToMidi-XPU-App` 和 `MusicToMidi-XPU-WebBackend` 目录，后端启动文件为 `MusicToMidiBackendXpu.exe`；前端目录和启动文件名不变。

## 源码版使用

第一次取得源码时，先按主 README 完成虚拟环境、依赖和模型安装。源码环境准备完成后，在仓库根目录运行：

```powershell
.\venv\Scripts\python.exe -m src.web
```

这一条命令会自动识别主要局域网 IPv4，依次启动 WebBackend 和 WebFrontend，配置两端使用的地址与 CORS 来源，等待 API 2.0 和前端真正就绪，并打开独立的 Edge 应用窗口。终端会显示以下实际地址：

- 前端：`http://<局域网IPv4>:5173`
- 后端健康检查：`http://<局域网IPv4>:8765/api/v1/health`
- 后端 OpenAPI 文档：`http://<局域网IPv4>:8765/docs`

按 `Ctrl+C` 或关闭 Edge 应用窗口会一并停止 WebFrontend 和 WebBackend。

### 允许同一局域网的其他电脑访问

统一入口默认使用操作系统默认路由选择的具体 RFC1918 局域网 IPv4。同一局域网的其他电脑只需在浏览器打开终端显示的前端地址。

多网卡、VPN 或路由选择不符合预期时，可以明确指定服务电脑的实际局域网 IPv4：

```powershell
.\venv\Scripts\python.exe -m src.web --host 192.168.1.50
```

只允许本机访问时可显式使用回环地址：

```powershell
.\venv\Scripts\python.exe -m src.web --host 127.0.0.1
```

无需在服务电脑上自动打开 Edge 时使用：

```powershell
.\venv\Scripts\python.exe -m src.web --no-window
```

统一入口不会绑定 `0.0.0.0`、链路本地地址或公网地址，避免服务同时监听 VPN、虚拟网卡或其他不在受信任局域网部署范围内的接口。

### Intel XPU 源码环境

Intel XPU 使用相同的统一入口，只需选择 `venv-xpu` 并声明加速器类型：

```powershell
$env:MUSIC_TO_MIDI_ACCELERATOR = "xpu"
.\venv-xpu\Scripts\python.exe -m src.web
```

### Linux / WSL2 源码环境

Linux / WSL2 可运行同一模块。当前自动应用窗口依赖 Windows Edge，因此使用 `--no-window` 后手动打开终端显示的地址：

```bash
./venv/bin/python -m src.web --no-window
```

## 构建 Windows 打包版

在仓库根目录运行：

```powershell
.\build_web_executables.ps1
```

Intel XPU 版本使用：

```powershell
.\build_web_executables.ps1 -Accelerator xpu
```

构建脚本先生成桌面 App 与完整 WebBackend，再生成不含推理依赖的 WebFrontend。默认 CUDA 结果就是上文列出的三个 `dist` 目录。

## 打包版同机使用

1. 解压 `MusicToMidi-WebBackend`，运行 `MusicToMidiBackend.exe`。默认监听 `127.0.0.1:8765`。
2. 解压 `MusicToMidi-WebFrontend`，运行 `MusicToMidiFrontend.exe`。默认打开独立 Edge 应用窗口，访问 `127.0.0.1:5173` 并连接上述后端。
3. 两个 EXE 首次运行会分别在自身目录生成 `MusicToMidiBackend.json` 和 `MusicToMidiFrontend.json`；同机默认值无需修改。
4. 关闭前端 Edge 应用窗口会停止前端宿主；在后端控制台按 `Ctrl+C` 停止后端。

打包的前端 EXE 本身不显示控制台。`--no-window` 或 `open_app_window=false` 模式对应的停止入口是任务管理器中的 `MusicToMidiFrontend.exe`，也可在 PowerShell 执行 `Stop-Process -Name MusicToMidiFrontend`；默认的独立 Edge 应用窗口模式在关窗后退出。

## 打包版供局域网其他电脑使用

最简单的部署是把 WebBackend 和 WebFrontend 都运行在同一台 GPU 服务电脑上，其他电脑只用浏览器访问。

### 1. 获取固定的服务地址

服务电脑的局域网 IPv4 可通过 `ipconfig` 确认。路由器 DHCP 地址保留可以避免重启或续租后地址改变。下文用 `<服务器局域网IPv4>` 表示这个地址。

### 2. 生成并修改后端配置

`MusicToMidiBackend.exe` 首次启动会在同目录生成 `MusicToMidiBackend.json`。控制台按 `Ctrl+C` 退出后，该文件可改为以下结构；尖括号占位符表示实际地址：

```json
{
  "host": "<服务器局域网IPv4>",
  "port": 8765,
  "data_dir": "",
  "allowed_origins": [
    "http://<服务器局域网IPv4>:5173"
  ],
  "log_level": "info",
  "retention_days": 30,
  "retention_max_jobs": 200
}
```

`allowed_origins` 的有效值与浏览器实际打开的前端协议、主机和端口完全一致，仅包含 origin，不包含末尾路径。

### 3. 生成并修改前端配置

`MusicToMidiFrontend.exe` 首次启动会生成同目录的 `MusicToMidiFrontend.json`。关闭打开的 Edge 应用窗口后，该文件可改为以下结构：

```json
{
  "host": "<服务器局域网IPv4>",
  "port": 5173,
  "public_host": "<服务器局域网IPv4>",
  "backend_url": "http://<服务器局域网IPv4>:8765",
  "open_app_window": true,
  "edge_path": ""
}
```

`open_app_window=false` 适用于服务电脑不自动打开窗口的场景。`public_host` 与 `backend_url` 使用浏览器能够访问的实际地址；配置校验会拒绝 `0.0.0.0` 作为公开地址。

### 4. Windows 防火墙范围

以下命令在管理员 PowerShell 中运行，示例变量分别表示服务电脑地址和受信任局域网网段：

```powershell
$ServerIp = "192.168.1.50"
$LanSubnet = "192.168.1.0/24"

New-NetFirewallRule `
  -DisplayName "MusicToMidi Web Frontend LAN TCP 5173" `
  -Direction Inbound -Action Allow -Enabled True -Profile Private `
  -Protocol TCP -LocalAddress $ServerIp -RemoteAddress $LanSubnet -LocalPort 5173

New-NetFirewallRule `
  -DisplayName "MusicToMidi Web Backend LAN TCP 8765" `
  -Direction Inbound -Action Allow -Enabled True -Profile Private `
  -Protocol TCP -LocalAddress $ServerIp -RemoteAddress $LanSubnet -LocalPort 8765
```

规则中的来源范围特意限定为 `$LanSubnet`。`RemoteAddress Any` 或路由器端口转发会超出本部署的受信任局域网边界。规则仅在 Windows 当前网络配置文件为 `Private` 时生效；其他配置文件状态表示需要重新确认网络信任级别，而不是扩大规则范围。

### 5. 启动与连通性验证

启动顺序为 `MusicToMidiBackend.exe`、`MusicToMidiFrontend.exe`。后端控制台会显示实际监听的局域网 IPv4。

在服务电脑 PowerShell 验证：

```powershell
$ServerIp = "192.168.1.50"
Invoke-RestMethod "http://${ServerIp}:8765/api/v1/health"
Get-NetTCPConnection -State Listen -LocalPort 5173,8765
```

在另一台局域网电脑验证：

```powershell
$ServerIp = "192.168.1.50"
Invoke-WebRequest -UseBasicParsing "http://${ServerIp}:5173/"
Invoke-RestMethod "http://${ServerIp}:8765/api/v1/health"
```

两条请求成功后，可在浏览器打开 `http://192.168.1.50:5173`，用一首真实歌曲完成转换并下载结果文件。远程连通性证据还包括后端日志中的请求来源为远端电脑局域网 IPv4，而不只是 `127.0.0.1`。

## 前后端分别部署在两台电脑

如果 GPU 后端和前端宿主不在同一台电脑：

1. 后端 `host` 写后端电脑的实际局域网 IPv4。
2. 后端 `allowed_origins` 写前端电脑的实际地址，例如 `http://192.168.1.60:5173`。
3. 前端 `host` 和 `public_host` 写前端电脑的实际局域网 IPv4。
4. 前端 `backend_url` 写后端电脑的实际地址，例如 `http://192.168.1.50:8765`。
5. 后端电脑仅放行 TCP `8765`；前端电脑仅放行 TCP `5173`，两条规则都限制为实际局域网网段。

浏览器始终打开前端电脑的 `5173` 地址，前端再访问后端电脑的 `8765` 地址。

## 设置、保留策略与故障判断

前端设置面板可分别编辑后端协议、IP/主机名与端口，并在保存前调用 `/api/v1/health` 测试连接和 API 版本。配置错误会明确报错，不会猜测网卡或静默改用其他地址。

后端默认保留 30 天、最多 200 条终态任务；任一值设为 `0` 只关闭对应的自动清理条件。父分离任务与逐轨子任务会一起清理，运行中任务不会被清理。结果页也可删除任务及相关文件，删除前会再次确认。

常见判断：

- 后端显示 `127.0.0.1:8765`：服务当前仅面向本机。局域网模式使用本机局域网 IPv4，配置变更在后端重启后生效。
- 前端能打开但显示后端离线：检查 `backend_url`、后端监听地址、TCP `8765` 防火墙和 `allowed_origins`。
- 浏览器提示 CORS：`allowed_origins` 与地址栏中的前端 origin 不完全一致。
- 服务电脑可用而其他电脑超时：检查实际局域网 IPv4、Windows 网络配置文件和入站规则。关闭整个防火墙会扩大暴露面，也无法验证规则是否配置正确。
- 前端或后端启动即退出：查看控制台的明确错误；前端日志位于运行数据目录的 `logs/web-frontend.log`，后端错误直接写入控制台。

桌面、Web 后端、Space 与 Colab 一次只运行一个加速器任务。多个入口同时提交时，其他任务会等待，并可在等待期间取消。
