# Docker 自托管

<p align="center">
  中文 | <a href="docker-deployment.en.md">English</a>
</p>

[Issue #9](https://github.com/mason369/music-to-midi/issues/9) 已在 `v1.6.0` 提供 Docker 自托管发行。使用者可以在自己的 NVIDIA 主机运行当前浏览器 Web 与 API；外置模型不包含在镜像中，而是按所选路线下载到持久卷。

Docker 入口使用与桌面版相同的 `MusicToMidiPipeline`、处理模式、模型路由、MIDI 生成和编辑语义。容器中提供浏览器 Web 与 API，不运行 Linux PyQt6 桌面窗口。

## 快速开始

### Linux

```bash
unzip MusicToMidi-Docker-vX.Y.Z.zip
cd MusicToMidi-Docker-vX.Y.Z
cp .env.selfhost.example .env
# 端口、GPU 和模型配置保存在 .env
bash scripts/docker_selfhost.sh setup
```

### Windows / Docker Desktop

```powershell
Expand-Archive .\MusicToMidi-Docker-vX.Y.Z.zip
Set-Location .\MusicToMidi-Docker-vX.Y.Z
Copy-Item .env.selfhost.example .env
# 端口、GPU 和模型配置保存在 .env
.\scripts\docker_selfhost.ps1 setup
```

安装完成后的默认地址是：

```text
http://127.0.0.1:7860
```

`setup` 会拉取固定版本镜像、检查 GPU 运行时、准备所选模型、启动服务并完成就绪验收。错误会以非零状态返回，同时保留相关诊断信息。

## 运行环境

| 项目 | 支持范围 |
|---|---|
| 容器平台 | `linux/amd64` |
| GPU | NVIDIA CUDA；当前没有 CPU、AMD/ROCm、Intel XPU 或 `linux/arm64` 容器变体 |
| 推理栈 | PyTorch `2.7.0+cu128`、CUDA runtime 12.8、ONNX Runtime GPU `1.23.2` |
| 宿主软件 | Docker Engine 24+、Docker Compose v2.17+、NVIDIA Container Toolkit |
| NVIDIA 驱动 | 能够运行 CUDA 12.8 容器的驱动 |
| 作业空间 | 默认保留至少 20 GiB 空闲空间；模型卷和镜像空间另计 |

Windows 使用 Docker Desktop 的 Linux 容器与 WSL2 GPU 支持。Linux 使用 Docker Engine 和 NVIDIA Container Toolkit。

以下命令可以单独确认宿主机的 Docker GPU 通道：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

显存需求会随模型、音频长度和处理模式变化，因此没有一个适用于全部 15 个配置的最低显存值。`setup` 会在下载模型前用实际后端镜像检查 PyTorch CUDA、GPU 可见性和 ONNX Runtime `CUDAExecutionProvider`。

## 下载与校验

正式版本在 [GitHub Releases](https://github.com/mason369/music-to-midi/releases) 提供以下文件：

```text
MusicToMidi-Docker-vX.Y.Z.zip
MusicToMidi-Docker-vX.Y.Z.tar.gz
MusicToMidi-Docker-compose-vX.Y.Z.yaml
MusicToMidi-Docker-env-vX.Y.Z.example
MusicToMidi-Docker-image-digests-vX.Y.Z.txt
MusicToMidi-Docker-SBOM-vX.Y.Z.txt
MusicToMidi-Docker-SHA256SUMS-vX.Y.Z.txt
```

Linux 可以这样核对 ZIP：

```bash
sha256sum MusicToMidi-Docker-vX.Y.Z.zip
grep 'MusicToMidi-Docker-vX.Y.Z.zip$' MusicToMidi-Docker-SHA256SUMS-vX.Y.Z.txt
```

Windows PowerShell 可以这样核对 ZIP：

```powershell
Get-FileHash .\MusicToMidi-Docker-vX.Y.Z.zip -Algorithm SHA256
Select-String .\MusicToMidi-Docker-SHA256SUMS-vX.Y.Z.txt -Pattern 'MusicToMidi-Docker-vX.Y.Z.zip$'
```

两处 SHA-256 值一致后，下载文件与 Release 清单对应。ZIP 与 tar.gz 内的 Compose 已固定到发布流水线匿名拉取验证过的镜像 digest，因此后续移动的 `latest` 标签不会改变这次安装。

解压后的目录包含：

```text
MusicToMidi-Docker-vX.Y.Z/
  compose.yaml
  .env.selfhost.example
  scripts/docker_selfhost.sh
  scripts/docker_selfhost.ps1
  README.md
  README_EN.md
  IMAGE_DIGESTS.txt
  RELEASE-MANIFEST.json
  LICENSE
  THIRD_PARTY_NOTICES.md
  THIRD_PARTY_SBOM.txt
```

## 配置

`.env.selfhost.example` 提供可直接启动的安全默认值。复制为 `.env` 后可以调整以下项目：

| 变量 | 默认值 | 含义 |
|---|---:|---|
| `MUSIC_TO_MIDI_PORT` | `7860` | 宿主机 loopback 访问端口，范围 `1-65535` |
| `GPU_DEVICE_ID` | `0` | 传给容器的 NVIDIA GPU 编号，编号可从 `nvidia-smi` 查看 |
| `MUSIC_TO_MIDI_ENABLED_PROFILES` | 两个默认配置 | 逗号分隔的模型配置；空值或未知配置会使初始化停止 |
| `MAX_UPLOAD_BYTES` | `4294967296` | API 单文件上传上限，单位为字节，默认 4 GiB |
| `MAX_REQUEST_BODY_SIZE` | `4GiB` | 网关请求体上限；通常与 API 上传上限保持一致 |
| `MAX_QUEUED_JOBS` | `4` | 等待中的作业上限；达到上限时返回 HTTP 429，`0` 表示不限制等待数量 |
| `MIN_FREE_BYTES` | `21474836480` | 作业卷最小保留空间，单位为字节，默认 20 GiB；`0` 表示不保留额外空间 |
| `RETENTION_DAYS` | `7` | 已结束作业的最长保留天数；`0` 表示不按天数清理 |
| `RETENTION_MAX_JOBS` | `50` | 已结束作业的数量上限；`0` 表示不按数量清理 |
| `RETENTION_MAX_BYTES` | `107374182400` | 作业数据总量上限，单位为字节，默认 100 GiB；`0` 表示不按总量清理 |
| `SHM_SIZE` | `2gb` | 后端与模型初始化容器的共享内存大小 |
| `LOG_LEVEL` | `info` | `critical`、`error`、`warning`、`info`、`debug` 或 `trace` |

端口只映射到 `127.0.0.1`。该默认值不会把服务发布到局域网或互联网。

## 模型配置

`MUSIC_TO_MIDI_ENABLED_PROFILES` 使用逗号分隔的配置 ID。默认组合兼顾常用多乐器和钢琴转写，同时控制首次下载量：

```dotenv
MUSIC_TO_MIDI_ENABLED_PROFILES=yourmt3:yptf_moe_multi_nops,piano_transkun
```

可用配置如下：

| 配置 ID | 路线 |
|---|---|
| `yourmt3:ymt3_plus` | YourMT3+ YMT3+ |
| `yourmt3:yptf_single_nops` | YourMT3+ YPTF Single (noPS) |
| `yourmt3:yptf_multi_ps` | YourMT3+ YPTF Multi (PS) |
| `yourmt3:yptf_moe_multi_nops` | YourMT3+ YPTF MoE Multi (noPS)，默认多乐器路线 |
| `yourmt3:yptf_moe_multi_ps` | YourMT3+ YPTF MoE Multi (PS) |
| `miros` | MIROS / MusicFM |
| `muscriptor` | MuScriptor Large |
| `muscriptor:medium` | MuScriptor Medium |
| `muscriptor:small` | MuScriptor Small |
| `piano_transkun` | TransKun 默认 V2（包内资源） |
| `piano_transkun_v2_aug` | TransKun V2 Aug |
| `piano_aria_amt` | Aria-AMT |
| `piano_bytedance_pedal` | ByteDance Piano Pedal |
| `vocal_split` | Leap XE vocals + PolarFormer accompaniment |
| `six_stem_split` | BS-RoFormer SW Fixed 六声部 |

修改列表后，以下命令会补齐并校验新选择的模型，然后重新启动服务：

```bash
bash scripts/docker_selfhost.sh models
bash scripts/docker_selfhost.sh start
```

```powershell
.\scripts\docker_selfhost.ps1 models
.\scripts\docker_selfhost.ps1 start
```

MuScriptor Small、Medium 和 Large 使用 Hugging Face gated 仓库。相应账户需要接受所选模型页面的条款。令牌只在模型初始化进程中使用：

```bash
export HF_TOKEN='...'
bash scripts/docker_selfhost.sh models
unset HF_TOKEN
```

```powershell
$env:HF_TOKEN = '...'
.\scripts\docker_selfhost.ps1 models
Remove-Item Env:HF_TOKEN
```

`.env` 不保存 `HF_TOKEN`，常驻后端也不会收到该令牌。模型初始化容器在任务结束后删除。

## `setup` 的工作内容

一次 `setup` 包含以下检查：

1. Docker Engine、Compose 版本和 Compose 配置。
2. 后端与网关镜像拉取。
3. 实际后端镜像中的 PyTorch CUDA 12.8、GPU 可见性和 ONNX Runtime CUDA provider。
4. 镜像支持的模型配置与 `.env` 选择的一致性。
5. 所选模型、Beat This、MuseScore General SoundFont 和 FluidSynth 的准备与校验。
6. 非 root、只读根文件系统和最小 capability 的后端与网关启动。
7. 内部及浏览器入口 `/api/v1/ready`、API `2.0`、用户 `10001:10001` 和内部推理网络验收。

常驻后端设置为 Hugging Face/Transformers 离线，并连接到无互联网出口的内部推理网络。运行期间缺失模型会显示为就绪失败，不会触发自动下载或切换到其他模型。

## 日常管理与日志

Linux：

```bash
bash scripts/docker_selfhost.sh status
bash scripts/docker_selfhost.sh verify
bash scripts/docker_selfhost.sh stop
bash scripts/docker_selfhost.sh start
```

Windows：

```powershell
.\scripts\docker_selfhost.ps1 status
.\scripts\docker_selfhost.ps1 verify
.\scripts\docker_selfhost.ps1 stop
.\scripts\docker_selfhost.ps1 start
```

实时日志：

```bash
docker compose --env-file .env -f compose.yaml logs --follow backend gateway
```

最近 200 行日志：

```bash
docker compose --env-file .env -f compose.yaml logs --tail 200 backend gateway
```

`stop` 使用 `docker compose down`，容器停止后四个命名卷仍然保留。`docker compose down -v` 会同时删除模型、作业和 Caddy 状态，适合完整重置而不是日常停止。

## 数据、备份与恢复

| 命名卷 | 内容 |
|---|---|
| `music-to-midi_model_data` | 下载并校验后的模型与模型缓存 |
| `music-to-midi_job_data` | 上传文件、进度、结果和编辑试听资产 |
| `music-to-midi_caddy_data` | 网关运行状态 |
| `music-to-midi_caddy_config` | 网关配置状态 |

重要结果可以直接从 Web 下载到普通文件。升级前停止服务并备份 `music-to-midi_job_data`，可以避免升级期间继续写入。

Linux 的 Docker 卷备份示例：

```bash
stamp=$(date +%Y%m%d-%H%M%S)
docker run --rm \
  -v music-to-midi_job_data:/source:ro \
  -v "$PWD:/backup" \
  alpine:3.22 \
  tar -czf "/backup/music-to-midi-job-data-${stamp}.tar.gz" -C /source .
```

Windows PowerShell 的 Docker 卷备份示例：

```powershell
$stamp = Get-Date -Format yyyyMMdd-HHmmss
docker run --rm `
  -v music-to-midi_job_data:/source:ro `
  -v "${PWD}:/backup" `
  alpine:3.22 `
  tar -czf "/backup/music-to-midi-job-data-$stamp.tar.gz" -C /source .
```

恢复时使用一个空的 `music-to-midi_job_data` 卷，再把备份解压到该卷。已有卷中的同名文件会被覆盖，而备份中不存在的旧文件仍会保留，因此空卷恢复的结果最明确。下面的 `docker volume create` 适用于该卷不存在的情况；同名卷已经存在时，Docker 会复用原卷：

```bash
docker volume create music-to-midi_job_data
docker run --rm \
  -v music-to-midi_job_data:/target \
  -v "$PWD:/backup:ro" \
  alpine:3.22 \
  tar -xzf /backup/<备份文件名>.tar.gz -C /target
```

## 升级与回滚

发行包中的镜像 digest 是实际运行版本。升级流程如下：

1. 查看新版本 Release notes 中的平台、许可和数据兼容说明。
2. 保留旧发行目录、旧 `.env` 和 `IMAGE_DIGESTS.txt`，并备份重要作业数据。
3. 下载并校验新版本 Docker 发行包。
4. 把旧 `.env` 复制到新目录，与新模板比较新增配置。
5. 在新目录运行 `setup`。
6. 使用实际音频检查每个已启用配置、分离后的逐轨转换、编辑试听和下载。

如果新版本验收未通过且 Release notes 没有不可逆迁移说明，旧发行目录中的 `start` 会按旧 digest 重建容器并复用原命名卷。含不可逆迁移的版本以对应 Release notes 和升级前备份为准。

## 本机、远程和公网访问

默认数据路径如下：

```text
Browser -> 127.0.0.1:7860 -> gateway -> internal backend:8765 -> NVIDIA GPU
                                             |-> model_data
                                             +-> job_data
```

本机自托管不涉及以下信息：

- 公网域名用于互联网访问。
- ACME 邮箱用于公开 HTTPS 证书的签发与续期通知。
- SSH 密码或密钥属于使用者登录远程主机的凭据，应用不会读取或保存。

远程服务器可以继续保持 loopback 绑定，并通过使用者自己的 SSH 连接转发端口：

```bash
ssh -L 7860:127.0.0.1:7860 -p <SSH端口> <用户>@<服务器>
```

隧道建立后，本机浏览器仍访问 `http://127.0.0.1:7860`。

公开互联网部署使用仓库中的 `compose.production.yaml`、`.env.production.example` 和 `scripts/setup_docker_production.ps1`。该模式包含公网 DNS、80/443、ACME 联系邮箱、Caddy Argon2id Basic Auth、HTTPS 终止和内部推理网络隔离。它面向单一所有者服务，不包含账号系统、租户隔离、配额归属或逐任务授权。

## 常见问题

### Docker 或 Compose 版本不符合要求

```bash
docker version
docker compose version
```

`setup` 会显示检测到的版本和最低版本要求。

### Docker 看不到 NVIDIA GPU

宿主机的 `nvidia-smi` 与前面的 NVIDIA CUDA 容器测试都需要正常返回。Windows Docker Desktop 同时需要 WSL2 后端和 GPU 支持；Linux 需要 NVIDIA Container Toolkit。

### `CUDAExecutionProvider` 不可用

该错误表示实际后端镜像没有获得当前固定的 ONNX Runtime CUDA provider。常见原因是宿主驱动、Docker GPU runtime 或 GPU 暴露配置异常。`setup` 会在模型下载前停止并显示 provider 列表。

### 端口已占用

在 `.env` 中更换 `MUSIC_TO_MIDI_PORT`，然后重新执行 `start`。端口范围为 `1-65535`。

### MuScriptor 返回 401 或 403

Hugging Face 账户需要接受对应 Small、Medium 或 Large 模型页面的条款，并在当前终端提供具有读取权限的 `HF_TOKEN`。令牌只用于 `models` 操作。

### 模型初始化中断

重新执行 `models` 会再次校验所选配置；已经完整且身份匹配的资产会继续使用，缺失或不完整的资产会显示具体失败原因。

### HTTP 507 或磁盘空间不足

`MIN_FREE_BYTES` 是作业卷的保留空间。`status`、`verify` 和后端日志会显示当前容量状态。已下载的重要结果可以移出作业卷，保留策略也可以在 `.env` 中调整。

### 服务启动但页面未就绪

```bash
bash scripts/docker_selfhost.sh verify
docker compose --env-file .env -f compose.yaml logs --tail 200 backend gateway
```

Windows 使用同名 PowerShell `verify` 操作；Docker Compose 日志命令保持相同。

## 停止、重置与移除

日常停止会保留全部数据：

```bash
bash scripts/docker_selfhost.sh stop
```

完整重置会删除四个命名卷及其中的模型、作业和网关状态：

```bash
docker compose --env-file .env -f compose.yaml down -v
```

容器与卷移除后，发行目录可以作为普通文件夹删除。镜像保留在 Docker 本地缓存中，后续安装可以复用；如需释放镜像空间，可在确认没有其他容器使用后通过 Docker Desktop 或 `docker image rm` 删除 `IMAGE_DIGESTS.txt` 中列出的镜像。

## 发布验证范围

容器发布工作流覆盖以下项目：

1. Docker、许可、Compose 和管理脚本契约测试。
2. `linux/amd64` 后端与网关构建，以及非 root、只读根文件系统的镜像级运行检查。
3. OCI SBOM、provenance、GitHub registry attestation 和固定版本标签。
4. 无 GHCR 登录状态下的匿名拉取。
5. 后端的 15 个模型配置、离线资源入口和网关动态同源配置。
6. digest 固定 Compose、Bash/PowerShell 语法、ZIP/tar.gz 和 SHA-256。

这些检查覆盖源码、镜像内容、公开分发和无模型启动契约。GitHub 托管 runner 不提供 NVIDIA GPU，因此自动化结果不代表所有模型已经在每种实体 GPU 上完成真实音频推理。首次部署时的 `setup` 会验证实际 GPU 与 provider；最终的模型质量、显存占用和长音频表现仍取决于使用者启用的配置、硬件和输入音频。
