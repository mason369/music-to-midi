# Docker 自托管发行

这套发行用于落实 [Issue #9](https://github.com/mason369/music-to-midi/issues/9)。交付物不是一个孤立的公开镜像，也不是由项目维护者代建公网网站，而是一套可下载、可验证、可升级和可回滚的单机自托管发行包：

| 交付物 | 用途 |
|---|---|
| `music-to-midi-backend` | 当前独立 Web API、共享 `MusicToMidiPipeline`、固定 CUDA 12.8 推理环境；不包含外置模型 |
| `music-to-midi-gateway` | 与当前仓库一致的浏览器前端、同源 API 反向代理和安全响应头 |
| `compose.yaml` | 固定同一版本的两个镜像；正式 Release 中进一步固定到不可变 digest |
| `.env.selfhost.example` | 端口、GPU、模型路线、容量与保留策略 |
| `scripts/docker_selfhost.sh` / `.ps1` | 拉取、GPU 实测、模型初始化、启动、就绪验收、状态与停止 |
| `IMAGE_DIGESTS.txt` / `RELEASE-MANIFEST.json` | 源码提交、平台、API 版本和实际镜像 digest |
| `SHA256SUMS`、SBOM、provenance、attestation | 下载文件和镜像供应链核验 |

模型按 Issue #9 的要求外置。用户在 `.env` 中明确选择路线后，由一次性 `model-init` 下载并严格校验到持久卷；没有选择的 checkpoint 不下载。常驻后端强制 Hugging Face/Transformers 离线并且只连接无互联网出口的内部网络，推理请求不会偷偷补模型。

Docker 提供当前浏览器 Web 与 API 能力，不在 Linux 容器中运行 PyQt6 桌面窗口。桌面版、Web、Space 和 Colab 继续共享处理流水线、模式、模型路由、MIDI 生成与编辑语义；容器镜像直接复制当前 `src/` 和 `web/`，发布测试会守护两端版本与功能契约。

## 支持范围

- 平台：`linux/amd64`。
- 加速器：NVIDIA GPU，PyTorch `2.7.0+cu128`，CUDA runtime 12.8，ONNX Runtime GPU `1.23.2`。
- 宿主机：Docker Engine 24+、Docker Compose v2.17+、NVIDIA Container Toolkit，以及兼容 CUDA 12.8 的 NVIDIA 驱动。
- 当前不提供 CPU、AMD/ROCm、Intel XPU 或 `linux/arm64` 容器变体，也不会静默降级。
- 镜像无外置 checkpoint；`transkun==2.0.1` 自带的默认 V2 包资源是唯一明确记录的包内模型资源。
- 默认作业卷至少需要 20 GiB 可用空间。模型所需空间随选择的路线变化；完整 15 配置远大于默认两配置。

安装前应先让 NVIDIA 官方容器测试成功；一键脚本还会在实际后端镜像中执行 `torch.cuda` 和 `CUDAExecutionProvider` 检查：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## 推荐：下载同版本自托管发行包

每个正式版本的 GitHub Release 都附带：

```text
MusicToMidi-Docker-vX.Y.Z.zip
MusicToMidi-Docker-vX.Y.Z.tar.gz
MusicToMidi-Docker-compose-vX.Y.Z.yaml
MusicToMidi-Docker-env-vX.Y.Z.example
MusicToMidi-Docker-image-digests-vX.Y.Z.txt
MusicToMidi-Docker-SBOM-vX.Y.Z.txt
MusicToMidi-Docker-SHA256SUMS-vX.Y.Z.txt
```

推荐下载 ZIP 或 tar.gz 以及对应 `SHA256SUMS`，先验证 SHA-256，再解压。包内 Compose 已从版本标签改写为发布流水线刚刚匿名拉取验证过的镜像 digest；以后 `latest` 移动也不会改变当前安装。

解压后目录为：

```text
MusicToMidi-Docker-vX.Y.Z/
  compose.yaml
  .env.selfhost.example
  scripts/docker_selfhost.sh
  scripts/docker_selfhost.ps1
  README.md
  IMAGE_DIGESTS.txt
  RELEASE-MANIFEST.json
  LICENSE
  THIRD_PARTY_NOTICES.md
  THIRD_PARTY_SBOM.txt
```

### Linux

```bash
cd MusicToMidi-Docker-vX.Y.Z
cp .env.selfhost.example .env
# 按需修改模型配置、端口、GPU 或容量限制
bash scripts/docker_selfhost.sh setup
```

### Windows / Docker Desktop

```powershell
Set-Location MusicToMidi-Docker-vX.Y.Z
Copy-Item .env.selfhost.example .env
# 按需修改模型配置、端口、GPU 或容量限制
.\scripts\docker_selfhost.ps1 setup
```

默认完成后只监听：

```text
http://127.0.0.1:7860
```

脚本不会询问公网域名、ACME 邮箱、登录密码或 SSH 密码。它依次执行：

1. 验证 Docker Engine、Compose 和 Compose 配置。
2. 拉取发行包固定的两个镜像。
3. 在实际后端镜像与所选 GPU 上验证 PyTorch CUDA 12.8、GPU 可见性和 ONNX Runtime CUDA provider。
4. 核对镜像真实支持的 15 个模型配置 ID。
5. 只下载并校验 `.env` 所选模型；直接转 MIDI 路线同时准备 Beat This、MuseScore General SoundFont，并验证 FluidSynth。
6. 启动只读、非 root、最小 capability 的后端和网关。
7. 等待内部及浏览器入口 `/api/v1/ready`，并核对 API `2.0`、只读根文件系统、用户 `10001:10001` 与内部推理网络。

任何一步失败都会返回非零并保留诊断；不会改用 CPU、另一模型、旧试听或假结果。

## 模型选择

`.env` 中的 `MUSIC_TO_MIDI_ENABLED_PROFILES` 是必填逗号列表。只有选择且严格就绪的配置会由 API 宣告可用：

| 配置 ID | 路线 |
|---|---|
| `yourmt3:ymt3_plus` | YourMT3+ YMT3+ |
| `yourmt3:yptf_single_nops` | YourMT3+ YPTF Single (noPS) |
| `yourmt3:yptf_multi_ps` | YourMT3+ YPTF Multi (PS) |
| `yourmt3:yptf_moe_multi_nops` | YourMT3+ YPTF MoE Multi (noPS)，推荐默认 |
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

默认配置是兼顾功能与下载量的最小组合：

```dotenv
MUSIC_TO_MIDI_ENABLED_PROFILES=yourmt3:yptf_moe_multi_nops,piano_transkun
```

增减列表后重新执行模型初始化，再启动或重启：

```bash
bash scripts/docker_selfhost.sh models
bash scripts/docker_selfhost.sh start
```

```powershell
.\scripts\docker_selfhost.ps1 models
.\scripts\docker_selfhost.ps1 start
```

MuScriptor Small/Medium/Large 是 gated 仓库。先用同一 Hugging Face 账号分别接受所选模型页条款，再只在当前终端临时提供令牌：

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

令牌不写入 `.env`，不传给常驻后端；一次性 model-init 容器退出后被删除。

## 日常管理

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

`stop` 使用普通 `docker compose down`，不会删除卷。不要把 `docker compose down -v` 当停止命令；`-v` 会删除模型、作业和 Caddy 状态。

四个稳定命名卷为：

```text
music-to-midi_model_data
music-to-midi_job_data
music-to-midi_caddy_data
music-to-midi_caddy_config
```

其中 `model_data` 可重新下载，`job_data` 保存上传、进度、结果与编辑试听资产。重要结果仍应从 Web 下载到普通文件备份；升级前应停止写入并用宿主机现有的卷备份方案保存 `job_data`。不要在容器运行时直接修改卷内文件。

## 升级与回滚

版本标签便于识别，Release 包中的 digest 才是实际运行锁。升级按以下顺序进行：

1. 阅读新版本 Release notes，确认平台、模型许可和数据兼容说明。
2. 保留旧发行目录、旧 `.env` 与 `IMAGE_DIGESTS.txt`，备份重要 `job_data`。
3. 下载并校验新版本 Docker 发行包。
4. 把旧 `.env` 复制到新目录，并与新模板逐项比较新增配置。
5. 在新目录执行 `setup`；它会拉取新 digest、验证 GPU、补齐所选模型并通过 readiness 后才报告成功。
6. 用至少一段真实音频验收每个启用配置，以及分离后的逐轨转换、编辑试听和下载。

若新版本验收失败且 Release notes 未声明不可逆迁移，回到保留的旧发行目录执行 `start`，Compose 会按旧 digest 重建容器并复用原命名卷。不要删除卷来“回滚”。若未来版本引入不可逆迁移，必须按该版本说明恢复升级前备份，不能假定旧程序能读取新数据。

## 为什么默认不需要域名、ACME 或 SSH 密码

默认 `compose.yaml` 把网关硬绑定到宿主机 `127.0.0.1`，仅本机可访问，所以不申请公网证书，也不收集登录密码：

```text
Browser -> 127.0.0.1:7860 -> Caddy -> internal backend:8765 -> NVIDIA GPU
                                      |-> model_data
                                      +-> job_data
```

- 公网域名：只用于把服务公开到互联网。
- ACME 邮箱：只用于 Caddy 自动申请/续期公开 HTTPS 证书时接收通知。
- SSH 密码/密钥：只是用户自己登录远程服务器或建立 SSH 隧道的凭据，不属于镜像或应用配置；项目不会收集或保存。

如果容器运行在远程服务器，推荐保持 loopback 绑定并由用户自己建立 SSH 隧道，例如把服务器 `127.0.0.1:7860` 转发到本机。应用发行包本身不需要知道 SSH 密码。

## 高级：真正的公网单所有者部署

仓库另保留 `compose.production.yaml`、`.env.production.example` 与 `scripts/setup_docker_production.ps1`。只有明确要把服务公开到互联网时才使用它们。该模式要求：

- 真实公网 DNS 和可达的 80/443；
- ACME 联系邮箱；
- Caddy Argon2id Basic Auth；
- HTTPS 终止与后端公共部署门禁；
- 公网入口与内部推理网络隔离。

它仍是“经认证的单一所有者”服务，不是多人互不信任的 SaaS：没有账号系统、租户隔离、配额归属或逐任务授权。普通 Issue #9 使用者不需要这套公网参数。

## 就绪、容量与失败语义

- `/api/v1/health`：API 进程与单 GPU worker 存活。
- `/api/v1/ready`：worker、CUDA runtime、所选模型、Beat This、SoundFont、FluidSynth 和数据盘容量全部通过。
- 上传超过限制：HTTP 413。
- 等待队列达到上限：HTTP 429。
- 数据盘低于最小空闲量：HTTP 507。
- GPU 作业单 worker 串行；队列限制是容量保护，不是横向扩容。
- 模型下载、身份、CUDA provider、SoundFont、FluidSynth、磁盘或配置失败都会显式终止。

## 发布与验收证据

`.github/workflows/container.yml` 在版本标签上执行：

1. 容器、许可证、自托管 Compose 和脚本静态契约测试。
2. 构建 `linux/amd64` 后端/网关，并以非 root、只读方式做镜像级 smoke。
3. 推送固定版本、commit SHA 与 `latest` 标签，同时生成 OCI SBOM、provenance 和 GitHub registry attestation。
4. 在一个没有 GHCR 登录状态的新 job 中匿名拉取刚发布的版本；私有或不可拉取会让发布失败。
5. 对实际拉取的后端检查 15 配置、最新 MIDI 量化与离线 SoundFont 入口；真实启动网关并检查动态同源配置。
6. 把实际 digest 写入 Compose 和清单，验证 Compose、Bash/PowerShell 语法，生成 ZIP/tar.gz 与 SHA-256 后上传到同版本 Release。

这些证明覆盖源码、构建、镜像内容、公开分发和无模型启动契约。GitHub 托管 runner 没有 NVIDIA GPU，因此“镜像成功构建”仍不等同于全部模型真实推理成功；正式发版还必须在兼容的 Linux NVIDIA 主机上，对所选模型执行真实音频端到端验收，并分别记录输出、模型配置与镜像 digest。
