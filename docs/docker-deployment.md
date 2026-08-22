# Docker 生产部署

这套部署用于落实 [Issue #9](https://github.com/mason369/music-to-midi/issues/9)：容器镜像包含应用源码和固定推理环境，外置模型由运维者按配置显式下载到持久卷。它部署当前浏览器前端与独立 Web API，不是桌面 GUI 的容器化版本。

## 已确定的运行边界

- 目标平台只有 `linux/amd64 + NVIDIA CUDA 12.8`。需要可运行 Docker Compose GPU reservation 的 Docker Engine/Compose v2、NVIDIA Container Toolkit，以及兼容 CUDA 12.8 的 NVIDIA 驱动。
- 不支持 CPU、AMD/ROCm、Intel XPU 或 `linux/arm64` 静默降级。GPU、PyTorch flavor、ONNX Runtime provider 或所选模型不满足条件时，后端保持未就绪并明确退出。
- Caddy 是唯一公网入口，只映射宿主机 `80/tcp`、`443/tcp` 与 `443/udp`。后端 `8765` 只有容器内部网络可访问。
- Caddy 为整个站点执行自动 HTTPS 与 Argon2id Basic Auth。后端将此识别为“经认证的单一所有者”部署；它没有账号系统、租户隔离或逐任务所有权授权，不能作为多人互不信任的 SaaS 使用。
- 推理请求永远不会自动下载缺失模型。模型只能由运维者执行 `model-init` 准备；任一下载、身份校验或运行时检查失败都会让命令返回非零。
- 镜像不复制项目模型目录、缓存、音频、MIDI、本地 JSON 配置、`.env`、Git 元数据或 agent 指令。`transkun==2.0.1` 的默认 V2 权重是上游 Python 包自带资源，这是“外置 checkpoint”规则中唯一明确记录的包内资源。

## 生产拓扑

```text
Internet
   |
   | 80/443 (HTTPS + Basic Auth + request-body limit)
   v
Caddy gateway  ---- serves ----> /srv/web
   |
   | internal Docker network only
   v
FastAPI backend :8765  ----> NVIDIA GPU
   |                         one inference worker
   +---- /models  (model_data volume)
   +---- /data    (job_data volume)
```

常驻 `backend` 只连接 `internal: true` 的推理网络，没有互联网出口；只有一次性 `model-init` 连接独立下载网络。外部模型必须在初始化阶段完整落盘并通过身份校验，运行中的用户请求没有联网补模型的通道。

## 前置条件

1. 把一个公开 DNS `A`/`AAAA` 记录指向部署主机，确认公网入站 `80/tcp`、`443/tcp` 和 `443/udp` 能到达该主机。家庭网络若没有公网 IPv4，可只配置可达的公网 IPv6；DNS 与端口转发必须在运行脚本前真实可用。
2. 安装 Docker Engine 24+ 与 Compose v2。`docker version` 和 `docker compose version` 必须成功。
3. 按 NVIDIA 官方方式安装 NVIDIA Container Toolkit，并先验证：

   ```bash
   docker run --rm --gpus device=0 \
     nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 \
     nvidia-smi
   ```

4. 为模型卷预留足够空间。默认作业门禁还要求 `/data` 至少保留 20 GiB 可用空间，并按 7 天、50 个终态作业或 100 GiB 三个上限中最先达到者清理完整任务族。
5. 如果启用 MuScriptor 任一档，先在 Hugging Face 分别接受 Small/Medium/Large 中实际所选仓库的条款，并只在执行模型初始化的终端会话设置 `HF_TOKEN`。令牌不会写进 `.env.production`，也不会传给常驻后端。

## 模型配置

`MUSIC_TO_MIDI_ENABLED_PROFILES` 是必填的逗号分隔列表。只有被选择且严格就绪的路线会在浏览器中启用：

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

任一直接转 MIDI 配置还会准备并校验 Beat This `final0.ckpt`，因为导出 BPM/tempo 需要同一个固定检测器。没有选择的配置不会被下载，也不会被标成可用。

推荐先以最小可用组合上线：

```dotenv
MUSIC_TO_MIDI_ENABLED_PROFILES=yourmt3:yptf_moe_multi_nops,piano_transkun
```

确认空间、显存和实际音频验收后，再显式增加其他配置。改变列表后必须重跑 `model-init`，随后重启后端；缺少新增模型时后端不会带病启动。

## Windows 一键部署

从仓库根目录运行：

```powershell
.\scripts\setup_docker_production.ps1
```

脚本会依次执行以下硬门禁：

1. 检查 Docker Engine、Compose v2 和指定 GPU 的 CUDA 容器。
2. 交互读取域名、ACME 邮箱和密码；密码仅经标准输入交给固定 Caddy 镜像生成 Argon2id 哈希，磁盘只保存哈希。
3. 生成被 Git 忽略的 `.env.production`，校验 Compose 配置并从当前工作区构建两个镜像。
4. 运行一次 `model-init`，严格准备所选模型。
5. 启动后端和网关，等待内部 `/api/v1/ready` 真正通过。
6. 再次读取密码，从公网 `https://<域名>/api/v1/ready` 验证证书、认证、反向代理、GPU 与模型就绪状态。

任一步返回非零都会停止。脚本不会把失败包装成成功，也不会改用 CPU 或另一条模型路线。

也可以明确传参：

```powershell
.\scripts\setup_docker_production.ps1 `
  -PublicAddress "midi.example.com" `
  -AcmeEmail "admin@example.com" `
  -BasicAuthUser "mason" `
  -Profiles "yourmt3:yptf_moe_multi_nops,piano_transkun" `
  -GpuDeviceId "0"
```

## Linux 手动部署

复制示例配置并填写真实值：

```bash
cp .env.production.example .env.production
chmod 600 .env.production
```

生成密码哈希时不要把明文密码放进命令参数或 `.env.production`：

```bash
read -rsp 'Basic Auth password: ' BASIC_PASSWORD; echo
printf '%s' "$BASIC_PASSWORD" | docker run --rm -i \
  caddy:2.11.4-alpine \
  caddy hash-password --algorithm argon2id
unset BASIC_PASSWORD
```

把输出作为单引号值写入 `BASIC_AUTH_HASH`。随后执行：

```bash
docker compose --env-file .env.production -f compose.production.yaml config --quiet
docker compose --env-file .env.production -f compose.production.yaml build --pull

# 只有启用 gated MuScriptor 时才在当前进程临时提供：export HF_TOKEN=...
docker compose --env-file .env.production -f compose.production.yaml \
  --profile tools run --rm model-init

docker compose --env-file .env.production -f compose.production.yaml up -d
docker compose --env-file .env.production -f compose.production.yaml ps
docker compose --env-file .env.production -f compose.production.yaml \
  exec -T backend python /app/docker/healthcheck.py
```

最后必须从外部网络携带 Basic Auth 验证：

```bash
curl --fail --user 'mason' https://midi.example.com/api/v1/ready
```

`curl` 会交互读取密码；不要把密码直接写进 shell 历史。

## 使用 GHCR 发布镜像

`.github/workflows/container.yml` 在 `master`、版本标签和手动触发时构建两个无模型的 `linux/amd64` 镜像：

- `ghcr.io/mason369/music-to-midi-backend`
- `ghcr.io/mason369/music-to-midi-gateway`

工作流使用固定 commit 的 GitHub Actions，先对本地加载的构建结果执行镜像级烟雾检查，通过后再推送 OCI provenance 与 SBOM，并为实际推送 digest 生成 GitHub attestation。首次发布后，仓库维护者还需在 GitHub Packages 设置中确认这两个 package 的可见性是 `Public`；在确认前，不应把匿名 `docker pull` 成功当作已公开发布。

生产机若只使用已发布镜像，可保留 Compose 文件的 `image:`，先执行 `docker compose pull`，再运行同一个 `model-init` 与 `up -d` 流程。部署记录应保存镜像 digest，而不只保存会移动的 `latest` 标签。

## 就绪、容量与故障行为

- `/api/v1/health` 只表示 API 进程与推理 worker 存活。
- `/api/v1/ready` 同时检查 worker、CUDA 运行时、所选模型、公共部署 TLS/认证声明和数据盘最低可用空间。只有这个端点返回 HTTP 200 才能接流量。
- 上传超过后端 `MAX_UPLOAD_BYTES` 时返回明确的 HTTP 413；Caddy `MAX_REQUEST_BODY_SIZE` 提前执行同级门禁。
- 等待队列达到 `MAX_QUEUED_JOBS` 时返回 HTTP 429；磁盘低于 `MIN_FREE_BYTES` 时返回 HTTP 507。
- GPU 作业仍是单 worker 串行执行。队列限制解决容量保护，不等同于横向扩容。
- 模型下载、模型身份、CUDA provider、磁盘或认证配置失败都不会触发替代算法、自动降级或伪成功结果。

## 数据、备份与升级

四个命名卷分别保存模型、作业/Caddy 状态：

```text
music-to-midi_model_data
music-to-midi_job_data
music-to-midi_caddy_data
music-to-midi_caddy_config
```

升级镜像不会删除这些卷。备份前先停止写入，并分别归档 `model_data`、`job_data` 与 Caddy 数据卷；恢复后先运行 `verify-models`，再允许后端接流量：

```bash
docker compose --env-file .env.production -f compose.production.yaml stop backend gateway
docker compose --env-file .env.production -f compose.production.yaml \
  --profile tools run --rm model-init verify-models
```

不要把 `docker compose down -v` 当作普通停止命令；`-v` 会删除模型、作业和证书状态。普通停止使用：

```bash
docker compose --env-file .env.production -f compose.production.yaml down
```

升级验收至少包含：镜像 digest、`nvidia-smi` 容器检查、内部 readiness、公网 HTTPS+认证 readiness，以及每个启用配置的一段真实音频端到端结果。仅有镜像构建成功不能证明模型推理正确。
