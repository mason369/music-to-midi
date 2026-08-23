# Docker Self-Hosting

<p align="center">
  <a href="docker-deployment.md">中文</a> | English
</p>

[Issue #9](https://github.com/mason369/music-to-midi/issues/9) is delivered in `v1.6.0` as a Docker self-host package. It runs the current browser UI and API on the user's NVIDIA host. External model checkpoints are not baked into the images; selected profiles are downloaded into a persistent volume.

The Docker entry point uses the same `MusicToMidiPipeline`, processing modes, model routes, MIDI generation, and editing semantics as the desktop application. It provides the browser UI and API, not a Linux PyQt6 desktop window.

## Quick start

### Linux

```bash
unzip MusicToMidi-Docker-vX.Y.Z.zip
cd MusicToMidi-Docker-vX.Y.Z
cp .env.selfhost.example .env
# Port, GPU, and model profile selection is stored in .env
bash scripts/docker_selfhost.sh setup
```

### Windows / Docker Desktop

```powershell
Expand-Archive .\MusicToMidi-Docker-vX.Y.Z.zip
Set-Location .\MusicToMidi-Docker-vX.Y.Z
Copy-Item .env.selfhost.example .env
# Port, GPU, and model profile selection is stored in .env
.\scripts\docker_selfhost.ps1 setup
```

The default local address is:

```text
http://127.0.0.1:7860
```

`setup` pulls the pinned images, validates the GPU runtime, prepares the selected models, starts the services, and completes readiness checks. An error returns a non-zero status and keeps the relevant diagnostics visible.

## Supported environment

| Area | Supported range |
|---|---|
| Container platform | `linux/amd64` |
| GPU | NVIDIA CUDA; no CPU, AMD/ROCm, Intel XPU, or `linux/arm64` container variant is currently published |
| Inference stack | PyTorch `2.7.0+cu128`, CUDA runtime 12.8, ONNX Runtime GPU `1.23.2` |
| Host software | Docker Engine 24+, Docker Compose v2.17+, NVIDIA Container Toolkit |
| NVIDIA driver | A driver capable of running CUDA 12.8 containers |
| Job storage | At least 20 GiB remains free by default; image and model storage are additional |

Windows uses Docker Desktop with Linux containers and WSL2 GPU support. Linux uses Docker Engine with NVIDIA Container Toolkit.

The host GPU path can be checked independently with:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

VRAM use varies by model, audio length, and processing mode, so one minimum cannot describe all 15 profiles. Before any model download, `setup` checks PyTorch CUDA, GPU visibility, and ONNX Runtime `CUDAExecutionProvider` in the actual backend image.

## Downloads and checksums

Stable releases provide these files on [GitHub Releases](https://github.com/mason369/music-to-midi/releases):

```text
MusicToMidi-Docker-vX.Y.Z.zip
MusicToMidi-Docker-vX.Y.Z.tar.gz
MusicToMidi-Docker-compose-vX.Y.Z.yaml
MusicToMidi-Docker-env-vX.Y.Z.example
MusicToMidi-Docker-image-digests-vX.Y.Z.txt
MusicToMidi-Docker-SBOM-vX.Y.Z.txt
MusicToMidi-Docker-SHA256SUMS-vX.Y.Z.txt
```

Linux ZIP verification:

```bash
sha256sum MusicToMidi-Docker-vX.Y.Z.zip
grep 'MusicToMidi-Docker-vX.Y.Z.zip$' MusicToMidi-Docker-SHA256SUMS-vX.Y.Z.txt
```

Windows PowerShell ZIP verification:

```powershell
Get-FileHash .\MusicToMidi-Docker-vX.Y.Z.zip -Algorithm SHA256
Select-String .\MusicToMidi-Docker-SHA256SUMS-vX.Y.Z.txt -Pattern 'MusicToMidi-Docker-vX.Y.Z.zip$'
```

Matching SHA-256 values associate the downloaded archive with the Release manifest. The packaged Compose file pins the anonymously verified image digests, so a later `latest` tag change does not alter an existing installation.

The archive contains:

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

## Configuration

`.env.selfhost.example` contains local-only defaults. Copying it to `.env` exposes these settings:

| Variable | Default | Meaning |
|---|---:|---|
| `MUSIC_TO_MIDI_PORT` | `7860` | Host loopback port, in the range `1-65535` |
| `GPU_DEVICE_ID` | `0` | NVIDIA GPU index shown by `nvidia-smi` |
| `MUSIC_TO_MIDI_ENABLED_PROFILES` | Two default profiles | Comma-separated model profiles; an empty or unknown value stops initialization |
| `MAX_UPLOAD_BYTES` | `4294967296` | API upload limit in bytes; default 4 GiB |
| `MAX_REQUEST_BODY_SIZE` | `4GiB` | Gateway body limit, normally kept aligned with the API upload limit |
| `MAX_QUEUED_JOBS` | `4` | Waiting-job limit; HTTP 429 is returned at capacity; `0` removes this limit |
| `MIN_FREE_BYTES` | `21474836480` | Reserved free space in the job volume; default 20 GiB; `0` disables the reserve |
| `RETENTION_DAYS` | `7` | Maximum age of completed jobs; `0` disables age-based cleanup |
| `RETENTION_MAX_JOBS` | `50` | Maximum completed-job count; `0` disables count-based cleanup |
| `RETENTION_MAX_BYTES` | `107374182400` | Job-data limit in bytes; default 100 GiB; `0` disables size-based cleanup |
| `SHM_SIZE` | `2gb` | Shared memory for backend and model initialization containers |
| `LOG_LEVEL` | `info` | `critical`, `error`, `warning`, `info`, `debug`, or `trace` |

The port is bound to `127.0.0.1`, so the default stack is not published to the LAN or Internet.

## Model profiles

`MUSIC_TO_MIDI_ENABLED_PROFILES` is a comma-separated list. The default combines a common multi-instrument route with the bundled TransKun model while limiting the initial download:

```dotenv
MUSIC_TO_MIDI_ENABLED_PROFILES=yourmt3:yptf_moe_multi_nops,piano_transkun
```

| Profile ID | Route |
|---|---|
| `yourmt3:ymt3_plus` | YourMT3+ YMT3+ |
| `yourmt3:yptf_single_nops` | YourMT3+ YPTF Single (noPS) |
| `yourmt3:yptf_multi_ps` | YourMT3+ YPTF Multi (PS) |
| `yourmt3:yptf_moe_multi_nops` | YourMT3+ YPTF MoE Multi (noPS), the default multi-instrument route |
| `yourmt3:yptf_moe_multi_ps` | YourMT3+ YPTF MoE Multi (PS) |
| `miros` | MIROS / MusicFM |
| `muscriptor` | MuScriptor Large |
| `muscriptor:medium` | MuScriptor Medium |
| `muscriptor:small` | MuScriptor Small |
| `piano_transkun` | TransKun default V2, bundled in its Python package |
| `piano_transkun_v2_aug` | TransKun V2 Aug |
| `piano_aria_amt` | Aria-AMT |
| `piano_bytedance_pedal` | ByteDance Piano Pedal |
| `vocal_split` | Leap XE vocals + PolarFormer accompaniment |
| `six_stem_split` | BS-RoFormer SW Fixed six-stem separation |

After changing the list, these commands prepare the additional profiles and restart the stack:

```bash
bash scripts/docker_selfhost.sh models
bash scripts/docker_selfhost.sh start
```

```powershell
.\scripts\docker_selfhost.ps1 models
.\scripts\docker_selfhost.ps1 start
```

MuScriptor Small, Medium, and Large are gated Hugging Face repositories. The account needs access to each selected model page. The token is scoped to model initialization:

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

`HF_TOKEN` is not stored in `.env` or sent to the long-running backend. The one-shot model initialization container is removed when it exits.

## What `setup` checks

One `setup` run covers:

1. Docker Engine, Compose version, and Compose configuration.
2. Backend and gateway image pulls.
3. PyTorch CUDA 12.8, GPU visibility, and ONNX Runtime CUDA provider in the actual backend image.
4. Agreement between image-supported profiles and the `.env` selection.
5. Selected models, Beat This, MuseScore General SoundFont, and FluidSynth.
6. Non-root, read-only backend and gateway containers with minimal capabilities.
7. Internal and browser-facing `/api/v1/ready`, API `2.0`, user `10001:10001`, and the internal inference network.

The long-running backend sets Hugging Face and Transformers to offline mode and only joins the internal inference network. A missing model appears as a readiness error rather than triggering an automatic download or a different model.

## Routine operations and logs

Linux:

```bash
bash scripts/docker_selfhost.sh status
bash scripts/docker_selfhost.sh verify
bash scripts/docker_selfhost.sh stop
bash scripts/docker_selfhost.sh start
```

Windows:

```powershell
.\scripts\docker_selfhost.ps1 status
.\scripts\docker_selfhost.ps1 verify
.\scripts\docker_selfhost.ps1 stop
.\scripts\docker_selfhost.ps1 start
```

Live logs:

```bash
docker compose --env-file .env -f compose.yaml logs --follow backend gateway
```

Last 200 log lines:

```bash
docker compose --env-file .env -f compose.yaml logs --tail 200 backend gateway
```

`stop` uses `docker compose down` and preserves all four named volumes. `docker compose down -v` also removes models, jobs, and gateway state, so it represents a full reset rather than a routine stop.

## Data, backup, and restore

| Named volume | Contents |
|---|---|
| `music-to-midi_model_data` | Downloaded and verified models and caches |
| `music-to-midi_job_data` | Uploads, progress, results, and preview assets |
| `music-to-midi_caddy_data` | Gateway runtime state |
| `music-to-midi_caddy_config` | Gateway configuration state |

Important outputs can be downloaded through the Web UI. Stopping the stack before backing up `music-to-midi_job_data` prevents writes during the archive operation.

Linux backup example:

```bash
stamp=$(date +%Y%m%d-%H%M%S)
docker run --rm \
  -v music-to-midi_job_data:/source:ro \
  -v "$PWD:/backup" \
  alpine:3.22 \
  tar -czf "/backup/music-to-midi-job-data-${stamp}.tar.gz" -C /source .
```

Windows PowerShell backup example:

```powershell
$stamp = Get-Date -Format yyyyMMdd-HHmmss
docker run --rm `
  -v music-to-midi_job_data:/source:ro `
  -v "${PWD}:/backup" `
  alpine:3.22 `
  tar -czf "/backup/music-to-midi-job-data-$stamp.tar.gz" -C /source .
```

Restore into an empty `music-to-midi_job_data` volume for a deterministic result. Existing files with matching names are overwritten, while unrelated old files remain when restoring into a non-empty volume. The following `docker volume create` applies when the volume does not exist; Docker reuses an existing volume with the same name:

```bash
docker volume create music-to-midi_job_data
docker run --rm \
  -v music-to-midi_job_data:/target \
  -v "$PWD:/backup:ro" \
  alpine:3.22 \
  tar -xzf /backup/<archive-name>.tar.gz -C /target
```

## Upgrade and rollback

The image digests in each release package are the effective runtime version:

1. Review the new Release notes for platform, licensing, and data-compatibility changes.
2. The old release directory, `.env`, and `IMAGE_DIGESTS.txt` remain available, together with a backup of important job data.
3. The new Docker release package is downloaded and verified.
4. The old `.env` is copied into the new directory and compared with the new template.
5. `setup` runs in the new directory.
6. Check each enabled profile, per-track conversion, editing, preview, and download with representative audio.

When a new version does not pass acceptance and its Release notes describe no irreversible migration, `start` from the retained old directory recreates containers with the old digests and reuses the named volumes. A release that introduces an irreversible migration defines its own restore path in its Release notes.

## Local, remote, and public access

The default path is:

```text
Browser -> 127.0.0.1:7860 -> gateway -> internal backend:8765 -> NVIDIA GPU
                                             |-> model_data
                                             +-> job_data
```

Local self-hosting does not use a public domain, ACME email, application password, or SSH credential. An SSH password or key belongs to the user's own remote-host login and is never read or stored by the application.

A remote server can retain the loopback binding and expose it through the user's SSH connection:

```bash
ssh -L 7860:127.0.0.1:7860 -p <ssh-port> <user>@<server>
```

The local browser then continues to use `http://127.0.0.1:7860`.

Internet-facing deployment uses `compose.production.yaml`, `.env.production.example`, and `scripts/setup_docker_production.ps1` from the repository. That variant adds public DNS, ports 80/443, an ACME contact address, Caddy Argon2id Basic Auth, HTTPS termination, and isolation from the inference network. It is a single-owner service rather than a multi-tenant account system.

## Troubleshooting

### Docker or Compose version

```bash
docker version
docker compose version
```

`setup` reports the detected and minimum versions.

### Docker cannot see the NVIDIA GPU

Both host `nvidia-smi` and the NVIDIA CUDA container test above need to return normally. Windows Docker Desktop additionally uses its WSL2 backend and GPU support. Linux uses NVIDIA Container Toolkit.

### `CUDAExecutionProvider` is unavailable

This means the fixed ONNX Runtime CUDA provider is not available in the actual backend container. Typical causes are the host driver, Docker GPU runtime, or GPU exposure settings. `setup` stops before model download and prints the provider list.

### Port conflict

`MUSIC_TO_MIDI_PORT` accepts values from `1` to `65535`; the next `start` operation applies a changed value.

### MuScriptor returns 401 or 403

The Hugging Face account needs accepted terms for each selected Small, Medium, or Large model and a read-capable `HF_TOKEN` in the current shell. The token is used by `models` only.

### Model initialization was interrupted

A repeated `models` operation revalidates the selected profiles. Complete assets with matching identities are reused; a missing or incomplete asset remains a visible error with its cause.

### HTTP 507 or low disk space

`MIN_FREE_BYTES` defines the job-volume reserve. `status`, `verify`, and backend logs show the current capacity state. Completed outputs can be moved out of the job volume, and retention settings can be adjusted in `.env`.

### The page starts but is not ready

```bash
bash scripts/docker_selfhost.sh verify
docker compose --env-file .env -f compose.yaml logs --tail 200 backend gateway
```

Windows uses the PowerShell `verify` operation; the Docker Compose log command is the same.

## Stop, reset, and removal

A routine stop preserves all data:

```bash
bash scripts/docker_selfhost.sh stop
```

A full reset removes all four named volumes, including models, jobs, and gateway state:

```bash
docker compose --env-file .env -f compose.yaml down -v
```

After containers and volumes are removed, the release directory is an ordinary folder. Images remain in Docker's local cache for reuse. Docker Desktop or `docker image rm` can remove the image digests listed in `IMAGE_DIGESTS.txt` once no container depends on them.

## Release validation scope

The container workflow covers:

1. Docker, licensing, Compose, and management-script contracts.
2. `linux/amd64` backend and gateway builds, including non-root and read-only runtime checks.
3. OCI SBOM, provenance, GitHub registry attestations, and versioned tags.
4. Anonymous pulls without a GHCR login state.
5. All 15 backend profiles, offline resource entry points, and gateway same-origin runtime configuration.
6. Digest-pinned Compose, Bash/PowerShell syntax, ZIP/tar.gz packages, and SHA-256 checksums.

These checks cover source, image contents, public distribution, and the model-free startup contract. GitHub-hosted runners do not provide an NVIDIA GPU, so automation does not represent real-audio inference across every physical GPU and model. First-run `setup` validates the host GPU and providers; final quality, VRAM use, and long-audio behavior remain specific to the enabled profiles, hardware, and input audio.
