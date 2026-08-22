#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/compose.yaml"
ENV_EXAMPLE="$ROOT_DIR/.env.selfhost.example"
ENV_FILE="$ROOT_DIR/.env"
ACTION="${1:-setup}"
COMPOSE=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE")

fail() {
    printf 'Docker 自托管操作失败：%s\n' "$*" >&2
    exit 2
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "缺少命令：$1"
}

env_value() {
    local name="$1"
    local value
    value="$(awk -F= -v key="$name" '
        $0 !~ /^[[:space:]]*#/ && $1 == key { sub(/^[^=]*=/, ""); gsub(/\r$/, ""); print }
    ' "$ENV_FILE")"
    [[ "$(printf '%s\n' "$value" | sed '/^$/d' | wc -l)" -le 1 ]] \
        || fail "$ENV_FILE 中 $name 重复"
    printf '%s' "$value"
}

ensure_environment() {
    [[ -f "$COMPOSE_FILE" ]] || fail "缺少 $COMPOSE_FILE"
    [[ -f "$ENV_EXAMPLE" ]] || fail "缺少 $ENV_EXAMPLE"
    if [[ ! -f "$ENV_FILE" ]]; then
        cp -- "$ENV_EXAMPLE" "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        printf '已从安全默认模板创建 %s。\n' "$ENV_FILE"
    fi

    local port gpu profiles
    port="$(env_value MUSIC_TO_MIDI_PORT)"
    gpu="$(env_value GPU_DEVICE_ID)"
    profiles="$(env_value MUSIC_TO_MIDI_ENABLED_PROFILES)"
    if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
        fail "MUSIC_TO_MIDI_PORT 必须是 1-65535"
    fi
    [[ "$gpu" =~ ^[0-9]+$ ]] || fail "GPU_DEVICE_ID 必须是非负整数"
    [[ -n "$profiles" ]] || fail "MUSIC_TO_MIDI_ENABLED_PROFILES 不能为空"
}

version_at_least() {
    local actual="$1" minimum="$2"
    [[ "$(printf '%s\n%s\n' "$minimum" "$actual" | sort -V | head -n 1)" == "$minimum" ]]
}

check_docker() {
    require_command docker
    require_command curl
    local engine_version compose_version
    engine_version="$(docker version --format '{{.Server.Version}}')" \
        || fail "Docker Engine 不可用"
    compose_version="$(docker compose version --short)" \
        || fail "Docker Compose v2 不可用"
    compose_version="${compose_version#v}"
    version_at_least "$engine_version" "24.0.0" \
        || fail "Docker Engine $engine_version 过旧，要求 24.0.0+"
    version_at_least "$compose_version" "2.17.0" \
        || fail "Docker Compose $compose_version 过旧，要求 2.17.0+"
    "${COMPOSE[@]}" config --quiet \
        || fail "Compose 或 .env 配置无效"
}

check_profile_selection() {
    local supported profiles profile seen needs_hf
    supported="$("${COMPOSE[@]}" --profile tools run --rm --no-deps \
        --entrypoint python model-init -m src.model_profiles list)" \
        || fail "无法从后端镜像读取模型配置清单"
    profiles="$(env_value MUSIC_TO_MIDI_ENABLED_PROFILES)"
    IFS=',' read -r -a selected <<< "$profiles"
    seen='|'
    needs_hf=0
    for profile in "${selected[@]}"; do
        profile="${profile//[[:space:]]/}"
        [[ -n "$profile" ]] || fail "模型配置列表包含空项"
        [[ "$seen" != *"|${profile}|"* ]] || fail "模型配置重复：$profile"
        seen="${seen}${profile}|"
        printf '%s\n' "$supported" | grep -Fqx -- "$profile" \
            || fail "后端镜像不支持模型配置：$profile"
        [[ "$profile" != muscriptor* ]] || needs_hf=1
    done
    if (( needs_hf == 1 )); then
        [[ -n "${HF_TOKEN:-}" ]] \
            || fail "MuScriptor 是 gated 模型；请只在当前终端 export HF_TOKEN 后重试"
    fi
}

check_gpu_runtime() {
    local probe
    probe='import torch; import onnxruntime as ort; assert torch.version.cuda == "12.8", torch.version.cuda; assert torch.cuda.is_available(); assert torch.cuda.device_count() > 0; assert "CUDAExecutionProvider" in ort.get_available_providers(); print(torch.cuda.get_device_name(0)); print(ort.get_available_providers())'
    "${COMPOSE[@]}" --profile tools run --rm --no-deps \
        --entrypoint python model-init -c "$probe" \
        || fail "实际后端镜像未通过 NVIDIA CUDA 12.8 与 ONNX Runtime CUDA provider 验证"
}

prepare_models() {
    check_profile_selection
    printf '正在显式下载并严格校验所选模型；常驻服务不会下载模型。\n'
    "${COMPOSE[@]}" --profile tools run --rm model-init \
        || fail "模型初始化失败"
}

wait_ready() {
    local port deadline response last_error
    port="$(env_value MUSIC_TO_MIDI_PORT)"
    deadline=$((SECONDS + 600))
    last_error="尚未收到响应"
    while (( SECONDS < deadline )); do
        if response="$(curl --fail --silent --show-error \
            --max-time 20 "http://127.0.0.1:${port}/api/v1/ready" 2>&1)"; then
            printf '%s' "$response" | grep -Eq '"status"[[:space:]]*:[[:space:]]*"ready"' \
                || fail "readiness 返回 200，但内容不是 ready：$response"
            return 0
        fi
        last_error="$response"
        sleep 5
    done
    "${COMPOSE[@]}" ps || true
    "${COMPOSE[@]}" logs --tail 200 backend gateway || true
    fail "10 分钟内未通过 readiness：$last_error"
}

verify_stack() {
    local port runtime ready backend_id gateway_id internal profiles profile
    port="$(env_value MUSIC_TO_MIDI_PORT)"
    "${COMPOSE[@]}" exec -T backend python /app/docker/healthcheck.py \
        || fail "后端内部 readiness 失败"
    runtime="$(curl --fail --silent --show-error --max-time 20 \
        "http://127.0.0.1:${port}/runtime-config.json")" \
        || fail "前端运行时配置不可访问"
    printf '%s' "$runtime" | grep -Eq '"expected_api_version"[[:space:]]*:[[:space:]]*"2\.0"' \
        || fail "前端运行时 API 版本不是 2.0：$runtime"
    ready="$(curl --fail --silent --show-error --max-time 20 \
        "http://127.0.0.1:${port}/api/v1/ready")" \
        || fail "外部 readiness 不可访问"
    printf '%s' "$ready" | grep -Eq '"status"[[:space:]]*:[[:space:]]*"ready"' \
        || fail "外部 readiness 内容无效：$ready"
    profiles="$(env_value MUSIC_TO_MIDI_ENABLED_PROFILES)"
    IFS=',' read -r -a selected <<< "$profiles"
    for profile in "${selected[@]}"; do
        profile="${profile//[[:space:]]/}"
        printf '%s' "$ready" | grep -Fq -- "\"$profile\"" \
            || fail "readiness 未返回所选模型配置：$profile"
    done

    backend_id="$("${COMPOSE[@]}" ps -q backend)"
    gateway_id="$("${COMPOSE[@]}" ps -q gateway)"
    [[ -n "$backend_id" && -n "$gateway_id" ]] || fail "后端或网关容器未运行"
    [[ "$(docker inspect --format '{{.HostConfig.ReadonlyRootfs}}' "$backend_id")" == "true" ]] \
        || fail "后端根文件系统不是只读"
    [[ "$(docker inspect --format '{{.Config.User}}' "$backend_id")" == "10001:10001" ]] \
        || fail "后端不是固定非 root 用户"
    [[ "$(docker inspect --format '{{.HostConfig.ReadonlyRootfs}}' "$gateway_id")" == "true" ]] \
        || fail "网关根文件系统不是只读"
    [[ "$(docker inspect --format '{{.Config.User}}' "$gateway_id")" == "10001:10001" ]] \
        || fail "网关不是固定非 root 用户"
    internal="$(docker network inspect music-to-midi_inference --format '{{.Internal}}')"
    [[ "$internal" == "true" ]] || fail "推理网络不是 internal 网络"

    printf 'Docker 自托管验收通过：http://127.0.0.1:%s\n' "$port"
    "${COMPOSE[@]}" images
}

case "$ACTION" in
    setup)
        ensure_environment
        check_docker
        "${COMPOSE[@]}" pull
        check_gpu_runtime
        prepare_models
        "${COMPOSE[@]}" up -d --remove-orphans
        wait_ready
        verify_stack
        ;;
    models)
        ensure_environment
        check_docker
        "${COMPOSE[@]}" pull backend model-init
        check_gpu_runtime
        prepare_models
        ;;
    start)
        ensure_environment
        check_docker
        "${COMPOSE[@]}" up -d --remove-orphans
        wait_ready
        verify_stack
        ;;
    verify)
        ensure_environment
        check_docker
        verify_stack
        ;;
    status)
        ensure_environment
        check_docker
        "${COMPOSE[@]}" ps
        "${COMPOSE[@]}" images
        ;;
    stop)
        ensure_environment
        check_docker
        "${COMPOSE[@]}" down
        printf '容器已停止；模型、作业和 Caddy 命名卷均已保留。\n'
        ;;
    *)
        fail "未知操作 '$ACTION'；可用操作：setup、models、start、verify、status、stop"
        ;;
esac
