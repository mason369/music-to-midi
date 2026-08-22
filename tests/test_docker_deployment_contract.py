from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _yaml(relative: str) -> dict:
    payload = yaml.safe_load(_read(relative))
    assert isinstance(payload, dict)
    return payload


def test_quantization_controls_and_backend_are_baked_into_both_docker_images():
    backend = _read("docker/backend.Dockerfile")
    gateway = _read("docker/gateway.Dockerfile")
    engine = _read("src/web_api/engine.py")
    web_html = _read("web/index.html")
    web_js = _read("web/app.js")

    assert "COPY src ./src" in backend
    assert "from src.core.midi_quantization import quantize_midi_notes" in engine
    assert "_apply_requested_quantization" in engine
    assert "COPY web /srv/web" in gateway
    assert 'id="quantizeNotes"' in web_html
    assert 'id="quantizeGridSelect"' in web_html
    assert "quantize_notes:" in web_js
    assert "quantize_grid:" in web_js


def test_backend_image_is_pinned_headless_non_root_and_model_free():
    dockerfile = _read("docker/backend.Dockerfile")
    constraints = _read("docker/backend-constraints.txt")
    dockerignore = _read(".dockerignore")

    assert (
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:"
        "ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc" in dockerfile
    )
    assert "torch==2.7.0" in dockerfile
    assert "torchaudio==2.7.0" in dockerfile
    assert "torchvision==0.22.0" in dockerfile
    assert "--index-url https://download.pytorch.org/whl/cu128" in dockerfile
    assert "onnxruntime-gpu==1.23.2" in dockerfile
    assert "audio-separator==0.44.1 --no-deps" in dockerfile
    for identity in (
        "numpy==1.26.4",
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "torchvision==0.22.0",
        "onnxruntime-gpu==1.23.2",
        "audio-separator==0.44.1",
        "transkun==2.0.1",
        "muscriptor==0.3.0",
    ):
        assert identity in constraints
    assert dockerfile.count("--constraint /tmp/backend-constraints.txt") == 2
    assert "USER 10001:10001" in dockerfile
    assert 'ENTRYPOINT ["/app/docker/backend-entrypoint.sh"]' in dockerfile
    assert "python -m src.utils.source_runtime" in dockerfile
    assert "validate_patched_yourmt3_source" in dockerfile
    assert "validate_default_transkun_runtime" in dockerfile
    assert "THIRD_PARTY_SBOM.txt" in dockerfile
    assert "PyQt6" in dockerfile and "python-rtmidi" in dockerfile

    # Build stages must never prepare external checkpoints. The explicit
    # operator-only command lives in backend-entrypoint.sh instead.
    assert "src.model_profiles prepare" not in dockerfile
    assert not re.search(r"RUN\s+python\s+download_[A-Za-z0-9_]+\.py", dockerfile)
    for ignored in (
        ".env.*",
        "config/*.json",
        "models",
        "external",
        "*.ckpt",
        "*.safetensors",
        "*.onnx",
        "*.wav",
        "*.mid",
        "AGENTS.md",
    ):
        assert ignored in dockerignore


def test_entrypoint_fails_closed_and_model_download_is_operator_only():
    entrypoint = _read("docker/backend-entrypoint.sh")

    assert "set -euo pipefail" in entrypoint
    assert "model-init)" in entrypoint
    assert "verify-models)" in entrypoint
    assert "server)" in entrypoint
    assert "server-selfhost)" in entrypoint
    assert "src.model_profiles prepare" in entrypoint
    assert "src.model_profiles verify" in entrypoint
    assert "--require-ready" in entrypoint
    assert "MUSIC_TO_MIDI_REQUIRE_ENABLED_PROFILES must be 1" in entrypoint
    assert "MUSIC_TO_MIDI_PUBLIC_DEPLOYMENT must be 1" in entrypoint
    assert "MUSIC_TO_MIDI_EDGE_AUTH must be basic" in entrypoint
    assert "MUSIC_TO_MIDI_TLS_TERMINATED_AT_EDGE must be 1" in entrypoint
    assert "server-selfhost requires HF_HUB_OFFLINE=1" in entrypoint
    assert "server-selfhost requires TRANSFORMERS_OFFLINE=1" in entrypoint
    assert '--cors-origin "http://127.0.0.1:$SELF_HOST_PORT"' in entrypoint
    assert '--cors-origin "http://localhost:$SELF_HOST_PORT"' in entrypoint
    assert "unsupported command" in entrypoint
    assert 'exec "$@"' not in entrypoint


def test_compose_exposes_only_the_authenticated_gateway_and_reserves_one_gpu():
    compose = _yaml("compose.production.yaml")
    services = compose["services"]
    backend = services["backend"]
    model_init = services["model-init"]
    gateway = services["gateway"]

    assert "ports" not in backend
    assert backend["expose"] == ["8765"]
    assert gateway["ports"] == ["80:8080", "443:8443", "443:8443/udp"]
    assert compose["networks"]["inference"]["internal"] is True
    assert compose["networks"]["model-download"].get("internal", False) is False
    assert set(gateway["networks"]) == {"edge", "inference"}
    assert backend["networks"] == ["inference"]
    assert model_init["networks"] == ["model-download"]

    for service in (backend, model_init):
        device = service["deploy"]["resources"]["reservations"]["devices"][0]
        assert device["driver"] == "nvidia"
        assert device["capabilities"] == ["gpu"]
        assert device["device_ids"] == ["${GPU_DEVICE_ID:-0}"]
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert "no-new-privileges:true" in service["security_opt"]
        assert "model_data:/models" in service["volumes"]
        assert "job_data:/data" in service["volumes"]

    assert "HF_TOKEN" not in backend["environment"]
    assert model_init["environment"]["HF_TOKEN"] == "${HF_TOKEN:-}"
    assert model_init["profiles"] == ["tools"]
    assert model_init["command"] == ["model-init"]
    assert backend["environment"]["MUSIC_TO_MIDI_REQUIRE_ENABLED_PROFILES"] == "1"
    assert backend["environment"]["MUSIC_TO_MIDI_PUBLIC_DEPLOYMENT"] == "1"
    assert backend["environment"]["HF_HUB_OFFLINE"] == "1"
    assert backend["environment"]["TRANSFORMERS_OFFLINE"] == "1"
    assert backend["healthcheck"]["test"] == [
        "CMD",
        "python",
        "/app/docker/healthcheck.py",
    ]
    assert gateway["depends_on"]["backend"]["condition"] == "service_healthy"


def test_gateway_is_pinned_non_root_same_origin_tls_and_argon2id_protected():
    dockerfile = _read("docker/gateway.Dockerfile")
    caddyfile = _read("docker/Caddyfile")
    entrypoint = _read("docker/gateway-entrypoint.sh")

    assert (
        "caddy:2.11.4-alpine@sha256:"
        "5f5c8640aae01df9654968d946d8f1a56c497f1dd5c5cda4cf95ab7c14d58648" in dockerfile
    )
    assert "USER 10001:10001" in dockerfile
    assert "http_port 8080" in caddyfile
    assert "https_port 8443" in caddyfile
    assert "basic_auth argon2id" in caddyfile
    assert "request_body" in caddyfile and "max_size {$MAX_REQUEST_BODY_SIZE}" in caddyfile
    assert "reverse_proxy backend:8765" in caddyfile
    assert "root * /srv/web" in caddyfile
    assert "handle /runtime-config.json" in caddyfile
    assert '"backend_url":"{$PUBLIC_ORIGIN}"' in caddyfile
    assert "Strict-Transport-Security" in caddyfile
    assert "Content-Security-Policy" in caddyfile
    assert "frame-ancestors 'none'" in caddyfile
    assert "admin off" in caddyfile
    assert "BASIC_AUTH_HASH must be an Argon2id hash" in entrypoint
    assert "BASIC_AUTH_HASH still contains the example placeholder" in entrypoint
    assert "PUBLIC_ADDRESS must be a public DNS hostname with valid labels" in entrypoint
    assert "ACME_EMAIL must be a syntactically valid email address" in entrypoint
    assert "MAX_REQUEST_BODY_SIZE must be a positive byte size" in entrypoint
    assert "PUBLIC_ORIGIN must exactly equal https://PUBLIC_ADDRESS" in entrypoint
    assert "caddy validate" in entrypoint


def test_default_selfhost_compose_is_versioned_loopback_only_and_model_external():
    compose = _yaml("compose.yaml")
    project_version = tomllib.loads(_read("pyproject.toml"))["project"]["version"]
    services = compose["services"]
    backend = services["backend"]
    model_init = services["model-init"]
    gateway = services["gateway"]

    assert "build" not in backend
    assert "build" not in model_init
    assert "build" not in gateway
    assert backend["image"] == f"ghcr.io/mason369/music-to-midi-backend:v{project_version}"
    assert model_init["image"] == backend["image"]
    assert gateway["image"] == f"ghcr.io/mason369/music-to-midi-gateway:v{project_version}"
    assert all(":latest" not in service["image"] for service in services.values())
    assert backend["command"] == ["server-selfhost"]
    assert "ports" not in backend
    assert backend["networks"] == ["inference"]
    assert compose["networks"]["inference"]["internal"] is True
    assert model_init["networks"] == ["model-download"]
    assert compose["networks"]["model-download"].get("internal", False) is False
    assert gateway["ports"] == ["127.0.0.1:${MUSIC_TO_MIDI_PORT:-7860}:8080"]
    assert gateway["environment"]["MUSIC_TO_MIDI_GATEWAY_MODE"] == "selfhost"
    assert backend["environment"]["MUSIC_TO_MIDI_PUBLIC_DEPLOYMENT"] == "0"
    assert backend["environment"]["HF_HUB_OFFLINE"] == "1"
    assert backend["environment"]["TRANSFORMERS_OFFLINE"] == "1"
    assert "HF_HUB_OFFLINE" not in model_init["environment"]
    assert "TRANSFORMERS_OFFLINE" not in model_init["environment"]
    assert model_init["profiles"] == ["tools"]
    assert model_init["environment"]["HF_TOKEN"] == "${HF_TOKEN:-}"
    assert backend["volumes"] == ["model_data:/models", "job_data:/data"]
    assert model_init["volumes"] == backend["volumes"]
    assert gateway["depends_on"]["backend"]["condition"] == "service_healthy"

    for service in (backend, model_init):
        device = service["deploy"]["resources"]["reservations"]["devices"][0]
        assert device == {
            "driver": "nvidia",
            "device_ids": ["${GPU_DEVICE_ID:-0}"],
            "capabilities": ["gpu"],
        }
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert "no-new-privileges:true" in service["security_opt"]


def test_selfhost_gateway_is_http_only_same_origin_and_not_publicly_bound():
    dockerfile = _read("docker/gateway.Dockerfile")
    caddyfile = _read("docker/Caddyfile.selfhost")
    entrypoint = _read("docker/gateway-entrypoint.sh")

    assert "COPY docker/Caddyfile.selfhost /etc/caddy/Caddyfile.selfhost" in dockerfile
    assert "auto_https off" in caddyfile
    assert ":8080" in caddyfile
    assert '"frontend_url":"http://{http.request.hostport}"' in caddyfile
    assert '"backend_url":"http://{http.request.hostport}"' in caddyfile
    assert "reverse_proxy backend:8765" in caddyfile
    assert "Content-Security-Policy" in caddyfile
    assert "basic_auth" not in caddyfile
    assert "Strict-Transport-Security" not in caddyfile
    assert "MUSIC_TO_MIDI_GATEWAY_MODE" in entrypoint
    assert "selfhost)" in entrypoint
    assert "Caddyfile.selfhost" in entrypoint


def test_selfhost_management_scripts_validate_real_gpu_models_and_readiness():
    bash = _read("scripts/docker_selfhost.sh")
    powershell = _read("scripts/docker_selfhost.ps1")

    for script in (bash, powershell):
        assert "src.model_profiles" in script
        assert "model-init" in script
        assert "torch.cuda.is_available" in script
        assert "CUDAExecutionProvider" in script
        assert "/api/v1/ready" in script
        assert "runtime-config.json" in script
        assert "ReadonlyRootfs" in script
        assert "10001:10001" in script
        assert "down -v" not in script
        assert "ACME" not in script
        assert "PUBLIC_ADDRESS" not in script
        assert "SSH" not in script


def test_direct_midi_assets_are_prepared_explicitly_and_resolved_offline_at_runtime():
    profiles = _read("src/model_profiles.py")
    result_assets = _read("src/core/muscriptor_result_assets.py")
    downloader = _read("src/utils/muscriptor_soundfont_downloader.py")

    assert "download_muscriptor_soundfont" in profiles
    assert "get_fluidsynth_executable" in profiles
    assert "validate_muscriptor_soundfont" in result_assets
    assert "download_muscriptor_soundfont" not in result_assets
    assert "local_files_only=True" in downloader
    assert "run the explicit model initialization command" in downloader


def test_production_example_contains_no_token_or_plaintext_password():
    example = _read(".env.production.example")
    gitignore = _read(".gitignore")
    setup = _read("scripts/setup_docker_production.ps1")

    assert ".env.production" in gitignore
    assert "HF_TOKEN=" not in example
    assert "PASSWORD=" not in example.upper()
    assert "BASIC_AUTH_HASH='$argon2id$" in example
    assert "caddy hash-password --algorithm argon2id" in setup
    assert "不会写入明文" in setup
    assert "HF_TOKEN" not in re.search(r"\$Lines = @\((.*?)\n    \)", setup, flags=re.DOTALL).group(
        1
    )


def test_container_workflow_publishes_amd64_images_with_supply_chain_evidence():
    workflow = _read(".github/workflows/container.yml")

    for action in (
        "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
        "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
        "docker/setup-buildx-action@37fe631027851001ddb9b187196cc803df7f5f0e",
        "docker/login-action@dbcb813823bdd20940b903addbd779551569679f",
        "docker/metadata-action@dc802804100637a589fabce1cb79ff13a1411302",
        "docker/build-push-action@53b7df96c91f9c12dcc8a07bcb9ccacbed38856a",
        "actions/attest@1e69f48acb82d1966a394da916b4c1698aa569d6",
    ):
        assert action in workflow
    assert "platforms: linux/amd64" in workflow
    assert "ghcr.io" in workflow
    assert "provenance: mode=max" in workflow
    assert "sbom: true" in workflow
    assert "push-to-registry: true" in workflow
    assert "artifact-metadata: write" in workflow
    assert "镜像级烟雾验收" in workflow
    assert "tests/test_third_party_notice_packaging.py" in workflow
    assert "HF_TOKEN" not in workflow
    assert "验收公开镜像并发布自托管发行包" in workflow
    assert "docker pull --platform linux/amd64" in workflow
    assert "Public anonymous pull failed permanently" in workflow
    assert "IMAGE_DIGESTS.txt" in workflow
    assert "RELEASE-MANIFEST.json" in workflow
    assert "MusicToMidi-Docker-SHA256SUMS-${TAG}.txt" in workflow
    assert "gh release upload" in workflow
    assert "docker compose --env-file" in workflow
    assert "bash -n" in workflow
    assert "Language.Parser]::ParseFile" in workflow


def test_deployment_document_lists_every_supported_profile_and_real_acceptance_boundary():
    document = _read("docs/docker-deployment.md")
    from src.model_profiles import ALL_PROFILE_IDS

    for profile_id in ALL_PROFILE_IDS:
        assert f"`{profile_id}`" in document
    assert "`linux/amd64`" in document
    assert "当前不提供 CPU、AMD/ROCm、Intel XPU" in document
    assert "推理请求不会偷偷补模型" in document
    assert "经认证的单一所有者" in document
    assert "镜像成功构建”仍不等同于全部模型真实推理成功" in document
    assert "docker compose down -v" in document
