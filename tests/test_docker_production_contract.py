from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _compose() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "compose.production.yaml").read_text(encoding="utf-8")
    )


def test_model_initializer_has_download_egress_without_runtime_network_access():
    compose = _compose()
    networks = compose["networks"]
    model_init = compose["services"]["model-init"]

    assert networks["inference"]["internal"] is True
    assert networks["model-download"].get("internal", False) is False
    assert model_init["networks"] == ["model-download"]
    assert "inference" not in model_init["networks"]
    assert "ports" not in model_init


def test_public_gateway_is_the_only_bridge_to_the_internal_backend():
    compose = _compose()
    backend = compose["services"]["backend"]
    gateway = compose["services"]["gateway"]

    assert backend["networks"] == ["inference"]
    assert "ports" not in backend
    assert gateway["networks"] == ["edge", "inference"]
    assert gateway["depends_on"]["backend"]["condition"] == "service_healthy"
    assert gateway["ports"] == ["80:8080", "443:8443", "443:8443/udp"]


def test_production_containers_keep_runtime_assets_out_of_the_image_context():
    ignored = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    backend = (REPO_ROOT / "docker" / "backend.Dockerfile").read_text(encoding="utf-8")
    compose = _compose()

    for required in ("models", "external", "MidiOutput", "*.ckpt", "*.onnx", "*.sf2"):
        assert required in ignored
    assert "COPY models" not in backend
    assert "COPY external" not in backend
    assert compose["services"]["backend"]["volumes"] == [
        "model_data:/models",
        "job_data:/data",
    ]
