from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import pytest

from src.web_api.server_config import (
    BackendServerConfig,
    load_backend_config,
    write_backend_config,
)
from src.web_frontend.config import (
    FrontendServerConfig,
    load_frontend_config,
    write_frontend_config,
)
from src.web_frontend.server import FrontendServer, find_edge_executable

REPO_ROOT = Path(__file__).resolve().parents[1]


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_backend_config_round_trip_and_rejects_unknown_or_invalid_network_values(tmp_path):
    path = tmp_path / "backend.json"
    expected = BackendServerConfig(
        host="0.0.0.0",
        port=18765,
        data_dir=str(tmp_path / "jobs"),
        allowed_origins=("http://192.168.8.5:15173",),
        log_level="debug",
        retention_days=14,
        retention_max_jobs=50,
    )
    write_backend_config(path, expected)
    assert load_backend_config(path) == expected
    assert load_backend_config(path).resolved_data_dir() == (tmp_path / "jobs").resolve()

    path.write_text('{"host":"127.0.0.1","unknown":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_backend_config(path)
    with pytest.raises(ValueError, match="between 1 and 65535"):
        BackendServerConfig(port=0).validated()
    with pytest.raises(ValueError, match="only scheme, host and port"):
        BackendServerConfig(allowed_origins=("http://127.0.0.1:5173/path",)).validated()
    with pytest.raises(ValueError, match="retention_days"):
        BackendServerConfig(retention_days=-1).validated()
    with pytest.raises(ValueError, match="retention_max_jobs"):
        BackendServerConfig(retention_max_jobs=100001).validated()


def test_missing_backend_config_is_created_with_explicit_defaults(tmp_path):
    path = tmp_path / "nested" / "backend.json"
    config = load_backend_config(path)
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == config.to_json_dict()


def test_tracked_web_configs_use_portable_loopback_defaults():
    backend = load_backend_config(REPO_ROOT / "config" / "web-backend.json")
    frontend = load_frontend_config(REPO_ROOT / "config" / "web-frontend.json")

    assert backend.host == "127.0.0.1"
    assert backend.allowed_origins == (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    )
    assert frontend.host == "127.0.0.1"
    assert frontend.public_host == "127.0.0.1"
    assert frontend.backend_url == "http://127.0.0.1:8765"


def test_frontend_config_round_trip_requires_real_public_address_for_wildcard(tmp_path):
    path = tmp_path / "frontend.json"
    expected = FrontendServerConfig(
        host="0.0.0.0",
        port=15173,
        public_host="192.168.8.5",
        backend_url="http://192.168.8.73:18765",
        open_app_window=False,
    )
    write_frontend_config(path, expected)
    assert load_frontend_config(path) == expected
    assert load_frontend_config(path).frontend_url == "http://192.168.8.5:15173"

    with pytest.raises(ValueError, match="cannot be a wildcard"):
        FrontendServerConfig(host="0.0.0.0", public_host="0.0.0.0").validated()
    with pytest.raises(ValueError, match="must not contain a path"):
        FrontendServerConfig(backend_url="http://127.0.0.1:8765/api").validated()


def test_frontend_server_publishes_static_ui_and_exact_runtime_config(tmp_path):
    web_root = tmp_path / "web"
    web_root.mkdir()
    (web_root / "index.html").write_text(
        "<!doctype html><title>test frontend</title>", encoding="utf-8"
    )
    port = _free_tcp_port()
    config = FrontendServerConfig(
        host="127.0.0.1",
        port=port,
        public_host="127.0.0.1",
        backend_url="http://127.0.0.1:18765",
        open_app_window=False,
    )
    server = FrontendServer(config, web_root=web_root)
    server.start_background()
    try:
        with urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
            assert b"test frontend" in response.read()
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert (
                "connect-src 'self' http://127.0.0.1:18765"
                in response.headers["Content-Security-Policy"]
            )
            assert response.headers["Permissions-Policy"].startswith("camera=()")
        with urlopen(f"http://127.0.0.1:{port}/runtime-config.json", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload == {
            "managed": True,
            "frontend_url": f"http://127.0.0.1:{port}",
            "backend_url": "http://127.0.0.1:18765",
            "expected_api_version": "2.0",
        }
    finally:
        server.close()


def test_edge_discovery_is_explicit_and_never_opens_an_unrelated_browser(tmp_path):
    edge = tmp_path / "msedge.exe"
    edge.write_bytes(b"edge")
    assert find_edge_executable(str(edge)) == edge.resolve()
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(FileNotFoundError, match="Microsoft Edge is required"):
            find_edge_executable()


def test_web_specs_produce_three_separate_portable_roles():
    backend_spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")
    frontend_spec = (REPO_ROOT / "MusicToMidiFrontend.spec").read_text(encoding="utf-8")
    build_script = (REPO_ROOT / "build_web_executables.ps1").read_text(encoding="utf-8")

    assert "name='MusicToMidiBackend'" in backend_spec
    assert "console=True" in backend_spec
    assert 'name="MusicToMidiFrontend"' in frontend_spec
    assert '"torch"' in frontend_spec and "excludes=" in frontend_spec
    assert "MusicToMidi-App" in build_script
    assert "MusicToMidi-WebBackend" in build_script
    assert "MusicToMidi-WebFrontend" in build_script
    assert "MusicToMidiBackend.exe" in build_script
    assert "MusicToMidiFrontend.exe" in build_script
    assert 'Join-Path $ResolvedDistRoot "MusicToMidi-WebFrontend"' in build_script
    assert "WebFrontend\\MusicToMidiFrontend.exe" not in build_script

    release_workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert ".\\build_web_executables.ps1" in release_workflow
    assert "MusicToMidi-WebFrontend" in release_workflow
    assert "expected_api_version -ne '2.0'" in release_workflow
    assert "构建桌面、独立后端与独立 Web 前端 (Linux)" in release_workflow
    assert "MusicToMidiFrontend.spec" in release_workflow
    assert 'cp -al "$COMBINED_ROOT" "$APP_ROOT"' in release_workflow
    assert 'cp -al "$COMBINED_ROOT" "$BACKEND_ROOT"' in release_workflow
    assert 'test -x "$FRONTEND_ROOT/MusicToMidiFrontend"' in release_workflow
    assert "Packaged Linux App, WebBackend, and WebFrontend smoke passed" in release_workflow


@pytest.mark.skipif(os.name != "nt", reason="Windows hard-link packaging helper")
def test_portable_role_separator_creates_independent_hardlinked_trees(tmp_path):
    combined = tmp_path / "MusicToMidi"
    internal = combined / "_internal"
    internal.mkdir(parents=True)
    (combined / "MusicToMidi.exe").write_bytes(b"desktop")
    (combined / "MusicToMidiBackend.exe").write_bytes(b"backend")
    (internal / "runtime.bin").write_bytes(b"shared-runtime")

    helper = REPO_ROOT / "scripts" / "separate_windows_portable_roles.ps1"
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(helper),
            "-CombinedRoot",
            str(combined),
            "-DistRoot",
            str(tmp_path),
            "-Accelerator",
            "cuda",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    app_root = tmp_path / "MusicToMidi-App"
    backend_root = tmp_path / "MusicToMidi-WebBackend"
    assert not combined.exists()
    assert (app_root / "MusicToMidi.exe").is_file()
    assert not (app_root / "MusicToMidiBackend.exe").exists()
    assert (backend_root / "MusicToMidiBackend.exe").is_file()
    assert not (backend_root / "MusicToMidi.exe").exists()
    assert os.path.samefile(
        app_root / "_internal" / "runtime.bin",
        backend_root / "_internal" / "runtime.bin",
    )


def test_web_connection_state_uses_accelerator_neutral_runtime_contract():
    source = (REPO_ROOT / "web" / "app.js").read_text(encoding="utf-8")
    zh = json.loads((REPO_ROOT / "web" / "locales" / "zh_CN.json").read_text(encoding="utf-8"))
    en = json.loads((REPO_ROOT / "web" / "locales" / "en_US.json").read_text(encoding="utf-8"))

    assert "runtime.accelerator_devices" in source
    assert "runtime.accelerator_device" in source
    assert "runtime.accelerator_ready" in source
    assert 'accelerator === "xpu"' in source
    assert "runtime.cuda_devices" not in source
    assert "runtime.cuda_available" not in source
    assert zh["backend.accelerator_unavailable"] == "GPU 不可用"
    assert en["backend.accelerator_unavailable"] == "GPU unavailable"
    assert "Intel XPU" in zh["config.compute_help"]
    assert "Intel XPU" in en["config.compute_help"]


def test_frontend_app_window_uses_a_dedicated_persistent_edge_profile():
    source = (REPO_ROOT / "src" / "web_frontend" / "server.py").read_text(encoding="utf-8")

    assert '"web-frontend" / "edge-profile"' in source
    assert 'f"--user-data-dir={profile_dir}"' in source
    assert '"--no-first-run"' in source


def test_frozen_backend_executable_dispatches_before_gui_or_torch_preload(monkeypatch):
    import src.main as main_module
    import src.web_api.__main__ as backend_main

    monkeypatch.setattr(sys, "argv", ["MusicToMidiBackend.exe", "--port", "18765"])
    monkeypatch.setattr(sys, "executable", "C:/portable/MusicToMidiBackend.exe")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with (
        patch.object(backend_main, "main", return_value=0) as run_backend,
        patch.object(main_module, "_prepare_torch_runtime_before_pyqt") as preload,
        pytest.raises(SystemExit) as raised,
    ):
        main_module.main()

    assert raised.value.code == 0
    run_backend.assert_called_once_with(["--port", "18765"])
    preload.assert_not_called()


def test_frozen_xpu_backend_executable_dispatches_before_gui_or_torch_preload(monkeypatch):
    import src.main as main_module
    import src.web_api.__main__ as backend_main

    monkeypatch.setattr(sys, "argv", ["MusicToMidiBackendXpu.exe", "--port", "18765"])
    monkeypatch.setattr(sys, "executable", "C:/portable/MusicToMidiBackendXpu.exe")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with (
        patch.object(backend_main, "main", return_value=0) as run_backend,
        patch.object(main_module, "_prepare_torch_runtime_before_pyqt") as preload,
        pytest.raises(SystemExit) as raised,
    ):
        main_module.main()

    assert raised.value.code == 0
    run_backend.assert_called_once_with(["--port", "18765"])
    preload.assert_not_called()


def test_frozen_backend_dispatches_native_profile_probe_before_web_server(monkeypatch):
    import src.main as main_module
    import src.model_profile_runtime_probe as probe_module
    import src.web_api.__main__ as backend_main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "MusicToMidiBackend.exe",
            probe_module.MODEL_PROFILE_RUNTIME_PROBE_SWITCH,
            "vocal_split",
        ],
    )
    monkeypatch.setattr(sys, "executable", "C:/portable/MusicToMidiBackend.exe")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with (
        patch.object(probe_module, "run_model_profile_runtime_probe", return_value=0) as probe,
        patch.object(backend_main, "main", return_value=0) as run_backend,
        patch.object(main_module.os, "_exit") as hard_exit,
        patch.object(main_module, "_prepare_torch_runtime_before_pyqt") as preload,
    ):
        main_module.main()

    probe.assert_called_once_with("vocal_split")
    hard_exit.assert_called_once_with(0)
    run_backend.assert_not_called()
    preload.assert_not_called()


def test_web_api_runtime_preserves_backend_logging_and_dependency_warnings():
    environment = os.environ.copy()
    log_environment_names = (
        "TF_CPP_MIN_LOG_LEVEL",
        "TF_ENABLE_ONEDNN_OPTS",
        "ABSL_MIN_LOG_LEVEL",
    )
    for name in log_environment_names:
        environment.pop(name, None)
    script = """
import json
import logging
import os
import sys
sys.argv = ["src.main", "--web-api"]
import src.main  # noqa: F401
print(json.dumps({
    "root_level": logging.getLogger().level,
    "tf_log_level": os.environ.get("TF_CPP_MIN_LOG_LEVEL"),
    "absl_log_level": os.environ.get("ABSL_MIN_LOG_LEVEL"),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["root_level"] != logging.ERROR
    assert payload["tf_log_level"] is None
    assert payload["absl_log_level"] is None
