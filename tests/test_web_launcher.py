from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

from src.web import __main__ as web_main
from src.web_frontend import __main__ as frontend_main
from src.web_frontend import server as frontend_server
from src.web_frontend.config import FrontendServerConfig
from src.web_frontend.server import FrontendServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _RouteProbe:
    def __init__(self, address: str) -> None:
        self.address = address

    def __enter__(self) -> "_RouteProbe":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def connect(self, target: tuple[str, int]) -> None:
        assert target == web_main._ROUTE_PROBE_ADDRESS

    def getsockname(self) -> tuple[str, int]:
        return self.address, 49152


def test_detect_lan_ipv4_uses_the_concrete_private_default_route() -> None:
    with patch.object(web_main.socket, "socket", return_value=_RouteProbe("192.168.8.16")):
        assert web_main.detect_lan_ipv4() == "192.168.8.16"


@pytest.mark.parametrize("address", ("0.0.0.0", "169.254.1.2", "203.0.113.8"))
def test_web_host_rejects_non_lan_addresses(address: str) -> None:
    with pytest.raises(ValueError):
        web_main._validate_host(address)


def test_unified_commands_share_the_detected_host_and_open_window_by_default() -> None:
    backend = web_main._backend_command("192.168.8.16", 8765, 5173, 30, 200)
    frontend = web_main._frontend_command(
        "192.168.8.16",
        8765,
        5173,
        no_window=False,
    )

    assert backend[:3] == [web_main.sys.executable, "-m", "src.web_api"]
    assert frontend[:3] == [web_main.sys.executable, "-m", "src.web_frontend"]
    assert backend[backend.index("--host") + 1] == "192.168.8.16"
    assert backend[backend.index("--cors-origin") + 1] == "http://192.168.8.16:5173"
    assert frontend[frontend.index("--public-host") + 1] == "192.168.8.16"
    assert frontend[frontend.index("--api-url") + 1] == "http://192.168.8.16:8765"
    assert "--no-window" not in frontend


def test_windows_children_use_the_real_python_process_with_venv_identity() -> None:
    venv_python = r"C:\project\venv\Scripts\python.exe"
    base_python = r"C:\Python311\python.exe"
    command = [venv_python, "-m", "src.web_frontend"]
    windows_os = SimpleNamespace(name="nt", environ=web_main.os.environ)
    with (
        patch.object(web_main, "os", windows_os),
        patch.object(web_main.sys, "executable", venv_python),
        patch.object(web_main.sys, "_base_executable", base_python),
        patch.object(web_main.subprocess, "Popen") as popen,
    ):
        web_main._start_process(command)

    assert popen.call_args.args == (command,)
    options = popen.call_args.kwargs
    assert options["executable"] == base_python
    assert options["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP
    assert options["env"]["__PYVENV_LAUNCHER__"] == venv_python


def test_unified_main_starts_backend_before_frontend_and_cleans_up() -> None:
    backend = Mock(spec=subprocess.Popen)
    frontend = Mock(spec=subprocess.Popen)
    with (
        patch.object(web_main, "require_source_runtime_identity") as runtime_gate,
        patch.object(web_main, "detect_lan_ipv4", return_value="192.168.8.16"),
        patch.object(web_main, "_start_process", side_effect=[backend, frontend]) as start,
        patch.object(web_main, "_wait_for_backend") as wait_backend,
        patch.object(web_main, "_wait_for_frontend") as wait_frontend,
        patch.object(web_main, "_monitor_processes", return_value=0) as monitor,
        patch.object(web_main, "_stop_process") as stop,
    ):
        assert web_main.main([]) == 0

    runtime_gate.assert_called_once_with()
    assert start.call_count == 2
    assert start.call_args_list[0].args[0][2] == "src.web_api"
    assert start.call_args_list[1].args[0][2] == "src.web_frontend"
    assert "--no-window" not in start.call_args_list[1].args[0]
    wait_backend.assert_called_once_with(backend, "192.168.8.16", 8765)
    wait_frontend.assert_called_once_with(frontend, "192.168.8.16", 5173, 8765)
    monitor.assert_called_once_with(backend, frontend)
    assert stop.call_args_list == [
        call(frontend, label="Web 前端"),
        call(backend, label="处理服务"),
    ]


def test_frontend_converts_windows_break_into_keyboard_interrupt() -> None:
    with (
        patch.object(frontend_main.sys, "platform", "win32"),
        patch.object(frontend_main.signal, "SIGBREAK", 21, create=True),
        patch.object(frontend_main.signal, "signal") as install,
    ):
        frontend_main._install_windows_break_handler()

    install.assert_called_once_with(21, frontend_main.signal.default_int_handler)


def test_frontend_main_treats_window_break_as_clean_exit(tmp_path: Path) -> None:
    config = FrontendServerConfig(
        host="127.0.0.1",
        port=_free_port(),
        public_host="127.0.0.1",
        backend_url="http://127.0.0.1:8765",
    )
    with (
        patch.object(frontend_main, "_install_windows_break_handler"),
        patch.object(frontend_main, "load_frontend_config", return_value=config),
        patch.object(frontend_main, "FrontendServer") as server_type,
    ):
        server_type.return_value.run_app_window.side_effect = KeyboardInterrupt()
        result = frontend_main.main(["--config", str(tmp_path / "frontend.json")])

    assert result == 0


def test_frontend_window_is_closed_when_the_host_is_interrupted(tmp_path: Path) -> None:
    edge = tmp_path / "msedge.exe"
    edge.write_bytes(b"edge")
    web_root = tmp_path / "web"
    web_root.mkdir()
    (web_root / "index.html").write_text("ok", encoding="utf-8")
    server = FrontendServer(
        FrontendServerConfig(
            host="127.0.0.1",
            port=_free_port(),
            public_host="127.0.0.1",
            backend_url="http://127.0.0.1:8765",
            edge_path=str(edge),
        ),
        web_root=web_root,
    )
    edge_process = Mock(spec=subprocess.Popen)
    edge_process.wait.side_effect = [
        subprocess.TimeoutExpired(cmd="msedge.exe", timeout=0.25),
        KeyboardInterrupt(),
        0,
    ]
    edge_process.pid = 1234
    edge_process.poll.return_value = None

    windows_os = SimpleNamespace(name="nt")
    with (
        patch.object(frontend_server, "os", windows_os),
        patch("src.web_frontend.server.subprocess.Popen", return_value=edge_process),
        patch(
            "src.web_frontend.server.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, "SUCCESS"),
        ) as taskkill,
        pytest.raises(KeyboardInterrupt),
    ):
        server.run_app_window()

    taskkill.assert_called_once()
    assert taskkill.call_args.args[0] == [
        "taskkill",
        "/PID",
        "1234",
        "/T",
        "/F",
    ]
    edge_process.terminate.assert_not_called()
    assert edge_process.wait.call_args_list[:2] == [
        call(timeout=0.25),
        call(timeout=0.25),
    ]
    assert edge_process.wait.call_args_list[-1] == call(timeout=5)
    assert server._thread is None
