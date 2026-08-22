"""Start the Web backend, frontend host, and app window with one command."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.utils.runtime_paths import get_project_root
from src.utils.source_runtime import require_source_runtime_identity
from src.web_contract import API_VERSION

DEFAULT_BACKEND_PORT = 8765
DEFAULT_FRONTEND_PORT = 5173
DEFAULT_RETENTION_DAYS = 30
DEFAULT_RETENTION_MAX_JOBS = 200
DEFAULT_READY_TIMEOUT_SECONDS = 60.0
_ROUTE_PROBE_ADDRESS = ("192.0.2.1", 9)
_TRUSTED_LAN_NETWORKS = tuple(
    ipaddress.ip_network(network) for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("端口范围为 1 到 65535")
    return port


def _validate_host(value: str) -> str:
    try:
        address = ipaddress.IPv4Address(str(value).strip())
    except ipaddress.AddressValueError as exc:
        raise ValueError(f"Web 监听地址必须是 IPv4：{value!r}") from exc
    if address.is_unspecified or address.is_multicast or address.is_link_local:
        raise ValueError(f"Web 监听地址不能是通配、组播或链路本地地址：{address}")
    if address.is_loopback or any(address in network for network in _TRUSTED_LAN_NETWORKS):
        return str(address)
    raise ValueError(
        f"Web 入口只允许回环或受信任局域网 IPv4：{address}；"
        "当前版本不包含互联网部署所需的认证、授权和 TLS"
    )


def detect_lan_ipv4() -> str:
    """Return the concrete IPv4 selected by the operating system's default route."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            # UDP connect performs route selection without sending application data.
            probe.connect(_ROUTE_PROBE_ADDRESS)
            candidate = str(probe.getsockname()[0])
    except OSError as exc:
        raise RuntimeError(
            "无法从默认路由识别局域网 IPv4；可使用 --host 指定服务电脑的实际 IPv4"
        ) from exc
    try:
        host = _validate_host(candidate)
    except ValueError as exc:
        raise RuntimeError(
            f"默认路由选择的地址不可用于受信任局域网部署：{candidate}；"
            "可使用 --host 指定服务电脑的实际 IPv4"
        ) from exc
    if ipaddress.IPv4Address(host).is_loopback:
        raise RuntimeError("默认路由只返回了本机回环地址；可使用 --host 指定服务电脑的实际 IPv4")
    return host


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="启动 Music to MIDI Web 前端、处理服务和浏览器窗口"
    )
    parser.add_argument(
        "--host",
        type=_validate_host,
        default=None,
        help="监听 IPv4；默认自动识别主要局域网 IPv4",
    )
    parser.add_argument("--backend-port", type=_port, default=DEFAULT_BACKEND_PORT)
    parser.add_argument("--frontend-port", type=_port, default=DEFAULT_FRONTEND_PORT)
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="只提供 Web 地址，不自动打开 Edge 应用窗口",
    )
    parser.add_argument("--retention-days", type=int, default=DEFAULT_RETENTION_DAYS)
    parser.add_argument(
        "--retention-max-jobs",
        type=int,
        default=DEFAULT_RETENTION_MAX_JOBS,
    )
    return parser


def _backend_command(
    host: str,
    backend_port: int,
    frontend_port: int,
    retention_days: int,
    retention_max_jobs: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.web_api",
        "--host",
        host,
        "--port",
        str(backend_port),
        "--cors-origin",
        f"http://{host}:{frontend_port}",
        "--retention-days",
        str(retention_days),
        "--retention-max-jobs",
        str(retention_max_jobs),
    ]


def _frontend_command(
    host: str,
    backend_port: int,
    frontend_port: int,
    *,
    no_window: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.web_frontend",
        "--host",
        host,
        "--port",
        str(frontend_port),
        "--public-host",
        host,
        "--api-url",
        f"http://{host}:{backend_port}",
    ]
    if no_window:
        command.append("--no-window")
    return command


def _start_process(command: list[str]) -> subprocess.Popen[bytes]:
    options: dict[str, object] = {
        "cwd": Path(get_project_root()),
        "env": os.environ.copy(),
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    return subprocess.Popen(command, **options)


def _wait_for_json_service(
    process: subprocess.Popen[bytes],
    url: str,
    *,
    label: str,
    validate: Callable[[dict[str, object]], bool],
    timeout_seconds: float = DEFAULT_READY_TIMEOUT_SECONDS,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "尚未响应"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"{label}在就绪前退出，退出码：{return_code}")
        try:
            request = Request(url, headers={"Accept": "application/json"})
            with urlopen(request, timeout=2) as response:
                if response.status != 200:
                    raise RuntimeError(f"{label}就绪检查返回 HTTP {response.status}")
                payload = json.load(response)
            if not isinstance(payload, dict) or not validate(payload):
                raise RuntimeError(f"{label}返回了无法识别的就绪状态：{payload!r}")
            return payload
        except HTTPError as exc:
            raise RuntimeError(f"{label}就绪检查返回 HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
            time.sleep(0.25)
    raise RuntimeError(f"{label}在 {timeout_seconds:g} 秒内未就绪：{last_error}")


def _wait_for_backend(process: subprocess.Popen[bytes], host: str, port: int) -> dict[str, object]:
    return _wait_for_json_service(
        process,
        f"http://{host}:{port}/api/v1/health",
        label="处理服务",
        validate=lambda payload: payload.get("status") == "ok"
        and payload.get("api_version") == API_VERSION,
    )


def _wait_for_frontend(
    process: subprocess.Popen[bytes], host: str, frontend_port: int, backend_port: int
) -> dict[str, object]:
    frontend_url = f"http://{host}:{frontend_port}"
    backend_url = f"http://{host}:{backend_port}"
    return _wait_for_json_service(
        process,
        f"{frontend_url}/runtime-config.json",
        label="Web 前端",
        validate=lambda payload: payload.get("frontend_url") == frontend_url
        and payload.get("backend_url") == backend_url
        and payload.get("expected_api_version") == API_VERSION,
    )


def _monitor_processes(backend: subprocess.Popen[bytes], frontend: subprocess.Popen[bytes]) -> int:
    while True:
        backend_code = backend.poll()
        if backend_code is not None:
            raise RuntimeError(f"处理服务意外退出，退出码：{backend_code}")
        frontend_code = frontend.poll()
        if frontend_code is not None:
            if frontend_code != 0:
                raise RuntimeError(f"Web 前端意外退出，退出码：{frontend_code}")
            return 0
        time.sleep(0.25)


def _stop_process(process: subprocess.Popen[bytes] | None, *, label: str) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=8)
        return
    except (OSError, ProcessLookupError):
        return
    except subprocess.TimeoutExpired:
        print(f"{label}没有按时停止，正在结束该进程。", file=sys.stderr, flush=True)
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def main(argv: list[str] | None = None) -> int:
    require_source_runtime_identity()
    args = _parser().parse_args(argv)
    host = args.host or detect_lan_ipv4()
    frontend_url = f"http://{host}:{args.frontend_port}"
    backend_url = f"http://{host}:{args.backend_port}"
    backend: subprocess.Popen[bytes] | None = None
    frontend: subprocess.Popen[bytes] | None = None
    print(f"局域网 IPv4：{host}", flush=True)
    print(f"正在启动处理服务：{backend_url}", flush=True)
    try:
        backend = _start_process(
            _backend_command(
                host,
                args.backend_port,
                args.frontend_port,
                args.retention_days,
                args.retention_max_jobs,
            )
        )
        _wait_for_backend(backend, host, args.backend_port)
        print("处理服务已就绪，正在启动 Web 前端。", flush=True)
        frontend = _start_process(
            _frontend_command(
                host,
                args.backend_port,
                args.frontend_port,
                no_window=args.no_window,
            )
        )
        _wait_for_frontend(frontend, host, args.frontend_port, args.backend_port)
        print(f"Web 地址：{frontend_url}", flush=True)
        print("同一局域网的电脑可直接打开该地址；按 Ctrl+C 停止前后端。", flush=True)
        return _monitor_processes(backend, frontend)
    except KeyboardInterrupt:
        print("\n正在停止 Web 前后端。", flush=True)
        return 0
    finally:
        _stop_process(frontend, label="Web 前端")
        _stop_process(backend, label="处理服务")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Web 启动失败：{exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
