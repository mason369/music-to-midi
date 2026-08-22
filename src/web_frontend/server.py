"""Static frontend server with an explicit runtime configuration endpoint."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlsplit

from src.utils.runtime_paths import get_resource_path, get_runtime_data_dir
from src.web_contract import API_VERSION
from src.web_frontend.config import FrontendServerConfig

LOGGER = logging.getLogger(__name__)


def find_edge_executable(configured_path: str = "") -> Path:
    if configured_path:
        path = Path(configured_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"configured Microsoft Edge executable does not exist: {path}")
        return path
    candidates = []
    for variable in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
        root = os.environ.get(variable)
        if root:
            candidates.append(Path(root) / "Microsoft" / "Edge" / "Application" / "msedge.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates) or "<no Windows program roots>"
    raise FileNotFoundError(
        "Microsoft Edge is required for the standalone app window; "
        f"searched: {searched}. Set edge_path explicitly after installing Edge."
    )


class _FrontendRequestHandler(SimpleHTTPRequestHandler):
    server_version = "MusicToMidiFrontend/2.0"

    def __init__(self, *args, runtime_config: dict[str, object], **kwargs):
        self._runtime_config = runtime_config
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        if urlsplit(self.path).path == "/runtime-config.json":
            body = json.dumps(self._runtime_config, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self) -> None:
        backend_origin = str(self._runtime_config["backend_url"])
        self.send_header(
            "Content-Security-Policy",
            "; ".join(
                [
                    "default-src 'self'",
                    "base-uri 'none'",
                    "object-src 'none'",
                    "frame-ancestors 'none'",
                    "form-action 'self'",
                    "script-src 'self'",
                    "style-src 'self' 'unsafe-inline'",
                    "img-src 'self' data:",
                    "font-src 'self'",
                    f"connect-src 'self' {backend_origin} http: https:",
                    f"media-src 'self' blob: {backend_origin} http: https:",
                    "worker-src 'none'",
                ]
            ),
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
        )
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        LOGGER.info("frontend http: " + format, *args)


class FrontendServer:
    def __init__(
        self,
        config: FrontendServerConfig,
        *,
        web_root: str | Path | None = None,
    ) -> None:
        self.config = config.validated()
        self.web_root = Path(web_root or get_resource_path("web")).resolve()
        if not (self.web_root / "index.html").is_file():
            raise FileNotFoundError(f"frontend index is missing: {self.web_root / 'index.html'}")
        runtime_config = {
            "managed": True,
            "frontend_url": self.config.frontend_url,
            "backend_url": self.config.backend_url,
            "expected_api_version": API_VERSION,
        }
        handler = partial(
            _FrontendRequestHandler,
            directory=str(self.web_root),
            runtime_config=runtime_config,
        )
        try:
            self._httpd = ThreadingHTTPServer(
                (self.config.host, self.config.port),
                handler,
            )
        except OSError as exc:
            raise RuntimeError(
                f"frontend could not bind {self.config.host}:{self.config.port}: {exc}"
            ) from exc
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def app_url(self) -> str:
        return f"{self.config.frontend_url}/?api={quote(self.config.backend_url, safe='')}"

    def start_background(self) -> None:
        if self._thread is not None:
            raise RuntimeError("frontend server is already running")
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="music-to-midi-frontend-http",
            daemon=True,
        )
        self._thread.start()

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def close(self) -> None:
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("frontend HTTP thread did not stop")
            self._thread = None
        self._httpd.server_close()

    def run_app_window(self) -> int:
        edge = find_edge_executable(self.config.edge_path)
        profile_dir = get_runtime_data_dir() / "web-frontend" / "edge-profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        process: subprocess.Popen[bytes] | None = None
        self.start_background()
        try:
            process = subprocess.Popen(
                [
                    str(edge),
                    f"--app={self.app_url}",
                    f"--user-data-dir={profile_dir}",
                    "--no-first-run",
                    "--start-maximized",
                ],
                close_fds=True,
            )
            return int(process.wait())
        finally:
            try:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            finally:
                self.close()


__all__ = ["FrontendServer", "find_edge_executable"]
