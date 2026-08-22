"""Validated persistent configuration for the standalone frontend host."""

from __future__ import annotations

import ipaddress
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit

from src.utils.runtime_paths import get_executable_dir, get_project_root

_CONFIG_KEYS = {
    "host",
    "port",
    "public_host",
    "backend_url",
    "open_app_window",
    "edge_path",
}


def _validate_host(value: str, *, field: str, allow_wildcard: bool) -> str:
    host = str(value).strip()
    if not host:
        raise ValueError(f"{field} must not be empty")
    if host == "localhost":
        return host
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError(f"{field} must be an IPv4/IPv6 address or localhost") from exc
    if not allow_wildcard and address.is_unspecified:
        raise ValueError(f"{field} cannot be a wildcard address")
    return host


def _validate_port(value: int) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError(f"frontend port must be between 1 and 65535: {port}")
    return port


def normalize_backend_url(value: str) -> str:
    url = str(value).strip().rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"backend_url must be an absolute HTTP(S) URL: {value!r}")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("backend_url must not contain a path, query or fragment")
    if parsed.username or parsed.password:
        raise ValueError("backend_url must not contain credentials")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"backend_url contains an invalid port: {value!r}") from exc
    return url


def default_frontend_config_path() -> Path:
    if getattr(sys, "frozen", False):
        return get_executable_dir() / "MusicToMidiFrontend.json"
    return get_project_root() / "config" / "web-frontend.json"


def _url_host(host: str) -> str:
    try:
        return f"[{host}]" if ipaddress.ip_address(host).version == 6 else host
    except ValueError:
        return host


@dataclass(frozen=True)
class FrontendServerConfig:
    host: str = "127.0.0.1"
    port: int = 5173
    public_host: str = "127.0.0.1"
    backend_url: str = "http://127.0.0.1:8765"
    open_app_window: bool = True
    edge_path: str = ""

    def validated(self) -> "FrontendServerConfig":
        host = _validate_host(self.host, field="frontend host", allow_wildcard=True)
        public_host = _validate_host(
            self.public_host,
            field="frontend public_host",
            allow_wildcard=False,
        )
        if host not in {"0.0.0.0", "::"} and public_host in {"0.0.0.0", "::"}:
            raise ValueError("frontend public_host cannot be a wildcard address")
        edge_path = str(self.edge_path).strip()
        if edge_path and not Path(edge_path).expanduser().is_file():
            raise FileNotFoundError(
                f"configured Microsoft Edge executable does not exist: {edge_path}"
            )
        if not isinstance(self.open_app_window, bool):
            raise ValueError("frontend open_app_window must be true or false")
        return replace(
            self,
            host=host,
            port=_validate_port(self.port),
            public_host=public_host,
            backend_url=normalize_backend_url(self.backend_url),
            edge_path=edge_path,
        )

    @property
    def frontend_url(self) -> str:
        validated = self.validated()
        return f"http://{_url_host(validated.public_host)}:{validated.port}"

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self.validated())


def load_frontend_config(path: str | Path) -> FrontendServerConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        config = FrontendServerConfig().validated()
        write_frontend_config(config_path, config)
        return config
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read frontend config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"frontend config must be a JSON object: {config_path}")
    unknown = sorted(set(payload) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"frontend config contains unknown keys: {', '.join(unknown)}")
    return FrontendServerConfig(
        host=payload.get("host", FrontendServerConfig.host),
        port=payload.get("port", FrontendServerConfig.port),
        public_host=payload.get("public_host", FrontendServerConfig.public_host),
        backend_url=payload.get("backend_url", FrontendServerConfig.backend_url),
        open_app_window=payload.get("open_app_window", FrontendServerConfig.open_app_window),
        edge_path=payload.get("edge_path", FrontendServerConfig.edge_path),
    ).validated()


def write_frontend_config(path: str | Path, config: FrontendServerConfig) -> Path:
    config_path = Path(path).expanduser().resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        config_path.write_text(
            json.dumps(config.to_json_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"failed to write frontend config {config_path}: {exc}") from exc
    return config_path


__all__ = [
    "FrontendServerConfig",
    "default_frontend_config_path",
    "load_frontend_config",
    "normalize_backend_url",
    "write_frontend_config",
]
