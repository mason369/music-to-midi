"""Validated, persistent network configuration for the inference server."""

from __future__ import annotations

import ipaddress
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit

from src.utils.runtime_paths import get_executable_dir, get_project_root

_LOG_LEVELS = {"critical", "error", "warning", "info", "debug", "trace"}
_CONFIG_KEYS = {
    "host",
    "port",
    "data_dir",
    "allowed_origins",
    "log_level",
    "max_upload_bytes",
    "max_queued_jobs",
    "min_free_bytes",
    "retention_days",
    "retention_max_jobs",
    "retention_max_bytes",
}


def _validate_bind_host(value: str) -> str:
    host = str(value).strip()
    if not host:
        raise ValueError("backend host must not be empty")
    if host == "localhost":
        return host
    try:
        ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("backend host must be an IPv4/IPv6 address or localhost") from exc
    return host


def _validate_port(value: int) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError(f"backend port must be between 1 and 65535: {port}")
    return port


def _validate_retention(value: int, *, field: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"backend {field} must be an integer")
    normalized = int(value)
    if not 0 <= normalized <= maximum:
        raise ValueError(f"backend {field} must be between 0 and {maximum}: {value}")
    return normalized


def _validate_positive(value: int, *, field: str, maximum: int) -> int:
    normalized = _validate_retention(value, field=field, maximum=maximum)
    if normalized == 0:
        raise ValueError(f"backend {field} must be positive")
    return normalized


def _normalize_origin(value: str) -> str:
    origin = str(value).strip().rstrip("/")
    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid allowed origin: {value!r}")
    if parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError(f"allowed origin must contain only scheme, host and port: {value!r}")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid allowed origin port: {value!r}") from exc
    return origin


def default_backend_config_path() -> Path:
    if getattr(sys, "frozen", False):
        return get_executable_dir() / "MusicToMidiBackend.json"
    return get_project_root() / "config" / "web-backend.json"


@dataclass(frozen=True)
class BackendServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    data_dir: str = ""
    allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    )
    log_level: str = "info"
    max_upload_bytes: int = 4 * 1024 * 1024 * 1024
    max_queued_jobs: int = 0
    min_free_bytes: int = 0
    retention_days: int = 30
    retention_max_jobs: int = 200
    retention_max_bytes: int = 0

    def validated(self) -> "BackendServerConfig":
        host = _validate_bind_host(self.host)
        port = _validate_port(self.port)
        log_level = str(self.log_level).strip().lower()
        if log_level not in _LOG_LEVELS:
            raise ValueError(
                f"backend log_level must be one of {sorted(_LOG_LEVELS)}: {self.log_level!r}"
            )
        origins = tuple(dict.fromkeys(_normalize_origin(item) for item in self.allowed_origins))
        if not origins:
            raise ValueError("backend allowed_origins must contain at least one explicit origin")
        data_dir = str(self.data_dir).strip()
        return replace(
            self,
            host=host,
            port=port,
            data_dir=data_dir,
            allowed_origins=origins,
            log_level=log_level,
            max_upload_bytes=_validate_positive(
                self.max_upload_bytes,
                field="max_upload_bytes",
                maximum=2**63 - 1,
            ),
            max_queued_jobs=_validate_retention(
                self.max_queued_jobs,
                field="max_queued_jobs",
                maximum=100000,
            ),
            min_free_bytes=_validate_retention(
                self.min_free_bytes,
                field="min_free_bytes",
                maximum=2**63 - 1,
            ),
            retention_days=_validate_retention(
                self.retention_days,
                field="retention_days",
                maximum=3650,
            ),
            retention_max_jobs=_validate_retention(
                self.retention_max_jobs,
                field="retention_max_jobs",
                maximum=100000,
            ),
            retention_max_bytes=_validate_retention(
                self.retention_max_bytes,
                field="retention_max_bytes",
                maximum=2**63 - 1,
            ),
        )

    def resolved_data_dir(self) -> Path:
        if self.data_dir:
            return Path(os.path.expandvars(self.data_dir)).expanduser().resolve()
        return (get_executable_dir() / "runtime" / "web-api").resolve()

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self.validated())
        payload["allowed_origins"] = list(payload["allowed_origins"])
        return payload


def load_backend_config(path: str | Path) -> BackendServerConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        config = BackendServerConfig().validated()
        write_backend_config(config_path, config)
        return config
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read backend config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"backend config must be a JSON object: {config_path}")
    unknown = sorted(set(payload) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"backend config contains unknown keys: {', '.join(unknown)}")
    origins = payload.get("allowed_origins", BackendServerConfig.allowed_origins)
    if not isinstance(origins, list) or not all(isinstance(item, str) for item in origins):
        raise ValueError("backend allowed_origins must be a JSON string array")
    return BackendServerConfig(
        host=payload.get("host", BackendServerConfig.host),
        port=payload.get("port", BackendServerConfig.port),
        data_dir=payload.get("data_dir", BackendServerConfig.data_dir),
        allowed_origins=tuple(origins),
        log_level=payload.get("log_level", BackendServerConfig.log_level),
        max_upload_bytes=payload.get(
            "max_upload_bytes",
            BackendServerConfig.max_upload_bytes,
        ),
        max_queued_jobs=payload.get(
            "max_queued_jobs",
            BackendServerConfig.max_queued_jobs,
        ),
        min_free_bytes=payload.get("min_free_bytes", BackendServerConfig.min_free_bytes),
        retention_days=payload.get("retention_days", BackendServerConfig.retention_days),
        retention_max_jobs=payload.get(
            "retention_max_jobs",
            BackendServerConfig.retention_max_jobs,
        ),
        retention_max_bytes=payload.get(
            "retention_max_bytes",
            BackendServerConfig.retention_max_bytes,
        ),
    ).validated()


def write_backend_config(path: str | Path, config: BackendServerConfig) -> Path:
    config_path = Path(path).expanduser().resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        config_path.write_text(
            json.dumps(config.to_json_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"failed to write backend config {config_path}: {exc}") from exc
    return config_path


__all__ = [
    "BackendServerConfig",
    "default_backend_config_path",
    "load_backend_config",
    "write_backend_config",
]
