"""Deterministic identity for the patched YourMT3 inference source tree."""

from __future__ import annotations

import hashlib
from pathlib import Path

_IGNORED_NAMES = {"__pycache__", ".pytest_cache", ".coverage", ".DS_Store"}
_TEXT_SUFFIXES = {"", ".ini", ".md", ".py", ".sh", ".txt"}
PATCHED_YOURMT3_MANIFEST_SHA256 = "28fde351b4fe0f0519571fcd64e128df48a5076e7c46a9c1a604165798fe986e"
PATCHED_YOURMT3_MANIFEST_FILE_COUNT = 106


def _identity_bytes(path: Path) -> bytes:
    content = path.read_bytes()
    if path.suffix.lower() in _TEXT_SUFFIXES or path.name == ".coveragerc":
        return content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return content


def calculate_yourmt3_source_manifest(source_dir: Path | str) -> tuple[str, int]:
    """Return a stable SHA256 manifest and file count for a YourMT3 source tree."""
    root = Path(source_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"YourMT3 source directory is missing: {root}")

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() not in {".pyc", ".pyo"}
        and not any(part in _IGNORED_NAMES for part in path.relative_to(root).parts)
    )
    if not files:
        raise RuntimeError(f"YourMT3 source directory contains no identity files: {root}")

    digest = hashlib.sha256()
    for path in files:
        relative_path = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative_path)
        digest.update(b"\0")
        digest.update(_identity_bytes(path))
        digest.update(b"\0")
    return digest.hexdigest(), len(files)


def validate_patched_yourmt3_source(source_dir: Path | str) -> tuple[str, int]:
    """Require the exact cross-platform manifest of the project's patched tree."""

    manifest_sha256, file_count = calculate_yourmt3_source_manifest(source_dir)
    if file_count != PATCHED_YOURMT3_MANIFEST_FILE_COUNT:
        raise RuntimeError(
            "Patched YourMT3 source file-count mismatch: "
            f"expected {PATCHED_YOURMT3_MANIFEST_FILE_COUNT}, got {file_count} "
            f"({Path(source_dir).resolve()})"
        )
    if manifest_sha256 != PATCHED_YOURMT3_MANIFEST_SHA256:
        raise RuntimeError(
            "Patched YourMT3 source manifest mismatch: "
            f"expected {PATCHED_YOURMT3_MANIFEST_SHA256}, got {manifest_sha256} "
            f"({Path(source_dir).resolve()})"
        )
    return manifest_sha256, file_count
