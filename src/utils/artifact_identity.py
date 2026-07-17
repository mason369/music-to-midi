"""Strict file identity validation for pinned model artifacts."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for *path* without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file_identity(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> Path:
    """Require an exact file size and SHA-256 identity, or raise explicitly."""

    candidate = Path(path)
    if not candidate.is_file():
        raise RuntimeError(f"{label} is missing: {candidate}")

    actual_size = candidate.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"{label} size mismatch: expected {expected_size} bytes, "
            f"got {actual_size} bytes ({candidate})"
        )

    actual_sha256 = sha256_file(candidate)
    if not hmac.compare_digest(actual_sha256.lower(), expected_sha256.lower()):
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256} ({candidate})"
        )

    return candidate
