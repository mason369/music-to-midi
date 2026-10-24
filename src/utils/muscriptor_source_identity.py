"""Strict source identity for the released MuScriptor v0.4.0 runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path

MUSCRIPTOR_PACKAGE_VERSION = "0.4.0"
MUSCRIPTOR_SOURCE_COMMIT = "8a1ef88672cc23c26619adaff37ec48d82c0b906"
MUSCRIPTOR_SOURCE_REQUIREMENT = (
    "muscriptor @ https://github.com/muscriptor/muscriptor/archive/"
    f"{MUSCRIPTOR_SOURCE_COMMIT}.zip"
)

# These hashes pin the released v0.4.0 inference, hard instrument mask,
# BeatGrid onset correction, and tempo-aware MIDI writer together.
_SOURCE_IDENTITY_FILES = {
    "transcription_model.py": "1f12a7da7b12b876eb6104823a63629f12b60ef0570b52f5da93239c414e7f30",
    "tokenizer/mt3.py": "c37c7d9b101f6498bf8cf5964ab90df8b4df59f2e9960a192ed43b7fa9ae4a62",
    "models/lm.py": "8600f40d7b5242b37d0db5dbf16eddde6b493d2243ac4177bc86750bcea554a2",
    "accelerator.py": "1bc8001093c7fbc888191fc649c874c238d3966b3363d6458833157761ae40e5",
    "tokenizer/notes.py": "c8ddbaab0d6db7f7eb9886d2adf17052c6d643c95ac1db6421dfa9f0effa2a84",
    "utils/beats.py": "463cdfabc65a7fe2b2a0d2437f3bcfee9c0f6412a126375ffac3b3fb7de98c4f",
    "utils/midi.py": "42b9565d1f965a6102d9f3befdc778afca4554e958c9e047382552842e3aded2",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_package_root(package_root: str | Path | None = None) -> Path:
    if package_root is not None:
        root = Path(package_root).resolve()
    else:
        spec = importlib.util.find_spec("muscriptor")
        locations = [] if spec is None else list(spec.submodule_search_locations or [])
        if len(locations) != 1:
            raise RuntimeError(
                "Unable to locate one installed MuScriptor package directory: "
                f"locations={locations!r}"
            )
        root = Path(locations[0]).resolve()
    if not root.is_dir():
        raise RuntimeError(f"MuScriptor package directory does not exist: {root}")
    return root


def validate_muscriptor_runtime_identity(
    package_root: str | Path | None = None,
) -> str:
    """Return an explicit identity error, or an empty string when exact."""

    try:
        installed_version = importlib.metadata.version("muscriptor")
    except importlib.metadata.PackageNotFoundError:
        return "MuScriptor package metadata is not installed"
    if installed_version != MUSCRIPTOR_PACKAGE_VERSION:
        return (
            "MuScriptor package version mismatch: "
            f"expected {MUSCRIPTOR_PACKAGE_VERSION}, got {installed_version}"
        )
    try:
        root = _installed_package_root(package_root)
    except Exception as exc:
        return str(exc)

    for relative, expected_sha256 in _SOURCE_IDENTITY_FILES.items():
        path = root / relative
        if not path.is_file():
            return f"MuScriptor identity file is missing: {path}"
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            return (
                "MuScriptor source identity mismatch: "
                f"path={path}, expected={expected_sha256}, actual={actual_sha256}"
            )
    return ""


__all__ = [
    "MUSCRIPTOR_PACKAGE_VERSION",
    "MUSCRIPTOR_SOURCE_COMMIT",
    "MUSCRIPTOR_SOURCE_REQUIREMENT",
    "validate_muscriptor_runtime_identity",
]
