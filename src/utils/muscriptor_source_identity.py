"""Strict source identity for the released MuScriptor v0.3.0 runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path

MUSCRIPTOR_PACKAGE_VERSION = "0.3.0"
MUSCRIPTOR_SOURCE_COMMIT = "d73147e75e5b9b0c0a79ebe154587db4fd603e0c"
MUSCRIPTOR_SOURCE_REQUIREMENT = (
    "muscriptor @ https://github.com/muscriptor/muscriptor/archive/"
    f"{MUSCRIPTOR_SOURCE_COMMIT}.zip"
)

# These hashes pin the released v0.3.0 inference, hard instrument mask,
# BeatGrid onset correction, and MuseScore-compatible MIDI writer together.
_SOURCE_IDENTITY_FILES = {
    "transcription_model.py": "d301cee911ad0e2fdf90bf4ad659a7e2a55e30d9028fa3604df3bb31b12fc88d",
    "tokenizer/mt3.py": "c37c7d9b101f6498bf8cf5964ab90df8b4df59f2e9960a192ed43b7fa9ae4a62",
    "models/lm.py": "8600f40d7b5242b37d0db5dbf16eddde6b493d2243ac4177bc86750bcea554a2",
    "accelerator.py": "1bc8001093c7fbc888191fc649c874c238d3966b3363d6458833157761ae40e5",
    "tokenizer/notes.py": "686ee5a6d336bf74a67390c72af376c00e42061e0d72354346e2e52b95363f33",
    "utils/beats.py": "58da8484da7c0372aa54690d5d6a801c9ae14c6897ac52aaf81acdee85af6bd9",
    "utils/midi.py": "21d16e15b073ec0522b31c6468a166dac4ae7c76f5ea318077945330ca572348",
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
