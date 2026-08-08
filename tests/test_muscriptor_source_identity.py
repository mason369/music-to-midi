from __future__ import annotations

import hashlib
from pathlib import Path

from src.utils import muscriptor_source_identity


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fake_official_identity(monkeypatch, package_root: Path) -> dict[str, bytes]:
    files = {
        "transcription_model.py": b"official v0.3.0 transcription model\n",
        "tokenizer/mt3.py": b"official v0.3.0 tokenizer\n",
        "utils/beats.py": b"official v0.3.0 beat grid\n",
    }
    monkeypatch.setattr(
        muscriptor_source_identity,
        "_SOURCE_IDENTITY_FILES",
        {relative: _sha256(payload) for relative, payload in files.items()},
    )
    monkeypatch.setattr(
        muscriptor_source_identity.importlib.metadata,
        "version",
        lambda _package: muscriptor_source_identity.MUSCRIPTOR_PACKAGE_VERSION,
    )
    for relative, payload in files.items():
        target = package_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return files


def test_runtime_identity_accepts_only_the_exact_official_source(tmp_path, monkeypatch):
    package_root = tmp_path / "muscriptor"
    _fake_official_identity(monkeypatch, package_root)

    assert muscriptor_source_identity.validate_muscriptor_runtime_identity(package_root) == ""


def test_runtime_identity_rejects_a_modified_official_file(tmp_path, monkeypatch):
    package_root = tmp_path / "muscriptor"
    _fake_official_identity(monkeypatch, package_root)
    target = package_root / "transcription_model.py"
    target.write_bytes(b"local mutation\n")

    error = muscriptor_source_identity.validate_muscriptor_runtime_identity(package_root)

    assert "source identity mismatch" in error
    assert str(target.resolve()) in error


def test_runtime_identity_rejects_a_missing_official_file(tmp_path, monkeypatch):
    package_root = tmp_path / "muscriptor"
    _fake_official_identity(monkeypatch, package_root)
    target = package_root / "utils/beats.py"
    target.unlink()

    error = muscriptor_source_identity.validate_muscriptor_runtime_identity(package_root)

    assert "identity file is missing" in error
    assert str(target.resolve()) in error


def test_runtime_identity_rejects_old_package_version(tmp_path, monkeypatch):
    package_root = tmp_path / "muscriptor"
    package_root.mkdir()
    monkeypatch.setattr(
        muscriptor_source_identity.importlib.metadata,
        "version",
        lambda _package: "0.3.0a1",
    )

    error = muscriptor_source_identity.validate_muscriptor_runtime_identity(package_root)

    assert "expected 0.3.0, got 0.3.0a1" in error
