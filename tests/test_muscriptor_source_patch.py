from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.utils import muscriptor_source_patch


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fake_identity(monkeypatch, package_root: Path) -> dict[str, bytes]:
    base = {
        "transcription_model.py": b"base transcription model\n",
        "tokenizer/mt3.py": b"base tokenizer\n",
    }
    patched = {
        "transcription_model.py": b"patched overlap restart model\n",
        "tokenizer/mt3.py": b"patched overlap tokenizer\n",
    }
    unchanged = {
        "models/lm.py": b"current main inference mode\n",
        "accelerator.py": b"current main accelerator\n",
    }
    monkeypatch.setattr(
        muscriptor_source_patch,
        "_PATCH_FILES",
        {
            relative: {
                "base_sha256": _sha256(base[relative]),
                "patched_sha256": _sha256(patched[relative]),
                "url": f"https://example.invalid/{relative}",
            }
            for relative in base
        },
    )
    monkeypatch.setattr(
        muscriptor_source_patch,
        "_MAIN_IDENTITY_FILES",
        {relative: _sha256(payload) for relative, payload in unchanged.items()},
    )
    monkeypatch.setattr(
        muscriptor_source_patch.importlib.metadata,
        "version",
        lambda _package: muscriptor_source_patch.MUSCRIPTOR_PACKAGE_VERSION,
    )
    for relative, payload in {**base, **unchanged}.items():
        target = package_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return patched


def test_quality_patch_replaces_only_verified_base_files(tmp_path, monkeypatch):
    package_root = tmp_path / "muscriptor"
    patched = _fake_identity(monkeypatch, package_root)
    log: list[str] = []

    result = muscriptor_source_patch.apply_muscriptor_quality_patch(
        package_root=package_root,
        fetcher=lambda url: patched[url.removeprefix("https://example.invalid/")],
        printer=log.append,
    )

    assert result == package_root.resolve()
    assert muscriptor_source_patch.validate_muscriptor_runtime_identity(package_root) == ""
    assert (package_root / "transcription_model.py").read_bytes() == patched[
        "transcription_model.py"
    ]
    assert (package_root / "tokenizer/mt3.py").read_bytes() == patched["tokenizer/mt3.py"]
    assert any("quality_patch=" in message for message in log)


def test_quality_patch_refuses_unknown_source_without_overwriting(tmp_path, monkeypatch):
    package_root = tmp_path / "muscriptor"
    patched = _fake_identity(monkeypatch, package_root)
    target = package_root / "transcription_model.py"
    target.write_bytes(b"unexpected local mutation\n")
    original = target.read_bytes()

    with pytest.raises(RuntimeError, match="Refusing to patch an unknown"):
        muscriptor_source_patch.apply_muscriptor_quality_patch(
            package_root=package_root,
            fetcher=lambda url: patched[url.removeprefix("https://example.invalid/")],
        )

    assert target.read_bytes() == original


def test_quality_patch_rolls_back_if_a_replacement_fails(tmp_path, monkeypatch):
    package_root = tmp_path / "muscriptor"
    patched = _fake_identity(monkeypatch, package_root)
    original = {relative: (package_root / relative).read_bytes() for relative in patched}
    real_replace = muscriptor_source_patch.os.replace
    calls = 0

    def fail_second_replacement(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(muscriptor_source_patch.os, "replace", fail_second_replacement)

    with pytest.raises(OSError, match="simulated second replacement failure"):
        muscriptor_source_patch.apply_muscriptor_quality_patch(
            package_root=package_root,
            fetcher=lambda url: patched[url.removeprefix("https://example.invalid/")],
        )

    assert {relative: (package_root / relative).read_bytes() for relative in patched} == original


def test_runtime_identity_rejects_old_package_version(tmp_path, monkeypatch):
    package_root = tmp_path / "muscriptor"
    package_root.mkdir()
    monkeypatch.setattr(
        muscriptor_source_patch.importlib.metadata,
        "version",
        lambda _package: "0.2.2a1",
    )

    error = muscriptor_source_patch.validate_muscriptor_runtime_identity(package_root)

    assert "expected 0.2.2, got 0.2.2a1" in error
