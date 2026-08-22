from __future__ import annotations

from pathlib import Path

import pytest

import src.utils.muscriptor_soundfont_downloader as soundfont_module


def test_runtime_soundfont_resolution_is_strictly_local(tmp_path, monkeypatch):
    cached = tmp_path / soundfont_module.MUSCRIPTOR_SF2_FILENAME
    cached.write_bytes(b"cached")
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        soundfont_module,
        "get_resource_path",
        lambda _relative: tmp_path / "missing-bundled.sf2",
    )
    monkeypatch.setattr(
        soundfont_module,
        "_validate_soundfont_path",
        lambda path, *, filename: Path(path).resolve(),
    )

    def resolve_cached(**kwargs):
        calls.append(kwargs)
        return str(cached)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", resolve_cached)

    assert soundfont_module.validate_muscriptor_soundfont() == cached.resolve()
    assert calls == [
        {
            "repo_id": soundfont_module.MUSCRIPTOR_ASSETS_REPO_ID,
            "filename": soundfont_module.MUSCRIPTOR_SF2_FILENAME,
            "revision": soundfont_module.MUSCRIPTOR_ASSETS_REVISION,
            "local_files_only": True,
        }
    ]


def test_missing_runtime_soundfont_reports_the_explicit_initializer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        soundfont_module,
        "get_resource_path",
        lambda _relative: tmp_path / "missing-bundled.sf2",
    )

    def missing(**_kwargs):
        raise FileNotFoundError("not cached")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", missing)

    with pytest.raises(RuntimeError, match="explicit model initialization"):
        soundfont_module.validate_muscriptor_soundfont()


def test_explicit_soundfont_preparation_is_allowed_to_download(tmp_path, monkeypatch):
    downloaded = tmp_path / soundfont_module.MUSCRIPTOR_SF2_FILENAME
    downloaded.write_bytes(b"downloaded")
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        soundfont_module,
        "get_resource_path",
        lambda _relative: tmp_path / "missing-bundled.sf2",
    )
    monkeypatch.setattr(
        soundfont_module,
        "_validate_soundfont_path",
        lambda path, *, filename: Path(path).resolve(),
    )

    def download(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)

    assert soundfont_module.download_muscriptor_soundfont() == downloaded.resolve()
    assert calls and "local_files_only" not in calls[0]
