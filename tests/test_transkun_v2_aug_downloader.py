from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

import download_transkun_v2_aug_model as downloader


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_archive(checkpoint: bytes, config: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("checkpointMSimplerAug/", b"")
        archive.writestr("checkpointMSimplerAug/model.conf", config)
        archive.writestr("checkpointMSimplerAug/checkpoint.pt", checkpoint)
    return buffer.getvalue()


def _patch_fixture_metadata(monkeypatch, archive: bytes, checkpoint: bytes, config: bytes):
    monkeypatch.setattr(downloader, "TRANSKUN_V2_AUG_ARCHIVE_SIZE", len(archive))
    monkeypatch.setattr(downloader, "TRANSKUN_V2_AUG_ARCHIVE_SHA256", _sha256(archive))
    monkeypatch.setattr(downloader, "TRANSKUN_V2_AUG_CHECKPOINT_SIZE", len(checkpoint))
    monkeypatch.setattr(
        downloader,
        "TRANSKUN_V2_AUG_CHECKPOINT_SHA256",
        _sha256(checkpoint),
    )
    monkeypatch.setattr(downloader, "TRANSKUN_V2_AUG_CONFIG_SIZE", len(config))
    monkeypatch.setattr(
        downloader,
        "TRANSKUN_V2_AUG_CONFIG_SHA256",
        _sha256(config),
    )

    def validate(model_dir: Path) -> str:
        model_dir = Path(model_dir)
        checkpoint_path = model_dir / "checkpoint.pt"
        config_path = model_dir / "model.conf"
        if not checkpoint_path.is_file() or not config_path.is_file():
            return "missing"
        if checkpoint_path.read_bytes() != checkpoint:
            return "checkpoint mismatch"
        if config_path.read_bytes() != config:
            return "config mismatch"
        return ""

    monkeypatch.setattr(downloader, "validate_transkun_v2_aug_model_files", validate)


def test_official_archive_identity_and_layout_constants_are_pinned():
    assert (
        downloader.TRANSKUN_V2_AUG_DOWNLOAD_URL == "https://drive.usercontent.google.com/download"
        "?id=1Hg5ua8vYdtg1Y-MnXD0mLyhRK9Srd7hm&export=download&confirm=t"
    )
    assert downloader.TRANSKUN_V2_AUG_ARCHIVE_NAME == "checkpointTransformerAug.zip"
    assert downloader.TRANSKUN_V2_AUG_ARCHIVE_SIZE == 50_694_377
    assert (
        downloader.TRANSKUN_V2_AUG_ARCHIVE_SHA256
        == "f61ebf6467d89081fde9728b659895a3e3d65b4c89516964178967167fae6590"
    )
    assert downloader.TRANSKUN_V2_AUG_MODEL_DIR_NAME == "checkpointMSimplerAug"
    assert downloader.TRANSKUN_V2_AUG_CHECKPOINT_NAME == "checkpoint.pt"
    assert downloader.TRANSKUN_V2_AUG_CONFIG_NAME == "model.conf"


def test_download_extracts_only_verified_official_layout(tmp_path, monkeypatch):
    checkpoint = b"official-aug-checkpoint"
    config = b'{"Model": {"module": "transkun.ModelTransformer"}}'
    archive = _build_archive(checkpoint, config)
    _patch_fixture_metadata(monkeypatch, archive, checkpoint, config)
    requested_urls = []

    def fake_download(url: str, output_path: Path) -> None:
        requested_urls.append(url)
        output_path.write_bytes(archive)

    result = downloader.download_transkun_v2_aug_model(
        cache_root=tmp_path,
        downloader=fake_download,
        printer=None,
    )

    assert result == tmp_path / "checkpointMSimplerAug"
    assert (result / "checkpoint.pt").read_bytes() == checkpoint
    assert (result / "model.conf").read_bytes() == config
    assert (tmp_path / "checkpointTransformerAug.zip").read_bytes() == archive
    assert requested_urls == [downloader.TRANSKUN_V2_AUG_DOWNLOAD_URL]


def test_download_rejects_archive_checksum_mismatch_without_publishing(tmp_path, monkeypatch):
    bad_archive = b"not-the-official-archive"
    monkeypatch.setattr(downloader, "TRANSKUN_V2_AUG_ARCHIVE_SIZE", len(bad_archive))
    monkeypatch.setattr(downloader, "TRANSKUN_V2_AUG_ARCHIVE_SHA256", "0" * 64)
    monkeypatch.setattr(
        downloader,
        "validate_transkun_v2_aug_model_files",
        lambda _model_dir: "missing",
    )

    def fake_download(_url: str, output_path: Path) -> None:
        output_path.write_bytes(bad_archive)

    with pytest.raises(RuntimeError, match="SHA256 不匹配"):
        downloader.download_transkun_v2_aug_model(
            cache_root=tmp_path,
            downloader=fake_download,
            printer=None,
        )

    assert not (tmp_path / "checkpointTransformerAug.zip").exists()
    assert not (tmp_path / "checkpointTransformerAug.zip.download").exists()
    assert not (tmp_path / "checkpointMSimplerAug" / "checkpoint.pt").exists()


def test_verified_cached_model_skips_network(tmp_path, monkeypatch):
    model_dir = tmp_path / "checkpointMSimplerAug"
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(
        downloader,
        "validate_transkun_v2_aug_model_files",
        lambda candidate: "" if Path(candidate) == model_dir else "unexpected",
    )

    def fail_download(_url: str, _output_path: Path) -> None:
        raise AssertionError("network download must not run")

    assert (
        downloader.download_transkun_v2_aug_model(
            cache_root=tmp_path,
            downloader=fail_download,
            printer=None,
        )
        == model_dir
    )
