"""Pinned Leap Instrumental assets and historical installer compatibility."""

import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest

import download_accompaniment_model as assets
import download_vocal_harmony_model as legacy


@pytest.fixture
def tiny_assets(monkeypatch):
    payloads = {
        assets.LEAP_INSTRUMENTAL_CHECKPOINT_NAME: b"leap-checkpoint",
        assets.LEAP_INSTRUMENTAL_CONFIG_NAME: b"leap-config",
    }
    for prefix, name in (
        ("LEAP_INSTRUMENTAL_CHECKPOINT", assets.LEAP_INSTRUMENTAL_CHECKPOINT_NAME),
        ("LEAP_INSTRUMENTAL_CONFIG", assets.LEAP_INSTRUMENTAL_CONFIG_NAME),
    ):
        monkeypatch.setattr(assets, prefix + "_SIZE", len(payloads[name]))
        monkeypatch.setattr(assets, prefix + "_SHA256", hashlib.sha256(payloads[name]).hexdigest())
    return payloads


def test_historical_entrypoints_select_current_instrumental_model():
    assert legacy.CHORUS_MODEL == "bs_roformer_leap_inst.ckpt"
    assert legacy.download_accompaniment_model is assets.download_accompaniment_model
    assert legacy.is_chorus_model_available is assets.is_chorus_model_available
    assert assets.CHORUS_MODELS == ("bs_roformer_leap_inst.ckpt", "bs_leap_inst_conf.yaml")


def test_default_cache_dir_is_under_user_home():
    assert assets.DEFAULT_CACHE_DIR == Path.home() / ".music-to-midi/models/audio-separator"


def test_downloads_and_verifies_both_assets_at_exact_revision(tmp_path, tiny_assets):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        path = tmp_path / kwargs["filename"]
        path.write_bytes(tiny_assets[path.name])
        return str(path)

    result = assets.download_accompaniment_model(
        tmp_path, downloader=download, printer=lambda _: None
    )
    assert result == tmp_path / assets.LEAP_INSTRUMENTAL_CHECKPOINT_NAME
    assert [call["filename"] for call in calls] == list(tiny_assets)
    assert all(call["repo_id"] == "pcunwa/BS-Roformer-Leap" for call in calls)
    assert all(call["revision"] == "4e47d6662ae82eaa8b4ac4329fe66099a843b48e" for call in calls)
    assert assets.is_accompaniment_model_available(tmp_path)
    assert legacy.is_chorus_model_available(tmp_path)


def test_valid_nested_assets_are_reused_without_network(tmp_path, tiny_assets):
    nested = tmp_path / "snapshot"
    nested.mkdir()
    for name, data in tiny_assets.items():
        (nested / name).write_bytes(data)
    downloader = Mock(side_effect=AssertionError("unexpected download"))
    result = legacy.download_chorus_model(tmp_path, downloader=downloader, printer=lambda _: None)
    assert result.parent == nested
    assert assets.resolve_chorus_model_paths(tmp_path) == tuple(
        nested / name for name in tiny_assets
    )
    downloader.assert_not_called()


@pytest.mark.parametrize("filename", assets.CHORUS_MODELS)
def test_same_size_corruption_is_rejected_without_overwrite(tmp_path, tiny_assets, filename):
    for name, data in tiny_assets.items():
        (tmp_path / name).write_bytes(data)
    corrupt = b"x" * len(tiny_assets[filename])
    (tmp_path / filename).write_bytes(corrupt)
    downloader = Mock()
    assert not assets.is_accompaniment_model_available(tmp_path)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        assets.download_accompaniment_model(tmp_path, downloader=downloader, printer=lambda _: None)
    assert (tmp_path / filename).read_bytes() == corrupt
    downloader.assert_not_called()


def test_failed_download_stops_before_config(tmp_path, tiny_assets):
    downloader = Mock(side_effect=ConnectionError("download failed"))
    with pytest.raises(ConnectionError, match="download failed"):
        assets.download_accompaniment_model(tmp_path, downloader=downloader, printer=lambda _: None)
    assert downloader.call_count == 1
    assert not assets.is_accompaniment_model_available(tmp_path)


def test_wrong_downloaded_model_stops_before_config(tmp_path, tiny_assets):
    calls = []

    def download(**kwargs):
        calls.append(kwargs["filename"])
        path = tmp_path / kwargs["filename"]
        path.write_bytes(b"wrong")
        return path

    with pytest.raises(RuntimeError, match="size mismatch"):
        assets.download_accompaniment_model(tmp_path, downloader=download, printer=lambda _: None)
    assert calls == [assets.LEAP_INSTRUMENTAL_CHECKPOINT_NAME]


def test_polarformer_files_do_not_satisfy_leap_readiness(tmp_path):
    (tmp_path / "bs_polarformer_fp16.onnx").write_bytes(b"old model")
    (tmp_path / "model_bs_polarformer_float16.yaml").write_bytes(b"old config")
    assert not assets.is_accompaniment_model_available(tmp_path)
    with pytest.raises(ValueError):
        assets.download_accompaniment_model(tmp_path, model_name="bs_polarformer_fp16.onnx")


def test_ambiguous_nested_models_are_rejected(tmp_path):
    for directory in ("first", "second"):
        root = tmp_path / directory
        root.mkdir()
        (root / assets.CHORUS_MODEL).write_bytes(b"data")
    with pytest.raises(RuntimeError):
        assets.resolve_accompaniment_model_path(tmp_path)
    assert not assets.is_accompaniment_model_available(tmp_path)


def test_cli_reports_invalid_assets(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        assets, "download_accompaniment_model", Mock(side_effect=RuntimeError("bad identity"))
    )
    assert assets.main(["--cache-dir", str(tmp_path)]) == 1
    assert "bad identity" in capsys.readouterr().out
