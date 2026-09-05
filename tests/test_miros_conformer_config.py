"""The MusicFM configuration is a required, pinned, offline-capable asset."""

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

import download_miros_model as downloader
import src.core.miros_transcriber as miros

FIXTURE = Path(__file__).parent / "fixtures/miros_conformer_config.json"


def test_conformer_fixture_is_exact_pinned_upstream_file():
    payload = FIXTURE.read_bytes()
    assert len(payload) == miros.MIROS_CONFORMER_CONFIG_EXACT_BYTES
    assert hashlib.sha256(payload).hexdigest() == miros.MIROS_CONFORMER_CONFIG_SHA256
    assert miros.MIROS_CONFORMER_CONFIG_COMMIT in downloader.MIROS_CONFORMER_CONFIG_URL


def test_pinned_config_loads_offline_with_upstream_relative_id(tmp_path, monkeypatch):
    destination = tmp_path / miros.MIROS_CONFORMER_CONFIG_REL_PATH
    destination.parent.mkdir(parents=True)
    destination.write_bytes(FIXTURE.read_bytes())
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hub"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    # A fresh interpreter ensures Transformers cannot use an already imported
    # module's cache constants. This is the exact identifier used by MusicFM.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from transformers import Wav2Vec2ConformerConfig; "
            "from unittest.mock import patch; "
            "patch('transformers.utils.hub.hf_hub_download', side_effect=AssertionError('unexpected Hub lookup')).start(); "
            "patch('socket.socket.connect', side_effect=AssertionError('unexpected network')).start(); "
            "c=Wav2Vec2ConformerConfig.from_pretrained('facebook/wav2vec2-conformer-rope-large-960h-ft'); "
            "assert c.hidden_size==1024 and c.num_hidden_layers==24 and c.position_embeddings_type=='rotary'",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    cache_files = {
        path.relative_to(tmp_path / "empty-hub").as_posix()
        for path in (tmp_path / "empty-hub").rglob("*")
        if path.is_file()
    }
    # Transformers may create its one-byte cache-format version marker on import.
    assert cache_files <= {"hub/version.txt"}


def test_only_the_independently_validated_config_is_excluded_from_source_hash(tmp_path):
    (tmp_path / "main.py").write_text("pass\n")
    original = miros.compute_miros_source_tree_sha256(tmp_path)
    config = tmp_path / miros.MIROS_CONFORMER_CONFIG_REL_PATH
    config.parent.mkdir(parents=True)
    config.write_bytes(FIXTURE.read_bytes())
    assert miros.compute_miros_source_tree_sha256(tmp_path) == original
    (config.parent / "unexpected.json").write_text("{}")
    assert miros.compute_miros_source_tree_sha256(tmp_path) != original


def test_missing_and_same_size_corrupt_config_fail_required_asset_gate(tmp_path, monkeypatch):
    for relative, prefix in [
        (miros.MirosTranscriber.PRETRAINED_REL_PATH, "MIROS_PRETRAINED"),
        (miros.MirosTranscriber.CHECKPOINT_REL_PATH, "MIROS_FINETUNED"),
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(b"test-weight")
        monkeypatch.setattr(miros, prefix + "_EXACT_BYTES", len(b"test-weight"))
        monkeypatch.setattr(miros, prefix + "_SHA256", hashlib.sha256(b"test-weight").hexdigest())
    assert "Conformer" in miros.get_miros_weight_identity_error(tmp_path)
    config = tmp_path / miros.MIROS_CONFORMER_CONFIG_REL_PATH
    config.parent.mkdir(parents=True)
    config.write_bytes(FIXTURE.read_bytes())
    assert miros.get_miros_weight_identity_error(tmp_path) == ""
    config.write_bytes(b"x" * miros.MIROS_CONFORMER_CONFIG_EXACT_BYTES)
    assert "SHA256 mismatch" in miros.get_miros_weight_identity_error(tmp_path)


def test_invalid_existing_config_is_not_replaced_or_redownloaded(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_bytes(b"x" * miros.MIROS_CONFORMER_CONFIG_EXACT_BYTES)

    def unexpected_network(*args, **kwargs):
        raise AssertionError("An invalid existing config must fail, not redownload")

    monkeypatch.setattr(downloader, "_run", unexpected_network)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        downloader._download_http_file(
            downloader.MIROS_CONFORMER_CONFIG_URL,
            config,
            miros.MIROS_CONFORMER_CONFIG_EXACT_BYTES,
            miros.MIROS_CONFORMER_CONFIG_SHA256,
            None,
            "Conformer config",
        )
    assert config.read_bytes() == b"x" * miros.MIROS_CONFORMER_CONFIG_EXACT_BYTES
