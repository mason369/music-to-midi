from pathlib import Path

import huggingface_hub
import pytest
from huggingface_hub.errors import GatedRepoError

from src.utils import muscriptor_downloader


def test_gated_preflight_skips_network_for_complete_cached_models(monkeypatch):
    messages = []

    monkeypatch.setattr(
        muscriptor_downloader,
        "get_cached_muscriptor_paths",
        lambda model_size, validate_hashes: (
            Path(f"/cache/{model_size}/model.safetensors"),
            Path(f"/cache/{model_size}/config.json"),
        ),
    )

    def unexpected_metadata_request(*args, **kwargs):
        raise AssertionError("complete caches must not trigger a gated network preflight")

    monkeypatch.setattr(
        huggingface_hub,
        "get_hf_file_metadata",
        unexpected_metadata_request,
    )

    muscriptor_downloader.preflight_muscriptor_download_access(printer=messages.append)

    assert len(messages) == 3
    assert all("完整本地缓存" in message for message in messages)


def test_gated_preflight_reports_every_required_repository(monkeypatch):
    monkeypatch.setattr(
        muscriptor_downloader,
        "get_cached_muscriptor_paths",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("not cached")),
    )

    def reject_metadata_request(*args, **kwargs):
        raise GatedRepoError("gated")

    monkeypatch.setattr(
        huggingface_hub,
        "get_hf_file_metadata",
        reject_metadata_request,
    )

    with pytest.raises(RuntimeError, match="无法匿名全自动下载") as raised:
        muscriptor_downloader.preflight_muscriptor_download_access()

    error = str(raised.value)
    assert "命令行登录不能代替网页接受条款" in error
    assert "HF_TOKEN" in error
    for artifact in muscriptor_downloader.MUSCRIPTOR_ARTIFACTS.values():
        assert f"https://huggingface.co/{artifact.repo_id}" in error
