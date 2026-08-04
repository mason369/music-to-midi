from __future__ import annotations

import io
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

import download_beat_this_model as downloader


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def _validate_test_checkpoint(path: Path | None = None) -> Path:
    candidate = Path(path)
    if not candidate.is_file() or candidate.read_bytes() != b"exact-final0":
        raise RuntimeError("identity mismatch")
    return candidate


def test_download_beat_this_model_publishes_only_validated_partial(tmp_path):
    target = tmp_path / "beat_this" / "final0.ckpt"
    with (
        mock.patch.object(downloader, "get_beat_this_download_path", return_value=target),
        mock.patch.object(
            downloader,
            "validate_beat_this_checkpoint",
            side_effect=_validate_test_checkpoint,
        ),
        mock.patch.object(
            downloader.urllib.request,
            "urlopen",
            return_value=_Response(b"exact-final0"),
        ) as urlopen,
    ):
        result = downloader.download_beat_this_model()

    assert result == target
    assert target.read_bytes() == b"exact-final0"
    assert not target.with_suffix(".ckpt.part").exists()
    assert urlopen.call_count == 1


def test_download_beat_this_model_does_not_retry_nontransient_http_error(tmp_path):
    target = tmp_path / "beat_this" / "final0.ckpt"
    error = urllib.error.HTTPError(
        downloader.BEAT_THIS_CHECKPOINT_URL,
        403,
        "forbidden",
        hdrs=None,
        fp=None,
    )
    with (
        mock.patch.object(downloader, "get_beat_this_download_path", return_value=target),
        mock.patch.object(
            downloader,
            "validate_beat_this_checkpoint",
            side_effect=_validate_test_checkpoint,
        ),
        mock.patch.object(
            downloader.urllib.request,
            "urlopen",
            side_effect=error,
        ) as urlopen,
    ):
        with pytest.raises(urllib.error.HTTPError) as raised:
            downloader.download_beat_this_model()

    assert raised.value.code == 403
    assert urlopen.call_count == 1
    assert not target.exists()
    assert not target.with_suffix(".ckpt.part").exists()
