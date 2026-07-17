import hashlib
from pathlib import Path
import urllib.error
from unittest import mock

import download_bytedance_piano_model as downloader


class _FakeResponse:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


def test_bytedance_piano_download_retries_transient_http_5xx(tmp_path):
    target = tmp_path / "note_F1=0.9677_pedal_F1=0.9186.pth"
    attempts = []
    payload = b"complete-checkpoint"

    def fake_urlopen(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.HTTPError(
                downloader.BYTEDANCE_PIANO_CHECKPOINT_URL,
                502,
                "Bad Gateway",
                {},
                None,
            )
        return _FakeResponse([payload])

    with (
        mock.patch.object(
            downloader,
            "get_bytedance_piano_model_path",
            return_value=target,
        ),
        mock.patch.object(downloader, "BYTEDANCE_PIANO_RETRY_DELAY_SECONDS", 0, create=True),
        mock.patch(
            "src.core.bytedance_piano_transcriber.BYTEDANCE_PIANO_CHECKPOINT_SIZE",
            len(payload),
        ),
        mock.patch(
            "src.core.bytedance_piano_transcriber.BYTEDANCE_PIANO_CHECKPOINT_SHA256",
            hashlib.sha256(payload).hexdigest(),
        ),
        mock.patch.object(
            downloader.urllib.request,
            "urlopen",
            side_effect=fake_urlopen,
        ),
    ):
        result = downloader.download_bytedance_piano_model()

    assert result == target
    assert target.read_bytes() == payload
    assert len(attempts) == 2


def test_bytedance_piano_download_does_not_retry_http_404(tmp_path):
    target = tmp_path / "note_F1=0.9677_pedal_F1=0.9186.pth"
    error = urllib.error.HTTPError(
        downloader.BYTEDANCE_PIANO_CHECKPOINT_URL,
        404,
        "Not Found",
        {},
        None,
    )

    with (
        mock.patch.object(downloader, "get_bytedance_piano_model_path", return_value=target),
        mock.patch.object(
            downloader.urllib.request,
            "urlopen",
            side_effect=error,
        ) as urlopen_mock,
    ):
        try:
            downloader.download_bytedance_piano_model()
        except urllib.error.HTTPError:
            pass
        else:
            raise AssertionError("Expected HTTP 404 to be raised without retry")

    assert urlopen_mock.call_count == 1
    assert not target.exists()
    assert not Path(str(target) + ".part").exists()


def test_bytedance_piano_download_keeps_existing_target_when_identity_check_fails(tmp_path):
    target = tmp_path / "note_F1=0.9677_pedal_F1=0.9186.pth"
    target.write_bytes(b"existing-invalid-cache")
    expected_payload = b"expected-checkpoint"

    with (
        mock.patch.object(
            downloader,
            "get_bytedance_piano_model_path",
            return_value=target,
        ),
        mock.patch(
            "src.core.bytedance_piano_transcriber.BYTEDANCE_PIANO_CHECKPOINT_SIZE",
            len(expected_payload),
        ),
        mock.patch(
            "src.core.bytedance_piano_transcriber.BYTEDANCE_PIANO_CHECKPOINT_SHA256",
            hashlib.sha256(expected_payload).hexdigest(),
        ),
        mock.patch.object(
            downloader.urllib.request,
            "urlopen",
            return_value=_FakeResponse([b"corrupt-checkpoint"]),
        ),
    ):
        try:
            downloader.download_bytedance_piano_model()
        except RuntimeError as exc:
            assert "ByteDance Piano checkpoint" in str(exc)
        else:
            raise AssertionError("Expected strict checkpoint identity failure")

    assert target.read_bytes() == b"existing-invalid-cache"
    assert not Path(str(target) + ".part").exists()
