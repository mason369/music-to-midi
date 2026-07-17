import hashlib
from pathlib import Path
from unittest import mock

import pytest

import download_aria_amt_model as downloader


def test_aria_download_validates_part_before_atomic_replace(tmp_path):
    payload = b"official-aria-checkpoint"
    observed_paths = []

    def fake_download(_url: str, output_path: Path) -> None:
        observed_paths.append(output_path)
        output_path.write_bytes(payload)

    with (
        mock.patch(
            "src.core.aria_amt_transcriber.ARIA_AMT_CHECKPOINT_SIZE",
            len(payload),
        ),
        mock.patch(
            "src.core.aria_amt_transcriber.ARIA_AMT_CHECKPOINT_SHA256",
            hashlib.sha256(payload).hexdigest(),
        ),
    ):
        result = downloader.download_aria_model(
            cache_dir=tmp_path,
            downloader=fake_download,
            printer=lambda _message: None,
        )

    assert result.read_bytes() == payload
    assert observed_paths == [result.with_suffix(result.suffix + ".part")]
    assert not observed_paths[0].exists()


def test_aria_download_preserves_existing_target_when_part_identity_is_wrong(tmp_path):
    target = tmp_path / downloader.ARIA_AMT_CHECKPOINT_NAME
    target.write_bytes(b"existing-invalid-cache")
    expected_payload = b"official-aria-checkpoint"

    def fake_download(_url: str, output_path: Path) -> None:
        output_path.write_bytes(expected_payload[:-1] + b"X")

    with (
        mock.patch(
            "src.core.aria_amt_transcriber.ARIA_AMT_CHECKPOINT_SIZE",
            len(expected_payload),
        ),
        mock.patch(
            "src.core.aria_amt_transcriber.ARIA_AMT_CHECKPOINT_SHA256",
            hashlib.sha256(expected_payload).hexdigest(),
        ),
        pytest.raises(RuntimeError, match="Aria-AMT checkpoint SHA-256 mismatch"),
    ):
        downloader.download_aria_model(
            cache_dir=tmp_path,
            downloader=fake_download,
            printer=lambda _message: None,
        )

    assert target.read_bytes() == b"existing-invalid-cache"
    assert not target.with_suffix(target.suffix + ".part").exists()
