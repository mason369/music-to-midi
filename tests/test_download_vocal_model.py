import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from download_vocal_model import (
    DEFAULT_CACHE_DIR,
    LEAP_CHECKPOINT_NAME,
    LEAP_CHECKPOINT_REPO_PATH,
    LEAP_CONFIG_NAME,
    LEAP_CONFIG_REPO_PATH,
    LEAP_REPO_ID,
    LEAP_REVISION,
    download_vocal_model,
    is_vocal_model_available,
    resolve_vocal_config_path,
    resolve_vocal_model_path,
    resolve_vocal_model_paths,
)


class TestDownloadVocalModel(unittest.TestCase):
    def test_default_cache_dir_is_under_user_home(self):
        self.assertIn(".music-to-midi", str(DEFAULT_CACHE_DIR))
        self.assertIn("audio-separator", str(DEFAULT_CACHE_DIR))

    def test_resolve_vocal_model_paths_find_nested_huggingface_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            nested = cache_dir / "Xe"
            nested.mkdir()
            checkpoint = nested / LEAP_CHECKPOINT_NAME
            config = nested / LEAP_CONFIG_NAME
            checkpoint.write_bytes(b"checkpoint")
            config.write_bytes(b"config")

            self.assertEqual(resolve_vocal_model_path(cache_dir), checkpoint)
            self.assertEqual(resolve_vocal_config_path(cache_dir), config)
            self.assertEqual(
                resolve_vocal_model_paths(cache_dir),
                (checkpoint, config),
            )

    def test_model_available_requires_expected_checkpoint_size_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            checkpoint = cache_dir / LEAP_CHECKPOINT_NAME
            config = cache_dir / LEAP_CONFIG_NAME
            checkpoint.write_bytes(b"model")
            config.write_bytes(b"config")

            with (
                patch("download_vocal_model.LEAP_CHECKPOINT_SIZE", 5),
                patch("download_vocal_model.LEAP_CHECKPOINT_SHA256", hashlib.sha256(b"model").hexdigest()),
                patch("download_vocal_model.LEAP_CONFIG_SIZE", 6),
                patch("download_vocal_model.LEAP_CONFIG_SHA256", hashlib.sha256(b"config").hexdigest()),
            ):
                self.assertTrue(is_vocal_model_available(cache_dir))
            with patch("download_vocal_model.LEAP_CHECKPOINT_SIZE", 6):
                self.assertFalse(is_vocal_model_available(cache_dir))

    def test_download_uses_pinned_repo_revision_and_verifies_checkpoint(self):
        payload = b"leap-model"
        expected_hash = hashlib.sha256(payload).hexdigest()
        config_payload = b"yaml"
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                repo_path = kwargs["filename"]
                path = cache_dir / repo_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload if repo_path == LEAP_CHECKPOINT_REPO_PATH else config_payload)
                return str(path)

            with (
                patch("download_vocal_model.LEAP_CHECKPOINT_SIZE", len(payload)),
                patch("download_vocal_model.LEAP_CHECKPOINT_SHA256", expected_hash),
                patch("download_vocal_model.LEAP_CONFIG_SIZE", len(config_payload)),
                patch("download_vocal_model.LEAP_CONFIG_SHA256", hashlib.sha256(config_payload).hexdigest()),
            ):
                result = download_vocal_model(
                    cache_dir=cache_dir,
                    downloader=fake_download,
                    printer=lambda *_: None,
                )

            self.assertEqual(result, cache_dir / LEAP_CHECKPOINT_REPO_PATH)
            self.assertEqual(
                [call["filename"] for call in calls],
                [LEAP_CHECKPOINT_REPO_PATH, LEAP_CONFIG_REPO_PATH],
            )
            self.assertTrue(all(call["repo_id"] == LEAP_REPO_ID for call in calls))
            self.assertTrue(all(call["revision"] == LEAP_REVISION for call in calls))

    def test_download_skips_only_when_both_assets_have_valid_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            checkpoint = cache_dir / LEAP_CHECKPOINT_NAME
            config = cache_dir / LEAP_CONFIG_NAME
            checkpoint.write_bytes(b"model")
            config.write_bytes(b"config")
            downloader = Mock(side_effect=AssertionError("download must not run"))

            with (
                patch("download_vocal_model.LEAP_CHECKPOINT_SIZE", 5),
                patch("download_vocal_model.LEAP_CHECKPOINT_SHA256", hashlib.sha256(b"model").hexdigest()),
                patch("download_vocal_model.LEAP_CONFIG_SIZE", 6),
                patch("download_vocal_model.LEAP_CONFIG_SHA256", hashlib.sha256(b"config").hexdigest()),
            ):
                result = download_vocal_model(
                    cache_dir=cache_dir,
                    downloader=downloader,
                    printer=lambda *_: None,
                )
            self.assertEqual(result, checkpoint)
            downloader.assert_not_called()

    def test_download_rejects_checkpoint_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            def fake_download(**kwargs):
                path = cache_dir / kwargs["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"bad" if path.suffix == ".ckpt" else b"yaml")
                return str(path)

            with (
                patch("download_vocal_model.LEAP_CHECKPOINT_SIZE", 3),
                patch("download_vocal_model.LEAP_CHECKPOINT_SHA256", "0" * 64),
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    download_vocal_model(
                        cache_dir=cache_dir,
                        downloader=fake_download,
                        printer=lambda *_: None,
                    )

    def test_download_rejects_unrelated_model_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unsupported vocal model"):
                download_vocal_model(
                    cache_dir=Path(tmp),
                    model_name="ensemble:vocal_rvc",
                    downloader=Mock(),
                    printer=lambda *_: None,
                )


if __name__ == "__main__":
    unittest.main()
