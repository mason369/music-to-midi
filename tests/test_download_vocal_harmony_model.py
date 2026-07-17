import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from download_vocal_harmony_model import (
    DEFAULT_CACHE_DIR,
    POLARFORMER_CONFIG_NAME,
    POLARFORMER_ONNX_NAME,
    POLARFORMER_REPO_ID,
    POLARFORMER_REVISION,
    download_accompaniment_model,
    download_chorus_model,
    is_accompaniment_model_available,
    is_chorus_model_available,
    resolve_accompaniment_config_path,
    resolve_accompaniment_model_path,
    resolve_chorus_model_paths,
)


class TestDownloadVocalHarmonyModel(unittest.TestCase):
    def test_default_cache_dir_is_under_user_home(self):
        self.assertIn(".music-to-midi", str(DEFAULT_CACHE_DIR))
        self.assertIn("audio-separator", str(DEFAULT_CACHE_DIR))

    def test_resolve_paths_find_nested_huggingface_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            nested = cache_dir / "polar"
            nested.mkdir()
            model = nested / POLARFORMER_ONNX_NAME
            config = nested / POLARFORMER_CONFIG_NAME
            model.write_bytes(b"model")
            config.write_bytes(b"config")

            self.assertEqual(resolve_accompaniment_model_path(cache_dir), model)
            self.assertEqual(resolve_accompaniment_config_path(cache_dir), config)
            self.assertEqual(resolve_chorus_model_paths(cache_dir), (model, config))

    def test_availability_requires_expected_onnx_size_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / POLARFORMER_ONNX_NAME).write_bytes(b"model")
            (cache_dir / POLARFORMER_CONFIG_NAME).write_bytes(b"config")

            with (
                patch("download_vocal_harmony_model.POLARFORMER_ONNX_SIZE", 5),
                patch("download_vocal_harmony_model.POLARFORMER_ONNX_SHA256", hashlib.sha256(b"model").hexdigest()),
                patch("download_vocal_harmony_model.POLARFORMER_CONFIG_SIZE", 6),
                patch("download_vocal_harmony_model.POLARFORMER_CONFIG_SHA256", hashlib.sha256(b"config").hexdigest()),
            ):
                self.assertTrue(is_accompaniment_model_available(cache_dir))
                self.assertTrue(is_chorus_model_available(cache_dir))
            with patch("download_vocal_harmony_model.POLARFORMER_ONNX_SIZE", 6):
                self.assertFalse(is_accompaniment_model_available(cache_dir))

    def test_download_uses_pinned_repo_revision_and_verifies_onnx(self):
        payload = b"polar-model"
        expected_hash = hashlib.sha256(payload).hexdigest()
        config_payload = b"yaml"
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                path = cache_dir / kwargs["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload if path.suffix == ".onnx" else config_payload)
                return str(path)

            with (
                patch("download_vocal_harmony_model.POLARFORMER_ONNX_SIZE", len(payload)),
                patch("download_vocal_harmony_model.POLARFORMER_ONNX_SHA256", expected_hash),
                patch("download_vocal_harmony_model.POLARFORMER_CONFIG_SIZE", len(config_payload)),
                patch("download_vocal_harmony_model.POLARFORMER_CONFIG_SHA256", hashlib.sha256(config_payload).hexdigest()),
            ):
                result = download_accompaniment_model(
                    cache_dir=cache_dir,
                    downloader=fake_download,
                    printer=lambda *_: None,
                )

            self.assertEqual(result, cache_dir / POLARFORMER_ONNX_NAME)
            self.assertEqual(
                [call["filename"] for call in calls],
                [POLARFORMER_ONNX_NAME, POLARFORMER_CONFIG_NAME],
            )
            self.assertTrue(all(call["repo_id"] == POLARFORMER_REPO_ID for call in calls))
            self.assertTrue(all(call["revision"] == POLARFORMER_REVISION for call in calls))

    def test_legacy_download_entrypoint_routes_to_polarformer(self):
        with patch(
            "download_vocal_harmony_model.download_accompaniment_model",
            return_value=Path("polar.onnx"),
        ) as download:
            result = download_chorus_model(printer=lambda *_: None)
        self.assertEqual(result, Path("polar.onnx"))
        download.assert_called_once()

    def test_download_skips_only_when_both_assets_have_valid_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            model = cache_dir / POLARFORMER_ONNX_NAME
            config = cache_dir / POLARFORMER_CONFIG_NAME
            model.write_bytes(b"model")
            config.write_bytes(b"config")
            downloader = Mock(side_effect=AssertionError("download must not run"))

            with (
                patch("download_vocal_harmony_model.POLARFORMER_ONNX_SIZE", 5),
                patch("download_vocal_harmony_model.POLARFORMER_ONNX_SHA256", hashlib.sha256(b"model").hexdigest()),
                patch("download_vocal_harmony_model.POLARFORMER_CONFIG_SIZE", 6),
                patch("download_vocal_harmony_model.POLARFORMER_CONFIG_SHA256", hashlib.sha256(b"config").hexdigest()),
            ):
                result = download_accompaniment_model(
                    cache_dir=cache_dir,
                    downloader=downloader,
                    printer=lambda *_: None,
                )
            self.assertEqual(result, model)
            downloader.assert_not_called()

    def test_download_rejects_onnx_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            def fake_download(**kwargs):
                path = cache_dir / kwargs["filename"]
                path.write_bytes(b"bad" if path.suffix == ".onnx" else b"yaml")
                return str(path)

            with (
                patch("download_vocal_harmony_model.POLARFORMER_ONNX_SIZE", 3),
                patch("download_vocal_harmony_model.POLARFORMER_ONNX_SHA256", "0" * 64),
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    download_accompaniment_model(
                        cache_dir=cache_dir,
                        downloader=fake_download,
                        printer=lambda *_: None,
                    )

    def test_download_rejects_old_karaoke_preset(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unsupported accompaniment model"):
                download_accompaniment_model(
                    cache_dir=Path(tmp),
                    model_name="ensemble:karaoke",
                    downloader=Mock(),
                    printer=lambda *_: None,
                )


if __name__ == "__main__":
    unittest.main()
