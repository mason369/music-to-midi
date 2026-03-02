import tempfile
import unittest
from pathlib import Path

from download_vocal_model import (
    DEFAULT_CACHE_DIR,
    ROFORMER_MODEL,
    download_vocal_model,
    is_vocal_model_available,
    resolve_vocal_model_path,
)


class TestDownloadVocalModel(unittest.TestCase):
    def test_default_cache_dir_is_under_user_home(self):
        self.assertIn(".music-to-midi", str(DEFAULT_CACHE_DIR))
        self.assertIn("audio-separator", str(DEFAULT_CACHE_DIR))

    def test_resolve_vocal_model_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            expected = cache_dir / ROFORMER_MODEL
            self.assertEqual(resolve_vocal_model_path(cache_dir), expected)

    def test_is_vocal_model_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            self.assertFalse(is_vocal_model_available(cache_dir))
            model_file = cache_dir / ROFORMER_MODEL
            model_file.write_bytes(b"ok")
            self.assertTrue(is_vocal_model_available(cache_dir))

    def test_download_skips_when_model_already_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            existing = cache_dir / ROFORMER_MODEL
            existing.write_bytes(b"already-downloaded")

            class ShouldNotBeCalledSeparator:
                def __init__(self, *args, **kwargs):
                    raise AssertionError("Separator should not be initialized")

            result = download_vocal_model(
                cache_dir=cache_dir,
                separator_cls=ShouldNotBeCalledSeparator,
                printer=lambda *_: None,
            )
            self.assertEqual(result, existing)

    def test_download_invokes_separator_and_returns_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            class FakeSeparator:
                def __init__(self, output_dir, model_file_dir, output_format):
                    self.output_dir = output_dir
                    self.model_file_dir = model_file_dir
                    self.output_format = output_format

                def load_model(self, model_name):
                    nested = Path(self.model_file_dir) / "nested"
                    nested.mkdir(parents=True, exist_ok=True)
                    (nested / model_name).write_bytes(b"x" * 16)

            logs = []
            result = download_vocal_model(
                cache_dir=cache_dir,
                separator_cls=FakeSeparator,
                printer=logs.append,
            )

            self.assertTrue(result.exists())
            self.assertEqual(result.name, ROFORMER_MODEL)
            self.assertTrue(any("下载成功" in line for line in logs))

    def test_download_raises_if_checkpoint_still_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            class BrokenSeparator:
                def __init__(self, output_dir, model_file_dir, output_format):
                    self.output_dir = output_dir
                    self.model_file_dir = model_file_dir
                    self.output_format = output_format

                def load_model(self, model_name):
                    return None

            with self.assertRaises(RuntimeError):
                download_vocal_model(
                    cache_dir=cache_dir,
                    separator_cls=BrokenSeparator,
                    printer=lambda *_: None,
                )


if __name__ == "__main__":
    unittest.main()
