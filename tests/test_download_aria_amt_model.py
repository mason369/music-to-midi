import tempfile
import unittest
from pathlib import Path

from download_aria_amt_model import (
    ARIA_AMT_CHECKPOINT_NAME,
    download_aria_model,
    is_aria_model_available,
    resolve_aria_model_path,
)


class TestDownloadAriaAmtModel(unittest.TestCase):
    def test_resolve_aria_model_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            model_path = resolve_aria_model_path(cache_dir)
            self.assertEqual(model_path, cache_dir / ARIA_AMT_CHECKPOINT_NAME)

    def test_is_aria_model_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            self.assertFalse(is_aria_model_available(cache_dir))

            model_path = resolve_aria_model_path(cache_dir)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"x")
            self.assertTrue(is_aria_model_available(cache_dir))

    def test_download_aria_model_skips_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            model_path = resolve_aria_model_path(cache_dir)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"exists")

            calls = []

            def fake_downloader(*_args, **_kwargs):
                calls.append("called")

            result = download_aria_model(
                cache_dir=cache_dir,
                downloader=fake_downloader,
                printer=lambda *_: None,
            )
            self.assertEqual(result, model_path)
            self.assertEqual(calls, [])

    def test_download_aria_model_invokes_downloader(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            calls = []

            def fake_downloader(url, output_path):
                calls.append((url, Path(output_path).name))
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"weights")

            model_path = download_aria_model(
                cache_dir=cache_dir,
                downloader=fake_downloader,
                printer=lambda *_: None,
            )
            self.assertTrue(model_path.exists())
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], ARIA_AMT_CHECKPOINT_NAME)


if __name__ == "__main__":
    unittest.main()

