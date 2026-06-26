import tempfile
import unittest
from pathlib import Path

from download_vocal_model import (
    DEFAULT_CACHE_DIR,
    ROFORMER_MODEL,
    ROFORMER_MODELS,
    ROFORMER_PRESET,
    download_vocal_model,
    is_vocal_model_available,
    resolve_vocal_model_path,
    resolve_vocal_model_paths,
)


class TestDownloadVocalModel(unittest.TestCase):
    def test_default_cache_dir_is_under_user_home(self):
        self.assertIn(".music-to-midi", str(DEFAULT_CACHE_DIR))
        self.assertIn("audio-separator", str(DEFAULT_CACHE_DIR))

    def test_resolve_vocal_model_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            self.assertEqual(resolve_vocal_model_path(cache_dir), cache_dir / ROFORMER_MODELS[0])
            self.assertEqual(
                resolve_vocal_model_paths(cache_dir),
                tuple(cache_dir / model_name for model_name in ROFORMER_MODELS),
            )

    def test_is_vocal_model_available_requires_all_ensemble_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            self.assertFalse(is_vocal_model_available(cache_dir))

            (cache_dir / ROFORMER_MODELS[0]).write_bytes(b"ok")
            self.assertFalse(is_vocal_model_available(cache_dir))

            (cache_dir / ROFORMER_MODELS[1]).write_bytes(b"ok")
            self.assertTrue(is_vocal_model_available(cache_dir))

    def test_download_skips_when_ensemble_already_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            for model_name in ROFORMER_MODELS:
                (cache_dir / model_name).write_bytes(b"already-downloaded")

            class ShouldNotBeCalledSeparator:
                def __init__(self, *args, **kwargs):
                    raise AssertionError("Separator should not be initialized")

            result = download_vocal_model(
                cache_dir=cache_dir,
                separator_cls=ShouldNotBeCalledSeparator,
                printer=lambda *_: None,
            )
            self.assertEqual(result, cache_dir / ROFORMER_MODELS[0])

    def test_download_invokes_ensemble_preset_and_returns_first_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            seen = {}

            class FakeSeparator:
                def __init__(self, output_dir, model_file_dir, output_format, ensemble_preset=None):
                    seen["output_dir"] = output_dir
                    seen["model_file_dir"] = model_file_dir
                    seen["output_format"] = output_format
                    seen["ensemble_preset"] = ensemble_preset

                def load_model(self):
                    nested = Path(seen["model_file_dir"]) / "nested"
                    nested.mkdir(parents=True, exist_ok=True)
                    for model_name in ROFORMER_MODELS:
                        (nested / model_name).write_bytes(b"x" * 16)

            logs = []
            result = download_vocal_model(
                cache_dir=cache_dir,
                separator_cls=FakeSeparator,
                printer=logs.append,
            )

            self.assertTrue(result.exists())
            self.assertEqual(result.name, ROFORMER_MODELS[0])
            self.assertEqual(seen["ensemble_preset"], ROFORMER_PRESET)
            self.assertTrue(any("模型就绪" in line for line in logs))

    def test_download_rejects_old_separator_without_ensemble_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            class OldSeparator:
                def __init__(self, output_dir, model_file_dir, output_format):
                    self.output_dir = output_dir
                    self.model_file_dir = model_file_dir
                    self.output_format = output_format

                def load_model(self, model_name):
                    raise AssertionError(model_name)

            with self.assertRaisesRegex(RuntimeError, "ensemble_preset"):
                download_vocal_model(
                    cache_dir=cache_dir,
                    model_name=ROFORMER_MODEL,
                    separator_cls=OldSeparator,
                    printer=lambda *_: None,
                )

    def test_download_raises_if_required_checkpoint_still_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            class BrokenSeparator:
                def __init__(self, output_dir, model_file_dir, output_format, ensemble_preset=None):
                    self.model_file_dir = model_file_dir

                def load_model(self):
                    nested = Path(self.model_file_dir) / "nested"
                    nested.mkdir(parents=True, exist_ok=True)
                    (nested / ROFORMER_MODELS[0]).write_bytes(b"x")

            with self.assertRaisesRegex(RuntimeError, ROFORMER_MODELS[1]):
                download_vocal_model(
                    cache_dir=cache_dir,
                    separator_cls=BrokenSeparator,
                    printer=lambda *_: None,
                )


if __name__ == "__main__":
    unittest.main()
