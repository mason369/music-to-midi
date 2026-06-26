import tempfile
import unittest
from pathlib import Path

from download_vocal_harmony_model import (
    CHORUS_MODEL,
    CHORUS_MODELS,
    CHORUS_PRESET,
    DEFAULT_CACHE_DIR,
    download_chorus_model,
    is_chorus_model_available,
    resolve_chorus_model_path,
    resolve_chorus_model_paths,
)


class TestDownloadVocalHarmonyModel(unittest.TestCase):
    def test_default_cache_dir_is_under_user_home(self):
        self.assertIn(".music-to-midi", str(DEFAULT_CACHE_DIR))
        self.assertIn("audio-separator", str(DEFAULT_CACHE_DIR))

    def test_resolve_chorus_model_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            self.assertEqual(resolve_chorus_model_path(cache_dir), cache_dir / CHORUS_MODELS[0])
            self.assertEqual(
                resolve_chorus_model_paths(cache_dir),
                tuple(cache_dir / model_name for model_name in CHORUS_MODELS),
            )

    def test_is_chorus_model_available_requires_all_ensemble_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            self.assertFalse(is_chorus_model_available(cache_dir))

            for model_name in CHORUS_MODELS[:-1]:
                (cache_dir / model_name).write_bytes(b"ok")
            self.assertFalse(is_chorus_model_available(cache_dir))

            (cache_dir / CHORUS_MODELS[-1]).write_bytes(b"ok")
            self.assertTrue(is_chorus_model_available(cache_dir))

    def test_download_invokes_karaoke_preset_and_returns_first_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            seen = {}

            class FakeSeparator:
                def __init__(self, output_dir, model_file_dir, output_format, ensemble_preset=None):
                    seen["model_file_dir"] = model_file_dir
                    seen["ensemble_preset"] = ensemble_preset
                    seen["downloaded"] = []

                def download_model_and_data(self, model_name):
                    seen["downloaded"].append(model_name)
                    nested = Path(seen["model_file_dir"]) / "nested"
                    nested.mkdir(parents=True, exist_ok=True)
                    (nested / model_name).write_bytes(b"x" * 16)

            result = download_chorus_model(
                cache_dir=cache_dir,
                separator_cls=FakeSeparator,
                printer=lambda *_: None,
            )

            self.assertTrue(result.exists())
            self.assertEqual(result.name, CHORUS_MODELS[0])
            self.assertEqual(seen["ensemble_preset"], CHORUS_PRESET)
            self.assertEqual(seen["downloaded"], list(CHORUS_MODELS))

    def test_download_rejects_old_separator_without_ensemble_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            class OldSeparator:
                def __init__(self, output_dir, model_file_dir, output_format):
                    pass

                def load_model(self, model_name):
                    raise AssertionError(model_name)

            with self.assertRaisesRegex(RuntimeError, "ensemble_preset"):
                download_chorus_model(
                    cache_dir=cache_dir,
                    model_name=CHORUS_MODEL,
                    separator_cls=OldSeparator,
                    printer=lambda *_: None,
                )

    def test_download_raises_if_required_checkpoint_still_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            class BrokenSeparator:
                def __init__(self, output_dir, model_file_dir, output_format, ensemble_preset=None):
                    self.model_file_dir = model_file_dir

                def download_model_and_data(self, model_name):
                    nested = Path(self.model_file_dir) / "nested"
                    nested.mkdir(parents=True, exist_ok=True)
                    if model_name == CHORUS_MODELS[0]:
                        (nested / model_name).write_bytes(b"x")

            with self.assertRaisesRegex(RuntimeError, CHORUS_MODELS[1]):
                download_chorus_model(
                    cache_dir=cache_dir,
                    separator_cls=BrokenSeparator,
                    printer=lambda *_: None,
                )

    def test_download_rejects_separator_without_download_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            class LoadOnlySeparator:
                def __init__(self, output_dir, model_file_dir, output_format, ensemble_preset=None):
                    pass

                def load_model(self):
                    raise AssertionError("load_model must not be used for downloads")

            with self.assertRaisesRegex(RuntimeError, "download_model_and_data"):
                download_chorus_model(
                    cache_dir=cache_dir,
                    separator_cls=LoadOnlySeparator,
                    printer=lambda *_: None,
                )


if __name__ == "__main__":
    unittest.main()
