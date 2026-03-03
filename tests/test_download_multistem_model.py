import tempfile
import unittest
from pathlib import Path

from download_multistem_model import (
    DEFAULT_CACHE_DIR,
    ROFORMER_SW_CONFIG,
    ROFORMER_SW_MODEL,
    download_multistem_model,
    is_multistem_model_available,
    resolve_multistem_model_paths,
)


class TestDownloadMultistemModel(unittest.TestCase):
    def test_default_cache_dir_is_under_user_home(self):
        self.assertIn('.music-to-midi', str(DEFAULT_CACHE_DIR))
        self.assertIn('audio-separator', str(DEFAULT_CACHE_DIR))

    def test_resolve_multistem_model_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            model_path, config_path = resolve_multistem_model_paths(cache_dir)
            self.assertEqual(model_path, cache_dir / ROFORMER_SW_MODEL)
            self.assertEqual(config_path, cache_dir / ROFORMER_SW_CONFIG)

    def test_is_multistem_model_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            self.assertFalse(is_multistem_model_available(cache_dir))

            model_path, config_path = resolve_multistem_model_paths(cache_dir)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b'x')
            config_path.write_text('ok', encoding='utf-8')

            self.assertTrue(is_multistem_model_available(cache_dir))

    def test_download_skips_when_assets_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            model_path, config_path = resolve_multistem_model_paths(cache_dir)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b'exists')
            config_path.write_text('exists', encoding='utf-8')

            calls = []

            def fake_downloader(*_args, **_kwargs):
                calls.append('called')

            result_model, result_config = download_multistem_model(
                cache_dir=cache_dir,
                downloader=fake_downloader,
                printer=lambda *_: None,
            )

            self.assertEqual(result_model, model_path)
            self.assertEqual(result_config, config_path)
            self.assertEqual(calls, [])

    def test_download_invokes_downloader_for_model_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            calls = []

            def fake_downloader(url, output_path, description):
                calls.append((url, Path(output_path).name, description))
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == '.ckpt':
                    path.write_bytes(b'ckpt-data')
                else:
                    path.write_text('yaml-data', encoding='utf-8')

            model_path, config_path = download_multistem_model(
                cache_dir=cache_dir,
                downloader=fake_downloader,
                printer=lambda *_: None,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue(config_path.exists())
            self.assertEqual(len(calls), 2)
            names = {name for _url, name, _desc in calls}
            self.assertIn(ROFORMER_SW_MODEL, names)
            self.assertIn(ROFORMER_SW_CONFIG, names)


if __name__ == '__main__':
    unittest.main()
