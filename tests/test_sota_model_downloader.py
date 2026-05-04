import unittest
from pathlib import Path
from unittest import mock

import download_sota_models
from src.utils.yourmt3_downloader import OFFICIAL_YOURMT3_MODEL_KEYS


class SotaModelDownloaderTests(unittest.TestCase):
    def test_downloads_every_official_selectable_yourmt3_model(self):
        calls = []

        def fake_download_model(model_key):
            calls.append(model_key)
            return Path("/tmp") / model_key / "model.ckpt"

        def fake_get_model_path(model_key):
            return Path("/tmp") / model_key / "model.ckpt"

        with (
            mock.patch.object(download_sota_models, "download_model", side_effect=fake_download_model),
            mock.patch.object(download_sota_models, "get_model_path", side_effect=fake_get_model_path),
        ):
            download_sota_models.download_official_yourmt3_models()

        self.assertEqual(calls, list(OFFICIAL_YOURMT3_MODEL_KEYS))


if __name__ == "__main__":
    unittest.main()
