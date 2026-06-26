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

    def test_download_sota_models_prepares_six_stem_and_vocal_assets(self):
        yourmt3_result = {"yptf_moe_multi_nops": Path("/tmp/yourmt3/model.ckpt")}
        six_stem_result = (
            Path("/tmp/audio-separator/BS-Rofo-SW-Fixed.ckpt"),
            Path("/tmp/audio-separator/BS-Rofo-SW-Fixed.yaml"),
        )
        vocal_result = Path("/tmp/audio-separator/melband_roformer_big_beta6x.ckpt")
        karaoke_result = Path("/tmp/audio-separator/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")

        with (
            mock.patch.object(
                download_sota_models,
                "download_official_yourmt3_models",
                return_value=yourmt3_result,
            ) as yourmt3_mock,
            mock.patch.object(
                download_sota_models,
                "download_multistem_model",
                return_value=six_stem_result,
            ) as multistem_mock,
            mock.patch.object(
                download_sota_models,
                "download_vocal_model",
                return_value=vocal_result,
            ) as vocal_mock,
            mock.patch.object(
                download_sota_models,
                "download_chorus_model",
                return_value=karaoke_result,
            ) as karaoke_mock,
        ):
            result = download_sota_models.download_sota_models()

        yourmt3_mock.assert_called_once_with()
        multistem_mock.assert_called_once_with()
        vocal_mock.assert_called_once_with()
        karaoke_mock.assert_called_once_with()
        self.assertEqual(result["yourmt3"], yourmt3_result)
        self.assertEqual(result["six_stem"]["model"], six_stem_result[0])
        self.assertEqual(result["six_stem"]["config"], six_stem_result[1])
        self.assertEqual(result["vocal_rvc"]["primary_model"], vocal_result)
        self.assertEqual(result["karaoke"]["primary_model"], karaoke_result)


if __name__ == "__main__":
    unittest.main()
