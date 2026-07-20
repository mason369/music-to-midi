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
            mock.patch.object(
                download_sota_models, "download_model", side_effect=fake_download_model
            ),
            mock.patch.object(
                download_sota_models, "get_model_path", side_effect=fake_get_model_path
            ),
        ):
            download_sota_models.download_official_yourmt3_models()

        self.assertEqual(calls, list(OFFICIAL_YOURMT3_MODEL_KEYS))

    def test_download_sota_models_prepares_six_stem_and_vocal_assets(self):
        transkun_result = {
            "package": "transkun",
            "version": "2.0.1",
            "weight_name": "2.0.pt",
            "config_name": "2.0.conf",
        }
        yourmt3_result = {"yptf_moe_multi_nops": Path("/tmp/yourmt3/model.ckpt")}
        miros_result = Path("/tmp/external/ai4m-miros")
        six_stem_result = (
            Path("/tmp/audio-separator/BS-Rofo-SW-Fixed.ckpt"),
            Path("/tmp/audio-separator/BS-Rofo-SW-Fixed.yaml"),
        )
        vocal_result = Path("/tmp/audio-separator/melband_roformer_big_beta6x.ckpt")
        karaoke_result = Path(
            "/tmp/audio-separator/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
        )
        transkun_v2_aug_result = Path("/tmp/transkun-v2-aug/checkpointMSimplerAug")
        aria_result = Path("/tmp/aria-amt/piano-medium-double-1.0.safetensors")
        bytedance_result = Path("/tmp/bytedance/note_F1=0.9677_pedal_F1=0.9186.pth")
        muscriptor_result = (
            Path("/tmp/muscriptor/model.safetensors"),
            Path("/tmp/muscriptor/config.json"),
        )
        soundfont_result = Path("/tmp/muscriptor-assets/MuseScore_General.sf2")
        fluidsynth_result = Path("/tmp/fluidsynth/bin/fluidsynth")

        with (
            mock.patch.object(
                download_sota_models,
                "validate_default_transkun_runtime",
                return_value=transkun_result,
            ) as transkun_mock,
            mock.patch.object(
                download_sota_models,
                "download_official_yourmt3_models",
                return_value=yourmt3_result,
            ) as yourmt3_mock,
            mock.patch.object(
                download_sota_models,
                "prepare_miros_model",
                return_value=miros_result,
            ) as miros_mock,
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
            mock.patch.object(
                download_sota_models,
                "download_transkun_v2_aug_model",
                return_value=transkun_v2_aug_result,
            ) as transkun_v2_aug_mock,
            mock.patch.object(
                download_sota_models,
                "download_aria_model",
                return_value=aria_result,
            ) as aria_mock,
            mock.patch.object(
                download_sota_models,
                "download_bytedance_piano_model",
                return_value=bytedance_result,
            ) as bytedance_mock,
            mock.patch.object(
                download_sota_models,
                "download_muscriptor_large_model",
                return_value=muscriptor_result,
            ) as muscriptor_mock,
            mock.patch.object(
                download_sota_models,
                "download_muscriptor_soundfont",
                return_value=soundfont_result,
            ) as soundfont_mock,
            mock.patch.object(
                download_sota_models,
                "download_fluidsynth_windows",
                return_value=fluidsynth_result,
            ) as fluidsynth_mock,
        ):
            result = download_sota_models.download_sota_models()

        transkun_mock.assert_called_once_with()
        yourmt3_mock.assert_called_once_with()
        miros_mock.assert_called_once_with()
        multistem_mock.assert_called_once_with()
        vocal_mock.assert_called_once_with()
        karaoke_mock.assert_called_once_with()
        transkun_v2_aug_mock.assert_called_once_with()
        aria_mock.assert_called_once_with()
        bytedance_mock.assert_called_once_with()
        muscriptor_mock.assert_called_once_with()
        soundfont_mock.assert_called_once_with()
        fluidsynth_mock.assert_called_once_with()
        self.assertEqual(result["transkun"], transkun_result)
        self.assertEqual(result["yourmt3"], yourmt3_result)
        self.assertEqual(result["miros"]["repo_dir"], miros_result)
        self.assertEqual(result["six_stem"]["model"], six_stem_result[0])
        self.assertEqual(result["six_stem"]["config"], six_stem_result[1])
        self.assertEqual(result["vocal_rvc"]["primary_model"], vocal_result)
        self.assertEqual(result["karaoke"]["primary_model"], karaoke_result)
        self.assertEqual(
            result["transkun_v2_aug"]["model_dir"],
            transkun_v2_aug_result,
        )
        self.assertEqual(result["aria_amt"]["checkpoint"], aria_result)
        self.assertEqual(
            result["bytedance_piano"]["checkpoint"],
            bytedance_result,
        )
        self.assertEqual(result["muscriptor"]["weights"], muscriptor_result[0])
        self.assertEqual(result["muscriptor"]["config"], muscriptor_result[1])
        self.assertEqual(result["muscriptor"]["soundfont"], soundfont_result)
        self.assertEqual(result["muscriptor"]["fluidsynth"], fluidsynth_result)

    def test_default_transkun_validation_rejects_unavailable_or_wrong_package(self):
        reason = "Transkun package version mismatch: expected 2.0.1, got 2.0.0"
        with mock.patch.object(
            download_sota_models.TranskunTranscriber,
            "get_unavailable_reason",
            return_value=reason,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Default TransKun 2.0.1 runtime validation failed.*version mismatch",
            ):
                download_sota_models.validate_default_transkun_runtime()

    def test_default_transkun_validation_rejects_invalid_packaged_resources(self):
        with mock.patch.object(
            download_sota_models,
            "TranskunTranscriber",
        ) as transkun_class:
            transkun_class.get_unavailable_reason.return_value = ""
            transkun_class.return_value.is_model_available.return_value = False

            with self.assertRaisesRegex(
                RuntimeError,
                "packaged resources are missing or failed exact identity validation",
            ) as raised:
                download_sota_models.validate_default_transkun_runtime()

        error = str(raised.exception)
        self.assertIn(download_sota_models.TRANSKUN_WEIGHT_SHA256, error)
        self.assertIn(download_sota_models.TRANSKUN_CONF_SHA256, error)

    def test_default_transkun_validation_reports_exact_runtime_identity(self):
        with mock.patch.object(
            download_sota_models,
            "TranskunTranscriber",
        ) as transkun_class:
            transkun_class.get_unavailable_reason.return_value = ""
            transkun_class.return_value.is_model_available.return_value = True

            result = download_sota_models.validate_default_transkun_runtime()

        self.assertEqual(result["package"], "transkun")
        self.assertEqual(result["version"], "2.0.1")
        self.assertEqual(result["weight_name"], "2.0.pt")
        self.assertEqual(result["weight_sha256"], download_sota_models.TRANSKUN_WEIGHT_SHA256)
        self.assertEqual(result["config_name"], "2.0.conf")
        self.assertEqual(result["config_sha256"], download_sota_models.TRANSKUN_CONF_SHA256)

    def test_download_sota_models_stops_on_default_transkun_validation_failure(self):
        failure = RuntimeError("default TransKun identity mismatch")
        with (
            mock.patch.object(
                download_sota_models,
                "validate_default_transkun_runtime",
                side_effect=failure,
            ),
            mock.patch.object(
                download_sota_models,
                "download_official_yourmt3_models",
            ) as yourmt3_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                download_sota_models.download_sota_models()

        yourmt3_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
