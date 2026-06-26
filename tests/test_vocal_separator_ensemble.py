import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core import vocal_separator
from src.core.vocal_separator import (
    KARAOKE_REQUIRED_MODELS,
    ROFORMER_REQUIRED_MODELS,
    VocalSeparator,
    _move_role_output,
)


class VocalSeparatorEnsembleTests(unittest.TestCase):
    def test_model_available_requires_vocal_rvc_and_karaoke_ensembles(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            with patch("src.core.vocal_separator.get_audio_separator_model_dir", return_value=cache_dir):
                self.assertFalse(VocalSeparator.is_model_available())

                for model_name in ROFORMER_REQUIRED_MODELS:
                    (cache_dir / model_name).write_bytes(b"ok")
                self.assertFalse(VocalSeparator.is_model_available())

                for model_name in KARAOKE_REQUIRED_MODELS:
                    (cache_dir / model_name).write_bytes(b"ok")
                self.assertTrue(VocalSeparator.is_model_available())

    def test_role_output_detection_fails_instead_of_order_guessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            unknown_one = out_dir / "song_part_a.wav"
            unknown_two = out_dir / "song_part_b.wav"
            unknown_one.write_bytes(b"wav")
            unknown_two.write_bytes(b"wav")

            with self.assertRaisesRegex(RuntimeError, "输出角色识别失败"):
                _move_role_output(
                    [unknown_one.name, unknown_two.name],
                    out_dir,
                    vocal_separator._classify_vocal_rvc_output,
                    {
                        "vocals": out_dir / "song_vocals.wav",
                        "accompaniment": out_dir / "song_accompaniment.wav",
                    },
                )


if __name__ == "__main__":
    unittest.main()
