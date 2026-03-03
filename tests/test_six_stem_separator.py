import tempfile
import unittest
from pathlib import Path

from src.core.multi_stem_separator import (
    BS_ROFORMER_SW_MODEL,
    HTDEMUCS_6S_MODEL,
    STEM_KEYS,
    SixStemSeparator,
)


class TestSixStemSeparator(unittest.TestCase):
    def test_separate_returns_all_stems_with_normalized_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / 'song.wav'
            audio_path.write_bytes(b'audio')
            out_dir = tmp_path / 'out'

            class FakeSeparator:
                loaded_models = []

                def __init__(self, output_dir, model_file_dir, output_format):
                    self.output_dir = Path(output_dir)

                def load_model(self, model_name):
                    self.__class__.loaded_models.append(model_name)

                def separate(self, _audio_path):
                    files = [
                        self.output_dir / 'song_(Vocals).wav',
                        self.output_dir / 'song_(Drums).wav',
                        self.output_dir / 'song_(Bass).wav',
                        self.output_dir / 'song_(Guitar).wav',
                        self.output_dir / 'song_(Piano).wav',
                        self.output_dir / 'song_(Other).wav',
                    ]
                    for path in files:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b'wav')
                    return [str(path) for path in files]

            sep = SixStemSeparator(
                separator_cls=FakeSeparator,
                ensure_assets_fn=lambda *_args, **_kwargs: None,
            )
            result = sep.separate(str(audio_path), str(out_dir))

            self.assertEqual(set(result.keys()), set(STEM_KEYS))
            for stem_key, stem_path in result.items():
                path = Path(stem_path)
                self.assertTrue(path.exists())
                self.assertTrue(path.name.endswith(f'_{stem_key}.wav'))

            self.assertEqual(FakeSeparator.loaded_models[0], BS_ROFORMER_SW_MODEL)

    def test_fallback_to_htdemucs_when_primary_load_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / 'song.wav'
            audio_path.write_bytes(b'audio')
            out_dir = tmp_path / 'out'

            class FakeSeparator:
                loaded_models = []

                def __init__(self, output_dir, model_file_dir, output_format):
                    self.output_dir = Path(output_dir)

                def load_model(self, model_name):
                    self.__class__.loaded_models.append(model_name)
                    if model_name == BS_ROFORMER_SW_MODEL:
                        raise ValueError('model not found')

                def separate(self, _audio_path):
                    files = [
                        self.output_dir / 'song_vocals.wav',
                        self.output_dir / 'song_drums.wav',
                        self.output_dir / 'song_bass.wav',
                        self.output_dir / 'song_guitar.wav',
                        self.output_dir / 'song_piano.wav',
                        self.output_dir / 'song_other.wav',
                    ]
                    for path in files:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b'wav')
                    return [str(path) for path in files]

            sep = SixStemSeparator(
                separator_cls=FakeSeparator,
                ensure_assets_fn=lambda *_args, **_kwargs: None,
                allow_fallback=True,
            )
            result = sep.separate(str(audio_path), str(out_dir))

            self.assertEqual(set(result.keys()), set(STEM_KEYS))
            self.assertEqual(FakeSeparator.loaded_models[0], BS_ROFORMER_SW_MODEL)
            self.assertIn(HTDEMUCS_6S_MODEL, FakeSeparator.loaded_models)

    def test_raise_when_stems_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / 'song.wav'
            audio_path.write_bytes(b'audio')
            out_dir = tmp_path / 'out'

            class BrokenSeparator:
                def __init__(self, output_dir, model_file_dir, output_format):
                    self.output_dir = Path(output_dir)

                def load_model(self, model_name):
                    return None

                def separate(self, _audio_path):
                    files = [
                        self.output_dir / 'song_vocals.wav',
                        self.output_dir / 'song_other.wav',
                    ]
                    for path in files:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b'wav')
                    return [str(path) for path in files]

            sep = SixStemSeparator(
                separator_cls=BrokenSeparator,
                ensure_assets_fn=lambda *_args, **_kwargs: None,
                allow_fallback=False,
            )
            with self.assertRaises(RuntimeError):
                sep.separate(str(audio_path), str(out_dir))


if __name__ == '__main__':
    unittest.main()
