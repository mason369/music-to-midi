import tempfile
import unittest
from pathlib import Path

from src.core.multi_stem_separator import (
    BS_ROFORMER_SW_MODEL,
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

    def test_raises_when_bs_roformer_sw_load_fails(self):
        """BS-RoFormer SW 加载失败必须直接报错，不静默回退。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / 'song.wav'
            audio_path.write_bytes(b'audio')
            out_dir = tmp_path / 'out'

            class FailingSeparator:
                def __init__(self, output_dir, model_file_dir, output_format):
                    pass

                def load_model(self, model_name):
                    raise ValueError('Unknown Roformer model type')

                def separate(self, _audio_path):
                    raise AssertionError('should not reach separate()')

            sep = SixStemSeparator(
                separator_cls=FailingSeparator,
                ensure_assets_fn=lambda *_args, **_kwargs: None,
            )
            with self.assertRaises(ValueError):
                sep.separate(str(audio_path), str(out_dir))

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
            )
            with self.assertRaises(RuntimeError):
                sep.separate(str(audio_path), str(out_dir))


if __name__ == '__main__':
    unittest.main()
