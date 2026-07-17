import tempfile
import unittest
from pathlib import Path

from src.core.multi_stem_separator import STEM_KEYS, SixStemSeparator


class SixStemSeparatorStrictOutputTests(unittest.TestCase):
    def _separator(self, cache_dir: Path) -> SixStemSeparator:
        return SixStemSeparator(separator_cls=object, cache_dir=cache_dir)

    def test_empty_return_does_not_scan_preexisting_wav_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "old_song_bass.wav").write_bytes(b"old")

            with self.assertRaisesRegex(RuntimeError, "returned no output files"):
                self._separator(output_dir)._normalize_outputs(
                    "song.wav",
                    output_dir,
                    [],
                )

    def test_returned_file_must_exist_inside_current_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            output_dir.mkdir()
            outside_path = Path(tmp) / "song_bass.wav"
            outside_path.write_bytes(b"outside")

            with self.assertRaisesRegex(RuntimeError, "outside the current output directory"):
                self._separator(output_dir)._normalize_outputs(
                    "song.wav",
                    output_dir,
                    [str(outside_path)],
                )

    def test_exact_six_returned_stems_are_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            returned = []
            for stem in STEM_KEYS:
                path = output_dir / f"song_({stem})_BS-Rofo-SW-Fixed.wav"
                path.write_bytes(stem.encode("ascii"))
                returned.append(str(path))

            result = self._separator(output_dir)._normalize_outputs(
                "song.wav",
                output_dir,
                returned,
            )

            self.assertEqual(set(result), set(STEM_KEYS))
            for stem in STEM_KEYS:
                self.assertEqual(Path(result[stem]), output_dir / f"song_{stem}.wav")
                self.assertTrue(Path(result[stem]).is_file())

    def test_source_name_stem_words_do_not_override_separator_stem_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            input_stem = "piano vocals bass guitar drums other demo"
            returned = []
            for stem in STEM_KEYS:
                path = output_dir / f"{input_stem}_({stem})_BS-Rofo-SW-Fixed.wav"
                path.write_bytes(stem.encode("ascii"))
                returned.append(str(path))

            result = self._separator(output_dir)._normalize_outputs(
                f"{input_stem}.mp3",
                output_dir,
                returned,
            )

            self.assertEqual(set(result), set(STEM_KEYS))
            for stem in STEM_KEYS:
                self.assertEqual(Path(result[stem]), output_dir / f"{input_stem}_{stem}.wav")
                self.assertEqual(Path(result[stem]).read_bytes(), stem.encode("ascii"))


if __name__ == "__main__":
    unittest.main()
