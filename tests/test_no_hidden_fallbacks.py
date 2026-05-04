import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import math

from src.core.pipeline import MusicToMidiPipeline
from src.core.beat_detector import BeatDetector
from src.models.data_models import Config, NoteEvent
from src.utils import yourmt3_downloader


class NoHiddenFallbackTests(unittest.TestCase):
    def test_ensure_wav_reports_ffmpeg_failure_without_librosa_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "song.mp3"
            audio_path.write_bytes(b"not real audio")

            with patch("src.core.pipeline.get_ffmpeg_executable", return_value="ffmpeg"), patch(
                "src.core.pipeline.subprocess.run",
                side_effect=subprocess.CalledProcessError(
                    returncode=1,
                    cmd=["ffmpeg"],
                    stderr="invalid data found",
                ),
            ), patch.dict("sys.modules", {"librosa": None}):
                with self.assertRaisesRegex(RuntimeError, "(?s)FFmpeg.*invalid data found"):
                    MusicToMidiPipeline._ensure_wav(str(audio_path), str(tmp_path / "out"))

    def test_detect_beat_or_raise_reports_detector_failure(self):
        pipeline = MusicToMidiPipeline(Config())
        pipeline.beat_detector.detect = lambda _audio_path: (_ for _ in ()).throw(
            RuntimeError("beat backend unavailable")
        )

        with self.assertRaisesRegex(RuntimeError, "节拍检测失败.*beat backend unavailable"):
            pipeline._detect_beat_or_raise("song.wav")

    def test_detect_beat_or_raise_rejects_missing_beat_info(self):
        pipeline = MusicToMidiPipeline(Config())
        pipeline.beat_detector.detect = lambda _audio_path: None

        with self.assertRaisesRegex(RuntimeError, "未返回 BPM"):
            pipeline._detect_beat_or_raise("song.wav")

    def test_beat_detector_raises_when_all_methods_fail(self):
        detector = BeatDetector(Config())

        with self.assertRaisesRegex(RuntimeError, "所有 BPM 检测方法均失败"):
            detector._detect_multi_method([], 22050)

    def test_beat_detector_rejects_invalid_tempo_values(self):
        detector = BeatDetector(Config())

        for tempo in (0.0, -1.0, math.nan):
            with self.subTest(tempo=tempo):
                with self.assertRaisesRegex(RuntimeError, "无效 BPM"):
                    detector._correct_octave_error(tempo)

    def test_beat_detector_empty_primary_tempo_does_not_become_120(self):
        detector = BeatDetector(Config())

        class EmptyBeat:
            @staticmethod
            def beat_track(**_kwargs):
                return [], []

            @staticmethod
            def tempo(**_kwargs):
                raise RuntimeError("tempo unavailable")

        fake_librosa = type(
            "FakeLibrosa",
            (),
            {
                "beat": EmptyBeat,
                "onset": type(
                    "Onset",
                    (),
                    {"onset_strength": staticmethod(lambda **_kwargs: [])},
                ),
                "feature": type(
                    "Feature",
                    (),
                    {"tempogram": staticmethod(lambda **_kwargs: [])},
                ),
                "tempo_frequencies": staticmethod(lambda *_args, **_kwargs: []),
            },
        )

        with patch.dict("sys.modules", {"librosa": fake_librosa}):
            with self.assertRaisesRegex(RuntimeError, "所有 BPM 检测方法均失败"):
                detector._detect_multi_method([], 22050)

    def test_beat_detector_uses_current_librosa_tempo_api(self):
        detector = BeatDetector(Config())

        class Beat:
            @staticmethod
            def beat_track(**_kwargs):
                return 120.0, []

            @staticmethod
            def tempo(**_kwargs):
                raise AssertionError("deprecated librosa.beat.tempo should not be used")

        class Rhythm:
            @staticmethod
            def tempo(**_kwargs):
                return [118.0, 120.0, 122.0]

        fake_librosa = type(
            "FakeLibrosa",
            (),
            {
                "beat": Beat,
                "onset": type(
                    "Onset",
                    (),
                    {"onset_strength": staticmethod(lambda **_kwargs: [0.0, 1.0, 0.0])},
                ),
                "feature": type(
                    "Feature",
                    (),
                    {
                        "rhythm": Rhythm,
                        "tempogram": staticmethod(lambda **_kwargs: [[0.0], [1.0], [0.0]]),
                    },
                ),
                "tempo_frequencies": staticmethod(lambda *_args, **_kwargs: [60.0, 120.0, 240.0]),
            },
        )

        with patch.dict("sys.modules", {"librosa": fake_librosa}):
            tempo, candidates = detector._detect_multi_method([0.0, 1.0, 0.0], 22050)

        self.assertGreater(tempo, 0)
        self.assertIn(120.0, candidates)

    def test_vocal_filter_does_not_guess_from_largest_non_vocal_program(self):
        pipeline = MusicToMidiPipeline(Config())
        notes = {
            24: [NoteEvent(pitch=60, start_time=0.0, end_time=0.5)],
            40: [
                NoteEvent(pitch=64, start_time=0.0, end_time=0.5),
                NoteEvent(pitch=65, start_time=0.5, end_time=1.0),
            ],
        }

        filtered = pipeline._filter_vocal_melody_notes(notes, {})

        self.assertEqual(filtered, ({}, {}))

    def test_vocal_filter_can_preserve_isolated_vocal_stem_notes_explicitly(self):
        pipeline = MusicToMidiPipeline(Config())
        notes = {
            65: [
                NoteEvent(pitch=64, start_time=0.5, end_time=1.0, program=65),
                NoteEvent(pitch=60, start_time=0.0, end_time=0.4, program=65),
            ],
        }

        filtered_notes, filtered_drums = pipeline._filter_vocal_melody_notes(
            notes,
            {},
            preserve_isolated_vocal_stem=True,
        )

        self.assertEqual(filtered_drums, {})
        self.assertEqual(list(filtered_notes), [0])
        self.assertEqual([note.pitch for note in filtered_notes[0]], [60, 64])
        self.assertTrue(all(note.program == 0 for note in filtered_notes[0]))

    def test_ssl_fix_requires_explicit_insecure_opt_in(self):
        ssl_error = yourmt3_downloader.ssl.SSLCertVerificationError("bad cert")

        with patch(
            "src.utils.yourmt3_downloader.urllib.request.urlopen",
            side_effect=ssl_error,
        ), patch.dict(
            "src.utils.yourmt3_downloader.os.environ",
            {"ALLOW_INSECURE_HF_DOWNLOAD": ""},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "(?s)SSL.*ALLOW_INSECURE_HF_DOWNLOAD"):
                yourmt3_downloader._fix_ssl_if_needed()

    def test_ssl_fix_allows_insecure_mode_only_when_env_opted_in(self):
        ssl_error = yourmt3_downloader.ssl.SSLCertVerificationError("bad cert")

        with patch(
            "src.utils.yourmt3_downloader.urllib.request.urlopen",
            side_effect=ssl_error,
        ), patch.dict(
            "src.utils.yourmt3_downloader.os.environ",
            {"ALLOW_INSECURE_HF_DOWNLOAD": "1"},
            clear=False,
        ):
            yourmt3_downloader._fix_ssl_if_needed()
            self.assertEqual(yourmt3_downloader.os.environ["HF_HUB_DISABLE_SSL_VERIFY"], "1")


if __name__ == "__main__":
    unittest.main()
