import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.pipeline import MusicToMidiPipeline
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
