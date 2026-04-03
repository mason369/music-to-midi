import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

mido_stub = types.ModuleType("mido")
mido_stub.__spec__ = None


class _Dummy:
    pass


mido_stub.MidiFile = _Dummy
mido_stub.MidiTrack = _Dummy
mido_stub.Message = _Dummy
mido_stub.MetaMessage = _Dummy
sys.modules.setdefault("mido", mido_stub)

from src.core.aria_amt_transcriber import AriaAmtTranscriber
from src.core.pipeline import MusicToMidiPipeline
from src.core.transkun_transcriber import TranskunTranscriber
from src.models.data_models import Config


class _FakeProcess:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True


class _SlowCancelableTranscriber:
    def __init__(self):
        self.cancel_called = False
        self._cancel_event = threading.Event()

    def is_available(self):
        return True

    def is_model_available(self):
        return True

    def cancel(self):
        self.cancel_called = True
        self._cancel_event.set()

    def transcribe(self, audio_path, output_path, progress_callback=None):
        self._cancel_event.wait(5.0)
        if not self.cancel_called:
            raise TimeoutError("cancel was not propagated")
        Path(output_path).write_text("mid", encoding="utf-8")
        return output_path


class _FakeBeatInfo:
    bpm = 120.0


class _FakeBeatDetector:
    def detect(self, _audio_path):
        return _FakeBeatInfo()


class TestPianoBackendCancellation(unittest.TestCase):
    def test_pipeline_cancel_propagates_to_aria_and_transkun_transcribers(self):
        pipeline = MusicToMidiPipeline(Config())
        aria = _SlowCancelableTranscriber()
        transkun = _SlowCancelableTranscriber()
        pipeline.aria_amt_transcriber = aria
        pipeline.transkun_transcriber = transkun

        pipeline.cancel()

        self.assertTrue(aria.cancel_called)
        self.assertTrue(transkun.cancel_called)

    def test_pipeline_cancel_interrupts_piano_transkun_mode_promptly(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "song.wav"
            audio_path.write_bytes(b"wav")
            out_dir = Path(tmp) / "out"

            pipeline = MusicToMidiPipeline(Config(processing_mode="piano_transkun"))
            pipeline.beat_detector = _FakeBeatDetector()
            pipeline._count_midi_notes = lambda _path: 1
            pipeline.transkun_transcriber = _SlowCancelableTranscriber()

            error_holder = []

            thread = threading.Thread(
                target=lambda: self._run_and_capture(
                    lambda: pipeline._process_piano_transkun(str(audio_path), str(out_dir)),
                    error_holder,
                )
            )
            thread.start()
            time.sleep(0.2)
            pipeline.cancel()
            thread.join(timeout=1.0)

            self.assertFalse(thread.is_alive(), "pipeline thread should exit after cancellation")
            self.assertTrue(error_holder)
            self.assertIsInstance(error_holder[0], InterruptedError)

    def test_aria_cancel_terminates_running_process(self):
        transcriber = AriaAmtTranscriber()
        fake_process = _FakeProcess()
        transcriber._process = fake_process

        transcriber.cancel()

        self.assertTrue(fake_process.terminated)

    def test_transkun_cancel_terminates_running_process(self):
        transcriber = TranskunTranscriber()
        fake_process = _FakeProcess()
        transcriber._process = fake_process

        transcriber.cancel()

        self.assertTrue(fake_process.terminated)

    @staticmethod
    def _run_and_capture(fn, error_holder):
        try:
            fn()
        except Exception as exc:  # pragma: no cover - assertion happens in caller
            error_holder.append(exc)


if __name__ == "__main__":
    unittest.main()
