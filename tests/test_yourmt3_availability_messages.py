import sys
import types
import unittest
from unittest.mock import patch

mido_stub = types.ModuleType("mido")
mido_stub.__spec__ = None


class _Dummy:
    pass


mido_stub.MidiFile = _Dummy
mido_stub.MidiTrack = _Dummy
mido_stub.Message = _Dummy
mido_stub.MetaMessage = _Dummy
sys.modules.setdefault("mido", mido_stub)

from src.core.pipeline import MusicToMidiPipeline
from src.core.yourmt3_transcriber import YourMT3Transcriber
from src.models.data_models import Config


class YourMT3AvailabilityMessageTests(unittest.TestCase):
    def test_smart_mode_surfaces_actual_availability_reason(self):
        class FakeYourMT3Transcriber:
            def __init__(self, _config):
                pass

            def set_cancel_check(self, _callback):
                return None

            @staticmethod
            def is_available():
                return False

            @staticmethod
            def get_unavailable_reason():
                return "YourMT3+ ?????? pytorch-lightning"

        class FakeAriaAmtTranscriber:
            def __init__(self):
                pass

        class FakeBeatDetector:
            def __init__(self, _config):
                pass

        class FakeMidiGenerator:
            def __init__(self, _config):
                pass

        with patch("src.core.pipeline.YourMT3Transcriber", FakeYourMT3Transcriber), patch(
            "src.core.pipeline.AriaAmtTranscriber", FakeAriaAmtTranscriber
        ), patch("src.core.pipeline.BeatDetector", FakeBeatDetector), patch(
            "src.core.pipeline.MidiGenerator", FakeMidiGenerator
        ):
            pipeline = MusicToMidiPipeline(Config())
            with self.assertRaisesRegex(RuntimeError, "pytorch-lightning"):
                pipeline._process_smart("input.wav", "output")

    def test_yourmt3_import_error_surfaces_underlying_dependency(self):
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pytorch_lightning":
                raise ModuleNotFoundError("No module named 'PIL'")
            return real_import(name, globals, locals, fromlist, level)

        with patch("src.core.yourmt3_transcriber._import_torch", return_value=object()), patch(
            "builtins.__import__", side_effect=fake_import
        ):
            available = YourMT3Transcriber.is_available()

        self.assertFalse(available)
        reason = YourMT3Transcriber.get_unavailable_reason()
        self.assertIn("pytorch-lightning", reason)
        self.assertIn("No module named 'PIL'", reason)

    def test_yourmt3_missing_source_tree_reports_directory_message(self):
        with patch("src.core.yourmt3_transcriber._import_torch", return_value=object()), patch(
            "builtins.__import__", return_value=types.SimpleNamespace(__name__="pytorch_lightning")
        ), patch("src.core.yourmt3_transcriber._get_yourmt3_amt_src_path", return_value=None):
            available = YourMT3Transcriber.is_available()

        self.assertFalse(available)
        reason = YourMT3Transcriber.get_unavailable_reason()
        self.assertIn("YourMT3 ????", reason)


if __name__ == "__main__":
    unittest.main()
