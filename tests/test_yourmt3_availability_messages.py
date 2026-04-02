import sys
import types
import unittest
from pathlib import Path
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
                return "YourMT3+ 不可用：缺少 pytorch-lightning"

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
            with self.assertRaisesRegex(RuntimeError, "缺少 pytorch-lightning"):
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

    def test_yourmt3_source_import_failure_surfaces_underlying_dependency(self):
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pytorch_lightning":
                return types.SimpleNamespace(__name__="pytorch_lightning")
            if name == "yourmt3":
                raise ModuleNotFoundError("No module named 'yourmt3'")
            if name == "model.ymt3":
                raise ImportError(
                    "DLL load failed while importing onnxruntime_pybind11_state: initialization routine failed"
                )
            return real_import(name, globals, locals, fromlist, level)

        with patch("src.core.yourmt3_transcriber._import_torch", return_value=object()), patch(
            "src.core.yourmt3_transcriber._get_yourmt3_amt_src_path",
            return_value=str(Path("YourMT3/amt/src").resolve()),
        ), patch("builtins.__import__", side_effect=fake_import):
            available = YourMT3Transcriber.is_available()

        self.assertFalse(available)
        reason = YourMT3Transcriber.get_unavailable_reason()
        self.assertIn("onnxruntime_pybind11_state", reason)
        self.assertNotIn("未找到 YourMT3 代码目录", reason)


if __name__ == "__main__":
    unittest.main()
