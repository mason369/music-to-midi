import unittest
from unittest.mock import patch

from src.core.pipeline import MusicToMidiPipeline
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


if __name__ == "__main__":
    unittest.main()
