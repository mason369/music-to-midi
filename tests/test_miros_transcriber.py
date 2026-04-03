import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.miros_transcriber import MirosTranscriber
from src.models.data_models import Config


class _FakeNote:
    def __init__(self, pitch, start, end, velocity):
        self.pitch = pitch
        self.start = start
        self.end = end
        self.velocity = velocity


class _FakeInstrument:
    def __init__(self, program, is_drum, notes):
        self.program = program
        self.is_drum = is_drum
        self.notes = notes


class _FakePrettyMIDI:
    def __init__(self, _path):
        self.instruments = [
            _FakeInstrument(
                program=5,
                is_drum=False,
                notes=[
                    _FakeNote(pitch=60, start=0.5, end=1.25, velocity=90),
                    _FakeNote(pitch=64, start=1.5, end=2.0, velocity=70),
                ],
            ),
            _FakeInstrument(
                program=0,
                is_drum=True,
                notes=[
                    _FakeNote(pitch=36, start=0.25, end=0.35, velocity=110),
                ],
            ),
        ]


class MirosTranscriberTests(unittest.TestCase):
    def test_transcribe_precise_reads_seconds_from_pretty_midi(self):
        pretty_midi_stub = types.ModuleType("pretty_midi")
        pretty_midi_stub.PrettyMIDI = _FakePrettyMIDI

        transcriber = MirosTranscriber(Config())

        def fake_transcribe_to_midi(audio_path, output_path, progress_callback=None):
            Path(output_path).write_bytes(b"mid")
            return output_path

        with patch.object(transcriber, "transcribe_to_midi", side_effect=fake_transcribe_to_midi), patch.dict(
            sys.modules,
            {"pretty_midi": pretty_midi_stub},
        ):
            instrument_notes, drum_notes = transcriber.transcribe_precise("song.wav")

        self.assertIn(5, instrument_notes)
        self.assertEqual(len(instrument_notes[5]), 2)
        self.assertAlmostEqual(instrument_notes[5][0].start_time, 0.5)
        self.assertAlmostEqual(instrument_notes[5][0].end_time, 1.25)

        self.assertIn(36, drum_notes)
        self.assertEqual(len(drum_notes[36]), 1)
        self.assertAlmostEqual(drum_notes[36][0].start_time, 0.25)
        self.assertAlmostEqual(drum_notes[36][0].end_time, 0.35)


if __name__ == "__main__":
    unittest.main()
