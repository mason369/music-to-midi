import tempfile
import unittest
from pathlib import Path

from mido import MidiFile

from src.core.midi_generator import MidiGenerator
from src.models.data_models import Config, NoteEvent


def _track_names(midi: MidiFile) -> list[str]:
    names = []
    for track in midi.tracks:
        for message in track:
            if message.type == "track_name":
                names.append(message.name)
                break
    return names


class TestMidiTrackMode(unittest.TestCase):
    def _generate(self, midi_track_mode: str) -> MidiFile:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "layout.mid"
            config = Config(midi_track_mode=midi_track_mode)
            generator = MidiGenerator(config)
            generator.generate_from_precise_instruments_v2(
                instrument_notes={
                    0: [NoteEvent(pitch=60, start_time=0.0, end_time=0.4, velocity=88)],
                    24: [NoteEvent(pitch=64, start_time=0.1, end_time=0.5, velocity=76)],
                },
                drum_notes={
                    36: [NoteEvent(pitch=36, start_time=0.0, end_time=0.2, velocity=100)],
                },
                tempo=120.0,
                output_path=str(output_path),
                quality="best",
            )
            return MidiFile(str(output_path))

    def test_multi_track_mode_keeps_one_track_per_gm_program(self):
        midi = self._generate("multi_track")

        self.assertEqual(_track_names(midi), ["GM000", "GM024", "Drums"])
        self.assertEqual(len(midi.tracks), 4)

    def test_single_track_mode_merges_melodic_instruments_but_keeps_drums_separate(self):
        midi = self._generate("single_track")

        self.assertEqual(_track_names(midi), ["All Instruments", "Drums"])
        self.assertEqual(len(midi.tracks), 3)

        melodic_programs = {
            message.program
            for message in midi.tracks[1]
            if message.type == "program_change"
        }
        melodic_channels = {
            message.channel
            for message in midi.tracks[1]
            if message.type == "note_on" and message.velocity > 0
        }
        drum_channels = {
            message.channel
            for message in midi.tracks[2]
            if message.type == "note_on" and message.velocity > 0
        }

        self.assertEqual(melodic_programs, {0, 24})
        self.assertEqual(melodic_channels, {0, 1})
        self.assertEqual(drum_channels, {9})


if __name__ == "__main__":
    unittest.main()
