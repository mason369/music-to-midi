import tempfile
import unittest
from pathlib import Path

from mido import Message, MetaMessage, MidiFile, MidiTrack

from src.utils.midi_output import clip_midi_to_duration


class MidiOutputDurationTests(unittest.TestCase):
    def test_clips_padded_note_off_and_pedal_events_across_tempo_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            midi_path = Path(tmp) / "padded.mid"
            midi = MidiFile(type=1, ticks_per_beat=480)

            tempo_track = MidiTrack()
            tempo_track.append(MetaMessage("set_tempo", tempo=500_000, time=0))
            tempo_track.append(MetaMessage("set_tempo", tempo=1_000_000, time=480))
            midi.tracks.append(tempo_track)

            piano_track = MidiTrack()
            piano_track.append(Message("note_on", note=60, velocity=90, time=240))
            piano_track.append(
                Message("control_change", control=64, value=127, time=0)
            )
            piano_track.append(Message("note_off", note=60, velocity=0, time=720))
            piano_track.append(
                Message("control_change", control=64, value=0, time=0)
            )
            piano_track.append(Message("note_on", note=67, velocity=80, time=120))
            midi.tracks.append(piano_track)
            midi.save(str(midi_path))

            clip_midi_to_duration(midi_path, 1.0, "test backend")

            clipped = MidiFile(str(midi_path))
            self.assertLessEqual(clipped.length, 1.0 + 1e-9)

            absolute_tick = 0
            retained = []
            for message in clipped.tracks[1]:
                absolute_tick += message.time
                if not message.is_meta:
                    retained.append((absolute_tick, message))

            self.assertEqual(
                [tick for tick, message in retained if message.type == "note_off"],
                [720],
            )
            self.assertEqual(
                [
                    tick
                    for tick, message in retained
                    if message.type == "control_change" and message.value == 0
                ],
                [720],
            )
            self.assertFalse(
                any(
                    message.type == "note_on"
                    and message.velocity > 0
                    and message.note == 67
                    for _tick, message in retained
                )
            )


if __name__ == "__main__":
    unittest.main()
