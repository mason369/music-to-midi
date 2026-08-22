import tempfile
import unittest
from pathlib import Path

from mido import Message, MetaMessage, MidiFile, MidiTrack

from src.utils.midi_output import clip_midi_to_duration, unique_midi_temp_path


class MidiOutputDurationTests(unittest.TestCase):
    def test_temporary_name_is_bounded_and_does_not_repeat_a_long_final_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            final_path = Path(tmp) / ("x" * 180 + ".mid")

            first = unique_midi_temp_path(final_path, "duration-clipped")
            second = unique_midi_temp_path(final_path, "duration-clipped")

            self.assertEqual(first.parent, final_path.parent.resolve())
            self.assertEqual(second.parent, final_path.parent.resolve())
            self.assertNotEqual(first, second)
            self.assertNotIn(final_path.stem, first.name)
            self.assertLessEqual(len(first.name), 78)
            self.assertRegex(
                first.name,
                r"^\.mtm-duration-clipped-[0-9a-f]{32}\.tmp\.mid$",
            )

    def test_duration_clipping_publishes_a_midi_with_a_long_final_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            midi_path = Path(tmp) / ("long-" + "x" * 175 + ".mid")
            midi = MidiFile(type=1, ticks_per_beat=480)
            track = MidiTrack()
            track.append(MetaMessage("set_tempo", tempo=500_000, time=0))
            track.append(Message("note_on", note=60, velocity=90, time=0))
            track.append(Message("note_off", note=60, velocity=0, time=480))
            midi.tracks.append(track)
            midi.save(str(midi_path))

            clipped_path = clip_midi_to_duration(midi_path, 0.5, "test backend")

            self.assertEqual(clipped_path, midi_path)
            self.assertEqual(len(MidiFile(str(midi_path)).tracks), 1)
            self.assertEqual(list(midi_path.parent.glob(".*.tmp.mid")), [])

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
            piano_track.append(Message("control_change", control=64, value=127, time=0))
            piano_track.append(Message("note_off", note=60, velocity=0, time=720))
            piano_track.append(Message("control_change", control=64, value=0, time=0))
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
                    message.type == "note_on" and message.velocity > 0 and message.note == 67
                    for _tick, message in retained
                )
            )


if __name__ == "__main__":
    unittest.main()
