import unittest
from typing import List
from tempfile import NamedTemporaryFile
from assert_fns import assert_notes_almost_equal
from utils.note_event_dataclasses import Note

from utils.midi import note_event2midi
from utils.midi import midi2note
from utils.note2event import note2note_event
# yapf: disable

class TestNoteMidiConversion(unittest.TestCase):

    def test_note2midi2note_z(self):
        original_notes = [
            Note(is_drum=False, program=3, onset=0., offset=1., pitch=60, velocity=1),
            Note(is_drum=False, program=3, onset=1., offset=2., pitch=64, velocity=1),
        ]

        with NamedTemporaryFile(suffix=".mid", delete=True) as temp_file:
            # Convert original_notes to MIDI and save it to the temporary file
            note_events = note2note_event(notes=original_notes, sort=True)
            note_event2midi(note_events, temp_file.name, velocity=100)

            # Convert the MIDI back to notes
            converted_notes, _ = midi2note(temp_file.name)

            # Compare original notes and converted notes
            assert_notes_almost_equal(original_notes, converted_notes)

    def test_midi2note2midi2note_piano_z(self):
        file = 'extras/examples/piano.mid'
        # This MIDI file is missing the program change event, so we force it to be 0
        notes, _ = midi2note(file, quantize=False, force_all_program_to=0)[:1000]
        note_events = note2note_event(notes=notes, sort=True)
        note_event2midi(note_events, 'extras/examples/piano_converted.mid', velocity=100)
        reconverted_notes, _ = midi2note('extras/examples/piano_converted.mid', quantize=False)
        assert_notes_almost_equal(notes, reconverted_notes, delta=0.01)

    def test_midi2note2midi2note_force_drum_z(self):
        file = 'extras/examples/drum.mid'
        conv_file = 'extras/examples/drum_converted.mid'
        # This MIDI file is missing the program change event, so we force it to be 0
        notes, _ = midi2note(file, quantize=True, force_all_drum=True)[:100]
        note_events = note2note_event(notes=notes, sort=True)
        note_event2midi(note_events, conv_file, velocity=100, ticks_per_beat=960)
        reconverted_notes, _ = midi2note(conv_file, quantize=True, force_all_drum=True)
        assert_notes_almost_equal(notes, reconverted_notes, delta=0.005)

        # In drum, this is very inaccurate. We should fix this in the future.
        # Even for the first 100 notes, the timing is off by 170 ms.

    def test_midi2note_ignore_pedal_true_z(self):
        file = 'extras/examples/piano.mid'
        notes, _ = midi2note(file, quantize=False, ignore_pedal=True, force_all_program_to=0)
        note_events = note2note_event(notes=notes, sort=True)
        note_event2midi(note_events, 'extras/examples/piano_converted.mid', velocity=100)
        reconverted_notes, _ = midi2note('extras/examples/piano_converted.mid', quantize=False)
        assert_notes_almost_equal(notes, reconverted_notes, delta=0.01)


# yapf: enable

if __name__ == '__main__':
    unittest.main()
