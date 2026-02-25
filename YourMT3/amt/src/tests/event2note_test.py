# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""event2midi_test.py:

This file contains tests for the following classes:
• event2note_event
• note_event2note

"""
import unittest
import pytest
from numpy import random
from assert_fns import assert_notes_almost_equal
from utils.note_event_dataclasses import Event, Note, NoteEvent
from utils.event2note import event2note_event
from utils.event2note import note_event2note


# yapf: disable
class TestEvent2NoteEvent(unittest.TestCase):
    def test_event2note_event(self):
        events = [
            Event('program', 33), Event('pitch', 60),
            Event('program', 52), Event('pitch', 40),
            Event('tie', 0),
            Event('shift', 20), Event('velocity', 1), Event('drum', 36),
            Event('shift', 150), Event('program', 33), Event('velocity', 0), Event('pitch', 60),
            Event('shift', 160), Event('velocity', 1), Event('pitch', 62), Event('program', 100),
                                Event('pitch', 77),
            Event('shift', 200), Event('velocity', 0), Event('pitch', 77),
            Event('shift', 250), Event('velocity', 1), Event('drum', 38),
            Event('shift', 300), Event('velocity', 0), Event('program', 33), Event('pitch', 62)
        ]

        note_events, tie_note_events, last_activity, err_cnt = event2note_event(events, start_time=0, sort=False, tps=100)
        self.assertEqual(len(err_cnt), 0)
        expected_note_events = [NoteEvent(True, 128, 0.2, 1, 36),
                                NoteEvent(False, 33, 1.5, 0, 60),
                                NoteEvent(False, 33, 1.6, 1, 62),
                                NoteEvent(False, 100, 1.6, 1, 77),
                                NoteEvent(False, 100, 2.0, 0, 77),
                                NoteEvent(True, 128, 2.5, 1, 38),
                                NoteEvent(False, 33, 3.0, 0, 62)]
        expected_tie_note_events = [NoteEvent(False, 33, None, 1, 60),
                                    NoteEvent(False, 52, None, 1, 40)]
        expected_last_activity = [(52, 40)]
        self.assertSequenceEqual(note_events, expected_note_events)
        self.assertSequenceEqual(tie_note_events, expected_tie_note_events)
        self.assertSequenceEqual(last_activity, expected_last_activity)

class TestEvent2NoteEventInvalidInputWarn(unittest.TestCase):
    def test_event2note_event_with_invalid_shift_value(self):
        events = [Event('tie', 0), Event('shift', 0), Event('shift', 1050)] # shift: 0 <= value <= 1000
        _, _, _, err_cnt = event2note_event(events, start_time=0, sort=True, tps=100)
        self.assertEqual(err_cnt['Err/Shift out of range'], 2)

    def test_event2note_event_with_invalid_pitch_event(self):
        events = [Event('pitch', 60), Event('tie', 0)] # pitch event must follow a program event
        _, _, _, err_cnt = event2note_event(events, start_time=0, sort=True, tps=100)
        self.assertEqual(err_cnt['Err/Missing prg in tie'], 1)

    def test_event2note_event_with_invalid_tie_event(self):
        events = [Event('shift', 10)]
        _, _, _, err_cnt = event2note_event(events, start_time=0, sort=True, tps=100)
        self.assertEqual(err_cnt['Err/Missing tie'], 1)

class TestEvent2NoteEventSpecialEvent(unittest.TestCase):
    def test_note_event2note_special_events(self):
        events = [Event('program', 33), Event('pitch', 60),
                  Event('tie', 0),
                  Event('shift', 10), Event('program', 33), Event('velocity', 0), Event('pitch', 60),
                  Event('EOS', 0), Event('PAD', 0), # <- will stop decoding at this point...
                  Event('shift', 20), Event('velocity', 1), Event('pitch', 20),
                  Event('shift', 30), Event('velocity', 1), Event('pitch', 20),]
        note_events, tie_note_events, _, err_cnt = event2note_event(events, start_time=0)
        print(note_events)
        self.assertEqual(len(note_events), 1)
        self.assertEqual(len(tie_note_events), 1)
        self.assertEqual(len(err_cnt), 0)


class TestNoteEvent2Note(unittest.TestCase):

    def test_note_event2note(self):

        note_events = [NoteEvent(is_drum=False, program=33, time=0, velocity=1, pitch=60),
        NoteEvent(is_drum=False, program=33, time=1.5, velocity=0, pitch=60),
        NoteEvent(is_drum=False, program=33, time=1.6, velocity=1, pitch=62),
        NoteEvent(is_drum=False, program=33, time=3.0, velocity=0, pitch=62),
        NoteEvent(is_drum=False, program=100, time=1.6, velocity=1, pitch=77),
        NoteEvent(is_drum=False, program=100, time=2.0, velocity=0, pitch=77),
        NoteEvent(is_drum=True, program=128, time=0.2, velocity=1, pitch=36),
        NoteEvent(is_drum=True, program=128, time=2.5, velocity=1, pitch=38)
        ]
        notes, err_cnt = note_event2note(note_events, sort=True)

        expected_notes = [
            Note(is_drum=False, program=33, onset=0, offset=1.5, pitch=60, velocity=1),
            Note(is_drum=True, program=128, onset=0.2, offset=0.21, pitch=36, velocity=1),
            Note(is_drum=False, program=33, onset=1.6, offset=3.0, pitch=62, velocity=1),
            Note(is_drum=False, program=100, onset=1.6, offset=2.0, pitch=77, velocity=1),
            Note(is_drum=True, program=128, onset=2.5, offset=2.51, pitch=38, velocity=1)
            ]
        self.assertEqual(len(err_cnt), 0)
        assert_notes_almost_equal(notes, expected_notes, delta=5e-3)


    def test_note_event2note_simple_cases(self):
        # Case 1: Basic test case with two notes
        note_events = [
            NoteEvent(is_drum=False, program=0, time=0.1, velocity=1, pitch=60),
            NoteEvent(is_drum=False, program=0, time=0.5, velocity=0, pitch=60),
            NoteEvent(is_drum=False, program=0, time=0.7, velocity=1, pitch=62),
            NoteEvent(is_drum=False, program=0, time=1.5, velocity=0, pitch=62),
        ]

        expected_notes = [
            Note(is_drum=False, program=0, onset=0.1, offset=0.5, pitch=60, velocity=1),
            Note(is_drum=False, program=0, onset=0.7, offset=1.5, pitch=62, velocity=1),
        ]
        notes, err_cnt = note_event2note(note_events)
        self.assertEqual(len(err_cnt), 0)
        self.assertSequenceEqual(notes, expected_notes)

        # Case 2: Test with drum notes
        note_events = [
            NoteEvent(is_drum=True, program=128, time=0.2, velocity=1, pitch=36),
            NoteEvent(is_drum=True, program=128, time=0.3, velocity=1, pitch=38),
            NoteEvent(is_drum=True, program=128, time=0.4, velocity=0, pitch=36),
            NoteEvent(is_drum=True, program=128, time=0.5, velocity=0, pitch=38),
        ]

        expected_notes = [
            Note(is_drum=True, program=128, onset=0.2, offset=0.21, pitch=36, velocity=1),
            Note(is_drum=True, program=128, onset=0.3, offset=0.31, pitch=38, velocity=1),
        ]
        notes, err_cnt = note_event2note(note_events)
        self.assertEqual(len(err_cnt), 0)
        assert_notes_almost_equal(notes, expected_notes, delta=5.1e-3)


    def test_note_event2note_multiple_overlapping_notes(self):

        note_events = [
            NoteEvent(is_drum=False, program=1, time=0.0, velocity=1, pitch=60),
            NoteEvent(is_drum=False, program=1, time=0.5, velocity=0, pitch=60),
            NoteEvent(is_drum=False, program=1, time=1.0, velocity=1, pitch=62),
            NoteEvent(is_drum=False, program=1, time=1.5, velocity=0, pitch=62),
            NoteEvent(is_drum=False, program=2, time=0.25, velocity=1, pitch=60),
            NoteEvent(is_drum=False, program=2, time=0.75, velocity=0, pitch=60),
            NoteEvent(is_drum=False, program=2, time=1.25, velocity=1, pitch=62),
            NoteEvent(is_drum=False, program=2, time=1.75, velocity=0, pitch=62),
            NoteEvent(is_drum=False, program=3, time=0.0, velocity=1, pitch=64),
            NoteEvent(is_drum=False, program=3, time=1.0, velocity=0, pitch=64),
            NoteEvent(is_drum=False, program=4, time=0.5, velocity=1, pitch=66),
            NoteEvent(is_drum=False, program=4, time=1.5, velocity=0, pitch=66),
            NoteEvent(is_drum=False, program=4, time=0.75, velocity=1, pitch=67),
            NoteEvent(is_drum=False, program=4, time=1.75, velocity=0, pitch=67),
            NoteEvent(is_drum=False, program=4, time=1.0, velocity=1, pitch=69),
            NoteEvent(is_drum=False, program=4, time=2.0, velocity=0, pitch=69)
        ]

        expected_notes = [
            Note(is_drum=False, program=1, onset=0.0, offset=0.5, pitch=60, velocity=1),
            Note(is_drum=False, program=3, onset=0.0, offset=1.0, pitch=64, velocity=1),
            Note(is_drum=False, program=2, onset=0.25, offset=0.75, pitch=60, velocity=1),
            Note(is_drum=False, program=4, onset=0.5, offset=1.5, pitch=66, velocity=1),
            Note(is_drum=False, program=4, onset=0.75, offset=1.75, pitch=67, velocity=1),
            Note(is_drum=False, program=1, onset=1.0, offset=1.5, pitch=62, velocity=1),
            Note(is_drum=False, program=4, onset=1.0, offset=2.0, pitch=69, velocity=1),
            Note(is_drum=False, program=2, onset=1.25, offset=1.75, pitch=62, velocity=1)
        ]

        notes, err_cnt = note_event2note(note_events)
        self.assertEqual(len(err_cnt), 0)
        assert_notes_almost_equal(notes, expected_notes, delta=5e-3)

# yapf: enable
if __name__ == '__main__':
    unittest.main()
