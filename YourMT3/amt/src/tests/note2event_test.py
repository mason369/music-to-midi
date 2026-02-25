# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
import unittest
import pytest
import warnings
from numpy import random

from utils.note2event import note2note_event
from utils.note2event import slice_note_events_and_ties
from utils.note2event import slice_multiple_note_events_and_ties_to_bundle
from utils.note2event import trim_overlapping_notes
from utils.note2event import note_event2event
from utils.note2event import mix_notes
from utils.note2event import validate_notes
from utils.note_event_dataclasses import Note, NoteEvent, NoteEventListsBundle
from utils.note_event_dataclasses import Event


# yapf: disable
class TestNoteTools(unittest.TestCase):

    def test_trim_overlapping_notes(self):
        notes = [
            Note(is_drum=False, program=1, onset=0.0, offset=1.0, pitch=60, velocity=100),
            Note(is_drum=False, program=1, onset=0.5, offset=1.5, pitch=60, velocity=100),
            Note(is_drum=False, program=1, onset=2.0, offset=3.0, pitch=60, velocity=100)
        ]
        expected_notes = [
            Note(is_drum=False, program=1, onset=0.0, offset=0.5, pitch=60, velocity=100),
            Note(is_drum=False, program=1, onset=0.5, offset=1.5, pitch=60, velocity=100),
            Note(is_drum=False, program=1, onset=2.0, offset=3.0, pitch=60, velocity=100)
        ]

        trimmed_notes = trim_overlapping_notes(notes)

        self.assertEqual(len(expected_notes), len(trimmed_notes), "Number of notes should be equal")
        for e_note, t_note in zip(expected_notes, trimmed_notes):
            self.assertEqual(e_note, t_note, "Trimmed note should match the expected note")

    def test_mix_notes(self):
        notes1 = [
            Note(is_drum=False, program=33, onset=0.0, offset=0.5, pitch=60, velocity=1),
            Note(is_drum=False, program=33, onset=1.0, offset=1.5, pitch=62, velocity=1),
            Note(is_drum=True, program=128, onset=2.0, offset=2.1, pitch=36, velocity=1)
        ]
        notes2 = [
            Note(is_drum=False, program=52, onset=0.5, offset=1.0, pitch=40, velocity=1),
            Note(is_drum=False, program=100, onset=1.5, offset=2.0, pitch=77, velocity=1),
            Note(is_drum=True, program=128, onset=2.5, offset=2.6, pitch=38, velocity=1)
        ]
        mixed_notes = mix_notes((notes1, notes2), sort=True, trim_overlap=True, fix_offset=True)

        expected_mixed_notes = [
            Note(is_drum=False, program=33, onset=0.0, offset=0.5, pitch=60, velocity=1),
            Note(is_drum=False, program=52, onset=0.5, offset=1.0, pitch=40, velocity=1),
            Note(is_drum=False, program=33, onset=1.0, offset=1.5, pitch=62, velocity=1),
            Note(is_drum=False, program=100, onset=1.5, offset=2.0, pitch=77, velocity=1),
            Note(is_drum=True, program=128, onset=2.0, offset=2.1, pitch=36, velocity=1),
            Note(is_drum=True, program=128, onset=2.5, offset=2.6, pitch=38, velocity=1)
        ]
        self.assertSequenceEqual(mixed_notes, expected_mixed_notes)

    def test_validate_notes(self):
        DRUM_OFFSET_TIME = 0.01  # in seconds
        MINIMUM_OFFSET_TIME = 0.01  # this is used to avoid zero-length notes

        notes = [
            Note(is_drum=False, program=33, onset=0.0, offset=0.5, pitch=60, velocity=1),
            Note(is_drum=False, program=33, onset=1.0, offset=0.9, pitch=62, velocity=1),
            Note(is_drum=True, program=128, onset=2.0, offset=2.1, pitch=36, velocity=1),
            Note(is_drum=False, program=100, onset=1.5, offset=1.4, pitch=77, velocity=1)
        ]
        with self.assertWarns(UserWarning):
            validated_notes = validate_notes(notes, fix=True)

        expected_validated_notes = [
            Note(is_drum=False, program=33, onset=0.0, offset=0.5, pitch=60, velocity=1),
            Note(is_drum=False, program=33, onset=1.0, offset=1.0 + MINIMUM_OFFSET_TIME, pitch=62, velocity=1),
            Note(is_drum=True, program=128, onset=2.0, offset=2.1, pitch=36, velocity=1),
            Note(is_drum=False, program=100, onset=1.5, offset=1.5 + MINIMUM_OFFSET_TIME, pitch=77, velocity=1)
        ]

        self.assertSequenceEqual(validated_notes, expected_validated_notes)



class TestNoteEvent(unittest.TestCase):

    def test_NoteEvent(self):
        note_event = NoteEvent(is_drum=False, program=33, time=0, velocity=1, pitch=60)
        self.assertEqual(note_event.is_drum, False)
        self.assertEqual(note_event.program, 33)
        self.assertEqual(note_event.time, 0)
        self.assertEqual(note_event.velocity, 1)
        self.assertEqual(note_event.pitch, 60)

        ne1 = NoteEvent(True, 64, 0.5, 0, 60)
        ne2 = NoteEvent(True, 64, 0.5, 0, 61)
        self.assertEqual(ne1.equals_except(ne2, "pitch"), True)
        self.assertEqual(ne1.equals_except(ne2, "program"), False)
        self.assertEqual(ne1.equals_except(ne2, "time", "pitch"), True)

        ne1 = NoteEvent(True, 64, 0.5, 1, 60)
        ne2 = NoteEvent(True, 11, 0.5, 1, 61)
        self.assertEqual(ne1.equals_only(ne2, "velocity"), True)
        self.assertEqual(ne1.equals_only(ne2, "time", "velocity"), True)
        self.assertEqual(ne1.equals_only(ne2, "program", "velocity"), False)


class TestNote2NoteEvent(unittest.TestCase):
    def test_note2note_event(self):
        notes = [
            Note(is_drum=False, program=33, onset=0, offset=1.5, pitch=60, velocity=1),
            Note(is_drum=False, program=33, onset=1.6, offset=3.0, pitch=62, velocity=1),
            Note(is_drum=False, program=100, onset=1.6, offset=2.0, pitch=77, velocity=1),
            Note(is_drum=True, program=128, onset=0.2, offset=0.21, pitch=36, velocity=1),
            Note(is_drum=True, program=128, onset=2.5, offset=2.51, pitch=38, velocity=1)
            ]

        note_events = note2note_event(notes, sort=False, return_activity=False)
        self.assertSequenceEqual(note_events,
        [NoteEvent(is_drum=False, program=33, time=0, velocity=1, pitch=60),
        NoteEvent(is_drum=False, program=33, time=1.5, velocity=0, pitch=60),
        NoteEvent(is_drum=False, program=33, time=1.6, velocity=1, pitch=62),
        NoteEvent(is_drum=False, program=33, time=3.0, velocity=0, pitch=62),
        NoteEvent(is_drum=False, program=100, time=1.6, velocity=1, pitch=77),
        NoteEvent(is_drum=False, program=100, time=2.0, velocity=0, pitch=77),
        NoteEvent(is_drum=True, program=128, time=0.2, velocity=1, pitch=36),
        NoteEvent(is_drum=True, program=128, time=2.5, velocity=1, pitch=38)
        ])

        note_events = note2note_event(notes, sort=True, return_activity=True)
        self.assertSequenceEqual(note_events,
        [NoteEvent(is_drum=False, program=33, time=0, velocity=1, pitch=60, activity=set()),
        NoteEvent(is_drum=True, program=128, time=0.2, velocity=1, pitch=36, activity={0}),
        NoteEvent(is_drum=False, program=33, time=1.5, velocity=0, pitch=60, activity={0}),
        NoteEvent(is_drum=False, program=33, time=1.6, velocity=1, pitch=62, activity=set()),
        NoteEvent(is_drum=False, program=100, time=1.6, velocity=1, pitch=77, activity={3}),
        NoteEvent(is_drum=False, program=100, time=2.0, velocity=0, pitch=77, activity={3, 4}),
        NoteEvent(is_drum=True, program=128, time=2.5, velocity=1, pitch=38, activity={3}),
        NoteEvent(is_drum=False, program=33, time=3.0, velocity=0, pitch=62, activity={3})
        ])

    def test_note2note_event_invalid_velocity_value(self):
        notes = [Note(is_drum=0, program=1, onset=0, offset=127, pitch=60, velocity=100)]
        with self.assertRaises(ValueError):
            note2note_event(notes)

    def test_note2note_event_non_empty_notes_list(self):
        notes = [
            Note(is_drum=0, program=1, onset=0, offset=127, pitch=60, velocity=1),
            Note(is_drum=0, program=1, onset=20, offset=127, pitch=62, velocity=1),
            Note(is_drum=0, program=1, onset=40, offset=127, pitch=64, velocity=1)
        ]
        note_events = note2note_event(notes)
        assert len(note_events) == 6

    def test_note2note_event_sort_parameter(self):
        notes = [
            Note(is_drum=0, program=10, onset=0, offset=127, pitch=64, velocity=1),
            Note(is_drum=0, program=10, onset=20, offset=127, pitch=60, velocity=1),
            Note(is_drum=0, program=10, onset=0, offset=127, pitch=62, velocity=1)
        ]
        note_events = note2note_event(notes, sort=True)
        sorted_note_events = sorted(
            note_events, key=lambda n_ev: (n_ev.time, n_ev.is_drum, n_ev.program, \
                                           n_ev.velocity, n_ev.pitch))
        assert note_events == sorted_note_events

class TestNoteEventTools(unittest.TestCase):

    def test_slice_note_events_and_ties(self):
        note_events = [
            NoteEvent(is_drum=False, program=33, time=0, velocity=1, pitch=60, activity=set()),
            NoteEvent(is_drum=True, program=128, time=0.2, velocity=1, pitch=36, activity={0}),
            NoteEvent(is_drum=False, program=33, time=1.5, velocity=0, pitch=60, activity={0}),
            NoteEvent(is_drum=False, program=33, time=1.6, velocity=1, pitch=62, activity=set()),
            NoteEvent(is_drum=False, program=100, time=1.6, velocity=1, pitch=77, activity={3}),
            NoteEvent(is_drum=False, program=100, time=2.0, velocity=0, pitch=77, activity={3, 4}),
            NoteEvent(is_drum=True, program=128, time=2.5, velocity=1, pitch=38, activity={3}),
            NoteEvent(is_drum=False, program=33, time=3.5, velocity=0, pitch=62, activity={3})
        ]
        start_time = 1.5
        end_time = 3.5

        sliced_note_events, tie_note_events, _ = slice_note_events_and_ties(note_events, start_time, end_time)
        assert len(sliced_note_events) == 5
        assert len(tie_note_events) == 1

        # Check if the tie_note_events are as expected
        expected_tie_note_events = [
            NoteEvent(is_drum=False, program=33, time=0, velocity=1, pitch=60, activity=set()),
        ]
        self.assertSequenceEqual(tie_note_events, expected_tie_note_events)

        # Check if the note_events are as expected
        expected_sliced_note_events = [
            NoteEvent(is_drum=False, program=33, time=1.5, velocity=0, pitch=60, activity={0}),
            NoteEvent(is_drum=False, program=33, time=1.6, velocity=1, pitch=62, activity=set()),
            NoteEvent(is_drum=False, program=100, time=1.6, velocity=1, pitch=77, activity={3}),
            NoteEvent(is_drum=False, program=100, time=2.0, velocity=0, pitch=77, activity={3, 4}),
            NoteEvent(is_drum=True, program=128, time=2.5, velocity=1, pitch=38, activity={3})
        ]
        self.assertSequenceEqual(sliced_note_events, expected_sliced_note_events)

    def test_slice_note_events_and_ties_tidyup(self):
        note_events = [
            NoteEvent(is_drum=False, program=33, time=0, velocity=1, pitch=60, activity=set()),
            NoteEvent(is_drum=True, program=128, time=0.2, velocity=1, pitch=36, activity={0}),
            NoteEvent(is_drum=False, program=33, time=1.5, velocity=0, pitch=60, activity={0}),
            NoteEvent(is_drum=False, program=33, time=1.6, velocity=1, pitch=62, activity=set()),
            NoteEvent(is_drum=False, program=100, time=1.6, velocity=1, pitch=77, activity={3}),
            NoteEvent(is_drum=False, program=100, time=2.0, velocity=0, pitch=77, activity={3, 4}),
            NoteEvent(is_drum=True, program=128, time=2.5, velocity=1, pitch=38, activity={3}),
            NoteEvent(is_drum=False, program=33, time=3.5, velocity=0, pitch=62, activity={3})
        ]
        start_time = 1.5
        end_time = 3.5

        sliced_note_events, tie_note_events, _ = slice_note_events_and_ties(
             note_events, start_time, end_time, tidyup=True)
        assert len(sliced_note_events) == 5
        assert len(tie_note_events) == 1

        # Check if the tie_note_events are as expected
        expected_tie_note_events = [
            NoteEvent(is_drum=False, program=33, time=None, velocity=1, pitch=60, activity=None),
        ]
        self.assertSequenceEqual(tie_note_events, expected_tie_note_events)

        # Check if the note_events are as expected
        expected_sliced_note_events = [
            NoteEvent(is_drum=False, program=33, time=1.5, velocity=0, pitch=60, activity=None),
            NoteEvent(is_drum=False, program=33, time=1.6, velocity=1, pitch=62, activity=None),
            NoteEvent(is_drum=False, program=100, time=1.6, velocity=1, pitch=77, activity=None),
            NoteEvent(is_drum=False, program=100, time=2.0, velocity=0, pitch=77, activity=None),
            NoteEvent(is_drum=True, program=128, time=2.5, velocity=1, pitch=38, activity=None)
        ]
        self.assertSequenceEqual(sliced_note_events, expected_sliced_note_events)

    def test_slice_note_events_and_ties_empty_input(self):
        note_events = []
        start_time = 1.0
        end_time = 2.5
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sliced_note_events, tie_note_events, _ = slice_note_events_and_ties(
                note_events, start_time, end_time)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "empty note_events as input" in str(w[-1].message)

        assert sliced_note_events == []
        assert tie_note_events == []

    def test_slice_note_events_and_ties_index_out_of_range(self):
        note_events = [
            NoteEvent(is_drum=False, program=33, time=0.1, velocity=1, pitch=60, activity=set()),
            NoteEvent(is_drum=True, program=128, time=0.2, velocity=1, pitch=36, activity={0}),
            NoteEvent(is_drum=False, program=33, time=1.5, velocity=0, pitch=60, activity={0}),
            NoteEvent(is_drum=True, program=128, time=6, velocity=1, pitch=36, activity=set())
            ]

        start_time = 0
        end_time = 0.1
        sliced_note_events, tie_note_events, _ = slice_note_events_and_ties(
            note_events, start_time, end_time)
        self.assertEqual(len(sliced_note_events), 0)
        self.assertEqual(len(tie_note_events), 0)

        start_time = 0.3
        end_time = 2
        sliced_note_events, tie_note_events, _ = slice_note_events_and_ties(
            note_events, start_time, end_time)
        self.assertEqual(len(sliced_note_events), 1)
        self.assertEqual(len(tie_note_events), 1) # drum has no offset, and activity is not counted

        start_time = 0.3
        end_time = 1
        sliced_note_events, tie_note_events, _ = slice_note_events_and_ties(
            note_events, start_time, end_time)
        self.assertEqual(len(sliced_note_events), 0)
        self.assertEqual(len(tie_note_events), 1) # drum has no offset, and activity is not counted

        start_time = 2
        end_time = 4
        sliced_note_events, tie_note_events, _ = slice_note_events_and_ties(
            note_events, start_time, end_time)
        self.assertEqual(len(sliced_note_events), 0)
        self.assertEqual(len(tie_note_events), 0)

        start_time = 3
        end_time = 4
        sliced_note_events, tie_note_events, _ = slice_note_events_and_ties(
            note_events, start_time, end_time)
        self.assertEqual(len(sliced_note_events), 0)
        self.assertEqual(len(tie_note_events), 0)

        start_time = 3
        end_time = 7
        sliced_note_events, tie_note_events, _ = slice_note_events_and_ties(
            note_events, start_time, end_time)
        self.assertEqual(len(sliced_note_events), 1)
        self.assertEqual(len(tie_note_events), 0)

        start_time = 7
        end_time = 8
        sliced_note_events, tie_note_events, _ = slice_note_events_and_ties(
            note_events, start_time, end_time)
        self.assertEqual(len(sliced_note_events), 0)
        self.assertEqual(len(tie_note_events), 0)


class TestNoteEventToolsMultiSlice(unittest.TestCase):

    def setUp(self):
        self.note_events = [
            NoteEvent(is_drum=False, program=33, time=0, velocity=1, pitch=60, activity=set()),
            NoteEvent(is_drum=True, program=128, time=0.2, velocity=1, pitch=36, activity={0}),
            NoteEvent(is_drum=False, program=33, time=1.5, velocity=0, pitch=60, activity={0}),
            NoteEvent(is_drum=False, program=33, time=1.6, velocity=1, pitch=62, activity=set()),
            NoteEvent(is_drum=False, program=100, time=1.6, velocity=1, pitch=77, activity={3}),
            NoteEvent(is_drum=False, program=100, time=2.0, velocity=0, pitch=77, activity={3, 4}),
            NoteEvent(is_drum=True, program=128, time=2.5, velocity=1, pitch=38, activity={3}),
            NoteEvent(is_drum=False, program=33, time=3.5, velocity=0, pitch=62, activity={3}),
            NoteEvent(is_drum=False, program=33, time=4.0, velocity=0, pitch=62, activity={3}),
            NoteEvent(is_drum=False, program=50, time=5.5, velocity=1, pitch=55, activity=set()),
            NoteEvent(is_drum=False, program=33, time=6.1, velocity=1, pitch=64, activity={9}),
            NoteEvent(is_drum=True, program=128, time=6.5, velocity=1, pitch=36, activity={9, 10}),
            NoteEvent(is_drum=False, program=33, time=7.5, velocity=0, pitch=64, activity={9, 10})
        ]

    def test_slice_note_events_and_ties_continuous_slices(self):
        start_times = [0., 2, 4, 6, 8]
        end_times = [2., 4, 6, 8, 10]
        sliced_note_events_list = []
        sliced_tie_note_events_list = []
        for start_time, end_time in zip(start_times, end_times):
            sliced_note_events, tie_note_events, t = slice_note_events_and_ties(
                self.note_events, start_time, end_time)
            sliced_note_events_list.extend(sliced_note_events) # merge...
            sliced_tie_note_events_list.append(tie_note_events)
        self.assertSequenceEqual(sliced_note_events_list, self.note_events)
        self.assertEqual(len(sliced_tie_note_events_list), 5)
        self.assertEqual(sliced_tie_note_events_list[0], []) # first slice always empty
        self.assertEqual(sliced_tie_note_events_list[4], []) # last slice is empty in this example

    def test_slice_multiple_note_events_and_ties_to_bundle(self):
        start_times = [0., 1]
        duration_sec = 2.
        # Create a bundle from the sliced note events
        ne_bundle = slice_multiple_note_events_and_ties_to_bundle(
            self.note_events, start_times, duration_sec)
        # ne_bundle = NoteEventListsBundle({'note_events': sliced_note_events_list,
        #                                   'tie_note_events': sliced_tie_note_events_list,
        #                                   'start_times': start_times})
        expected_ne_0 = [
            NoteEvent(is_drum=False, program=33, time=0, velocity=1, pitch=60, activity=set()),
            NoteEvent(is_drum=True, program=128, time=0.2, velocity=1, pitch=36, activity={0}),
            NoteEvent(is_drum=False, program=33, time=1.5, velocity=0, pitch=60, activity={0}),
            NoteEvent(is_drum=False, program=33, time=1.6, velocity=1, pitch=62, activity=set()),
            NoteEvent(is_drum=False, program=100, time=1.6, velocity=1, pitch=77, activity={3})]
        self.assertSequenceEqual(ne_bundle['note_events'][0], expected_ne_0)
        expected_ne_1 = [
            NoteEvent(is_drum=False, program=33, time=1.5, velocity=0, pitch=60, activity={0}),
            NoteEvent(is_drum=False, program=33, time=1.6, velocity=1, pitch=62, activity=set()),
            NoteEvent(is_drum=False, program=100, time=1.6, velocity=1, pitch=77, activity={3}),
            NoteEvent(is_drum=False, program=100, time=2.0, velocity=0, pitch=77, activity={3, 4}),
            NoteEvent(is_drum=True, program=128, time=2.5, velocity=1, pitch=38, activity={3})]
        self.assertSequenceEqual(ne_bundle['note_events'][1], expected_ne_1)
        expected_tne_0 = []
        self.assertSequenceEqual(ne_bundle['tie_note_events'][0], expected_tne_0)
        expected_tne_1 = [NoteEvent(is_drum=False, program=33, time=0, velocity=1, pitch=60, activity=set())]
        self.assertSequenceEqual(ne_bundle['tie_note_events'][1], expected_tne_1)
        self.assertEqual(ne_bundle['start_times'], start_times)

    def test_slice_multiple_note_events_and_ties_to_bundle_overlength_case(self):
        # This is a case where the last slices are intended to be empty as in datasets_eval.py
        start_times = [10., 11.]
        duration_sec = 2.
        # Create a bundle from the sliced note events
        ne_bundle = slice_multiple_note_events_and_ties_to_bundle(
            self.note_events, start_times, duration_sec)
        expected_ne_0 = []
        expected_ne_1 = []
        self.assertSequenceEqual(ne_bundle['note_events'][0], expected_ne_0)
        self.assertSequenceEqual(ne_bundle['note_events'][1], expected_ne_1)




class TestNoteEvent2Event(unittest.TestCase):
    def test_note_event2event(self):
        note_events = [NoteEvent(True, 128, 0.2, 1, 36),
                       NoteEvent(False, 33, 1.5, 0, 60),
                       NoteEvent(False, 33, 1.6, 1, 62),
                       NoteEvent(False, 100, 1.6, 1, 77),
                       NoteEvent(False, 100, 2.0, 0, 77),
                       NoteEvent(True, 128, 2.5, 1, 38),
                       NoteEvent(False, 33, 3.0, 0, 62)]
        tie_note_events = [NoteEvent(False, 33, None, 1, 60),
                           NoteEvent(False, 52, None, 1, 40)]
        start_time = 0.0
        tps = 100
        sort = False
        events = note_event2event(note_events, tie_note_events, start_time, tps, sort)

        expected_events = [
            Event('program', 33), Event('pitch', 60),
            Event('program', 52), Event('pitch', 40),
            Event('tie', 0),
            Event('shift', 20), Event('velocity', 1), Event('drum', 36),
            Event('shift', 150), Event('program', 33), Event('velocity', 0), Event('pitch', 60),
            Event('shift', 160), Event('velocity', 1), Event('pitch', 62), Event('program', 100),
                                Event('pitch', 77),
            Event('shift', 200), Event('velocity', 0), Event('pitch', 77),
            Event('shift', 250), Event('velocity', 1), Event('drum', 38),
            Event('shift', 300), Event('program', 33), Event('velocity', 0), Event('pitch', 62)
        ]
        self.assertSequenceEqual(events, expected_events)

    def test_empty_input(self):
        events = note_event2event([])
        expected_events = [Event('tie', 0)]
        self.assertEqual(events, expected_events)

        events = note_event2event([], [])
        self.assertEqual(events, expected_events)

        events = note_event2event([], [], 0)
        self.assertEqual(events, expected_events)

    def test_single_note_event(self):
        note_events = [NoteEvent(False, 33, 1.0, 1, 60)]
        events = note_event2event(note_events)
        expected_events = [
            Event('tie', 0),
            Event('shift', 100), Event('program', 33), Event('velocity', 1), Event('pitch', 60)
        ]
        self.assertSequenceEqual(events, expected_events)

    def test_single_drum_event(self):
        note_events = [NoteEvent(True, 128, 1.0, 1, 36)]
        events = note_event2event(note_events)
        expected_events = [
            Event('tie', 0),
            Event('shift', 100), Event('velocity', 1), Event('drum', 36)
        ]
        self.assertSequenceEqual(events, expected_events)

    def test_multiple_drum_event(self):
        note_events = [NoteEvent(is_drum=True, program=128, time=0.105, velocity=1, pitch=38, activity=set()),
            NoteEvent(is_drum=True, program=128, time=0.11499999999999999, velocity=0, pitch=38, activity=set()),
            NoteEvent(is_drum=True, program=128, time=1.5, velocity=1, pitch=38, activity=set()),
            NoteEvent(is_drum=True, program=128, time=1.51, velocity=0, pitch=38, activity=set()),
            NoteEvent(is_drum=True, program=128, time=2.886, velocity=1, pitch=38, activity=set()),
            NoteEvent(is_drum=True, program=128, time=2.896, velocity=0, pitch=38, activity=set()),
            NoteEvent(is_drum=True, program=128, time=5.528, velocity=1, pitch=38, activity=set()),
            NoteEvent(is_drum=True, program=128, time=5.537999999999999, velocity=0, pitch=38, activity=set()),
            NoteEvent(is_drum=True, program=128, time=7.641, velocity=1, pitch=38, activity=set()),
            NoteEvent(is_drum=True, program=128, time=7.651, velocity=0, pitch=38, activity=set()),
            NoteEvent(is_drum=True, program=128, time=10.413, velocity=1, pitch=38, activity=set()),
            NoteEvent(is_drum=True, program=128, time=10.423, velocity=0, pitch=38, activity=set())]

        events = note_event2event(note_events, [])
        expected_events = [Event(type='tie', value=0),
            Event(type='shift', value=10),
            Event(type='velocity', value=1),
            Event(type='drum', value=38),
            Event(type='shift', value=150),
            Event(type='drum', value=38),
            Event(type='shift', value=289),
            Event(type='drum', value=38),
            Event(type='shift', value=553),
            Event(type='drum', value=38),
            Event(type='shift', value=764),
            Event(type='drum', value=38),
            Event(type='shift', value=1041),
            Event(type='drum', value=38)]
        print(events)
        self.assertSequenceEqual(events, expected_events)


    def test_tie_note_events(self):
        note_events = [NoteEvent(False, 33, 1.0, 1, 60),
                       NoteEvent(False, 33, 2.0, 0, 60)]
        tie_note_events = [NoteEvent(False, 33, None, 1, 60)]
        events = note_event2event(note_events, tie_note_events)
        expected_events = [
            Event('program', 33),
            Event('pitch', 60),
            Event('tie', 0),
            Event('shift', 100),
            Event('velocity', 1),
            Event('pitch', 60),
            Event('shift', 200),
            Event('velocity', 0),
            Event('pitch', 60)
        ]
        self.assertSequenceEqual(events, expected_events)

    def test_rounding_behavior_in_shift(self):
        note_events = [NoteEvent(False, 33, 1.000001, 1, 60),
                       NoteEvent(False, 12, 1.98900, 1, 60), # less than 10ms is ignored
                       NoteEvent(False, 11, 1.99000, 1, 60), # smaller program number first!
                       NoteEvent(False, 10, 1.99100, 1, 60),
                       NoteEvent(False, 33, 1.99999, 1, 60), # less than 10ms is ignored
                       NoteEvent(False, 33, 2.00001, 0, 60)] # offset first!

        events = note_event2event(note_events, sort=True)
        expected_events = [
            Event('tie', 0),
            Event('shift', 100), Event('program', 33), Event('velocity', 1), Event('pitch', 60),
            Event('shift', 199), Event('program', 10), Event('pitch', 60), # smaller program number first!
                                Event('program', 11), Event('pitch', 60),
                                Event('program', 12), Event('pitch', 60),
            Event('shift', 200), Event('program', 33), Event('velocity', 0), Event('pitch', 60),
                               Event('velocity', 1), Event('pitch', 60)] # offset first!
        self.assertSequenceEqual(events, expected_events)

    def test_rounding_behavior_in_shift_without_sort(self):
        note_events = [NoteEvent(False, 12, 1.98900, 1, 60), # less than 10ms is ignored
                       NoteEvent(False, 11, 1.99000, 1, 60)]

        # If sort=False, the order of events in quantized timing is not guaranteed.
        # To avoid sort(), midi2note(..., quantize=True) is default.
        events = note_event2event(note_events, sort=False)
        expected_events = [
            Event('tie', 0),
            Event('shift', 199), Event('program', 12), Event('velocity', 1), Event('pitch', 60),
            Event('program', 11), Event('pitch', 60)]
        self.assertSequenceEqual(events, expected_events)

    def test_rounding_behavior_in_shift_without_sort_quantized_note_event(self):
        note_events = [NoteEvent(False, 11, 1.99000, 1, 60),
                       NoteEvent(False, 12, 1.99000, 1, 60)]
        events = note_event2event(note_events, sort=False)
        expected_events = [
            Event('tie', 0),
            Event('shift', 199), Event('program', 11), Event('velocity', 1), Event('pitch', 60),
            Event('program', 12), Event('pitch', 60)]
        self.assertSequenceEqual(events, expected_events)


class TestNoteEvent2EventProcessTime(unittest.TestCase):

    def setUp(self):
        self.note_events = [
            NoteEvent(is_drum=False, program=i % 128, time=i / 10.0, velocity=64, pitch=i % 128)
            for i in range(333)
        ]

        self.rand_note_events = [
            NoteEvent(is_drum=bool(random.randint(2)), program=random.randint(128),
                      time=random.randint(500) / 10.0, velocity=random.randint(2),
                      pitch=random.randint(128))
            for i in range(333)
        ]

    @pytest.mark.timeout(0.1)  # Set a timeout of 30 ms
    def test_large_note_event_list(self):
        # B = 64, Sequence_length = 333. 64 * 333 = 21248 with single cpu process
        for i in range(64):
            events = note_event2event(self.note_events, sort=False)

    @pytest.mark.timeout(0.1)  # Set a timeout of 35 ms
    def test_large_random_note_event_list_with_sort(self):
        # B = 64, Sequence_length = 333. 64 * 333 = 21248 with single cpu process
        for i in range(64):
            events = note_event2event(self.rand_note_events, sort=True)

# yapf: enable
if __name__ == '__main__':
    unittest.main()
