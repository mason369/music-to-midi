# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""event_codec_test.py:

This file contains tests for the following classes:
• Event
• EventRange
• FastCodec equivalent to MT3 author's Codec

See tokenizer_test.py for the FastCodec performance benchmark

"""
import unittest
from utils.note_event_dataclasses import Event, EventRange
from utils.event_codec import FastCodec as Codec
# from utils.event_codec import Codec


class TestEvent(unittest.TestCase):

    def test_Event(self):
        e = Event(type='shift', value=0)
        self.assertEqual(e.type, 'shift')
        self.assertEqual(e.value, 0)


class TestEventRange(unittest.TestCase):

    def test_EventRange(self):
        er = EventRange('abc', min_value=0, max_value=500)
        self.assertEqual(er.type, 'abc')
        self.assertEqual(er.min_value, 0)
        self.assertEqual(er.max_value, 500)


class TestEventCodec(unittest.TestCase):

    def test_event_codec(self):
        ec = Codec(
            special_tokens=['asd'],
            max_shift_steps=1001,
            event_ranges=[
                EventRange('pitch', min_value=0, max_value=127),
                EventRange('velocity', min_value=0, max_value=1),
                EventRange('tie', min_value=0, max_value=0),
                EventRange('program', min_value=0, max_value=127),
                EventRange('drum', min_value=0, max_value=127),
            ],
        )

        events = [
            Event(type='shift', value=0),  # actually not needed
            Event(type='shift', value=1),  # 10 ms shift
            Event(type='shift', value=1000),  # 10 s shift
            Event(type='pitch', value=0),  # lowest pitch 8.18 Hz
            Event(type='pitch', value=60),  # C4 or 261.63 Hz
            Event(type='pitch', value=127),  # highest pitch G9 or 12543.85 Hz
            Event(type='velocity', value=0),  # lowest velocity)
            Event(type='velocity', value=1),  # lowest velocity)
            Event(type='tie', value=0),  # tie
            Event(type='program', value=0),  # program
            Event(type='program', value=127),  # program
            Event(type='drum', value=0),  # drum
            Event(type='drum', value=127),  # drum
        ]

        encoded = [ec.encode_event(e) for e in events]
        decoded = [ec.decode_event_index(idx) for idx in encoded]
        self.assertSequenceEqual(events, decoded)


class TestEventCodecErrorCases(unittest.TestCase):

    def setUp(self):
        self.event_ranges = [
            EventRange("program", 0, 127),
            EventRange("pitch", 0, 127),
            EventRange("velocity", 0, 3),
            EventRange("drum", 0, 127),
            EventRange("tie", 0, 1),
        ]
        self.ec = Codec([], 1000, self.event_ranges)

    def test_encode_event_with_invalid_event_type(self):
        with self.assertRaises(ValueError):
            self.ec.encode_event(Event("unknown_event_type", 50))

    def test_encode_event_with_invalid_event_value(self):
        with self.assertRaises(ValueError):
            self.ec.encode_event(Event("program", 200))

    def test_event_type_range_with_invalid_event_type(self):
        with self.assertRaises(ValueError):
            self.ec.event_type_range("unknown_event_type")

    def test_decode_event_index_with_invalid_index(self):
        with self.assertRaises(ValueError):
            self.ec.decode_event_index(1000000)


class TestEventCodecVocabulary(unittest.TestCase):

    def test_encode_event_using_program_vocabulary(self):
        prog_vocab = {"Piano": [0, 1, 2, 3, 4, 5, 6, 7], "xxx": [50, 30, 120]}
        ec = Codec(special_tokens=['asd'],
                   max_shift_steps=1001,
                   event_ranges=[
                       EventRange('pitch', min_value=0, max_value=127),
                       EventRange('velocity', min_value=0, max_value=1),
                       EventRange('tie', min_value=0, max_value=0),
                       EventRange('program', min_value=0, max_value=127),
                       EventRange('drum', min_value=0, max_value=127),
                   ],
                   program_vocabulary=prog_vocab)

        events = [
            Event(type='program', value=0),  # 0 --> 0
            Event(type='program', value=7),  # 7 --> 0
            Event(type='program', value=111),  # 111 --> 111
            Event(type='program', value=30),  # 30 --> 50
        ]
        encoded = [ec.encode_event(e) for e in events]
        expected = [1133, 1133, 1244, 1183]
        self.assertSequenceEqual(encoded, expected)

    def test_encode_event_using_drum_vocabulary(self):
        drum_vocab = {"Kick": [50, 51, 52], "Snare": [53, 54]}
        ec = Codec(special_tokens=['asd'],
                   max_shift_steps=1001,
                   event_ranges=[
                       EventRange('pitch', min_value=0, max_value=127),
                       EventRange('velocity', min_value=0, max_value=1),
                       EventRange('tie', min_value=0, max_value=0),
                       EventRange('program', min_value=0, max_value=127),
                       EventRange('drum', min_value=0, max_value=127),
                   ],
                   drum_vocabulary=drum_vocab)

        events = [
            Event(type='drum', value=50),
            Event(type='drum', value=51),
            Event(type='drum', value=53),
            Event(type='drum', value=54),
        ]
        encoded = [ec.encode_event(e) for e in events]
        self.assertEqual(encoded[0], encoded[1])
        self.assertEqual(encoded[2], encoded[3])


if __name__ == '__main__':
    unittest.main()
