import unittest
import pytest
from numpy import random

from utils.note_event_dataclasses import NoteEvent, Event, EventRange
from utils.event_codec import FastCodec as Codec
from utils.tokenizer import EventTokenizer
from utils.tokenizer import NoteEventTokenizer


class TestEventTokenizerBase(unittest.TestCase):

    def test_encode_and_decode(self):
        tokenizer = EventTokenizer()
        events = [
            Event('pitch', 64),
            Event('velocity', 1),
            Event('tie', 0),
            Event('program', 10),
            Event('drum', 0)
        ]
        tokens = tokenizer.encode(events)
        decoded_events = tokenizer.decode(tokens)
        self.assertEqual(events, decoded_events)

    def test_unknown_codec_name(self):
        with self.assertRaises(ValueError):
            EventTokenizer(base_codec='unknown')

    def test_unknown_codec_type(self):
        with self.assertRaises(TypeError):
            EventTokenizer(base_codec=123)

    def test_encode_and_decode_with_custom_codec(self):

        special_tokens = ['PAD', 'EOS', 'SOS', 'T']
        max_shift_steps = 100
        event_ranges = [
            EventRange('eat', min_value=0, max_value=9),
            EventRange('sleep', min_value=0, max_value=9),
            EventRange('play', min_value=0, max_value=1)
        ]

        my_codec = Codec(special_tokens, max_shift_steps, event_ranges)
        tokenizer = EventTokenizer(my_codec)
        events = [
            Event('eat', 3),
            Event('shift', 9),
            Event('sleep', 9),
            Event('shift', 20),
            Event('play', 1)
        ]
        tokens = tokenizer.encode(events)

        # 0~3: special tokens
        # 4~103: shift tokens
        # 104~112: eat tokens
        # 113~121: sleep tokens
        # 122~123: play tokens
        expected_tokens = [107, 13, 123, 24, 125]
        self.assertEqual(tokens, expected_tokens)
        decoded_events = tokenizer.decode(tokens)
        self.assertEqual(events, decoded_events)


class TestEventTokenizerBaseProcessTime(unittest.TestCase):

    def setUp(self) -> None:
        self.tokenizer = EventTokenizer('mt3')
        self.random_tokens = random.randint(0, 500, size=333)
        self.events = [
            Event(type='pitch', value=60),
            Event(type='velocity', value=1),
            Event(type='program', value=0),
            Event(type='shift', value=10),
            Event(type='tie', value=0),
            Event(type='drum', value=0),
        ] * 55

    @pytest.mark.timeout(0.008)  # 32 ms --> 8 ms
    def test_event_tokenizer_encode(self):
        for i in range(64):
            encoded = self.tokenizer.encode(self.events)

    @pytest.mark.timeout(0.01)  # 40 ms --> 10 ms
    def test_event_tokenizer_decode(self):
        for i in range(64):
            decoded = self.tokenizer.decode(self.random_tokens)


# yapf: disable
class NoteEventTokenizerTest(unittest.TestCase):

    def test_note_event_tokenizer_encode(self):
        tokenizer = NoteEventTokenizer()
        note_events = [
            NoteEvent(is_drum=False, program=33, time=0, velocity=1, pitch=60, activity=set()),
            NoteEvent(is_drum=True, program=128, time=0.2, velocity=1, pitch=36, activity=set()),
            NoteEvent(is_drum=False, program=33, time=1.5, velocity=0, pitch=60, activity=set())
            ]
        tokens = tokenizer.encode(note_events)
        decoded_events, decoded_tie_events, last_activity, err_cnt = tokenizer.decode(tokens)
        self.assertSequenceEqual(note_events, decoded_events)
        self.assertSequenceEqual([], decoded_tie_events)
        self.assertEqual(len(last_activity), 0)
        self.assertEqual(len(err_cnt), 0)

    def test_note_event_tokenizer_encode_plus(self):
        tokenizer = NoteEventTokenizer()
        note_events = [
            NoteEvent(is_drum=False, program=33, time=0, velocity=1, pitch=60, activity=set()),
            NoteEvent(is_drum=True, program=128, time=0.2, velocity=1, pitch=36, activity=set()),
            NoteEvent(is_drum=False, program=33, time=1.5, velocity=0, pitch=60, activity=set())
            ]
        tokens = tokenizer.encode_plus(note_events, max_length=30)
        decoded_events, decoded_tie_events, last_activity, err_cnt = tokenizer.decode(tokens)
        self.assertSequenceEqual(note_events, decoded_events)
        self.assertSequenceEqual([], decoded_tie_events)
        self.assertEqual(len(last_activity), 0)
        self.assertEqual(len(err_cnt), 0)



# yapf: enable
if __name__ == '__main__':
    unittest.main()