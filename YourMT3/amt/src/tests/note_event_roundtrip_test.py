# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
""" note_event_roundtrip_test.py:
This file contains tests for the round trip conversion between Note and 
NoteEvent and Event. 

Itinerary 1:
    NoteEvent → Event → Token → Event → NoteEvent 

Itinerary 2:
    Note → NoteEvent → Event → Token → Event → NoteEvent → Note

Training:
    (Dataloader) NoteEvent → (augmentation) → Event → Token
    
Evaluation :
    (Model side) Token → Event → NoteEvent → Note → (mir_eval)
    (Ground Truth) Note → (mir_eval)
    
 • This conversion may fail for unsorted and unquantized timing events.
 • Acitivity attribute of NoteEvent is often ignorable. 

"""
import unittest
import numpy as np
from assert_fns import assert_notes_almost_equal
from assert_fns import assert_note_events_almost_equal
from assert_fns import assert_track_metrics_score1

from utils.note_event_dataclasses import Note, NoteEvent, Event
from utils.note2event import note2note_event, note_event2event
from utils.note2event import validate_notes, trim_overlapping_notes
from utils.event2note import event2note_event, note_event2note
from utils.tokenizer import EventTokenizer, NoteEventTokenizer
from utils.midi import note_event2midi
from utils.midi import midi2note
from utils.note2event import slice_multiple_note_events_and_ties_to_bundle
from utils.event2note import merge_zipped_note_events_and_ties_to_notes
from utils.metrics import compute_track_metrics
from config.vocabulary import GM_INSTR_FULL, SINGING_SOLO_CLASS
# yapf: disable

class TestNoteEventRoundTrip1(unittest.TestCase):

    def setUp(self) -> None:
        self.note_events = [
            NoteEvent(is_drum=False, program=33, time=0, velocity=1, pitch=60, activity=set()),
            NoteEvent(is_drum=True, program=128, time=0.2, velocity=1, pitch=36, activity=set()),
            NoteEvent(is_drum=False, program=33, time=1.5, velocity=0, pitch=60, activity=set()),
            NoteEvent(is_drum=False, program=33, time=1.6, velocity=1, pitch=62, activity=set()),
            NoteEvent(is_drum=False, program=100, time=1.6, velocity=1, pitch=77, activity=set()),
            NoteEvent(is_drum=False, program=100, time=2.0, velocity=0, pitch=77, activity=set()),
            NoteEvent(is_drum=True, program=128, time=2.0, velocity=1, pitch=38, activity=set()),
            NoteEvent(is_drum=False, program=33, time=2.0, velocity=0, pitch=62, activity=set())
        ]
        self.tokenizer = EventTokenizer()

    def test_note_event_rt_ne2e2ne(self):
        """ NoteEvent → Event → NoteEvent """
        note_events = self.note_events.copy()
        events = note_event2event(note_events=note_events,
                                  tie_note_events=None,
                                  start_time=0, sort=True)
        recon_note_events, unused_tie_note_events, unsued_last_activity, err_cnt = event2note_event(
            events, start_time=0, sort=True, tps=100)

        self.assertSequenceEqual(note_events, recon_note_events)
        self.assertEqual(len(err_cnt), 0)

    def test_note_event_rt_ne2e2t2e2ne(self):
        """ NoteEvent → Event → Token → Event → NoteEvent """
        note_events = self.note_events.copy()
        events = note_event2event(
            note_events=note_events, tie_note_events=None, start_time=0, sort=True)
        tokens = self.tokenizer.encode(events)
        events = self.tokenizer.decode(tokens)
        recon_note_events, unused_tie_note_events, unsued_last_activity, err_cnt = event2note_event(
            events, start_time=0, sort=True, tps=100)

        self.assertSequenceEqual(note_events, recon_note_events)
        self.assertEqual(len(err_cnt), 0)

class TestNoteEvent2(unittest.TestCase):

    def setUp(self) -> None:
        notes = [
            Note(is_drum=False, program=33, onset=0, offset=1.5, pitch=60, velocity=1),
            Note(is_drum=True, program=128, onset=0.2, offset=0.21, pitch=36, velocity=1),
            Note(is_drum=False, program=25, onset=0.4, offset=1.1, pitch=55, velocity=1),
            Note(is_drum=True, program=128, onset=1, offset=1.01, pitch=42, velocity=1),
            Note(is_drum=False, program=33, onset=1.2, offset=1.8, pitch=80, velocity=1),
            Note(is_drum=False, program=33, onset=1.6, offset=2.0, pitch=62, velocity=1),
            Note(is_drum=False, program=100, onset=1.6, offset=2.0, pitch=77, velocity=1),
            Note(is_drum=False, program=98, onset=1.7, offset=2.0, pitch=77, velocity=1),
            Note(is_drum=True, program=128, onset=1.9, offset=1.91, pitch=38, velocity=1)
            ]

        # Validate and trim notes to make sure they are valid.
        _notes = validate_notes(notes, fix=True)
        self.assertSequenceEqual(notes, _notes)
        _notes = trim_overlapping_notes(notes, sort=True)
        self.assertSequenceEqual(notes, _notes)

        self.notes = notes
        self.tokenizer = EventTokenizer()


    def test_note_event_rt_n2ne2e2t2e2ne2n(self):
        """ Note → NoteEvent → Event → Token → Event → NoteEvent → Note """
        notes = self.notes.copy()
        note_events = note2note_event(notes=notes, sort=True)
        events = note_event2event(note_events=note_events,
                                    tie_note_events=None,
                                    start_time=0,
                                    tps=100,
                                    sort=True)
        tokens = self.tokenizer.encode(events)
        events = self.tokenizer.decode(tokens)
        recon_note_events, unused_tie_note_events, unsued_last_activity, err_cnt = event2note_event(
            events, start_time=0, sort=True, tps=100)
        self.assertEqual(len(err_cnt), 0)

        recon_notes, err_cnt = note_event2note(note_events=recon_note_events, sort=True)
        self.assertEqual(len(err_cnt), 0)
        assert_notes_almost_equal(notes, recon_notes, delta=5e-3) # 5 ms on/offset tolerance

    # def test_encoding_from_midi_without_slicing_zz(self):
    #     """ MIDI → Note → NoteEvent → Event → Token → Event → NoteEvent → Note → MIDI """
    #     src_midi_file = 'extras/examples/1727.mid'
    #     notes, _ = midi2note(src_midi_file, quantize=False)
    #     note_events = note2note_event(notes=notes, sort=True)
    #     events = note_event2event(note_events=note_events,
    #                                 tie_note_events=None,
    #                                 start_time=0,
    #                                 tps=100,
    #                                 sort=True)
    #     # check acculuated time by all the shift events
    #     last_shift = 0
    #     for ev in events:
    #         if ev.type == "shift":
    #             last_shift = ev.value
    #     last_shift_in_sec = last_shift / 100 # 447.04
    #     assert last_shift_in_sec == 447.04
    #     # compare with the last offset time)
    #     last_offset_time = 0.
    #     for n in notes:
    #         if last_offset_time < n.offset:
    #             last_offset_time = n.offset # 447.0395833...
    #     self.assertAlmostEqual(last_shift_in_sec, last_offset_time, delta=1e-3)

    #     tokens = self.tokenizer.encode(events)
    #     # reconustrction -----------------------------------------------------------
    #     recon_events = self.tokenizer.decode(tokens)
    #     self.assertSequenceEqual(events, recon_events)
    #     recon_note_events, unused_tie_note_events, err_cnt = event2note_event(recon_events)
    #     self.assertEqual(len(err_cnt), 0)
    #     assert_note_events_almost_equal(note_events, recon_note_events)
    #     recon_notes, err_cnt = note_event2note(note_events=recon_note_events, sort=True, fix_offset=False)
    #     self.assertEqual(len(err_cnt), 0)
    #     assert_notes_almost_equal(notes, recon_notes, delta=5e-3)
    #     # evaluation without MIDI
    #     drum_metric, non_drum_metric, instr_metric = compute_track_metrics(recon_notes, notes, eval_vocab=GM_INSTR_FULL, onset_threshold=0.5)
    #     assert_track_metrics_score1(drum_metric)
    #     assert_track_metrics_score1(non_drum_metric)
    #     assert_track_metrics_score1(instr_metric)

    #     # evaluation thourgh MIDI
    #     note_event2midi(recon_note_events, output_file='extras/examples/recon_1727.mid')
    #     re_recon_notes, _ = midi2note('extras/examples/recon_1727.mid', quantize=False)
    #     drum_metric, non_drum_metric, instr_metric = compute_track_metrics(re_recon_notes, notes, eval_vocab=GM_INSTR_FULL, onset_threshold=0.5)
    #     assert_track_metrics_score1(drum_metric)
    #     assert_track_metrics_score1(non_drum_metric)
    #     assert_track_metrics_score1(instr_metric)

    def test_encoding_from_midi_with_slicing_zz(self):
        src_midi_file = 'extras/examples/2106.mid' # 'extras/examples/1727.mid'# 'extras/examples/1733.mid' # these are from musicnet_em
        notes, max_time = midi2note(src_midi_file, quantize=False)
        note_events = note2note_event(notes=notes, sort=True)

        # slice note events
        num_segs = int(max_time * 16000 // 32757 + 1)
        seg_len_sec = 32767 / 16000
        start_times = [i * seg_len_sec for i in range(num_segs)]
        note_event_segments = slice_multiple_note_events_and_ties_to_bundle(
                note_events,
                start_times,
                seg_len_sec,
            )

        # encode
        tokenizer = NoteEventTokenizer()
        token_array = np.zeros((num_segs, 1024), dtype=np.int32)
        for i, tup in enumerate(list(zip(*note_event_segments.values()))):
            padded_tokens = tokenizer.encode_plus(*tup)
            token_array[i, :] = padded_tokens

        # decode: warning: Invalid pitch event without program or velocity --> solved
        zipped_note_events_and_tie, list_events, err_cnt = tokenizer.decode_list_batches(
                    [token_array], start_times, return_events=True)
        self.assertEqual(len(err_cnt), 0)

        # First check, the number of empty note_events and tie_note_events
        cnt_org_empty = 0
        cnt_recon_empty = 0
        for i, (recon_note_events, recon_tie_note_events, recon_last_activity, recon_start_times) in enumerate(zipped_note_events_and_tie):
            org_note_events = note_event_segments['note_events'][i]
            org_tie_note_events = note_event_segments['tie_note_events'][i]
            if org_note_events == []:
                cnt_org_empty += 1
            if recon_note_events == []:
                cnt_recon_empty += 1

        assert len(org_note_events) == len(recon_note_events) # passed after bug fix
        # self.assertEqual(len(org_tie_note_events), len(recon_tie_note_events))


        # Check the reconstruction of note_events
        for i, (recon_note_events, recon_tie_note_events, recon_last_activity, recon_start_times) in enumerate(zipped_note_events_and_tie):
            org_note_events = note_event_segments['note_events'][i]
            org_tie_note_events = note_event_segments['tie_note_events'][i]

            org_note_events.sort(key=lambda n_ev: (n_ev.time, n_ev.is_drum, n_ev.program, n_ev.velocity, n_ev.pitch))
            org_tie_note_events.sort(key=lambda n_ev: (n_ev.program, n_ev.pitch))
            recon_note_events.sort(key=lambda n_ev: (n_ev.time, n_ev.is_drum, n_ev.program, n_ev.velocity, n_ev.pitch))
            recon_tie_note_events.sort(key=lambda n_ev: (n_ev.program, n_ev.pitch))

            assert_note_events_almost_equal(org_note_events, recon_note_events)
            assert_note_events_almost_equal(org_tie_note_events, recon_tie_note_events, ignore_time=True)

        # Check notes
        recon_notes, err_cnt = merge_zipped_note_events_and_ties_to_notes(zipped_note_events_and_tie, fix_offset=False)
        self.assertEqual(len(err_cnt), 0)
        assert_notes_almost_equal(notes, recon_notes, delta=5.1e-3)

        # Check metric
        drum_metric, non_drum_metric, instr_metric = compute_track_metrics(
            recon_notes, notes, eval_vocab=GM_INSTR_FULL, onset_tolerance=0.005) # 5ms
        self.assertEqual(non_drum_metric['onset_f'], 1.0)
# yapf: enable

if __name__ == '__main__':
    unittest.main()
