# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
import sys
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
from collections import Counter
from scipy.stats import hmean
from mir_eval.transcription import precision_recall_f1_overlap
from mir_eval.multipitch import evaluate
from mir_eval.melody import to_cent_voicing, raw_pitch_accuracy, raw_chroma_accuracy, overall_accuracy
from mir_eval.util import midi_to_hz
from utils.note_event_dataclasses import Note

EPS = sys.float_info.epsilon


def f1_measure(p, r):
    return hmean([p + EPS, r + EPS]) - EPS


def round_float(l=[], ndigits=4):
    return [round(x, ndigits) for x in l]


# Convert Notes to pitches and intervals for mir_eval note-wise evaluation
def extract_pitches_intervals_from_notes(notes: List[Note], is_drum: bool = False) -> Dict[str, np.ndarray]:
    # drum offsets will be ignored anyways...
    pitches = [midi_to_hz(n.pitch) for n in notes]
    if is_drum:
        intervals = [[n.onset, n.onset + 0.008] for n in notes]
    else:
        intervals = [[n.onset, n.offset] for n in notes]
    return {
        "pitches": np.array(pitches),  # (L,)
        "intervals": np.array(intervals),  # (L, 2)
    }


# Convert Notes to time and freqs for mir_eval frame-wise evaluation
def extract_frame_time_freq_from_notes(notes: List[Note],
                                       is_drum: bool = False,
                                       hop_size_sec: float = 0.0625) -> Dict[str, np.ndarray]:
    if len(notes) == 0:
        return {
            "time": np.array([]),
            "freqs": [[]],
            "roll": np.zeros((0, 128)),
        }

    # drum offsets will be ignored anyways...
    note_pitches = [n.pitch for n in notes]
    last_offset = max([n.offset for n in notes[-20:]])
    shape = (int(last_offset / hop_size_sec), 128)
    roll = np.zeros(shape)
    if is_drum:
        frame_intervals = [[int(n.onset / hop_size_sec), int(n.onset / hop_size_sec) + 1] for n in notes]
    else:
        frame_intervals = [[
            int(n.onset / hop_size_sec),
            max(int(n.offset / hop_size_sec),
                int(n.onset / hop_size_sec) + 1)
        ] for n in notes]
    # create frame-level piano-roll
    for note_pitch, (frame_onset, frame_offset) in zip(note_pitches, frame_intervals):
        roll[frame_onset:frame_offset, note_pitch] = 1

    # take frequency in the range of [16, 110] due to the limitation of mir_eval
    roll[:, :16] = 0
    roll[:, 110:] = 0

    time = np.arange(shape[0])
    frame_pitches = [roll[t, :].nonzero()[0] for t in time]
    return {
        "time": time * hop_size_sec,
        "freqs": [np.array([midi_to_hz(p) for p in pitches]) for pitches in frame_pitches],
        "roll": roll,
    }


# Evaluation: Single instrument Note Onset F1 & OnsetOffset F1
def mir_eval_note_f1(est_pitches: np.ndarray,
                     est_intervals: np.ndarray,
                     ref_pitches: np.ndarray,
                     ref_intervals: np.ndarray,
                     is_drum: bool = False,
                     add_micro_metric: bool = False,
                     suffix: Optional[str] = None,
                     onset_tolerance: float = 0.05) -> Dict[str, Any]:
    """ Instrument-agnostic Note F1 score
    
    Args:
        est_pitches (np.ndarray): Estimated pitches (Hz) shape=(n,)
        est_intervals (np.ndarray): Estimated intervals (seconds) shape=(n, 2)
        ref_pitches (np.ndarray): Reference pitches (Hz) shape=(n,)
        ref_intervals (np.ndarray): Reference intervals (seconds) shape=(n, 2)
        is_drum (bool, optional): Whether the instrument is drum. Defaults to False.
        suffix (Optional[str], optional): Suffix to add to the metric names. Defaults to None.
    
    Returns:
        Dict[str, Any]: Instrument-agnostic Note F1 score. np.nan if empty.

    """
    if len(ref_pitches) == 0 and len(est_pitches) == 0:
        metrics = {
            'onset_f': np.nan,
            'offset_f': np.nan,
        }
        onset_p, onset_r, offset_p, offset_r = np.nan, np.nan, np.nan, np.nan
    elif len(ref_pitches) == 0 and len(est_pitches) != 0:
        metrics = {
            'onset_f': np.nan,  # No false negatives, recall and F1 will be NaN
            'offset_f': np.nan,  # No false negatives, recall and F1 will be NaN
        }
        onset_p, onset_r, offset_p, offset_r = 0., np.nan, 0., np.nan
    # Add the following elif case to handle the situation when there are reference pitches but no estimated pitches
    elif len(ref_pitches) != 0 and len(est_pitches) == 0:
        metrics = {
            'onset_f': 0.,  # No false positives, precision is NaN. recall and F1 are 0.
            'offset_f': 0.,  # No false positives, precision is NaN. recall and F1 are 0.
        }
        onset_p, onset_r, offset_p, offset_r = np.nan, 0., np.nan, 0.
    else:
        metrics = {}
        onset_p, onset_r, metrics['onset_f'], _ = precision_recall_f1_overlap(ref_intervals,
                                                                              ref_pitches,
                                                                              est_intervals,
                                                                              est_pitches,
                                                                              onset_tolerance=onset_tolerance,
                                                                              pitch_tolerance=50.,
                                                                              offset_ratio=None)
        if is_drum is not True:
            offset_p, offset_r, metrics['offset_f'], _ = precision_recall_f1_overlap(ref_intervals,
                                                                                     ref_pitches,
                                                                                     est_intervals,
                                                                                     est_pitches,
                                                                                     onset_tolerance=onset_tolerance,
                                                                                     pitch_tolerance=50.,
                                                                                     offset_ratio=0.2)

    if add_micro_metric is True:
        metrics['micro_onset_p'] = {'value': onset_p, 'weight': len(est_pitches)}
        metrics['micro_onset_r'] = {'value': onset_r, 'weight': len(ref_pitches)}
        if is_drum is not True:
            metrics['micro_offset_p'] = {'value': offset_p, 'weight': len(est_pitches)}
            metrics['micro_offset_r'] = {'value': offset_r, 'weight': len(ref_pitches)}

    if is_drum:
        # remove offset metrics, and add suffix '_drum' for drum onset metrics
        metrics = {k + '_drum' if 'onset' in k else k: v for k, v in metrics.items() if 'offset' not in k}

    if suffix:
        metrics = {k + '_' + suffix: v for k, v in metrics.items()}

    return metrics


# Evaluation: Frame F1
def mir_eval_frame_f1(est_time_freqs: Dict[str, List[np.ndarray]],
                      ref_time_freqs: Dict[str, List[np.ndarray]],
                      suffix: Optional[str] = None) -> Dict[str, float]:
    """ Instrument-agnostic Note F1 score
    
    Args:
        est_time_freqs Dict[str, List[np.ndarray]]: Estimated time, freqs and piano-roll
            {
                'time': np.ndarray, Estimated time indices in seconds.
                'freqs': List[np.ndarray], Estimated frequencies in Hz.
                'roll': np.ndarray, Estimated piano-roll.
            }
        ref_time_freqs Dict[str, List[np.ndarray]]: Reference time, freqs and piano-roll
            {
                'time': np.ndarray, Reference time indices in seconds.
                'freqs': List[np.ndarray], Reference frequencies in Hz.
                'roll': np.ndarray, Reference piano-roll.
            }
        suffix (Optional[str], optional): Suffix to add to the metric names. Defaults to None.
    
    Returns:
        Tuple[Counter, Dict]: Instrument-agnostic Note F1 score

    """
    if np.sum(ref_time_freqs['roll']) == 0 and np.sum(est_time_freqs['roll']) == 0:
        metrics = {
            'frame_f': np.nan,
            'frame_f_pc': np.nan,
        }
    elif np.sum(ref_time_freqs['roll']) == 0 and np.sum(est_time_freqs['roll']) != 0:
        metrics = {
            'frame_f': np.nan,  # F1-score will be NaN
            'frame_f_pc': np.nan,
        }
    # Add the following elif case to handle the situation when there are reference pitches but no estimated pitches
    elif np.sum(ref_time_freqs['roll']) != 0 and np.sum(est_time_freqs['roll']) == 0:
        metrics = {
            'frame_f': 0.,  # F1-score will be 0.
            'frame_f_pc': 0.,
        }
    else:
        # frame-wise evaluation
        res = evaluate(ref_time=ref_time_freqs['time'],
                       ref_freqs=ref_time_freqs['freqs'],
                       est_time=est_time_freqs['time'],
                       est_freqs=est_time_freqs['freqs'])
        frame_f = f1_measure(res['Precision'], res['Recall'])
        frame_f_pc = f1_measure(res['Chroma Precision'], res['Chroma Recall'])
        metrics = {
            'frame_f': frame_f,
            'frame_f_pc': frame_f_pc,
        }

    if suffix:
        metrics = {k + '_' + suffix: v for k, v in metrics.items()}

    return metrics


# Evaluation: Melody metrics
def mir_eval_melody_metric(est_pitches: np.ndarray,
                           est_intervals: np.ndarray,
                           ref_pitches: np.ndarray,
                           ref_intervals: np.ndarray,
                           cent_tolerance: float = 50,
                           suffix: Optional[str] = None) -> Dict[str, Any]:
    """ Melody metrics: Raw Pitch Accuracy, Raw Chroma Accuracy, Overall Accuracy
        
    Args:
        est_pitches (np.ndarray): Estimated pitches (Hz) shape=(n,)
        est_intervals (np.ndarray): Estimated intervals (seconds) shape=(n, 2)
        ref_pitches (np.ndarray): Reference pitches (Hz) shape=(n,)
        ref_intervals (np.ndarray): Reference intervals (seconds) shape=(n, 2)
        cent_tolerance (float, optional): Cent tolerance. Defaults to 50.
        suffix (Optional[str], optional): Suffix to add to the metric names. Defaults to None.

    Returns:
        Dict[str, Any]: RPA, RCA, OA
        
    """
    try:
        (ref_v, ref_c, est_v, est_c) = to_cent_voicing(ref_intervals[:, 0:1],
                                                       ref_pitches,
                                                       est_intervals[:, 0:1],
                                                       est_pitches,
                                                       hop=0.01)
        # Your code here to calculate rpa based on the outputs of to_cent_voicing
    except Exception as e:
        print(f"Error occurred: {e}")
        return {
            'melody_rpa' + ('_' + suffix if suffix else ''): np.nan,
            'melody_rca' + ('_' + suffix if suffix else ''): np.nan,
            'melody_oa' + ('_' + suffix if suffix else ''): np.nan,
        }

    rpa = raw_pitch_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance)
    rca = raw_chroma_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance)
    oa = overall_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance)
    return {
        'melody_rpa' + ('_' + suffix if suffix else ''): rpa,
        'melody_rca' + ('_' + suffix if suffix else ''): rca,
        'melody_oa' + ('_' + suffix if suffix else ''): oa,
    }


def test():
    ref_pitches = np.array([100, 100, 200, 300])  # in Hz
    ref_intervals = np.array([
        [0, 1],  # in seconds
        [2, 3],
        [5, 12],
        [1, 10]
    ])
    est_pitches = ref_pitches.copy()
    est_intervals = ref_intervals.copy()
    mir_eval_note_f1(ref_pitches, ref_intervals, ref_pitches, ref_intervals)
    """
    result:

    (Counter({
        'note_onset/precision': 1.0,
        'note_onset/recall': 1.0,
        'note_onset/f1': 1.0,
        'note_offset/precision': 1.0,
        'note_offset/recall': 1.0,
        'note_offset/f1': 1.0
    })
    """

    est_pitches = np.array([101, 100, 200, 300])  # in Hz
    est_intervals = np.array([
        [0.3, 1],  # wrong onset, thus on-offset incorrect too.
        [2, 3],
        [5, 12],
        [1, 10]
    ])
    mir_eval_note_f1(est_pitches, est_intervals, ref_pitches, ref_intervals)
    # note_onset/f1': 0.75,  'note_offset/f1': 0.75}),

    est_pitches = np.array([101, 100, 200, 300])  # in Hz
    est_intervals = np.array([
        [0, 0.5],  # correct onset, on-offset incorrect
        [2, 3],
        [5, 12],
        [1, 10]
    ])
    mir_eval_note_f1(est_pitches, est_intervals, ref_pitches, ref_intervals)
    # 'note_onset/f1': 1.0, 'note_offset/f1': 0.75}),
    """ Duplicated notes """
    est_pitches = ref_pitches.copy()
    est_intervals = ref_intervals.copy()
    np.append(est_pitches, 100)  # ref has 4 notes, while est has correct 4 notes + another 1 note.
    np.append(est_intervals, [1.5, 2.5])
    mir_eval_note_f1(est_pitches, est_intervals, ref_pitches, ref_intervals)
    # 'note_onset/f1': 1.0, 'note_offset/f1': 1.0}),
    # The duplicated note is not counted as a false positive
    # and thus we do not need to post-process multi-instrument tokens
    # to remove duplicated notes in instrument-agnostic metrics.
