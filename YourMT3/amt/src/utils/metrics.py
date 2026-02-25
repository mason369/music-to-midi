# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""metrics.py"""

from typing import List, Any, Dict, Optional, Tuple, Union
import numpy as np
import copy
from torch.nn import Module

from utils.note_event_dataclasses import NoteEvent, Note
from utils.note2event import sort_notes, notes2pc_notes
from utils.event2note import note_event2note
from sklearn.metrics import average_precision_score
from utils.metrics_helper import (f1_measure, round_float, mir_eval_note_f1, mir_eval_frame_f1, mir_eval_melody_metric,
                                  extract_pitches_intervals_from_notes, extract_frame_time_freq_from_notes)
from torchmetrics import MeanMetric, SumMetric


class UpdatedMeanMetric(MeanMetric):
    """
    A wrapper of torchmetrics.MeanMetric to support reset and update separately.
    """

    def __init__(self, nan_strategy: str = 'ignore', **kwargs) -> None:
        super().__init__(nan_strategy=nan_strategy, **kwargs)
        self._updated = False

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._updated = True

    def is_updated(self):
        return self._updated


class UpdatedSumMetric(SumMetric):
    """
    A wrapper of torchmetrics.SumMetric to support reset and update separately.
    """

    def __init__(self, nan_strategy: str = 'ignore', **kwargs) -> None:
        super().__init__(nan_strategy=nan_strategy, **kwargs)
        self._updated = False

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._updated = True

    def is_updated(self):
        return self._updated


class AMTMetrics(Module):
    """
    Automatic music transcription (AMT) evaluation metrics for music transcription 
    tasks with DDP support, following the convention of AMT. The average of file-wise
    metrics is calculated.
    
    Metrics:
    --------
    
    Instrument-agnostic note onset and note offset metrics:
    (Drum notes are generally excluded)
    
    - onset_f: the most conventional, often called Note F1
    - offset_f: a pair of onset + offset matching metric
    
    Multi-instrument note on-offset Macro-micro F1 metric, multi-F1 (of MT3):
    
    - multi_f: counts for onset + offset + program (instrument class) matching.
               For drum notes, we only count onset. macro-micro means that we
               calculate weighted precision and recall by counting each note 
               instrument class per file, and calcualte micro F1. We then 
               calculate average F1 for all files with equal weights (Macro).          

    Instrument-group note onset and offset metrics are defined by extra_classes:

    e.g. extra_classes = ['piano', 'guitar']
    - onset_f_piano: piano instrument
    - onset_f_guitar: guitar instrument
    - offset_f_piano: piano instrument
    - offset_f_guitar: guitar instrument
    also p, r metrics follow...


    Usage:
    ------
    
    Each metric instance can be individually updated and reset for computation.
    
    ```
    my_metric = AMTMetrics()
    my_metric.onset_f.update(0.5)
    my_metric.onset_f(0.5) # same
    my_metric.onset_f(0, weight=1.0) # same and weighted by 1.0 (default)
    my_metric.onset_f.compute() # return 0.333..
    my_metric.onset_f.reset() # reset the metric

    ```
    • {attribute}.update(value: float, weight: Optional[float]): Here weight is an
        optional argument for weighted average.
    • {attribute}.(...): Same as update method.
    • {attribute}.compute(): Return the average value of the metric.
    • {attribute}.reset(): Reset the metric.

    Class methods:
    ---------------
    
    ```
    d = {'onset_f': 0.5, 'offset_f': 0.5}
    my_metric.bulk_update(d)
    d = {'onset_f': {'value': 0.5, 'weight': 1.0}, 'offset_f': {'value': 0.5, 'weight': 1.0}}
    my_metric.onset_f.update(d)
    ```

    • bulk_update(metrics: Dict[str, Union[float, Dict[str, float]]]): Update metrics with a
        dictionary as an argument.
    • bulk_compute(): Return a dictionary of any non-empty metrics with average values. 
    • bulk_reset(): Reset all metrics.

    """

    def __init__(self,
                 prefix: str = '',
                 nan_strategy: str = 'ignore',
                 extra_classes: Optional[List[str]] = None,
                 extra_metrics: Optional[List[str]] = None,
                 error_types: Optional[List[str]] = None,
                 **kwargs) -> None:
        """
        Args:
            suffix: prefix for the metric name, e.g. 'val' or 'test'. '_' will be added automatically.
            nan_strategy: 'warn' or 'raise' or 'ignore'

        """
        super().__init__(**kwargs)
        self._prefix = prefix
        self.nan_strategy = nan_strategy

        # Instrument-agnostic Note onsets and Note on-offset metrics for non-drum notes
        self.onset_f = UpdatedMeanMetric(nan_strategy=nan_strategy)
        self.offset_f = UpdatedMeanMetric(nan_strategy=nan_strategy)

        # Instrument-agnostic Frame F1 (skip in validation)
        self.frame_f = UpdatedMeanMetric(nan_strategy=nan_strategy)
        self.frame_f_pc = UpdatedMeanMetric(nan_strategy=nan_strategy)

        # Drum Onset metrics
        self.onset_f_drum = UpdatedMeanMetric(nan_strategy=nan_strategy)

        # Multi F1 (Macro-micro F1 of MT3)
        self.multi_f = UpdatedMeanMetric(nan_strategy=nan_strategy)

        # Initialize extra metrics for instrument macro F1
        self.extra_classes = extra_classes
        if extra_classes is not None:
            for class_name in extra_classes:
                if not hasattr(self, class_name):
                    for onoff in ['onset', 'offset']:
                        for fpr in ['f']:
                            setattr(self, onoff + '_' + fpr + '_' + class_name,
                                    UpdatedMeanMetric(nan_strategy=nan_strategy))
                    # setattr(self, class_name, UpdatedMeanMetric(nan_strategy=nan_strategy))
                else:
                    raise ValueError(f"Metric '{class_name}' already exists.")

        # Initialize extra metrics for instruments(F is computed later)
        self.extra_classes = extra_classes
        if extra_classes is not None:
            for class_name in extra_classes:
                if not hasattr(self, class_name):
                    for onoff in ['micro_onset', 'micro_offset']:
                        for fpr in ['p', 'r']:
                            setattr(self, onoff + '_' + fpr + '_' + class_name,
                                    UpdatedMeanMetric(nan_strategy=nan_strategy))
                        # setattr(
                        #     self, onoff + '_f_' + class_name, None
                        # )  # micro_onset_f and micro_offset_f for each instrument
                else:
                    raise ValueError(f"Metric '{class_name}' already exists.")

        # Initialize drum micro P,R (F is computed later)
        self.micro_onset_p_drum = UpdatedMeanMetric(nan_strategy=nan_strategy)
        self.micro_onset_r_drum = UpdatedMeanMetric(nan_strategy=nan_strategy)

        # Initialize extra metrics directly
        if extra_metrics is not None:
            for metric_name in extra_metrics:
                setattr(self, metric_name, UpdatedMeanMetric(nan_strategy=nan_strategy))

        # Initialize error counters
        self.error_types = error_types
        if error_types is not None:
            for error_type in error_types:
                setattr(self, error_type, UpdatedMeanMetric(nan_strategy=nan_strategy))

    def bulk_update(self, metrics: Dict[str, Union[float, Dict[str, float], Tuple[float, ...]]]) -> None:
        """ Update metrics with a dictionary as an argument.

        metrics:
            {'onset_f': 0.5, 'offset_f': 0.5}
            or {'onset_f': {'value': 0.5, 'weight': 1.0}, 'offset_f': {'value': 0.5, 'weight': 1.0}} 
            or {'onset_p': (0.3, 5)}

        """
        for k, v in metrics.items():
            if isinstance(v, dict):
                getattr(self, k).update(**v)
            elif isinstance(v, tuple):
                getattr(self, k).update(*v)
            else:
                getattr(self, k).update(v)

    def bulk_update_errors(self, errors: Dict[str, Union[int, float]]) -> None:
        """ Update error counts with a dictionary as an argument. 

        errors:
            {'error_type_or_message_1': (int | float) count,
             'error_type_or_message_2': (int | float) count,}
        
        """
        for error_type, count in errors.items():
            # Update the error count
            if isinstance(count, int) or isinstance(count, float):
                getattr(self, error_type).update(count)
            else:
                raise ValueError(f"Count of error type '{error_type}' must be an integer or a float.")

    def bulk_compute(self) -> Dict[str, float]:
        computed_metrics = {}
        for k, v in self._modules.items():
            if isinstance(v, UpdatedMeanMetric) and v.is_updated():
                computed_metrics[self._prefix + k] = v.compute()
        # Create micro onset F1 for each instrument. Only when micro metrics are updated.
        extra_classes = self.extra_classes if self.extra_classes is not None else []
        for class_name in extra_classes + ['drum']:
            # micro onset F1 for each instrument.
            _micro_onset_p_instr = computed_metrics.get(self._prefix + 'micro_onset_p_' + class_name, None)
            _micro_onset_r_instr = computed_metrics.get(self._prefix + 'micro_onset_r_' + class_name, None)
            if _micro_onset_p_instr is not None and _micro_onset_r_instr is not None:
                computed_metrics[self._prefix + 'micro_onset_f_' + class_name] = f1_measure(
                    _micro_onset_p_instr.item(), _micro_onset_r_instr.item())
            # micro offset F1 for each instrument. 'drum' is usually not included.
            _micro_offset_p_instr = computed_metrics.get(self._prefix + 'micro_offset_p_' + class_name, None)
            _micro_offset_r_instr = computed_metrics.get(self._prefix + 'micro_offset_r_' + class_name, None)
            if _micro_offset_p_instr is not None and _micro_offset_r_instr is not None:
                computed_metrics[self._prefix + 'micro_offset_f_' + class_name] = f1_measure(
                    _micro_offset_p_instr.item(), _micro_offset_r_instr.item())

        # Remove micro onset and offset P,R (Now we have F1)
        for class_name in extra_classes + ['drum']:
            for onoff in ['micro_onset', 'micro_offset']:
                for pr in ['p', 'r']:
                    computed_metrics.pop(self._prefix + onoff + '_' + pr + '_' + class_name, None)

        return computed_metrics

    def bulk_reset(self) -> None:
        for k, v in self._modules.items():
            if isinstance(v, UpdatedMeanMetric):
                v.reset()
                v._updated = False


def compute_track_metrics(pred_notes: List[Note],
                          ref_notes: List[Note],
                          eval_vocab: Optional[Dict] = None,
                          eval_drum_vocab: Optional[Dict] = None,
                          onset_tolerance: float = 0.05,
                          add_pitch_class_metric: Optional[List[str]] = None,
                          add_melody_metric: Optional[List[str]] = None,
                          add_frame_metric: bool = False,
                          add_micro_metric: bool = False,
                          add_multi_f_metric: bool = False,
                          extra_info: Optional[Any] = None):
    """ Track metrics

    Args:
        pred_notes: (List[Note]) predicted sequence of notes for a track
        ref_notes: (List[Note]) reference sequence of notes for a track
        return_instr_metric: (bool) return instrument-specific metrics
        eval_vocab: (Dict or None) program group for instrument-specific metrics
            {
                instrument_or_group_name:
                  [program_number_0, program_number_1  ...]
            }
            If None, use default GM instruments.

            ex) eval_vocab = {"piano": np.arange(0, 8), ...}
        drum_vocab: (Dict or None) note (pitch) group for drum-specific metrics
            {
                instrument_or_group_name:
                    [note_number_0, note_number_1  ...]
            }
        add_pitch_class_metric: (List[str] or None) add pitch class metrics for the
            given instruments. The instrument names are defined in config/vocabulrary.py.
            ex) ['Bass', 'Guitar']
        add_singing_oa_metric: (bool) add melody overall accuracy for tje given instruments.
            The instrument names are defined in config/vocabulrary.py.
            ex) ['Singing Voice']
            (https://craffel.github.io/mir_eval/#mir_eval.melody.overall_accuracy
        add_frame_metric: (bool) add frame-wise metrics
        extra_info: (Any) extra information for debugging. Currently not implemented

    Returns:
        metrics: (Dict) track metrics in the AMTMetric format with attribute names such as 'onset_f_{instrument_or_group_name}'


    @dataclass
    class Note:
        is_drum: bool
        program: int
        onset: float
        offset: float
        pitch: int
        velocity: int

    Caution: Note is mutable instance, even if we use copy().

    """

    # Extract drum and non-drum notes
    def extract_drum_and_non_drum_notes(notes: List[Note]):
        drum_notes, non_drum_notes = [], []
        for note in notes:
            if note.is_drum:
                drum_notes.append(note)
            else:
                non_drum_notes.append(note)
        return drum_notes, non_drum_notes

    pns_drum, pns_non_drum = extract_drum_and_non_drum_notes(pred_notes)
    rns_drum, rns_non_drum = extract_drum_and_non_drum_notes(ref_notes)

    # Reduce drum notes to drum vocab
    def reduce_drum_notes_to_drum_vocab(notes: List[Note], drum_vocab: Dict):
        reduced_notes = []
        for note in notes:
            for drum_name, pitches in drum_vocab.items():
                if note.pitch in pitches:
                    new_note = copy.deepcopy(note)
                    new_note.pitch = pitches[0]
                    reduced_notes.append(new_note)
        return sort_notes(reduced_notes)

    if eval_drum_vocab != None:
        pns_drum = reduce_drum_notes_to_drum_vocab(pns_drum, eval_drum_vocab)
        rns_drum = reduce_drum_notes_to_drum_vocab(rns_drum, eval_drum_vocab)

    # Extract Pitches (freq) and Intervals
    pns_drum_pi = extract_pitches_intervals_from_notes(pns_drum, is_drum=True)
    pns_non_drum_pi = extract_pitches_intervals_from_notes(pns_non_drum)
    rns_drum_pi = extract_pitches_intervals_from_notes(rns_drum, is_drum=True)
    rns_non_drum_pi = extract_pitches_intervals_from_notes(rns_non_drum)

    # Compute file-wise PRF for drums
    drum_metric = mir_eval_note_f1(pns_drum_pi['pitches'],
                                   pns_drum_pi['intervals'],
                                   rns_drum_pi['pitches'],
                                   rns_drum_pi['intervals'],
                                   onset_tolerance=onset_tolerance,
                                   is_drum=True,
                                   add_micro_metric=add_micro_metric)

    # Compute file-wise PRF for non-drums
    non_drum_metric = mir_eval_note_f1(pns_non_drum_pi['pitches'],
                                       pns_non_drum_pi['intervals'],
                                       rns_non_drum_pi['pitches'],
                                       rns_non_drum_pi['intervals'],
                                       onset_tolerance=onset_tolerance,
                                       is_drum=False)

    # Compute file-wise frame PRF for non-drums
    if add_frame_metric is True:
        # Extract frame-level Pitches (freq) and Intervals
        pns_non_drum_tf = extract_frame_time_freq_from_notes(pns_non_drum)
        rns_non_drum_tf = extract_frame_time_freq_from_notes(rns_non_drum)

        res = mir_eval_frame_f1(pns_non_drum_tf, rns_non_drum_tf)
        non_drum_metric = {**non_drum_metric, **res}  # merge dicts

    ############## Compute instrument-wise PRF for non-drums ##############

    if eval_vocab is None:
        return drum_metric, non_drum_metric, {}
    else:
        instr_metric = {}
        for group_name, programs in eval_vocab.items():
            # Extract notes for each instrument
            # bug fix for piano/drum overlap on slakh
            pns_group = [note for note in pns_non_drum if note.program in programs]
            rns_group = [note for note in rns_non_drum if note.program in programs]

            # Compute PC instrument-wise PRF using pitch class (currently for bass)
            if add_pitch_class_metric is not None:
                if group_name.lower() in [g.lower() for g in add_pitch_class_metric]:
                    # pc: pitch information is converted to pitch classe e.g. 0-11
                    pns_pc_group = extract_pitches_intervals_from_notes(notes2pc_notes(pns_group))
                    rns_pc_group = extract_pitches_intervals_from_notes(notes2pc_notes(rns_group))

                    _instr_pc_metric = mir_eval_note_f1(pns_pc_group['pitches'],
                                                        pns_pc_group['intervals'],
                                                        rns_pc_group['pitches'],
                                                        rns_pc_group['intervals'],
                                                        onset_tolerance=onset_tolerance,
                                                        is_drum=False,
                                                        add_micro_metric=add_micro_metric,
                                                        suffix=group_name + '_pc')
                    # Add to instrument-wise PRF
                    for k, v in _instr_pc_metric.items():
                        instr_metric[k] = v

            # Extract Pitches (freq) and Intervals
            pns_group = extract_pitches_intervals_from_notes(pns_group)
            rns_group = extract_pitches_intervals_from_notes(rns_group)

            # Compute instrument-wise PRF
            _instr_metric = mir_eval_note_f1(pns_group['pitches'],
                                             pns_group['intervals'],
                                             rns_group['pitches'],
                                             rns_group['intervals'],
                                             onset_tolerance=onset_tolerance,
                                             is_drum=False,
                                             add_micro_metric=add_micro_metric,
                                             suffix=group_name)

            # Merge instrument-wise PRF
            for k, v in _instr_metric.items():
                instr_metric[k] = v

            # Optionally compute melody metrics: RPA, RCA, OA
            if add_melody_metric is not None:
                if group_name.lower() in [g.lower() for g in add_melody_metric]:
                    _melody_metric = mir_eval_melody_metric(pns_group['pitches'],
                                                            pns_group['intervals'],
                                                            rns_group['pitches'],
                                                            rns_group['intervals'],
                                                            cent_tolerance=50,
                                                            suffix=group_name)
                    for k, v in _melody_metric.items():
                        instr_metric[k] = v

        # Calculate multi_f metric for this track
        if add_multi_f_metric is True:
            drum_micro_onset_tp_sum, drum_micro_onset_tpfp_sum, drum_micro_onset_tpfn_sum = 0., 0., 0.
            non_drum_micro_offset_tp_sum, non_drum_micro_offset_tpfp_sum, non_drum_micro_offset_tpfn_sum = 0., 0., 0.
            # Collect offset metric for non-drum notes
            for k, v in instr_metric.items():
                if 'micro_offset_p_' in k and not np.isnan(v['value']):
                    non_drum_micro_offset_tp_sum += v['value'] * v['weight']
                    non_drum_micro_offset_tpfp_sum += v['weight']
                if 'micro_offset_r_' in k and not np.isnan(v['value']):
                    non_drum_micro_offset_tpfn_sum += v['weight']
            # Collect onset metric for drum notes
            for k, v in drum_metric.items():
                if 'micro_onset_p_drum' in k and not np.isnan(v['value']):
                    drum_micro_onset_tp_sum += v['value'] * v['weight']
                    drum_micro_onset_tpfp_sum += v['weight']
                if 'micro_onset_r_drum' in k and not np.isnan(v['value']):
                    drum_micro_onset_tpfn_sum += v['weight']

            tp = non_drum_micro_offset_tp_sum + drum_micro_onset_tp_sum
            tpfp = non_drum_micro_offset_tpfp_sum + drum_micro_onset_tpfp_sum
            tpfn = non_drum_micro_offset_tpfn_sum + drum_micro_onset_tpfn_sum
            multi_p_track = tp / tpfp if tpfp > 0 else np.nan
            multi_r_track = tp / tpfn if tpfn > 0 else np.nan
            multi_f_track = f1_measure(multi_p_track, multi_r_track)
            instr_metric['multi_f'] = multi_f_track

    return drum_metric, non_drum_metric, instr_metric
