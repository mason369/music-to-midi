# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""metrics_test.py:

This file contains tests for the following classes:
• AMTMetrics

"""
import unittest
import warnings
import torch
import numpy as np
from utils.metrics import AMTMetrics
from utils.metrics import compute_track_metrics


class TestAMTMetrics(unittest.TestCase):

    def test_individual_attributes(self):
        metric = AMTMetrics()

        # Test updating the metric using .update() method
        metric.onset_f.update(0.5)

        # Test updating the metric using __call__() method
        metric.onset_f(0.5)

        # Test updating the metric with a weight
        metric.onset_f(0, weight=1.0)

        # Test computing the average value of the metric
        computed_value = metric.onset_f.compute()
        self.assertAlmostEqual(computed_value, 0.3333333333333333)

        # Test resetting the metric
        metric.onset_f.reset()
        with self.assertWarns(UserWarning):
            torch._assert(metric.onset_f.compute(), torch.nan)

        # Test bulk_compute
        with self.assertWarns(UserWarning):
            computed_metrics = metric.bulk_compute()

    def test_bulk_update_and_compute(self):
        metric = AMTMetrics()

        # Test bulk_update with values only
        d1 = {'onset_f': 0.5, 'offset_f': 0.5}
        metric.bulk_update(d1)

        # Test bulk_update with values and weights
        d2 = {'onset_f': {'value': 0.5, 'weight': 1.0}, 'offset_f': {'value': 0.5, 'weight': 1.0}}
        metric.bulk_update(d2)

        # Test bulk_compute
        computed_metrics = metric.bulk_compute()

        # Ensure the 'onset_f' and 'offset_f' keys exist in the computed_metrics dictionary
        self.assertIn('onset_f', computed_metrics)
        self.assertIn('offset_f', computed_metrics)

        # Check the computed values
        self.assertAlmostEqual(computed_metrics['onset_f'], 0.5)
        self.assertAlmostEqual(computed_metrics['offset_f'], 0.5)

    def test_compute_track_metrics_singing(self):
        from config.vocabulary import SINGING_SOLO_CLASS, GM_INSTR_CLASS_PLUS
        from utils.event2note import note_event2note

        ref_notes_dict = np.load('extras/examples/singing_notes.npy', allow_pickle=True).tolist()
        ref_note_events_dict = np.load('extras/examples/singing_note_events.npy', allow_pickle=True).tolist()
        est_notes, _ = note_event2note(ref_note_events_dict['note_events'])
        ref_notes = ref_notes_dict['notes']

        metric = AMTMetrics(prefix=f'test/', extra_classes=[k for k in SINGING_SOLO_CLASS.keys()])
        drum_metric, non_drum_metric, instr_metric = compute_track_metrics(est_notes,
                                                                           ref_notes,
                                                                           eval_vocab=SINGING_SOLO_CLASS,
                                                                           eval_drum_vocab=None,
                                                                           onset_tolerance=0.05)
        metric.bulk_update(drum_metric)
        metric.bulk_update(non_drum_metric)
        metric.bulk_update(instr_metric)
        computed_metrics = metric.bulk_compute()
        cnt = 0
        for k, v in computed_metrics.items():
            if 'Singing Voice' in k:
                self.assertEqual(v, 1.0)
                cnt += 1
        self.assertEqual(cnt, 6)

        metric = AMTMetrics(prefix=f'test/', extra_classes=[k for k in GM_INSTR_CLASS_PLUS.keys()])
        drum_metric, non_drum_metric, instr_metric = compute_track_metrics(est_notes,
                                                                           ref_notes,
                                                                           eval_vocab=GM_INSTR_CLASS_PLUS,
                                                                           eval_drum_vocab=None,
                                                                           onset_tolerance=0.05)
        metric.bulk_update(drum_metric)
        metric.bulk_update(non_drum_metric)
        metric.bulk_update(instr_metric)
        computed_metrics = metric.bulk_compute()
        cnt = 0
        for k, v in computed_metrics.items():
            if 'Singing Voice' in k:
                self.assertEqual(v, 1.0)
                cnt += 1
        self.assertEqual(cnt, 6)


if __name__ == '__main__':
    unittest.main()
