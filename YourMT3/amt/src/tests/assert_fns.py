""" assert_fn.py """
import numpy as np


def assert_notes_almost_equal(actual_notes, predicted_notes, delta=5e-3):
    """
    Asserts that the given lists of Note instances are equal up to a small
    floating-point tolerance, similar to `assertAlmostEqual` of `unittest`.
    Tolerance is 5e-3 by default, which is 5 ms for 100 ticks-per-second.
    """
    assert len(actual_notes) == len(predicted_notes)
    for actual_note, predicted_note in zip(actual_notes, predicted_notes):
        assert abs(actual_note.onset - predicted_note.onset) < delta
        assert abs(actual_note.offset - predicted_note.offset) < delta
        assert actual_note.pitch == predicted_note.pitch
        if actual_note.is_drum is False and predicted_note.is_drum is False:
            assert actual_note.program == predicted_note.program
        assert actual_note.is_drum == predicted_note.is_drum
        assert actual_note.velocity == predicted_note.velocity


def assert_note_events_almost_equal(actual_note_events,
                                    predicted_note_events,
                                    ignore_time=False,
                                    ignore_activity=True,
                                    delta=5.1e-3):
    """
    Asserts that the given lists of Note instances are equal up to a small
    floating-point tolerance, similar to `assertAlmostEqual` of `unittest`.
    Tolerance is 5e-3 by default, which is 5 ms for 100 ticks-per-second.

    If `ignore_time` is True, then the time field is ignored. (useful for 
    comparing tie note events, default is False)

    If `ignore_activity` is True, then the activity field is ignored (default
    is True).
    """
    assert len(actual_note_events) == len(predicted_note_events)
    for j, (actual_note_event,
            predicted_note_event) in enumerate(zip(actual_note_events, predicted_note_events)):
        if ignore_time is False:
            assert abs(actual_note_event.time - predicted_note_event.time) <= delta
        assert actual_note_event.is_drum == predicted_note_event.is_drum
        if actual_note_event.is_drum is False and predicted_note_event.is_drum is False:
            assert actual_note_event.program == predicted_note_event.program
        assert actual_note_event.pitch == predicted_note_event.pitch
        assert actual_note_event.velocity == predicted_note_event.velocity
        if ignore_activity is False:
            assert actual_note_event.activity == predicted_note_event.activity


def assert_track_metrics_score1(metrics) -> None:
    for k, v in metrics.items():
        if np.isnan(v) is False:
            assert v == 1.0
