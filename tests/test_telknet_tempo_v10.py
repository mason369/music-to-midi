"""Acceptance contracts ported from the reviewed TelkNet v10 tempo chain."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.beat_this_tracker import analyze_beat_this_grid
from src.core.telknet_beat_grid_v10 import (
    MAX_FIXED_TEMPO_PHASE_ERROR_BEATS,
    normalize_beat_grid,
)
from src.core.telknet_tempo_map import (
    MAX_VARIABLE_TEMPO_PHASE_ERROR_BEATS,
    BeatThisTempoMapError,
    _tempo_map_from_beat_times,
    _tempo_map_from_downbeats,
    build_adaptive_tempo_map,
    tempo_map_max_phase_error,
)
from src.models.data_models import TempoMode


def _timeline(periods, *, start: float = 0.2) -> list[float]:
    times = [start]
    for period in periods:
        times.append(times[-1] + float(period))
    return times


def test_fixed_auto_is_the_default_single_tempo_export_contract():
    beats = _timeline([0.5] * 96)
    info = analyze_beat_this_grid(beats, beats[::4])

    assert info.bpm == pytest.approx(120.0)
    assert info.tempo_map == []
    assert info.time_signature == (4, 4)


def test_manual_mode_writes_only_the_requested_bpm_without_changing_evidence():
    beats = _timeline([0.5] * 96)
    info = analyze_beat_this_grid(
        beats,
        beats[::4],
        tempo_mode=TempoMode.FIXED_MANUAL.value,
        manual_bpm=87.25,
    )

    assert info.bpm == pytest.approx(87.25)
    assert info.source_bpm == pytest.approx(120.0)
    assert info.tempo_map == []
    assert info.beat_times == pytest.approx(beats)


def test_adaptive_mode_keeps_a_real_sustained_section_change():
    beats = _timeline([0.8] * 32 + [0.6] * 32 + [0.8] * 32)
    downbeats = beats[::4]

    info = analyze_beat_this_grid(
        beats,
        downbeats,
        tempo_mode=TempoMode.ADAPTIVE.value,
    )

    assert len(info.tempo_map) >= 3
    observed = [bpm for _start, bpm in info.tempo_map]
    assert observed[0] == pytest.approx(75.0, abs=0.3)
    assert max(observed) == pytest.approx(100.0, abs=0.3)
    assert observed[-1] == pytest.approx(75.0, abs=0.3)
    assert tempo_map_max_phase_error(info.beat_times, info.tempo_map) <= (
        MAX_VARIABLE_TEMPO_PHASE_ERROR_BEATS + 1e-8
    )


def test_bar_aligned_map_folds_sustained_half_and_double_time_labels():
    base_bpm = 66.66666666666667
    base_period = 60.0 / base_bpm
    downbeats = [0.2]
    intervals = [4 * base_period] * 6 + [2 * base_period] * 12 + [4 * base_period] * 6
    for interval in intervals:
        downbeats.append(downbeats[-1] + interval)

    tempo_map = _tempo_map_from_downbeats(
        downbeats,
        beats_per_bar=4,
        representative_bpm=base_bpm,
    )

    assert len(tempo_map) == 1
    assert tempo_map[0] == pytest.approx((0.0, base_bpm), abs=1e-8)


def test_bar_aligned_map_keeps_sustained_non_octave_sections():
    downbeats = [0.4]
    for bpm in [75.0] * 8 + [100.0] * 8 + [75.0] * 8:
        downbeats.append(downbeats[-1] + 4 * 60.0 / bpm)

    tempo_map = _tempo_map_from_downbeats(
        downbeats,
        beats_per_bar=4,
        representative_bpm=75.0,
    )

    assert len(tempo_map) == 3
    assert [bpm for _start, bpm in tempo_map] == pytest.approx([75.0, 100.0, 75.0], abs=0.25)


def test_beat_fallback_filters_alternating_detector_frame_jitter():
    beats = _timeline([0.82, 0.80] * 40)

    tempo_map = _tempo_map_from_beat_times(beats, representative_bpm=73.171)

    assert len(tempo_map) == 1
    assert tempo_map[0] == pytest.approx((0.0, 60.0 / 0.81), abs=1e-8)


def test_phase_bounded_map_clamps_real_300_bpm_detector_roundoff():
    beats = [72.84, 73.04, 73.24]
    raw_bpm = 60.0 * 2 / (beats[-1] - beats[0])
    assert raw_bpm == 300.0000000000064

    tempo_map = _tempo_map_from_beat_times(beats, representative_bpm=272.7272727272741)

    assert tempo_map == ((0.0, 300.0),)


def test_phase_bounded_map_still_rejects_a_real_above_300_bpm_segment():
    beats = [0.0, 60.0 / 300.000001]

    with pytest.raises(
        BeatThisTempoMapError,
        match="out-of-range phase-bounded tempo",
    ):
        _tempo_map_from_beat_times(beats, representative_bpm=300.0)


def test_long_song_adaptive_map_has_a_hard_one_sixteenth_beat_phase_bound():
    beats = _timeline([0.44] * 320 + [0.445] * 320 + [0.45] * 320, start=0.46)

    result = build_adaptive_tempo_map(
        beats,
        [],
        beats_per_bar=None,
        representative_bpm=136.363636,
    )

    assert len(result.events) >= 2
    assert result.strategy == "phase_bounded_beats"
    assert result.maximum_phase_error_beats <= MAX_VARIABLE_TEMPO_PHASE_ERROR_BEATS
    assert tempo_map_max_phase_error(beats, result.events) <= (
        MAX_VARIABLE_TEMPO_PHASE_ERROR_BEATS + 1e-8
    )


def test_v10_repairs_one_isolated_phase_zigzag_but_not_a_tempo_section():
    isolated = [index * 0.5 for index in range(128)]
    isolated[64] += 0.16
    repaired = normalize_beat_grid(isolated)

    sustained = _timeline([0.5] * 40 + [0.6] * 8 + [0.5] * 40, start=0.0)
    variable = normalize_beat_grid(sustained)

    assert repaired.isolated_phase_outliers_repaired == 1
    assert repaired.beat_times[64] == pytest.approx(32.0)
    assert repaired.fixed_tempo_reliable is True
    assert variable.isolated_phase_outliers_repaired == 0
    assert variable.fixed_tempo_reliable is False


def test_v10_recovers_sub_frame_bpm_without_long_song_phase_drift():
    true_bpm = 121.99149363221414
    beats = np.round(0.28 + np.arange(476) * (60.0 / true_bpm), 2).tolist()

    grid = normalize_beat_grid(beats)

    assert 60.0 / float(np.median(np.diff(beats))) == pytest.approx(122.4489795918)
    assert grid.fixed_tempo_reliable is True
    assert grid.bpm == pytest.approx(true_bpm, abs=0.001)
    maximum_phase_error = max(
        abs((timestamp - grid.beat_times[0]) * grid.bpm / 60.0 - ordinal)
        for ordinal, timestamp in enumerate(grid.beat_times)
    )
    assert maximum_phase_error <= MAX_FIXED_TEMPO_PHASE_ERROR_BEATS


def test_v10_does_not_flatten_a_real_tempo_change_into_one_bpm():
    beats = _timeline([0.60] * 32 + [0.50] * 32, start=0.0)

    grid = normalize_beat_grid(beats)

    assert grid.fixed_tempo_reliable is False
    assert grid.tempo_warning is not None
