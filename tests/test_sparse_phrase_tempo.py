"""Sparse phrases supply real intervals even when no phrase has eight beats."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.beat_this_tracker import analyze_beat_this_grid
from src.core.telknet_beat_grid_v12 import _fit_sparse_observed_period


@pytest.mark.parametrize("mode", ["fixed_auto", "fixed_manual", "adaptive"])
def test_three_short_phrases_establish_tempo_without_counting_silence(mode):
    beats = [start + i * 0.5 for start in (0.1, 20.1, 40.1) for i in range(4)]
    info = analyze_beat_this_grid(
        beats, [], tempo_mode=mode, manual_bpm=100 if mode == "fixed_manual" else None
    )
    assert info.bpm == pytest.approx(100 if mode == "fixed_manual" else 120)
    if mode == "fixed_manual":
        assert info.source_bpm == pytest.approx(120)


@pytest.mark.parametrize("bpm", [83.7, 113.3, 128.4, 157.2])
def test_quantized_short_phrases_pool_observations_without_median_selection_bias(bpm):
    period = 60 / bpm
    # Vary frame phase and gap length independently; gaps have no known ordinal.
    beats = np.concatenate(
        [
            np.round((start + np.arange(5) * period) / 0.02) * 0.02
            for start in (0.17, 20.33, 57.59, 103.81, 155.23, 213.49, 267.73, 303.91)
        ]
    )
    fitted = _fit_sparse_observed_period(beats, float(np.median(np.diff(beats))))
    assert 60 / fitted == pytest.approx(bpm, rel=0.004)


def test_long_silences_cannot_replace_missing_interval_evidence():
    beats = [start + i * 0.5 for start in (0.1, 80.1) for i in range(4)]
    with pytest.raises(RuntimeError, match="observed beat intervals"):
        _fit_sparse_observed_period(beats, 0.5)


def test_inconsistent_short_phrase_tempos_still_fail():
    beats = [start + offset for start in (0.0, 20.0, 40.0) for offset in (0.0, 0.4, 1.0, 1.4)]
    with pytest.raises(RuntimeError, match="consistent tempo"):
        _fit_sparse_observed_period(beats, 0.5)


@pytest.mark.parametrize("second_period", [0.52, 0.55, 0.6])
def test_observed_tempo_changes_across_a_gap_remain_supported(second_period):
    beats = [
        start + i * period
        for start, period in ((0.0, 0.5), (60.0, second_period))
        for i in range(30)
    ]
    info = analyze_beat_this_grid(beats, [])
    assert 60 / second_period <= info.bpm <= 120
    assert info.fixed_tempo_reliable is False
    assert info.tempo_warning


def test_real_sparse_piano_detection_retains_its_uncertainty():
    # Real final0 output from a 241 s piano recording gated to 3 s / 12 s.
    # The ungated recording measured 121.9195 BPM; this is not human beat truth.
    beats = json.loads((Path(__file__).parent / "fixtures/sparse_piano_phrases.json").read_text())
    info = analyze_beat_this_grid(beats, [])
    assert len(beats) == 146
    assert info.bpm == pytest.approx(121.9195, abs=0.7)
    assert info.fixed_tempo_reliable is False
    assert info.tempo_warning
