"""Regression for frame-quantized, sparse Beat This detections (no audio inference)."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.beat_this_tracker import analyze_beat_this_grid
from src.core.telknet_beat_grid_v12 import normalize_beat_grid
from src.models.data_models import BeatInfo, ProcessingResult
from src.web_api.engine import InferenceEngine


def test_real_sparse_detector_grid_does_not_count_a_long_gap_at_quantized_median():
    beats = json.loads((Path(__file__).parent / "fixtures/sparse_beat_this_ticks.json").read_text())
    grid = normalize_beat_grid(beats)
    # Same recording's mix / drums were measured at 128.39 / 128.45 BPM.
    # This is an observed consistency bound, not a claimed manual beat annotation.
    assert grid.bpm == pytest.approx(128.4, abs=0.12)
    assert grid.raw_beat_times == tuple(beats)
    assert grid.fixed_tempo_reliable is False
    assert grid.tempo_warning


@pytest.mark.parametrize("bpm", [83.7, 113.3, 128.4, 157.2])
def test_subframe_tempo_uses_observed_runs_not_synthetic_gap_ordinals(bpm):
    period = 60 / bpm
    ordinals = np.r_[np.arange(0, 100, 2), np.arange(230, 280)]
    beats = np.round((0.7 + ordinals * period) / 0.02) * 0.02
    grid = normalize_beat_grid(beats)
    assert grid.bpm == pytest.approx(bpm, abs=0.06)


def test_sparse_gaps_without_a_real_contiguous_run_fail_explicitly():
    with pytest.raises(RuntimeError, match="observed beats"):
        analyze_beat_this_grid([0.1, 0.6, 1.1, 45.1, 90.1], [])


def test_reliability_survives_auto_manual_and_adaptive_modes():
    beats = json.loads((Path(__file__).parent / "fixtures/sparse_beat_this_ticks.json").read_text())
    for mode in ("fixed_auto", "fixed_manual", "adaptive"):
        info = analyze_beat_this_grid(
            beats, [], tempo_mode=mode, manual_bpm=110 if mode == "fixed_manual" else None
        )
        assert info.fixed_tempo_reliable is False
        assert info.tempo_warning
        if mode == "fixed_manual":
            assert info.bpm == 110
            assert info.source_bpm == pytest.approx(128.4, abs=0.12)


def test_api_distinguishes_unknown_and_failed_tempo_evidence():
    for reliable in (None, False, True):
        beat = BeatInfo(
            128.4,
            fixed_tempo_reliable=reliable,
            tempo_warning="detail" if reliable is False else None,
        )
        result = ProcessingResult(midi_path="output.mid", beat_info=beat)
        payload = InferenceEngine._beat_payload(result)
        assert payload["fixed_tempo_reliable"] is reliable
        assert payload["tempo_warning"] == beat.tempo_warning


def test_duplicates_and_interpolation_cannot_create_reliable_evidence():
    beats = [0.1, 0.6, 1.1, 1.11, 1.6, 2.1, 3.1, 3.6]
    grid = normalize_beat_grid(beats)
    assert grid.duplicate_beats_removed == 1
    assert grid.missing_beats_interpolated == 1
    assert len(grid.beat_times) == 8
    assert grid.fixed_tempo_reliable is False
    assert "7 distinct observed beats" in grid.tempo_warning


@pytest.mark.parametrize("mode", ["fixed_auto", "fixed_manual", "adaptive"])
def test_all_tempo_modes_require_enough_distinct_observed_beats(mode):
    with pytest.raises(RuntimeError, match="7 distinct observed beats"):
        analyze_beat_this_grid(
            [0.1, 0.6, 1.1, 1.11, 1.6, 2.1, 3.1, 3.6],
            [],
            tempo_mode=mode,
            manual_bpm=110 if mode == "fixed_manual" else None,
        )
