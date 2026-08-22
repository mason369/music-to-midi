from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import torch

from src.core.beat_this_tracker import (
    BEAT_THIS_CHECKPOINT_NAME,
    BeatThisGridError,
    BeatThisTracker,
    analyze_beat_this_grid,
    count_musical_beat_positions,
    get_beat_this_checkpoint_path,
    remove_competing_beat_marks,
)
from src.models.data_models import Config, TempoMode


def test_linear_fit_removes_final0_frame_quantization_error():
    real_bpm = 118.6
    exact = np.arange(80, dtype=float) * 60.0 / real_bpm
    final0_quantized = np.round(exact / 0.02) * 0.02

    result = analyze_beat_this_grid(
        final0_quantized,
        final0_quantized[::4],
    )

    assert result.bpm == pytest.approx(real_bpm, abs=0.01)
    assert result.time_signature == (4, 4)
    assert result.tempo_map == []


def test_duplicate_mark_cleanup_keeps_mark_nearest_expected_local_pulse():
    beats = np.arange(16, dtype=float) * 0.5
    beats = np.sort(np.append(beats, 2.6))

    cleaned, removed = remove_competing_beat_marks(beats)

    assert removed == 1
    assert cleaned == pytest.approx(np.arange(16, dtype=float) * 0.5)


def test_first_duplicate_cluster_is_scored_against_following_marks():
    beats = np.array([0.0, 0.1, *np.arange(1, 16, dtype=float) * 0.5])

    cleaned, removed = remove_competing_beat_marks(beats)

    assert removed == 1
    assert cleaned[0] == pytest.approx(0.0)
    assert cleaned[1:] == pytest.approx(np.arange(1, 16, dtype=float) * 0.5)


def test_missing_beats_are_counted_before_global_tempo_fit():
    complete = np.arange(40, dtype=float) * 0.5
    detected = np.delete(complete, [7, 23])

    counted = count_musical_beat_positions(detected)
    result = analyze_beat_this_grid(
        detected,
        complete[::4],
    )

    assert counted.recovered_missing_beats == 2
    assert result.bpm == pytest.approx(120.0, abs=1e-8)
    assert result.time_signature == (4, 4)


def test_variable_performance_always_emits_beat_level_tempo_map():
    intervals = np.concatenate([np.full(12, 60.0 / 90.0), np.full(12, 60.0 / 140.0)])
    beats = np.concatenate([[0.0], np.cumsum(intervals)])

    result = analyze_beat_this_grid(
        beats,
        beats[::4],
        tempo_mode=TempoMode.ADAPTIVE.value,
    )

    assert result.is_variable_tempo
    assert result.tempo_map[0] == pytest.approx((0.0, 90.0))
    assert min(bpm for _, bpm in result.tempo_map) == pytest.approx(90.0)
    assert max(bpm for _, bpm in result.tempo_map) == pytest.approx(140.0)


def test_meter_uses_independent_sixty_percent_agreement_gate():
    beats = np.arange(36, dtype=float) * 0.5
    # Reliable meter evidence uses one stable three-beat bar lattice.
    downbeat_positions = [0, 3, 6, 9, 12, 15]
    downbeats = beats[downbeat_positions]

    result = analyze_beat_this_grid(
        beats,
        downbeats,
    )

    assert result.time_signature == (3, 4)
    assert result.downbeats == pytest.approx(downbeats)


def test_uncertain_meter_is_explicitly_unknown_instead_of_fake_four_four():
    beats = np.arange(32, dtype=float) * 0.5

    result = analyze_beat_this_grid(
        beats,
        [0.0, 2.0],
    )

    assert result.time_signature is None


@pytest.mark.parametrize(
    "beats",
    [
        [0.0, 0.5, float("nan"), 1.5, 2.0, 2.5, 3.0, 3.5],
        [0.0, 0.5, 1.0, 0.75, 1.5, 2.0, 2.5, 3.0],
        np.arange(7, dtype=float) * 0.5,
    ],
)
def test_invalid_or_too_short_grids_fail_explicitly(beats):
    with pytest.raises(BeatThisGridError):
        analyze_beat_this_grid(beats, [])


def test_runtime_uses_only_final0_without_dbn_or_half_precision(tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio-placeholder")
    checkpoint = tmp_path / "final0.ckpt"
    checkpoint.write_bytes(b"checkpoint-placeholder")
    factory_args = {}

    class FakeRuntime:
        def __call__(self, _audio_path):
            beats = np.arange(12, dtype=float) * 0.5
            return beats, beats[::4]

    def factory(**kwargs):
        factory_args.update(kwargs)
        return FakeRuntime()

    tracker = BeatThisTracker(
        Config(use_gpu=False, enable_tempo_map=False),
        checkpoint_path=checkpoint,
        tracker_factory=factory,
    )
    with mock.patch(
        "src.core.beat_this_tracker.validate_beat_this_checkpoint",
        return_value=checkpoint,
    ):
        result = tracker.detect(str(audio))

    assert result.bpm == pytest.approx(120.0)
    assert factory_args == {
        "checkpoint_path": str(checkpoint),
        "device": "cpu",
        "float16": False,
        "dbn": False,
    }


@pytest.mark.parametrize("raises", [False, True])
def test_cuda_tracking_disables_cudnn_tf32_only_for_final0_call(tmp_path, raises):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio-placeholder")
    observed_allow_tf32 = []

    class FakeRuntime:
        def __call__(self, _audio_path):
            observed_allow_tf32.append(torch.backends.cudnn.allow_tf32)
            if raises:
                raise RuntimeError("inference failed")
            beats = np.arange(12, dtype=float) * 0.5
            return beats, beats[::4]

    tracker = BeatThisTracker(Config(use_gpu=True, gpu_device=0))
    tracker._device = "cuda:0"
    previous_allow_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = True
    try:
        with mock.patch.object(tracker, "_load_tracker", return_value=FakeRuntime()):
            if raises:
                with pytest.raises(RuntimeError, match="inference failed"):
                    tracker.track_raw(str(audio))
            else:
                beats, downbeats = tracker.track_raw(str(audio))
                assert len(beats) == 12
                assert len(downbeats) == 3

        assert observed_allow_tf32 == [False]
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cudnn.allow_tf32 = previous_allow_tf32


def test_frozen_runtime_never_substitutes_a_user_cache_checkpoint(tmp_path):
    executable_root = tmp_path / "portable"
    internal_root = executable_root / "_internal"
    with (
        mock.patch("src.core.beat_this_tracker.is_frozen_app", return_value=True),
        mock.patch(
            "src.core.beat_this_tracker.get_bundle_roots",
            return_value=[executable_root, internal_root],
        ),
    ):
        resolved = get_beat_this_checkpoint_path()

    assert resolved == internal_root / "models" / "beat_this" / BEAT_THIS_CHECKPOINT_NAME


def test_invalid_checkpoint_stops_before_runtime_factory(tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"audio-placeholder")
    factory = mock.Mock()
    tracker = BeatThisTracker(
        Config(use_gpu=False),
        checkpoint_path=tmp_path / "invalid.ckpt",
        tracker_factory=factory,
    )

    with mock.patch(
        "src.core.beat_this_tracker.validate_beat_this_checkpoint",
        side_effect=RuntimeError("identity mismatch"),
    ):
        with pytest.raises(RuntimeError, match="identity mismatch"):
            tracker.detect(str(audio))

    factory.assert_not_called()
