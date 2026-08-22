"""Beat This ``final0`` tracking with strict, production-oriented grid analysis.

The neural tracker supplies beat and downbeat marks.  This module owns the
consumer-side work that MuScriptor PR #65 deliberately keeps minimal:

* competing sub-beat marks are resolved against a local pulse;
* isolated missed beats are represented as musical-position gaps before the
  global least-squares tempo fit;
* meter confidence is evaluated independently from tempo confidence; and
* expressive performances expose a beat-level tempo map automatically.

There is no alternate detector or placeholder-tempo fallback.  Missing or
invalid model assets and unusable grids are hard failures.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from src.i18n.translator import Translator
from src.core.telknet_beat_grid_v10 import (
    TEMPO_FIT_ID,
    infer_beats_per_bar as infer_telknet_beats_per_bar,
    normalize_beat_grid,
    normalize_downbeat_grid,
)
from src.core.telknet_tempo_map import build_adaptive_tempo_map
from src.models.data_models import (
    MAX_TEMPO_BPM,
    MIN_TEMPO_BPM,
    BeatInfo,
    Config,
    TempoMode,
)
from src.utils.artifact_identity import validate_file_identity
from src.utils.gpu_utils import (
    ensure_accelerator_runtime_compatibility,
    ensure_module_on_device,
    get_device,
)
from src.utils.runtime_paths import get_bundle_roots, get_runtime_data_dir, is_frozen_app

logger = logging.getLogger(__name__)

BEAT_THIS_CHECKPOINT_NAME = "final0.ckpt"
BEAT_THIS_CHECKPOINT_URL = (
    "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
)
BEAT_THIS_CHECKPOINT_SIZE = 81_058_141
BEAT_THIS_CHECKPOINT_SHA256 = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"

MIN_BEATS = 8
LOCAL_GAP_RADIUS = 5
DUPLICATE_GAP_FRACTION = 0.5
MISSING_BEAT_TOLERANCE = 0.20
MAX_INTERVAL_BEAT_COUNT = 8
MIN_SINGLE_INTERVAL_FRACTION = 0.60
CONSTANT_TEMPO_RESIDUAL_FRACTION = 0.05
MIN_METER_AGREEMENT = 0.60
MIN_BEATS_PER_BAR = 2
MAX_BEATS_PER_BAR = 12
DOWNBEAT_MATCH_FRACTION = 0.35
MIN_LOCAL_BPM = 20.0
MAX_LOCAL_BPM = 400.0


class BeatThisGridError(RuntimeError):
    """Beat This returned no grid that is safe to write into exported MIDI."""


@dataclass(frozen=True)
class CleanBeatGrid:
    """Clean beat marks and their monotonically increasing musical positions."""

    times: np.ndarray
    positions: np.ndarray
    interval_beat_counts: np.ndarray
    removed_duplicates: int
    recovered_missing_beats: int


def get_beat_this_checkpoint_path() -> Path:
    """Resolve exactly one checkpoint path for the current runtime shape.

    A frozen build is self-contained: it must use its bundled artifact and must
    never substitute a user-cache copy when that artifact is absent. A source
    checkout uses the prepared application-data artifact. Neither path performs
    a task-time download.
    """

    relative = Path("models") / "beat_this" / BEAT_THIS_CHECKPOINT_NAME
    if is_frozen_app():
        roots = get_bundle_roots()
        for root in roots:
            candidate = root / relative
            if candidate.is_file():
                return candidate
        return roots[-1] / relative
    return get_runtime_data_dir() / "models" / "beat_this" / BEAT_THIS_CHECKPOINT_NAME


def validate_beat_this_checkpoint(path: Path | None = None) -> Path:
    """Require the exact official ``final0`` artifact used by this project."""

    return validate_file_identity(
        Path(path) if path is not None else get_beat_this_checkpoint_path(),
        expected_size=BEAT_THIS_CHECKPOINT_SIZE,
        expected_sha256=BEAT_THIS_CHECKPOINT_SHA256,
        label="Beat This final0 checkpoint",
    )


def _validated_marks(values: Iterable[float], label: str) -> np.ndarray:
    marks = np.asarray(list(values), dtype=float).reshape(-1)
    if marks.size and (not np.all(np.isfinite(marks)) or np.any(marks < 0.0)):
        raise BeatThisGridError(f"{label} contain non-finite or negative times")
    if marks.size > 1 and np.any(np.diff(marks) < 0.0):
        raise BeatThisGridError(f"{label} are not ordered by time")
    return marks


def _local_spacing(times: np.ndarray, gap_index: int) -> float:
    gaps = np.diff(times)
    if gaps.size == 0:
        raise BeatThisGridError("Beat grid has no positive interval")
    lo = max(0, int(gap_index) - LOCAL_GAP_RADIUS)
    hi = min(gaps.size, int(gap_index) + LOCAL_GAP_RADIUS + 1)
    positive = gaps[lo:hi][gaps[lo:hi] > 0.0]
    if positive.size == 0:
        positive = gaps[gaps > 0.0]
    if positive.size == 0:
        raise BeatThisGridError("Beat grid contains no positive interval")
    spacing = float(np.median(positive))
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise BeatThisGridError(f"Beat grid has invalid local spacing: {spacing!r}")
    return spacing


def _first_cluster_candidate_score(
    candidate: float,
    following: np.ndarray,
    spacing: float,
) -> float:
    if following.size == 0:
        return 0.0
    relative = (following - candidate) / spacing
    residual = np.abs(relative - np.rint(relative))
    return float(np.median(residual))


def remove_competing_beat_marks(beats: Sequence[float]) -> tuple[np.ndarray, int]:
    """Resolve duplicate marks using the local pulse and expected next beat.

    Marks less than half a local beat apart compete for one musical position.
    Later clusters are resolved against ``previous + local spacing``.  The
    first cluster has no previous beat, so each candidate is scored against up
    to eight following marks, matching the production rule described in the
    MuScriptor PR #65 review.
    """

    raw = _validated_marks(beats, "Beat This beat marks")
    if raw.size < MIN_BEATS:
        raise BeatThisGridError(
            f"Beat This detected only {raw.size} beats; at least {MIN_BEATS} are required"
        )

    kept: list[float] = []
    index = 0
    while index < raw.size:
        gap_index = min(index, max(0, raw.size - 2))
        spacing = _local_spacing(raw, gap_index)
        cluster_end = index + 1
        while cluster_end < raw.size:
            gap = float(raw[cluster_end] - raw[cluster_end - 1])
            if gap >= DUPLICATE_GAP_FRACTION * spacing:
                break
            cluster_end += 1

        cluster = raw[index:cluster_end]
        if cluster.size == 1:
            selected = float(cluster[0])
        elif kept:
            expected = kept[-1] + spacing
            selected = float(min(cluster, key=lambda mark: abs(float(mark) - expected)))
        else:
            following = raw[cluster_end : min(raw.size, cluster_end + 8)]
            selected = float(
                min(
                    cluster,
                    key=lambda mark: (
                        _first_cluster_candidate_score(float(mark), following, spacing),
                        float(mark),
                    ),
                )
            )
        if kept and selected <= kept[-1]:
            raise BeatThisGridError("Duplicate-mark cleanup produced a non-increasing grid")
        kept.append(selected)
        index = cluster_end

    cleaned = np.asarray(kept, dtype=float)
    removed = int(raw.size - cleaned.size)
    maximum_safe_removals = max(6, int(math.ceil(raw.size * 0.10)))
    if removed > maximum_safe_removals:
        raise BeatThisGridError(
            "Beat This grid contains too many competing beat marks: "
            f"removed={removed}, raw={raw.size}, allowed={maximum_safe_removals}"
        )
    if cleaned.size < MIN_BEATS:
        raise BeatThisGridError(f"Only {cleaned.size} beats remain after duplicate-mark cleanup")
    return cleaned, removed


def count_musical_beat_positions(cleaned_beats: Sequence[float]) -> CleanBeatGrid:
    """Count missed beats from local spacing before fitting global tempo."""

    times = _validated_marks(cleaned_beats, "Clean Beat This beat marks")
    if times.size < MIN_BEATS:
        raise BeatThisGridError(
            f"Only {times.size} cleaned beats are available; at least {MIN_BEATS} are required"
        )
    gaps = np.diff(times)
    if np.any(gaps <= 0.0):
        raise BeatThisGridError("Clean Beat This beat marks are not strictly increasing")

    candidate_counts = np.ones(gaps.size, dtype=int)
    for gap_index, gap in enumerate(gaps):
        spacing = _local_spacing(times, gap_index)
        rounded = int(round(float(gap) / spacing))
        if rounded < 2 or rounded > MAX_INTERVAL_BEAT_COUNT:
            continue
        per_beat = float(gap) / rounded
        if abs(per_beat - spacing) <= MISSING_BEAT_TOLERANCE * spacing:
            candidate_counts[gap_index] = rounded

    single_fraction = float(np.mean(candidate_counts == 1))
    if single_fraction < MIN_SINGLE_INTERVAL_FRACTION:
        interval_counts = np.ones_like(candidate_counts)
    else:
        interval_counts = candidate_counts

    positions = np.concatenate([np.array([0], dtype=int), np.cumsum(interval_counts, dtype=int)])
    recovered = int(np.sum(interval_counts - 1))
    return CleanBeatGrid(
        times=times,
        positions=positions,
        interval_beat_counts=interval_counts,
        removed_duplicates=0,
        recovered_missing_beats=recovered,
    )


def fit_global_tempo(grid: CleanBeatGrid) -> tuple[float, float, float]:
    """Return ``(bpm, residual_rms_seconds, seconds_per_beat)``."""

    slope, intercept = np.polyfit(grid.positions.astype(float), grid.times, 1)
    slope = float(slope)
    intercept = float(intercept)
    if not math.isfinite(slope) or slope <= 0.0 or not math.isfinite(intercept):
        raise BeatThisGridError(
            f"Beat-position regression produced an invalid slope/intercept: {slope}, {intercept}"
        )
    residuals = grid.times - (intercept + slope * grid.positions)
    residual_rms = float(np.sqrt(np.mean(np.square(residuals))))
    bpm = 60.0 / slope
    if not math.isfinite(bpm) or bpm <= 0.0:
        raise BeatThisGridError(f"Beat-position regression produced invalid BPM: {bpm!r}")
    return bpm, residual_rms, slope


def infer_time_signature(
    grid: CleanBeatGrid,
    downbeats: Sequence[float],
) -> tuple[Optional[tuple[int, int]], list[float]]:
    """Infer beats per bar independently, returning ``None`` when uncertain."""

    raw_downbeats = _validated_marks(downbeats, "Beat This downbeat marks")
    if raw_downbeats.size < 3:
        return None, raw_downbeats.tolist()

    matched_positions: list[int] = []
    matched_times: list[float] = []
    for downbeat in raw_downbeats:
        nearest_index = int(np.argmin(np.abs(grid.times - downbeat)))
        spacing = _local_spacing(
            grid.times,
            min(nearest_index, max(0, grid.times.size - 2)),
        )
        if abs(float(grid.times[nearest_index] - downbeat)) > DOWNBEAT_MATCH_FRACTION * spacing:
            continue
        position = int(grid.positions[nearest_index])
        if matched_positions and position == matched_positions[-1]:
            continue
        matched_positions.append(position)
        matched_times.append(float(downbeat))

    if len(matched_positions) < 3:
        return None, matched_times

    gaps = np.diff(np.asarray(matched_positions, dtype=int))
    eligible = gaps[(gaps >= MIN_BEATS_PER_BAR) & (gaps <= MAX_BEATS_PER_BAR)]
    if eligible.size == 0:
        return None, matched_times
    values, counts = np.unique(eligible, return_counts=True)
    best_index = int(np.argmax(counts))
    beats_per_bar = int(values[best_index])
    agreement = float(counts[best_index] / gaps.size)
    if agreement < MIN_METER_AGREEMENT:
        return None, matched_times

    # Beat This reports pulse positions, not a notation denominator.  Following
    # the official MuScriptor integration, encode that pulse as quarter notes
    # instead of guessing 6/8 versus 6/4 from timing alone.
    return (beats_per_bar, 4), matched_times


def build_variable_tempo_map(
    grid: CleanBeatGrid,
    *,
    residual_rms: float,
    seconds_per_beat: float,
) -> list[tuple[float, float]]:
    """Build a beat-level tempo map only for confidently non-constant audio."""

    if residual_rms <= CONSTANT_TEMPO_RESIDUAL_FRACTION * seconds_per_beat:
        return []

    gaps = np.diff(grid.times)
    local_bpms = 60.0 * grid.interval_beat_counts.astype(float) / gaps
    if not np.all(np.isfinite(local_bpms)):
        raise BeatThisGridError("Beat This variable-tempo grid contains non-finite BPM values")
    invalid = local_bpms[(local_bpms < MIN_LOCAL_BPM) | (local_bpms > MAX_LOCAL_BPM)]
    if invalid.size:
        raise BeatThisGridError(
            "Beat This variable-tempo grid contains implausible local BPM values: "
            f"{invalid.tolist()}"
        )

    tempo_map: list[tuple[float, float]] = [(0.0, float(local_bpms[0]))]
    for interval_index in range(1, local_bpms.size):
        bpm = float(local_bpms[interval_index])
        if math.isclose(bpm, tempo_map[-1][1], rel_tol=1e-9, abs_tol=1e-9):
            continue
        tempo_map.append((float(grid.times[interval_index]), bpm))
    return tempo_map if len(tempo_map) >= 2 else []


def analyze_beat_this_grid(
    beats: Sequence[float],
    downbeats: Sequence[float],
    *,
    tempo_mode: str = TempoMode.FIXED_AUTO.value,
    manual_bpm: float | None = None,
) -> BeatInfo:
    """Apply the reviewed TelkNet v10 grid and one explicit tempo mode."""

    selected_mode = str(tempo_mode or TempoMode.FIXED_AUTO.value).strip().lower()
    if selected_mode not in {mode.value for mode in TempoMode}:
        raise BeatThisGridError(f"Unsupported MIDI tempo mode: {selected_mode!r}")
    try:
        normalized = normalize_beat_grid([float(value) for value in beats])
        if normalized.bpm is None or len(normalized.beat_times) < MIN_BEATS:
            raise RuntimeError(
                f"Beat This detected only {len(normalized.beat_times)} usable beats; "
                f"at least {MIN_BEATS} are required"
            )
        normalized_downbeats = normalize_downbeat_grid(
            [float(value) for value in downbeats],
            normalized.beat_times,
        )
        beats_per_bar = infer_telknet_beats_per_bar(
            normalized.beat_times,
            normalized_downbeats.downbeat_times,
        )
        time_signature = (beats_per_bar, 4) if beats_per_bar is not None else None

        detected_bpm = float(normalized.bpm)
        tempo_map: list[tuple[float, float]] = []
        strategy = "fixed_auto"
        maximum_phase_error = 0.0
        output_bpm = detected_bpm
        source_bpm: float | None = None
        if selected_mode == TempoMode.ADAPTIVE.value:
            if normalized.fixed_tempo_reliable:
                strategy = "fixed_reliable"
            else:
                adaptive = build_adaptive_tempo_map(
                    normalized.beat_times,
                    normalized_downbeats.downbeat_times,
                    beats_per_bar=beats_per_bar,
                    representative_bpm=detected_bpm,
                )
                tempo_map = list(adaptive.events)
                strategy = adaptive.strategy
                maximum_phase_error = adaptive.maximum_phase_error_beats
                output_bpm = float(tempo_map[0][1])
        elif selected_mode == TempoMode.FIXED_MANUAL.value:
            if (
                manual_bpm is None
                or not math.isfinite(float(manual_bpm))
                or not MIN_TEMPO_BPM <= float(manual_bpm) <= MAX_TEMPO_BPM
            ):
                raise RuntimeError(
                    f"Manual MIDI BPM must be between {MIN_TEMPO_BPM:g} " f"and {MAX_TEMPO_BPM:g}"
                )
            output_bpm = float(manual_bpm)
            source_bpm = detected_bpm
            strategy = "fixed_manual"
        elif manual_bpm is not None:
            raise RuntimeError("manual_bpm is only valid for fixed_manual tempo mode")
    except BeatThisGridError:
        raise
    except Exception as exc:
        raise BeatThisGridError(f"TelkNet v10 beat-grid validation failed: {exc}") from exc

    logger.info(
        "Beat This %s: mode=%s strategy=%s detected=%.6f BPM output=%.6f BPM, "
        "beats=%d, downbeats=%d, meter=%s, duplicates_removed=%d, "
        "missing_beats_interpolated=%d, isolated_phase_repairs=%d, "
        "octave_family_normalized=%s, residual=%.1f ms, tempo_points=%d, "
        "max_phase_error=%.6f beats",
        TEMPO_FIT_ID,
        selected_mode,
        strategy,
        detected_bpm,
        output_bpm,
        len(normalized.beat_times),
        len(normalized_downbeats.downbeat_times),
        time_signature,
        normalized.duplicate_beats_removed,
        normalized.missing_beats_interpolated,
        normalized.isolated_phase_outliers_repaired,
        normalized.octave_family_normalized,
        float(normalized.residual_seconds or 0.0) * 1000.0,
        len(tempo_map),
        maximum_phase_error,
    )
    return BeatInfo(
        bpm=output_bpm,
        beat_times=list(normalized.beat_times),
        downbeats=list(normalized_downbeats.downbeat_times) or None,
        time_signature=time_signature,
        tempo_map=tempo_map,
        source_bpm=source_bpm,
    )


class BeatThisTracker:
    """Lazy, local-checkpoint-only Beat This ``final0`` runtime wrapper."""

    def __init__(
        self,
        config: Config,
        *,
        checkpoint_path: Path | None = None,
        tracker_factory: Optional[Callable[..., object]] = None,
    ) -> None:
        self.config = config
        self.checkpoint_path = checkpoint_path
        self._tracker_factory = tracker_factory
        self._tracker: object | None = None
        self._device: str | None = None
        self._translator = Translator(getattr(self.config, "language", Translator.DEFAULT_LANGUAGE))

    def _target_device(self) -> str:
        device = get_device(
            bool(getattr(self.config, "use_gpu", True)),
            int(getattr(self.config, "gpu_device", 0)),
        )
        ensure_accelerator_runtime_compatibility(device)
        return device

    def _load_tracker(self) -> object:
        device = self._target_device()
        if self._tracker is not None:
            if device != self._device:
                raise RuntimeError(
                    f"Beat This device changed after model load: {self._device} -> {device}"
                )
            return self._tracker

        checkpoint = validate_beat_this_checkpoint(self.checkpoint_path)
        factory = self._tracker_factory
        if factory is None:
            try:
                from beat_this.inference import File2Beats
            except Exception as exc:
                raise RuntimeError(f"Beat This 1.1.0 runtime import failed: {exc}") from exc
            factory = File2Beats
        try:
            self._tracker = factory(
                checkpoint_path=str(checkpoint),
                device=device,
                float16=False,
                dbn=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Beat This final0 model loading failed: {exc}") from exc
        if device.startswith(("cuda", "xpu")):
            model = getattr(self._tracker, "model", None)
            if model is None:
                raise RuntimeError(
                    f"Beat This runtime did not expose its model for {device} residency validation"
                )
            ensure_module_on_device(model, device, "Beat This model")
        self._device = device
        return self._tracker

    def track_raw(self, audio_path: str):
        """Run final0 with a stable FP32 cuDNN path and return raw marks.

        Ampere and newer NVIDIA cards can otherwise use TF32 convolutions even
        when ``File2Beats(float16=False)`` is requested.  One borderline beat
        in the published TelkNet corpus then moves to a neighbouring final0
        frame.  Scope the precision change to this call and restore the prior
        process setting so the transcription backends keep their own policy.
        """

        source = Path(audio_path)
        if not source.is_file():
            raise FileNotFoundError(f"Beat This input audio does not exist: {source}")
        tracker = self._load_tracker()
        device = str(self._device or "")
        if not device.startswith("cuda"):
            return tracker(str(source))  # type: ignore[operator]

        import torch

        previous_allow_tf32 = bool(torch.backends.cudnn.allow_tf32)
        torch.backends.cudnn.allow_tf32 = False
        try:
            return tracker(str(source))  # type: ignore[operator]
        finally:
            torch.backends.cudnn.allow_tf32 = previous_allow_tf32

    def detect(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> BeatInfo:
        source = Path(audio_path)
        if not source.is_file():
            raise FileNotFoundError(f"Beat This input audio does not exist: {source}")
        if progress_callback is not None:
            progress_callback(0.0, self._translator.t("progress.loading_beat_this"))
        self._load_tracker()
        if progress_callback is not None:
            progress_callback(0.35, self._translator.t("progress.running_beat_this"))
        try:
            beats, downbeats = self.track_raw(str(source))
        except Exception as exc:
            raise RuntimeError(f"Beat This final0 inference failed: {exc}") from exc
        if progress_callback is not None:
            progress_callback(0.80, self._translator.t("progress.validating_beat_grid"))
        info = analyze_beat_this_grid(
            beats,
            downbeats,
            tempo_mode=getattr(self.config, "tempo_mode", TempoMode.FIXED_AUTO.value),
            manual_bpm=getattr(self.config, "custom_bpm", None),
        )
        if progress_callback is not None:
            progress_callback(
                1.0,
                self._translator.t("progress.beat_this_complete", bpm=f"{info.bpm:.1f}"),
            )
        return info


__all__ = [
    "BEAT_THIS_CHECKPOINT_NAME",
    "BEAT_THIS_CHECKPOINT_SHA256",
    "BEAT_THIS_CHECKPOINT_SIZE",
    "BEAT_THIS_CHECKPOINT_URL",
    "BeatThisGridError",
    "BeatThisTracker",
    "CleanBeatGrid",
    "analyze_beat_this_grid",
    "build_variable_tempo_map",
    "count_musical_beat_positions",
    "fit_global_tempo",
    "get_beat_this_checkpoint_path",
    "infer_time_signature",
    "remove_competing_beat_marks",
    "validate_beat_this_checkpoint",
]
