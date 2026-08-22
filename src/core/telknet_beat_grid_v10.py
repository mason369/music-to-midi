"""TelkNet v10 Beat This grid normalization for local MIDI generation.

This is the project-side implementation reviewed against the user private
TelkNet dev branch. It consumes Beat This final0 timestamps only; it does
not run another detector and it fails closed on invalid evidence.
"""

from __future__ import annotations

import math
import statistics
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

MIN_TEMPO_BPM = 30.0
MAX_TEMPO_BPM = 300.0
MIN_BEAT_PERIOD_SECONDS = 60.0 / MAX_TEMPO_BPM
MAX_BEAT_PERIOD_SECONDS = 60.0 / MIN_TEMPO_BPM
PERIOD_BOUNDARY_TOLERANCE_SECONDS = 1e-9
DUPLICATE_PERIOD_RATIO = 0.55
DOMINANT_PERIOD_TOLERANCE_RATIO = 0.05
MIN_DOMINANT_PERIOD_RATIO = 0.80
MAX_DOMINANT_PERIOD_RESIDUAL_RATIO = 0.04
MAX_GLOBAL_FIT_RESIDUAL_RATIO = 0.05
MAX_GLOBAL_FIT_DISAGREEMENT_RATIO = 0.05
MIN_SUSTAINED_CHANGE_BEATS = 8
OCTAVE_FAMILY_TOLERANCE_RATIO = 0.10
MIN_BEATS = 8
MIN_GLOBAL_FIT_BEATS = 32
MIN_METER_AGREEMENT = 0.90
MAX_FIXED_TEMPO_PHASE_ERROR_BEATS = 3.0 / 8.0
MIN_ISOLATED_PHASE_OUTLIER_BEATS = 1.0 / 16.0
MAX_ISOLATED_PHASE_OUTLIER_BEATS = 3.0 / 8.0
ISOLATED_PHASE_OUTER_PERIOD_TOLERANCE_RATIO = 0.05
TEMPO_FIT_ID = "beat-this-final0-origin-l2-minimax-phase-grid-v10"


@dataclass(frozen=True)
class NormalizedBeatGrid:
    raw_beat_times: tuple[float, ...]
    beat_times: tuple[float, ...]
    bpm: float | None
    residual_seconds: float | None
    fixed_tempo_reliable: bool
    tempo_warning: str | None
    duplicate_beats_removed: int
    missing_beats_interpolated: int
    isolated_phase_outliers_repaired: int = 0
    octave_family_normalized: bool = False
    tactus_period_seconds: float | None = None


@dataclass(frozen=True)
class NormalizedDownbeatGrid:
    raw_downbeat_times: tuple[float, ...]
    downbeat_times: tuple[float, ...]
    duplicate_downbeats_removed: int
    unaligned_downbeats_removed: int


def _validated_times(values: Sequence[float]) -> list[float]:
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("Beat This produced an invalid beat timestamp")
        timestamp = float(value)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise RuntimeError("Beat This produced an invalid beat timestamp")
        if normalized and timestamp <= normalized[-1]:
            raise RuntimeError("Beat This produced a non-monotonic beat timeline")
        normalized.append(timestamp)
    return normalized


def normalize_downbeat_grid(
    downbeat_times: Sequence[float],
    normalized_beat_times: Sequence[float],
) -> NormalizedDownbeatGrid:
    """Snap official downbeats to the accepted beat lattice and de-duplicate.

    Beat This can label both members of one close duplicate beat pair as
    downbeats.  The beat-grid pass already selects the lattice-consistent
    member, so publishing the untouched downbeat array would reintroduce a
    physically impossible bar interval.  Keep only downbeats supported by the
    normalized beat evidence and collapse two labels that resolve to one beat.
    """

    raw = _validated_times(downbeat_times)
    beats = _validated_times(normalized_beat_times)
    if not raw or not beats:
        return NormalizedDownbeatGrid(tuple(raw), (), 0, len(raw))

    beat_periods = [current - previous for previous, current in zip(beats, beats[1:])]
    representative_period = float(statistics.median(beat_periods)) if beat_periods else 0.5
    alignment_tolerance = min(
        0.12,
        max(0.05, representative_period * 0.25),
    )
    accepted: list[float] = []
    duplicate_count = 0
    unaligned_count = 0

    for downbeat in raw:
        insertion = bisect_left(beats, downbeat)
        candidate_indices = [
            index for index in (insertion - 1, insertion) if 0 <= index < len(beats)
        ]
        nearest = min(candidate_indices, key=lambda index: abs(beats[index] - downbeat))
        snapped = beats[nearest]
        if abs(snapped - downbeat) > alignment_tolerance:
            unaligned_count += 1
            continue
        if accepted and snapped <= accepted[-1]:
            duplicate_count += 1
            continue
        accepted.append(snapped)

    return NormalizedDownbeatGrid(
        raw_downbeat_times=tuple(raw),
        downbeat_times=tuple(accepted),
        duplicate_downbeats_removed=duplicate_count,
        unaligned_downbeats_removed=unaligned_count,
    )


def _period_error(interval: float, reference_period: float) -> float:
    steps = max(1, round(interval / reference_period))
    return abs(interval / steps - reference_period) / reference_period


def _longest_period_family_run(
    intervals: Sequence[float],
    target_period: float,
) -> int:
    longest = 0
    current = 0
    for interval in intervals:
        if abs(float(interval) - target_period) / target_period <= OCTAVE_FAMILY_TOLERANCE_RATIO:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _select_reference_period(intervals: Sequence[float]) -> tuple[float, bool]:
    """Select one stable tactus before gap interpolation.

    Beat This final0 can switch from quarter-note detections to half-time
    detections in the middle of an otherwise steady recording.  A raw median
    then selects whichever family occupies more of the file and turns the
    other half into a false tempo change.  Only normalize the octave when both
    the short and doubled families are independently sustained for at least
    ``MIN_SUSTAINED_CHANGE_BEATS`` consecutive intervals.  Isolated missed
    beats therefore remain ordinary gaps, and a song with only one detected
    family keeps the official median unchanged.
    """

    values = [float(value) for value in intervals]
    median_period = float(statistics.median(values))
    best: tuple[int, float, float] | None = None
    # Beat This timestamps sit on 20 ms frames. Candidate tactus periods can
    # therefore be grouped at a much finer 1 ms precision without evaluating
    # the same O(n) family lists hundreds of times.
    seeds = sorted({round(value, 3) for value in values})
    for seed in seeds:
        if not MIN_BEAT_PERIOD_SECONDS <= seed <= MAX_BEAT_PERIOD_SECONDS / 2.0:
            continue
        short_family = [
            value for value in values if abs(value - seed) / seed <= OCTAVE_FAMILY_TOLERANCE_RATIO
        ]
        if len(short_family) < MIN_SUSTAINED_CHANGE_BEATS:
            continue
        center = float(statistics.median(short_family))
        short_family = [
            value
            for value in values
            if abs(value - center) / center <= OCTAVE_FAMILY_TOLERANCE_RATIO
        ]
        long_center = center * 2.0
        long_family = [
            value
            for value in values
            if abs(value - long_center) / long_center <= OCTAVE_FAMILY_TOLERANCE_RATIO
        ]
        if (
            len(long_family) < MIN_SUSTAINED_CHANGE_BEATS
            or _longest_period_family_run(values, center) < MIN_SUSTAINED_CHANGE_BEATS
            or _longest_period_family_run(values, long_center) < MIN_SUSTAINED_CHANGE_BEATS
        ):
            continue
        score = len(short_family) + len(long_family)
        candidate = (score, -center, center)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return median_period, False
    return best[2], True


def infer_beats_per_bar(
    normalized_beat_times: Sequence[float],
    normalized_downbeat_times: Sequence[float],
) -> int | None:
    """Infer meter from beat ordinals while tolerating missed downbeats.

    Dividing downbeat seconds by one median period reproduces Beat This's
    half/double-time switch in the meter estimate.  The downbeats have already
    been snapped to the accepted beat lattice, so count exact beat ordinals
    instead.  A missing downbeat may span two or more complete bars; it is not
    evidence that the time signature itself doubled.
    """

    beats = _validated_times(normalized_beat_times)
    downbeats = _validated_times(normalized_downbeat_times)
    if len(beats) < 2 or len(downbeats) < 3:
        return None
    indices: list[int] = []
    for downbeat in downbeats:
        insertion = bisect_left(beats, downbeat)
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(beats)]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda index: abs(beats[index] - downbeat))
        if abs(beats[nearest] - downbeat) <= 1e-6:
            indices.append(nearest)
    counts = [
        current - previous
        for previous, current in zip(indices, indices[1:])
        if current - previous >= 2
    ]
    if not counts:
        return None
    candidates = sorted({count for count in counts if 2 <= count <= 16})
    ranked: list[tuple[int, float, int]] = []
    for candidate in candidates:
        explained = sum(count % candidate == 0 for count in counts)
        agreement = explained / len(counts)
        exact = sum(count == candidate for count in counts)
        if agreement >= MIN_METER_AGREEMENT and exact / len(counts) >= 0.40:
            ranked.append((exact, agreement, candidate))
    if not ranked:
        return None
    _exact, _agreement, candidate = max(ranked)
    return candidate


def _removal_score(times: Sequence[float], remove_index: int, period: float) -> float:
    remaining_count = len(times) - 1
    start = max(0, remove_index - 2)
    stop = min(remaining_count - 1, remove_index + 1)

    def remaining_at(index: int) -> float:
        return float(times[index if index < remove_index else index + 1])

    return sum(
        _period_error(remaining_at(index + 1) - remaining_at(index), period)
        for index in range(start, stop + 1)
        if index + 1 < remaining_count
    )


def _remove_duplicate_detections(
    beat_times: Sequence[float],
    *,
    reference_period: float,
) -> tuple[list[float], int]:
    """Remove only detections too close to represent a reviewed musical beat.

    Beat This final0 occasionally emits two timestamps around one onset.  The
    closer pair is not blindly truncated: the timestamp whose removal best
    restores the surrounding integer beat lattice is removed.  A true double-
    tempo passage remains valid because its half-period is above the strict
    45-percent duplicate boundary.
    """

    times = list(beat_times)
    removed = 0
    duplicate_threshold = max(
        MIN_BEAT_PERIOD_SECONDS,
        reference_period * DUPLICATE_PERIOD_RATIO,
    )
    close_index = 0
    while close_index + 1 < len(times):
        if times[close_index + 1] - times[close_index] >= duplicate_threshold:
            close_index += 1
            continue
        left_index = close_index
        right_index = close_index + 1
        if left_index == 0:
            remove_index = right_index
        elif right_index == len(times) - 1:
            remove_index = left_index
        else:
            left_score = _removal_score(times, left_index, reference_period)
            right_score = _removal_score(times, right_index, reference_period)
            remove_index = left_index if left_score < right_score else right_index
        del times[remove_index]
        removed += 1
        close_index = max(0, remove_index - 1)
    return times, removed


def _expand_missing_beats(
    beat_times: Sequence[float],
    *,
    reference_period: float,
) -> tuple[list[float], int]:
    """Represent long detector gaps as integer missing beats in linear time."""

    if len(beat_times) < 2:
        return list(beat_times), 0
    expanded = [float(beat_times[0])]
    interpolated = 0
    for previous, current in zip(beat_times, beat_times[1:]):
        interval = current - previous
        steps = _normalized_interval_steps(
            interval,
            reference_period=reference_period,
        )
        local_period = interval / steps
        if (
            local_period < MIN_BEAT_PERIOD_SECONDS - PERIOD_BOUNDARY_TOLERANCE_SECONDS
            or local_period > MAX_BEAT_PERIOD_SECONDS + PERIOD_BOUNDARY_TOLERANCE_SECONDS
        ):
            raise RuntimeError(
                "Beat This could not normalize a beat interval inside the reviewed "
                f"{MIN_TEMPO_BPM:.0f}-{MAX_TEMPO_BPM:.0f} BPM range"
            )
        for step in range(1, steps + 1):
            expanded.append(previous + interval * step / steps)
        interpolated += steps - 1
    return expanded, interpolated


def _repair_isolated_phase_outliers(
    beat_times: Sequence[float],
) -> tuple[list[float], int]:
    """Repair only one-beat timestamp zigzags surrounded by one stable clock.

    final0 timestamps use 20 ms frames. An isolated frame error can lengthen
    one interval and shorten the next by the same amount. Treating both as
    structural tempo changes creates a slow/fast one-beat Tempo Map spike even
    though the clock before and after never moved. A candidate is accepted
    only when the two untouched outer intervals agree, their two-beat span
    agrees with that local clock, and no adjacent timestamp is also a
    candidate. Sustained or abrupt tempo changes therefore remain untouched.
    """

    times = _validated_times(beat_times)
    if len(times) < 5:
        return times, 0
    candidates: list[int] = []
    for index in range(2, len(times) - 2):
        previous_outer = times[index - 1] - times[index - 2]
        following_outer = times[index + 2] - times[index + 1]
        outer_period = (previous_outer + following_outer) / 2.0
        if (
            outer_period <= 0
            or abs(previous_outer - following_outer) / outer_period
            > ISOLATED_PHASE_OUTER_PERIOD_TOLERANCE_RATIO
        ):
            continue
        pair_period = (times[index + 1] - times[index - 1]) / 2.0
        if (
            pair_period <= 0
            or abs(pair_period - outer_period) / outer_period
            > ISOLATED_PHASE_OUTER_PERIOD_TOLERANCE_RATIO
        ):
            continue
        midpoint = (times[index - 1] + times[index + 1]) / 2.0
        phase_error = abs(times[index] - midpoint) / pair_period
        if MIN_ISOLATED_PHASE_OUTLIER_BEATS < phase_error <= MAX_ISOLATED_PHASE_OUTLIER_BEATS:
            candidates.append(index)

    accepted = [
        index
        for index in candidates
        if all(abs(index - other) > 1 for other in candidates if other != index)
    ]
    repaired = list(times)
    for index in accepted:
        repaired[index] = (times[index - 1] + times[index + 1]) / 2.0
    return repaired, len(accepted)


def _normalized_interval_steps(
    interval: float,
    *,
    reference_period: float,
) -> int:
    """Choose the nearest integer lattice count that remains musically valid.

    A detector interval can sit exactly on the reviewed 300 BPM boundary but
    evaluate a few femtoseconds below 0.2 seconds after binary subtraction.
    It can also round to two reference beats even when one step is the only
    count that stays inside the accepted tempo range. Derive the complete valid
    integer range first, with a nanosecond boundary tolerance, then clamp the
    reference-preferred count into that range. A truly sub-0.2-second interval
    still fails closed after duplicate removal.
    """

    if not math.isfinite(interval) or interval <= 0:
        raise RuntimeError("Beat This produced an invalid beat interval")
    minimum_steps = max(
        1,
        math.ceil((interval - PERIOD_BOUNDARY_TOLERANCE_SECONDS) / MAX_BEAT_PERIOD_SECONDS),
    )
    maximum_steps = math.floor(
        (interval + PERIOD_BOUNDARY_TOLERANCE_SECONDS) / MIN_BEAT_PERIOD_SECONDS
    )
    if maximum_steps < minimum_steps:
        raise RuntimeError(
            "Beat This could not normalize a beat interval inside the reviewed "
            f"{MIN_TEMPO_BPM:.0f}-{MAX_TEMPO_BPM:.0f} BPM range"
        )
    preferred_steps = max(1, round(interval / reference_period))
    return min(max(preferred_steps, minimum_steps), maximum_steps)


def _fit_tempo(beat_times: Sequence[float]) -> tuple[float, float]:
    """Least-squares fit over the normalized, gap-aware beat ordinal."""

    import numpy as np

    beats = np.asarray(beat_times, dtype=float)
    index = np.arange(len(beats), dtype=float)
    slope, intercept = np.polyfit(index, beats, 1)
    if not np.isfinite(slope) or slope <= 0:
        raise RuntimeError("Beat This produced an invalid beat slope")
    residual = beats - (intercept + slope * index)
    return 60.0 / float(slope), float(residual.std())


def _fit_origin_tempo(beat_times: Sequence[float]) -> tuple[float, float]:
    """Fit the least-squares beat rate in the coordinate MIDI can publish.

    The statistical fixed/variable classifier deliberately permits a free
    intercept so an imprecise first detector frame does not make a stable song
    look variable.  An SMF tempo event cannot publish that fitted intercept,
    however: note seconds stay unchanged and the effective beat phase is
    measured from the first accepted beat.  Solve that origin-constrained
    least-squares problem explicitly before publishing a fixed BPM.
    """

    times = _validated_times(beat_times)
    if len(times) < 2:
        raise RuntimeError("Beat This produced too few beats for a tempo fit")
    origin = times[0]
    elapsed = [timestamp - origin for timestamp in times[1:]]
    ordinals = list(range(1, len(times)))
    denominator = sum(seconds * seconds for seconds in elapsed)
    if not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("Beat This produced an invalid origin-constrained slope")
    beats_per_second = (
        sum(seconds * ordinal for ordinal, seconds in zip(ordinals, elapsed)) / denominator
    )
    bpm = 60.0 * beats_per_second
    if not math.isfinite(bpm) or not MIN_TEMPO_BPM <= bpm <= MAX_TEMPO_BPM:
        raise RuntimeError("Beat This produced an invalid origin-constrained slope")
    max_phase_error = max(
        abs(seconds * beats_per_second - ordinal) for ordinal, seconds in zip(ordinals, elapsed)
    )
    return bpm, max_phase_error


def _fit_minimax_origin_tempo(
    beat_times: Sequence[float],
) -> tuple[float, float]:
    """Fit the constant BPM with the smallest worst-case beat-phase error.

    Least squares minimizes average timestamp error but can still leave one
    part of a long recording more than half a beat away from the DAW grid.
    For a constant tempo, every accepted beat constrains the admissible beat
    rate to one interval. Binary-search the smallest error whose intervals
    intersect, then choose the midpoint of that intersection. The result is
    deterministic and does not run another detector or decode the audio.
    """

    times = _validated_times(beat_times)
    if len(times) < 2:
        raise RuntimeError("Beat This produced too few beats for a tempo fit")
    origin = times[0]
    elapsed = [timestamp - origin for timestamp in times[1:]]
    ordinals = list(range(1, len(times)))

    lower_error = 0.0
    upper_error = float(len(times))
    # Forty halvings leave substantially finer than nanosecond-level beat
    # placement over the reviewed 30--300 BPM range; more iterations only add
    # Python work without changing an SMF tempo integer.
    for _iteration in range(40):
        error = (lower_error + upper_error) / 2.0
        minimum_rate = max(
            (ordinal - error) / seconds for ordinal, seconds in zip(ordinals, elapsed)
        )
        maximum_rate = min(
            (ordinal + error) / seconds for ordinal, seconds in zip(ordinals, elapsed)
        )
        if minimum_rate <= maximum_rate:
            upper_error = error
        else:
            lower_error = error

    error = upper_error
    minimum_rate = max((ordinal - error) / seconds for ordinal, seconds in zip(ordinals, elapsed))
    maximum_rate = min((ordinal + error) / seconds for ordinal, seconds in zip(ordinals, elapsed))
    beats_per_second = (minimum_rate + maximum_rate) / 2.0
    bpm = 60.0 * beats_per_second
    if not MIN_TEMPO_BPM <= bpm <= MAX_TEMPO_BPM:
        raise RuntimeError("Beat This produced an invalid minimax beat slope")
    max_phase_error = max(
        abs(seconds * beats_per_second - ordinal) for ordinal, seconds in zip(ordinals, elapsed)
    )
    return bpm, max_phase_error


def _origin_phase_violation_count(
    beat_times: Sequence[float],
    bpm: float,
    *,
    maximum_error_beats: float,
) -> int:
    times = _validated_times(beat_times)
    if len(times) < 2 or not math.isfinite(float(bpm)) or float(bpm) <= 0:
        raise RuntimeError("Beat This produced invalid phase-fit evidence")
    origin = times[0]
    beats_per_second = float(bpm) / 60.0
    return sum(
        abs((timestamp - origin) * beats_per_second - ordinal) > maximum_error_beats
        for ordinal, timestamp in enumerate(times)
    )


def _dominant_period(
    observed_periods: Sequence[float],
    *,
    reference_period: float,
    timeline_periods: Sequence[float],
) -> tuple[float, float, float, int]:
    """Return the dense beat-period cluster without amplifying frame jitter.

    Beat This timestamps are quantized to its inference frame.  Comparing each
    adjacent pair as an exact tempo turns a harmless 20 ms offset into a BPM
    jump.  Each observed interval gets one vote after missing-beat division;
    synthetic interpolation points therefore cannot outweigh the detector's
    actual evidence. This pass finds the densest bounded period family, then
    measures its support and RMS error.
    """

    periods = sorted(float(period) for period in observed_periods)
    if not periods:
        raise RuntimeError("Beat This produced too few beat periods")
    period_span_ratio = (1.0 + DOMINANT_PERIOD_TOLERANCE_RATIO) / (
        1.0 - DOMINANT_PERIOD_TOLERANCE_RATIO
    )
    best_left = 0
    best_right = 1
    right = 0
    for left, lower in enumerate(periods):
        right = max(right, left + 1)
        while right < len(periods) and periods[right] <= lower * period_span_ratio:
            right += 1
        best_count = best_right - best_left
        candidate_count = right - left
        if candidate_count > best_count:
            best_left, best_right = left, right
        elif candidate_count == best_count:
            best_center = statistics.median(periods[best_left:best_right])
            candidate_center = statistics.median(periods[left:right])
            if abs(candidate_center - reference_period) < abs(best_center - reference_period):
                best_left, best_right = left, right

    center = float(statistics.median(periods[best_left:best_right]))
    inliers = [
        period
        for period in periods
        if abs(period - center) / center <= DOMINANT_PERIOD_TOLERANCE_RATIO
    ]
    center = float(statistics.median(inliers))
    inliers = [
        period
        for period in periods
        if abs(period - center) / center <= DOMINANT_PERIOD_TOLERANCE_RATIO
    ]
    residual = math.sqrt(sum((period - center) ** 2 for period in inliers) / len(inliers))
    longest_change = 0
    current_change = 0
    for period in timeline_periods:
        if abs(period - center) / center > DOMINANT_PERIOD_TOLERANCE_RATIO:
            current_change += 1
            longest_change = max(longest_change, current_change)
        else:
            current_change = 0
    return center, len(inliers) / len(periods), residual, longest_change


def normalize_beat_grid(beat_times: Sequence[float]) -> NormalizedBeatGrid:
    raw = _validated_times(beat_times)
    if len(raw) < 2:
        return NormalizedBeatGrid(
            raw_beat_times=tuple(raw),
            beat_times=tuple(raw),
            bpm=None,
            residual_seconds=None,
            fixed_tempo_reliable=False,
            tempo_warning="Too few beats were detected to estimate a global BPM.",
            duplicate_beats_removed=0,
            missing_beats_interpolated=0,
            isolated_phase_outliers_repaired=0,
        )

    raw_intervals = [current - previous for previous, current in zip(raw, raw[1:])]
    reference_period, octave_family_normalized = _select_reference_period(raw_intervals)
    if not math.isfinite(reference_period) or reference_period <= 0:
        raise RuntimeError("Beat This produced an invalid representative beat period")

    deduplicated, duplicate_count = _remove_duplicate_detections(
        raw,
        reference_period=reference_period,
    )
    if len(deduplicated) < 2:
        raise RuntimeError("Beat This produced too few distinct beat detections")
    repaired, phase_outlier_count = _repair_isolated_phase_outliers(deduplicated)

    # Use final0's robust median interval only to identify the beat family and
    # normalize sparse gaps. Its timestamps are quantized to 20 ms frames, so
    # the median adjacent interval can land on 0.50 seconds even when the full
    # song clock is about 0.492 seconds. Publishing that local median as the
    # fixed BPM accumulates seconds of bar-grid drift over a long recording.
    expanded, missing_count = _expand_missing_beats(
        repaired,
        reference_period=reference_period,
    )
    representative_period = reference_period
    bpm = 60.0 / representative_period
    if not MIN_TEMPO_BPM <= bpm <= MAX_TEMPO_BPM:
        warning = (
            f"The robust global tempo {bpm:.1f} BPM is outside the reviewed "
            f"{MIN_TEMPO_BPM:.0f}-{MAX_TEMPO_BPM:.0f} BPM range."
        )
        return NormalizedBeatGrid(
            raw_beat_times=tuple(raw),
            beat_times=tuple(expanded),
            bpm=bpm,
            residual_seconds=0.0,
            fixed_tempo_reliable=False,
            tempo_warning=warning,
            duplicate_beats_removed=duplicate_count,
            missing_beats_interpolated=missing_count,
            isolated_phase_outliers_repaired=phase_outlier_count,
            octave_family_normalized=octave_family_normalized,
            tactus_period_seconds=reference_period,
        )

    fitted_bpm, lattice_residual = _fit_tempo(expanded)
    observed_periods = []
    for previous, current in zip(repaired, repaired[1:]):
        interval = current - previous
        steps = _normalized_interval_steps(
            interval,
            reference_period=representative_period,
        )
        observed_periods.append(interval / steps)
    timeline_periods = [current - previous for previous, current in zip(expanded, expanded[1:])]
    dominant_period, dominant_ratio, dominant_residual, longest_change = _dominant_period(
        observed_periods,
        reference_period=representative_period,
        timeline_periods=timeline_periods,
    )
    dominant_fixed = (
        dominant_ratio >= MIN_DOMINANT_PERIOD_RATIO
        and dominant_residual <= MAX_DOMINANT_PERIOD_RESIDUAL_RATIO * dominant_period
        and longest_change < MIN_SUSTAINED_CHANGE_BEATS
    )
    fitted_period = 60.0 / fitted_bpm
    dominant_bpm = 60.0 / dominant_period
    global_fit_disagreement = abs(fitted_bpm - dominant_bpm) / dominant_bpm
    use_global_fit = len(expanded) >= MIN_GLOBAL_FIT_BEATS
    global_fit_reliable = not use_global_fit or (
        lattice_residual <= MAX_GLOBAL_FIT_RESIDUAL_RATIO * fitted_period
        and global_fit_disagreement <= MAX_GLOBAL_FIT_DISAGREEMENT_RATIO
    )
    fixed_reliable = len(expanded) >= MIN_BEATS and dominant_fixed and global_fit_reliable
    # Beat This occasionally emits one isolated half-beat or a short run of
    # frame-jittered detections. Those outliers can fail the statistical fixed
    # classifier and previously forced a dense variable tempo map with
    # musically impossible one-beat BPM spikes. If one constant clock keeps
    # every accepted beat strictly inside the nearest-beat ambiguity boundary,
    # the source has no evidence that requires a variable DAW tempo map.
    phase_candidate = (
        not fixed_reliable
        and len(raw) >= MIN_GLOBAL_FIT_BEATS
        and len(expanded) >= MIN_BEATS
        and dominant_ratio >= 0.90
        and global_fit_disagreement <= MAX_GLOBAL_FIT_DISAGREEMENT_RATIO
        and longest_change < MIN_SUSTAINED_CHANGE_BEATS
    )
    minimax_bpm: float | None = None
    minimax_phase_error = math.inf
    origin_bpm: float | None = None
    if fixed_reliable and use_global_fit:
        origin_bpm, _origin_phase_error = _fit_origin_tempo(expanded)
    if phase_candidate:
        minimax_bpm, minimax_phase_error = _fit_minimax_origin_tempo(expanded)
    phase_fixed = phase_candidate and minimax_phase_error <= MAX_FIXED_TEMPO_PHASE_ERROR_BEATS
    fixed_reliable = fixed_reliable or phase_fixed
    minimax_selected = phase_fixed
    published_phase_violations = 0
    if fixed_reliable and not phase_fixed:
        published_bpm = origin_bpm if use_global_fit else dominant_bpm
        assert published_bpm is not None
        # One bad terminal detector frame is an outlier, not accumulated
        # clock drift. Validate the origin-constrained slope that will actually
        # be published, not the free-intercept classifier fit. Require repeated
        # violations before changing the robust fixed-tempo slope; this
        # preserves the dominant clock while still correcting a phase error
        # observed in separate parts of the song. Short clips publish the
        # dominant period directly, so the same published-slope check applies
        # to them as well.
        published_phase_violations = _origin_phase_violation_count(
            expanded,
            published_bpm,
            maximum_error_beats=MAX_FIXED_TEMPO_PHASE_ERROR_BEATS,
        )
        if published_phase_violations >= 2:
            minimax_bpm, minimax_phase_error = _fit_minimax_origin_tempo(expanded)
            minimax_selected = minimax_phase_error <= MAX_FIXED_TEMPO_PHASE_ERROR_BEATS
            if not minimax_selected:
                # The statistical classifier saw one stable tempo family, but
                # no constant clock keeps the accepted beats inside the
                # reviewed phase bound (for example a genuinely faster opening
                # section). Publish the honest variable tempo map instead of a
                # fixed BPM that downstream fail-closed validation must reject.
                fixed_reliable = False
    if fixed_reliable:
        # The dominant cluster establishes that this is one stable tempo
        # family. For long recordings, the full-song slope supplies the
        # sub-frame BPM that keeps the first and last bars on the same phase;
        # short clips retain the robust dominant period.
        if minimax_selected:
            assert minimax_bpm is not None
            bpm = minimax_bpm
            representative_period = 60.0 / minimax_bpm
            residual = lattice_residual
        elif use_global_fit:
            assert origin_bpm is not None
            representative_period = 60.0 / origin_bpm
            bpm = origin_bpm
            residual = lattice_residual
        else:
            representative_period = dominant_period
            bpm = dominant_bpm
            residual = dominant_residual
    else:
        residual = lattice_residual
    if len(expanded) < MIN_BEATS:
        warning = (
            f"Only {len(expanded)} normalized beats were detected; at least "
            f"{MIN_BEATS} are required to confirm a fixed global BPM."
        )
    elif not fixed_reliable:
        warning = (
            "The recording does not have a reliable fixed tempo "
            f"({lattice_residual * 1000:.0f} ms lattice RMS deviation; "
            f"dominant period support {dominant_ratio * 100:.1f}%; "
            f"global/dominant BPM disagreement "
            f"{global_fit_disagreement * 100:.2f}%; "
            f"longest sustained change {longest_change} beats)."
        )
        if published_phase_violations >= 2:
            warning += (
                " The stable-tempo cluster was overruled because no constant "
                "publishable BPM keeps every accepted beat within the reviewed "
                f"{MAX_FIXED_TEMPO_PHASE_ERROR_BEATS:.3f}-beat phase bound "
                f"({published_phase_violations} beats violated it; the best "
                f"constant fit still misses by {minimax_phase_error:.3f} beats)."
            )
    else:
        warning = None
    return NormalizedBeatGrid(
        raw_beat_times=tuple(raw),
        beat_times=tuple(expanded),
        bpm=bpm,
        residual_seconds=residual,
        fixed_tempo_reliable=fixed_reliable,
        tempo_warning=warning,
        duplicate_beats_removed=duplicate_count,
        missing_beats_interpolated=missing_count,
        isolated_phase_outliers_repaired=phase_outlier_count,
        octave_family_normalized=octave_family_normalized,
        tactus_period_seconds=reference_period,
    )
