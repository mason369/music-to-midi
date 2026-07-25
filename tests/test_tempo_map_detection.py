"""变速 tempo map 的分段算法与 BeatInfo 显示契约。"""

import numpy as np
import pytest

from src.core.beat_detector import BeatDetector
from src.models.data_models import BeatInfo, Config


class TestBeatInfoTempoMap:
    def test_constant_tempo_display_is_single_value(self):
        info = BeatInfo(bpm=120.0)

        assert not info.is_variable_tempo
        assert info.bpm_display == "120.0"

    def test_variable_tempo_display_is_range(self):
        info = BeatInfo(bpm=90.0, tempo_map=[(0.0, 69.8), (30.0, 128.4)])

        assert info.is_variable_tempo
        assert info.bpm_display == "69.8–128.4"

    def test_single_point_tempo_map_is_treated_as_constant(self):
        info = BeatInfo(bpm=100.0, tempo_map=[(0.0, 100.0)])

        assert not info.is_variable_tempo
        assert info.bpm_display == "100.0"


class TestSlidingMedian:
    def test_smooths_single_spike(self):
        values = np.array([100.0] * 8 + [200.0] + [100.0] * 8)

        smoothed = BeatDetector._sliding_median(values, 8)

        assert np.all(smoothed == 100.0)

    def test_preserves_step_change(self):
        values = np.array([80.0] * 16 + [120.0] * 16)

        smoothed = BeatDetector._sliding_median(values, 8)

        assert smoothed[0] == 80.0
        assert smoothed[-1] == 120.0


class TestSegmentTempoCurve:
    @staticmethod
    def _times(n: int, step: float = 0.5) -> np.ndarray:
        return np.arange(1, n + 1, dtype=float) * step

    def test_constant_curve_returns_empty(self):
        bpms = np.full(32, 100.0)

        result = BeatDetector._segment_tempo_curve(
            bpms, self._times(32), tolerance=0.05, min_run=4
        )

        assert result == []

    def test_single_change_point(self):
        bpms = np.array([100.0] * 16 + [130.0] * 16)
        times = self._times(32)

        result = BeatDetector._segment_tempo_curve(
            bpms, times, tolerance=0.05, min_run=4
        )

        assert len(result) == 2
        assert result[0] == (0.0, pytest.approx(100.0))
        # 变化点记为新段第一个区间(索引16)的起始拍时刻 = times[15]
        assert result[1][0] == pytest.approx(times[15])
        assert result[1][1] == pytest.approx(130.0)

    def test_short_deviation_is_ignored(self):
        bpms = np.array([100.0] * 16 + [140.0] * 2 + [100.0] * 14)

        result = BeatDetector._segment_tempo_curve(
            bpms, self._times(32), tolerance=0.05, min_run=4
        )

        assert result == []

    def test_insignificant_change_is_merged(self):
        # 3% 差异低于 5% 容差，视为恒速抖动
        bpms = np.array([100.0] * 16 + [103.0] * 16)

        result = BeatDetector._segment_tempo_curve(
            bpms, self._times(32), tolerance=0.05, min_run=4
        )

        assert result == []

    def test_two_change_points(self):
        bpms = np.array([100.0] * 12 + [130.0] * 12 + [70.0] * 12)
        times = self._times(36)

        result = BeatDetector._segment_tempo_curve(
            bpms, times, tolerance=0.05, min_run=4
        )

        assert len(result) == 3
        assert [bpm for _, bpm in result] == [
            pytest.approx(100.0),
            pytest.approx(130.0),
            pytest.approx(70.0),
        ]
        assert result[1][0] == pytest.approx(times[11])
        assert result[2][0] == pytest.approx(times[23])

    def test_too_few_intervals_returns_empty(self):
        bpms = np.array([100.0] * 4 + [130.0] * 3)

        result = BeatDetector._segment_tempo_curve(
            bpms, self._times(7), tolerance=0.05, min_run=4
        )

        assert result == []


class TestDetectTempoMapFallback:
    def test_short_audio_returns_empty(self):
        detector = BeatDetector(Config())
        sr = 22050
        y = np.zeros(sr, dtype=np.float32)  # 1 秒静音：拍数不足

        assert detector._detect_tempo_map(y, sr) == []


class TestGateTempoMap:
    """高置信门控：防止逐帧 tempo 抖动产生错误的变速点。"""

    def test_rejects_too_many_sections(self):
        sections = [(i * 3.0, 70.0 if i % 2 == 0 else 140.0) for i in range(10)]

        assert BeatDetector._gate_tempo_map(sections, 30.0) == []

    def test_rejects_short_first_section(self):
        sections = [(0.0, 60.0), (5.0, 120.0)]

        assert BeatDetector._gate_tempo_map(sections, 30.0) == []

    def test_rejects_short_tail_section(self):
        sections = [(0.0, 60.0), (25.0, 120.0)]

        assert BeatDetector._gate_tempo_map(sections, 30.0) == []

    def test_rejects_insignificant_jump(self):
        sections = [(0.0, 100.0), (10.0, 108.0)]

        assert BeatDetector._gate_tempo_map(sections, 30.0) == []

    def test_accepts_clear_two_section_change(self):
        sections = [(0.0, 60.0), (15.0, 120.0)]

        assert BeatDetector._gate_tempo_map(sections, 30.0) == sections

    def test_rejects_single_section(self):
        assert BeatDetector._gate_tempo_map([(0.0, 100.0)], 30.0) == []


class TestOctaveCorrection:
    """tempogram 投票与拍间隔中位数呈严格倍频时，以拍间隔为准。"""

    def test_corrects_half_tempo_vote(self):
        det = BeatDetector(Config())
        # 78.3 -> 161.5 呈 2.06x 倍频：半频误检被纠正
        assert det._resolve_octave_by_interval_median(78.3, 161.5) == pytest.approx(161.5)

    def test_corrects_double_tempo_vote(self):
        det = BeatDetector(Config())
        assert det._resolve_octave_by_interval_median(161.5, 78.3) == pytest.approx(78.3)

    def test_keeps_non_octave_vote(self):
        det = BeatDetector(Config())
        # 123.0/118.4 = 1.039 非倍频：保持投票结果
        assert det._resolve_octave_by_interval_median(118.4, 123.0) == pytest.approx(118.4)

    def test_keeps_vote_when_median_missing(self):
        det = BeatDetector(Config())
        assert det._resolve_octave_by_interval_median(120.0, None) == pytest.approx(120.0)

    def test_keeps_vote_outside_octave_window(self):
        det = BeatDetector(Config())
        # 1.8x 不在 [1.9, 2.1] 窗口内：保持投票结果
        assert det._resolve_octave_by_interval_median(100.0, 180.0) == pytest.approx(100.0)
