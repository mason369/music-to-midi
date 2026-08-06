"""Inline project-native MIDI piano roll, mixer, and exports."""

from __future__ import annotations

import logging
import math
import re
import shutil
import tempfile
import time
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import (
    QCoreApplication,
    QEvent,
    QLineF,
    QPointF,
    QRectF,
    QSignalBlocker,
    Qt,
    QThread,
    QTimer,
    QUrl,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QAbstractSlider,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QStyleOptionSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.core.midi_editor import export_edited_midi
from src.core.muscriptor_result_assets import (
    MuscriptorPlaybackAssets,
    MuscriptorPreviewAssets,
    MuscriptorRollNote,
    prepare_midi_playback_assets,
    prepare_midi_preview_assets,
    read_midi_roll_notes,
)
from src.i18n.translator import get_translator, t
from src.models.data_models import MAX_MIDI_BPM, MIN_MIDI_BPM, ProcessingResult
from src.models.gm_instruments import get_instrument_name
from src.models.muscriptor_instruments import (
    MUSCRIPTOR_REPRESENTATIVE_PROGRAMS,
    muscriptor_instrument_label,
)

_INSTRUMENT_COLORS = (
    "#4a9eff",
    "#ff8d66",
    "#7bd88f",
    "#c89bff",
    "#ffd166",
    "#ff70a6",
    "#62d2c3",
    "#b3e35d",
)
_BLACK_PITCH_CLASSES = frozenset({1, 3, 6, 8, 10})
_STREAM_CHUNK_SECONDS = 5.0
_PROGRESS_EMA_ALPHA = 0.4
_PROGRESS_INTERPOLATION_CAP = 0.95
_PROGRESS_INITIAL_CURVE_SECONDS = 4.0
_SLOW_CONVERSION_THRESHOLD_SECONDS = 10.0 * 60.0
_STREAM_PREVIEW_REFRESH_SECONDS = 15.0
_STREAM_PREVIEW_PLAYBACK_MARGIN_SECONDS = 3.0
_PLAYHEAD_TIMER_MS = 16
_PLAYHEAD_MAX_LEAD_MS = 120.0
_ROLL_FOLLOW_SCROLL_BLOCK_PX = 12
_ROLL_BASE_PIXELS_PER_SECOND = 92.0
_ROLL_MIN_PIXELS_PER_SECOND = _ROLL_BASE_PIXELS_PER_SECOND * 0.5
_ROLL_MAX_PIXELS_PER_SECOND = _ROLL_BASE_PIXELS_PER_SECOND * 4.0
_ROLL_ZOOM_STEP = 1.15
_ROLL_WHEEL_STEP_PX = 96
_ROLL_TILE_WIDTH = 512
_ROLL_TILE_CACHE_LIMIT = 8

logger = logging.getLogger(__name__)


def _compact_editor_error(error: object, *, limit: int = 360) -> str:
    """Keep status labels bounded while the complete diagnostic stays in logs."""

    normalized = " ".join(str(error).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _export_midi_with_bpm(
    source_path: str | Path,
    destination_path: str | Path,
    bpm: float,
) -> Path:
    """Build the reference-speed preview while retaining musical ticks."""

    from src.core.midi_tempo import rewrite_midi_tempo_preserving_ticks

    return rewrite_midi_tempo_preserving_ticks(
        source_path,
        destination_path,
        bpm,
        label="Result reference-speed MIDI preview",
    )


def _export_midi_at_project_speed(
    source_path: str | Path,
    destination_path: str | Path,
    reference_bpm: float,
    target_bpm: float,
) -> Path:
    """Publish target tempo from an already normalized reference tick grid."""

    from src.core.midi_tempo import (
        rewrite_midi_tempo_preserving_ticks,
        validated_midi_bpm,
    )

    validated_midi_bpm(reference_bpm, "reference")
    return rewrite_midi_tempo_preserving_ticks(
        source_path,
        destination_path,
        target_bpm,
        label="Result project-speed MIDI export",
    )


def _instrument_label(instrument: str) -> str:
    language = get_translator().get_language()
    if instrument in MUSCRIPTOR_REPRESENTATIVE_PROGRAMS or instrument == "drums":
        return muscriptor_instrument_label(instrument, language)
    if instrument.startswith("gm:"):
        try:
            return get_instrument_name(int(instrument.split(":", 1)[1]), language)
        except ValueError:
            pass
    return instrument.replace("_", " ")


def _format_seconds(value: float) -> str:
    return f"{max(0.0, value):.1f}s"


def _format_clock(value: float) -> str:
    seconds = max(0, int(round(value)))
    return f"{seconds // 60}:{seconds % 60:02d}"


class _ChunkProgressEstimator:
    """Official MuScriptor-style chunk interpolation and ETA estimation."""

    def __init__(self, *, started_at: float | None = None):
        self.started_at = time.monotonic() if started_at is None else float(started_at)
        self.total = 0
        self.completed = 0
        self.ema_chunk_seconds: float | None = None
        self.last_anchor_at = self.started_at
        self.active = False

    def on_anchor(self, completed: int, total: int, *, now: float | None = None) -> None:
        anchor_at = time.monotonic() if now is None else float(now)
        if total <= 0 or completed < 0 or completed > total:
            raise ValueError(
                f"Invalid MuScriptor chunk progress: completed={completed}, total={total}"
            )
        if self.active and completed > self.completed:
            sample = (anchor_at - self.last_anchor_at) / (completed - self.completed)
            if sample < 0:
                raise ValueError(f"Invalid MuScriptor chunk duration sample: {sample}")
            if sample > 0:
                self.ema_chunk_seconds = (
                    sample
                    if self.ema_chunk_seconds is None
                    else _PROGRESS_EMA_ALPHA * sample
                    + (1.0 - _PROGRESS_EMA_ALPHA) * self.ema_chunk_seconds
                )
        self.total = total
        self.completed = completed
        self.last_anchor_at = anchor_at
        self.active = True

    def fraction(self, *, now: float | None = None) -> float:
        if not self.active or self.total == 0:
            return 0.0
        if self.completed >= self.total:
            return 1.0
        current = time.monotonic() if now is None else float(now)
        elapsed = max(0.0, current - self.last_anchor_at)
        if self.ema_chunk_seconds is not None and self.ema_chunk_seconds > 0:
            within_chunk = min(
                elapsed / self.ema_chunk_seconds,
                _PROGRESS_INTERPOLATION_CAP,
            )
        else:
            within_chunk = (
                1.0 - pow(2.718281828459045, -elapsed / _PROGRESS_INITIAL_CURVE_SECONDS)
            ) * _PROGRESS_INTERPOLATION_CAP
        return min(
            self.completed / self.total + within_chunk / self.total,
            0.999,
        )

    def eta_seconds(self, *, now: float | None = None) -> float | None:
        if not self.active or self.ema_chunk_seconds is None:
            return None
        if self.completed >= self.total:
            return 0.0
        current = time.monotonic() if now is None else float(now)
        return max(
            0.0,
            self.ema_chunk_seconds * (self.total - self.completed)
            - (current - self.last_anchor_at),
        )

    def elapsed_seconds(self, *, now: float | None = None) -> float:
        current = time.monotonic() if now is None else float(now)
        return max(0.0, current - self.started_at)


class _SmoothPlaybackClock:
    """Interpolate coarse media-clock samples without hiding a real stall."""

    def __init__(self, *, max_lead_ms: float = _PLAYHEAD_MAX_LEAD_MS):
        if max_lead_ms <= 0:
            raise ValueError(f"Playback-clock lead must be positive: {max_lead_ms}")
        self.max_lead_ms = float(max_lead_ms)
        self._reported_ms = 0.0
        self._display_ms = 0.0
        self._sampled_at = time.monotonic()

    def reset(self, position_ms: float, *, now: float | None = None) -> None:
        position = max(0.0, float(position_ms))
        self._reported_ms = position
        self._display_ms = position
        self._sampled_at = time.monotonic() if now is None else float(now)

    def sample(
        self,
        reported_ms: float,
        *,
        playback_rate: float = 1.0,
        now: float | None = None,
    ) -> float:
        current = time.monotonic() if now is None else float(now)
        reported = max(0.0, float(reported_ms))
        rate = float(playback_rate)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"Invalid smooth playback-clock rate: {playback_rate!r}")
        elapsed_ms = max(0.0, (current - self._sampled_at) * 1000.0 * rate)
        self._sampled_at = current
        if reported > self._reported_ms:
            self._reported_ms = reported
        # Advance from the last painted position instead of snapping to every
        # coarse QMediaPlayer sample. A bounded 1.5x correction closes genuine
        # forward drift smoothly while the lead ceiling still exposes a stalled
        # multimedia backend after ``max_lead_ms``.
        forward_drift = max(0.0, self._reported_ms - self._display_ms)
        catch_up = min(forward_drift, elapsed_ms * 0.5)
        projected = self._display_ms + elapsed_ms + catch_up
        ceiling = max(self._display_ms, self._reported_ms + self.max_lead_ms)
        self._display_ms = max(self._display_ms, min(projected, ceiling))
        return self._display_ms


class _SeekSlider(QSlider):
    """Horizontal transport slider with absolute click and continuous drag seeking."""

    seek_requested = pyqtSignal(int)
    scrub_started = pyqtSignal()
    scrub_finished = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._scrubbing = False

    def _value_for_x(self, x_position: float) -> int:
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        handle = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )
        slider_min = groove.left()
        slider_max = groove.right() - handle.width() + 1
        if slider_max <= slider_min:
            slider_min = 0
            slider_max = max(1, self.width() - handle.width())
        slider_position = round(float(x_position) - handle.width() / 2.0)
        return QStyle.sliderValueFromPosition(
            self.minimum(),
            self.maximum(),
            max(0, min(slider_max - slider_min, slider_position - slider_min)),
            max(1, slider_max - slider_min),
            option.upsideDown,
        )

    def _seek_to_pointer(self, x_position: float) -> int:
        value = self._value_for_x(x_position)
        self.setValue(value)
        self.seek_requested.emit(value)
        return value

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.isEnabled() and event.button() == Qt.MouseButton.LeftButton:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self._scrubbing = True
            self.setSliderDown(True)
            self.scrub_started.emit()
            self._seek_to_pointer(event.position().x())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._scrubbing and event.buttons() & Qt.MouseButton.LeftButton:
            self._seek_to_pointer(event.position().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._scrubbing and event.button() == Qt.MouseButton.LeftButton:
            value = self._seek_to_pointer(event.position().x())
            self._scrubbing = False
            self.setSliderDown(False)
            self.scrub_finished.emit(value)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _PianoRollCanvas(QWidget):
    seek_requested = pyqtSignal(float)
    edit_committed = pyqtSignal(object, object)
    selection_changed = pyqtSignal(object)
    add_note_requested = pyqtSignal(float, int)
    delete_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    command_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._notes: tuple[MuscriptorRollNote, ...] = ()
        self._notes_by_start: tuple[MuscriptorRollNote, ...] = ()
        self._note_starts: tuple[float, ...] = ()
        self._note_prefix_max_ends: tuple[float, ...] = ()
        self._duration = 10.0
        self._position = 0.0
        self._render_offset_px = 0.0
        self._muted: set[str] = set()
        self._colors: dict[str, QColor] = {}
        self._instrument_order: tuple[str, ...] = ()
        self._tile_cache: OrderedDict[tuple[int, int], QPixmap] = OrderedDict()
        self._pixels_per_second = _ROLL_BASE_PIXELS_PER_SECOND
        self._keyboard_width = 72
        self._row_height = 7
        self._editable = False
        self._selected_index: int | None = None
        self._selected_indices: set[int] = set()
        self._grid_seconds = 0.125
        self._beat_times: tuple[float, ...] = ()
        self._downbeat_times: tuple[float, ...] = ()
        self._drag_mode: str | None = None
        self._drag_origin_position = None
        self._drag_origin_note: MuscriptorRollNote | None = None
        self._drag_origin_notes: dict[int, MuscriptorRollNote] = {}
        self._drag_before: tuple[MuscriptorRollNote, ...] | None = None
        self._marquee_origin: QPointF | None = None
        self._marquee_current: QPointF | None = None
        self._marquee_base: set[int] = set()
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(88 * self._row_height)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._update_size()

    def set_notes(
        self,
        notes: Iterable[MuscriptorRollNote],
        *,
        duration: float | None = None,
        selected_index: int | None = None,
        selected_indices: Iterable[int] | None = None,
    ) -> None:
        normalized = tuple(notes)
        for note in normalized:
            if (
                not math.isfinite(note.start)
                or not math.isfinite(note.end)
                or note.start < 0
                or note.end <= note.start
            ):
                raise ValueError(
                    "Piano roll contains an invalid note interval: "
                    f"start={note.start}, end={note.end}"
                )
        notes_by_start = tuple(sorted(normalized, key=lambda note: note.start))
        note_starts = tuple(note.start for note in notes_by_start)
        prefix_max_ends: list[float] = []
        max_end = 0.0
        for note in notes_by_start:
            max_end = max(max_end, note.end)
            prefix_max_ends.append(max_end)
        instruments = list(self._instrument_order)
        for note in normalized:
            if note.instrument not in instruments:
                instruments.append(note.instrument)
        colors = {
            name: QColor(_INSTRUMENT_COLORS[index % len(_INSTRUMENT_COLORS)])
            for index, name in enumerate(instruments)
        }
        if duration is not None:
            playable_duration = float(duration)
            if not math.isfinite(playable_duration) or playable_duration <= 0:
                raise ValueError(
                    f"Piano-roll duration must be finite and positive: {playable_duration}"
                )
            if any(note.end > playable_duration + 1e-6 for note in normalized):
                raise ValueError("Piano roll contains notes beyond playable audio")
            roll_duration = max(1.0, playable_duration)
        elif normalized:
            roll_duration = max(1.0, max(note.end for note in normalized))
        else:
            roll_duration = self._duration

        # Commit only after every invariant has passed so a rejected async
        # payload cannot leave half-applied roll state behind.
        self._notes = normalized
        self._notes_by_start = notes_by_start
        self._note_starts = note_starts
        self._note_prefix_max_ends = tuple(prefix_max_ends)
        self._colors = colors
        self._duration = roll_duration
        if selected_indices is None:
            normalized_selection = (
                {int(selected_index)}
                if selected_index is not None and 0 <= int(selected_index) < len(normalized)
                else set()
            )
        else:
            normalized_selection = {
                int(index) for index in selected_indices if 0 <= int(index) < len(normalized)
            }
        primary = (
            int(selected_index)
            if selected_index is not None and int(selected_index) in normalized_selection
            else (min(normalized_selection) if normalized_selection else None)
        )
        self._selected_indices = normalized_selection
        self._selected_index = primary
        self._tile_cache.clear()
        self._update_size()
        self.update()

    @property
    def notes(self) -> tuple[MuscriptorRollNote, ...]:
        return self._notes

    @property
    def selected_index(self) -> int | None:
        return self._selected_index

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._selected_indices))

    @property
    def selected_note(self) -> MuscriptorRollNote | None:
        if self._selected_index is None:
            return None
        return self._notes[self._selected_index]

    def set_editable(self, editable: bool) -> None:
        self._editable = bool(editable)
        if not self._editable:
            self._drag_mode = None
            self._drag_origin_note = None
            self._drag_origin_notes.clear()
            self._drag_before = None
            self._set_selected_indices(())
        self.setCursor(Qt.CursorShape.CrossCursor if self._editable else Qt.CursorShape.ArrowCursor)

    def set_selected_index(self, index: int | None) -> None:
        self._set_selected_index(index)

    def _set_selected_index(self, index: int | None) -> None:
        self._set_selected_indices(() if index is None else (int(index),), primary=index)

    def set_selected_indices(
        self,
        indices: Iterable[int],
        *,
        primary: int | None = None,
    ) -> None:
        self._set_selected_indices(indices, primary=primary)

    def _set_selected_indices(
        self,
        indices: Iterable[int],
        *,
        primary: int | None = None,
    ) -> None:
        normalized = {int(index) for index in indices if 0 <= int(index) < len(self._notes)}
        normalized_primary = (
            int(primary)
            if primary is not None and int(primary) in normalized
            else (min(normalized) if normalized else None)
        )
        if normalized == self._selected_indices and normalized_primary == self._selected_index:
            return
        self._selected_indices = normalized
        self._selected_index = normalized_primary
        self.update()
        self.selection_changed.emit(self.selected_note)

    def set_instrument_order(self, instruments: Iterable[str]) -> None:
        order = tuple(dict.fromkeys(str(item) for item in instruments if str(item)))
        if order == self._instrument_order:
            return
        self._instrument_order = order
        combined = list(order)
        for note in self._notes:
            if note.instrument not in combined:
                combined.append(note.instrument)
        self._colors = {
            name: QColor(_INSTRUMENT_COLORS[index % len(_INSTRUMENT_COLORS)])
            for index, name in enumerate(combined)
        }
        self._tile_cache.clear()
        self.update()

    def set_grid_seconds(self, seconds: float) -> None:
        grid = float(seconds)
        if not math.isfinite(grid) or grid <= 0:
            raise ValueError(f"Piano-roll grid must be finite and positive: {seconds}")
        self._grid_seconds = grid

    def set_beat_grid(
        self,
        beat_times: Iterable[float] | None,
        downbeats: Iterable[float] | None = None,
        *,
        beats_per_bar: int | None = None,
    ) -> None:
        """Set the final audio-aligned Beat This grid used by the roll."""

        if beat_times is None:
            beats: tuple[float, ...] = ()
            bars: tuple[float, ...] = ()
        else:
            beats = tuple(float(value) for value in beat_times)
            if (
                len(beats) < 2
                or any(not math.isfinite(value) or value < 0.0 for value in beats)
                or any(right <= left for left, right in zip(beats, beats[1:]))
            ):
                raise ValueError(
                    "Piano-roll beat grid must contain at least two finite, "
                    "strictly increasing times"
                )
            bars = tuple(float(value) for value in (downbeats or ()))
            if any(not math.isfinite(value) or value < 0.0 for value in bars) or any(
                right <= left for left, right in zip(bars, bars[1:])
            ):
                raise ValueError("Piano-roll downbeats must be finite and strictly increasing")
            if len(bars) < 2 and beats_per_bar is not None:
                meter = int(beats_per_bar)
                if meter <= 0:
                    raise ValueError(f"Piano-roll beats per bar must be positive: {meter}")
                anchor_time = bars[0] if bars else beats[0]
                anchor_index = min(
                    range(len(beats)),
                    key=lambda index: abs(beats[index] - anchor_time),
                )
                bars = tuple(
                    beats[index] for index in range(anchor_index % meter, len(beats), meter)
                )
        if beats == self._beat_times and bars == self._downbeat_times:
            return
        self._beat_times = beats
        self._downbeat_times = bars
        self._tile_cache.clear()
        self.update()

    def snap_time(self, seconds: float) -> float:
        return round(max(0.0, float(seconds)) / self._grid_seconds) * self._grid_seconds

    def set_position(self, seconds: float) -> None:
        position = max(0.0, float(seconds))
        if abs(position - self._position) < 1e-9:
            return
        previous_x = self.x_for_time_float(self._position) - self._render_offset_px
        self._position = position
        current_x = self.x_for_time_float(self._position) - self._render_offset_px
        # Repaint fractional movement too. At zoomed-out scales one 16 ms frame
        # is less than one device pixel; suppressing those frames visibly halves
        # the playhead animation rate.
        dirty_left = max(0, math.floor(min(previous_x, current_x)) - 3)
        dirty_right = min(self.width(), math.ceil(max(previous_x, current_x)) + 3)
        self.update(dirty_left, 0, max(1, dirty_right - dirty_left + 1), self.height())

    @property
    def position(self) -> float:
        return self._position

    def set_instrument_muted(self, instrument: str, muted: bool) -> None:
        if muted:
            self._muted.add(instrument)
        else:
            self._muted.discard(instrument)
        self._tile_cache.clear()
        self.update()

    def x_for_time(self, seconds: float) -> int:
        return int(self.x_for_time_float(seconds))

    def x_for_time_float(self, seconds: float) -> float:
        return self._keyboard_width + max(0.0, seconds) * self._pixels_per_second

    def time_for_x(self, x_position: float) -> float:
        return max(0.0, (float(x_position) - self._keyboard_width) / self._pixels_per_second)

    @property
    def pixels_per_second(self) -> float:
        return self._pixels_per_second

    @property
    def render_offset_px(self) -> float:
        return self._render_offset_px

    def set_pixels_per_second(self, value: float) -> None:
        pixels_per_second = float(value)
        if pixels_per_second <= 0:
            raise ValueError(f"Piano-roll zoom must be positive: {pixels_per_second}")
        if abs(pixels_per_second - self._pixels_per_second) < 1e-6:
            return
        self._pixels_per_second = pixels_per_second
        self._tile_cache.clear()
        self._update_size()
        self.update()

    def set_render_offset(self, pixels: float) -> None:
        offset = float(pixels)
        if offset < 0:
            raise ValueError(f"Piano-roll render offset cannot be negative: {offset}")
        if abs(offset - self._render_offset_px) < 0.01:
            return
        self._render_offset_px = offset
        self.update()

    def _update_size(self) -> None:
        width = int(self._keyboard_width + max(10.0, self._duration) * self._pixels_per_second + 80)
        self.setFixedSize(width, 88 * self._row_height)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        exposed = event.rect()
        painter.fillRect(exposed, QColor("#0f1a2d"))
        logical_left = exposed.left() + self._render_offset_px
        logical_right = exposed.right() + self._render_offset_px
        painter.translate(-self._render_offset_px, 0.0)
        first_tile = max(0, int(logical_left) // _ROLL_TILE_WIDTH)
        last_tile = min(
            (self.width() - 1) // _ROLL_TILE_WIDTH,
            max(first_tile, int(logical_right) // _ROLL_TILE_WIDTH),
        )
        for tile_index in range(first_tile, last_tile + 1):
            painter.drawPixmap(
                tile_index * _ROLL_TILE_WIDTH,
                0,
                self._static_tile(tile_index),
            )
        next_tile = last_tile + 1
        if next_tile * _ROLL_TILE_WIDTH < self.width():
            self._static_tile(next_tile)

        for index in self.selected_indices:
            selected = self._notes[index]
            if not 21 <= selected.pitch <= 108:
                continue
            selected_x = self.x_for_time_float(selected.start)
            selected_width = max(
                2.0,
                self.x_for_time_float(selected.end) - selected_x,
            )
            selected_y = (108 - selected.pitch) * self._row_height + 1
            selection_pen = QPen(QColor("#ffffff" if index == self._selected_index else "#b7d9ff"))
            selection_pen.setWidthF(1.5)
            selection_pen.setCosmetic(True)
            painter.setPen(selection_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(
                QRectF(
                    selected_x,
                    selected_y,
                    selected_width,
                    max(2, self._row_height - 2),
                )
            )

        if self._marquee_origin is not None and self._marquee_current is not None:
            marquee = QRectF(self._marquee_origin, self._marquee_current).normalized()
            painter.setPen(QPen(QColor("#9fc9ff"), 1, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(74, 158, 255, 38))
            painter.drawRect(marquee)

        playhead_x = self.x_for_time_float(self._position)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        playhead_pen = QPen(QColor("#ffffff"))
        playhead_pen.setWidthF(1.5)
        playhead_pen.setCosmetic(True)
        painter.setPen(playhead_pen)
        painter.drawLine(QLineF(playhead_x, 0.0, playhead_x, float(self.height())))

    def _static_tile(self, tile_index: int) -> QPixmap:
        device_ratio = max(1.0, self.devicePixelRatioF())
        cache_key = (int(tile_index), round(device_ratio * 100))
        cached = self._tile_cache.get(cache_key)
        if cached is not None:
            self._tile_cache.move_to_end(cache_key)
            return cached

        logical_left = tile_index * _ROLL_TILE_WIDTH
        logical_width = min(_ROLL_TILE_WIDTH, self.width() - logical_left)
        if logical_width <= 0:
            raise ValueError(f"Invalid piano-roll tile index: {tile_index}")
        tile = QPixmap(
            max(1, round(logical_width * device_ratio)),
            max(1, round(self.height() * device_ratio)),
        )
        tile.setDevicePixelRatio(device_ratio)
        tile.fill(QColor("#0f1a2d"))
        tile_painter = QPainter(tile)
        tile_painter.translate(-logical_left, 0.0)
        self._paint_static_content(
            tile_painter,
            float(logical_left),
            float(logical_left + logical_width - 1),
        )
        tile_painter.end()
        self._tile_cache[cache_key] = tile
        self._tile_cache.move_to_end(cache_key)
        while len(self._tile_cache) > _ROLL_TILE_CACHE_LIMIT:
            self._tile_cache.popitem(last=False)
        return tile

    def _paint_static_content(
        self,
        painter: QPainter,
        logical_left: float,
        logical_right: float,
    ) -> None:
        roll_left = self._keyboard_width
        for pitch in range(21, 109):
            row = 108 - pitch
            y = row * self._row_height
            is_black = pitch % 12 in _BLACK_PITCH_CLASSES
            piano_color = QColor("#23282e") if is_black else QColor("#e4e8eb")
            roll_color = QColor("#13213a") if is_black else QColor("#172842")
            if logical_left < roll_left:
                painter.fillRect(0, y, roll_left, self._row_height, piano_color)
            roll_fill_left = max(float(roll_left), logical_left)
            if roll_fill_left <= logical_right:
                painter.fillRect(
                    QRectF(
                        roll_fill_left,
                        y,
                        logical_right - roll_fill_left + 1.0,
                        self._row_height,
                    ),
                    roll_color,
                )
            painter.setPen(QPen(QColor("#2b3d5c"), 1))
            painter.drawLine(QLineF(logical_left, float(y), logical_right, float(y)))
            if pitch % 12 == 0 and logical_left < roll_left:
                painter.setPen(QColor("#22282e") if not is_black else QColor("#d2d7dc"))
                painter.setFont(QFont("Consolas", 7))
                painter.drawText(3, y + self._row_height - 1, f"C{pitch // 12 - 1}")

        painter.setFont(QFont("Consolas", 8))
        if self._beat_times:
            visible_start = max(0.0, (logical_left - roll_left) / self._pixels_per_second)
            visible_end = max(
                visible_start,
                (logical_right - roll_left) / self._pixels_per_second,
            )
            first_bar = max(0, bisect_right(self._downbeat_times, visible_start) - 1)
            last_bar = min(
                len(self._downbeat_times) - 1,
                bisect_right(self._downbeat_times, visible_end),
            )
            for bar_index in range(first_bar, last_bar):
                if bar_index % 2 == 0:
                    continue
                left = self.x_for_time_float(self._downbeat_times[bar_index])
                right = self.x_for_time_float(self._downbeat_times[bar_index + 1])
                painter.fillRect(
                    QRectF(left, 0.0, max(0.0, right - left), float(self.height())),
                    QColor(255, 255, 255, 8),
                )

            first_beat = bisect_left(self._beat_times, visible_start)
            last_beat = bisect_right(self._beat_times, visible_end)
            median_spacing = sorted(
                right - left for left, right in zip(self._beat_times, self._beat_times[1:])
            )[len(self._beat_times) // 2 - 1]
            beat_stride = 1
            while median_spacing * self._pixels_per_second * beat_stride < 12.0:
                beat_stride *= 2
            for beat_index in range(first_beat, last_beat):
                if beat_index % beat_stride:
                    continue
                x = self.x_for_time_float(self._beat_times[beat_index])
                painter.setPen(QPen(QColor("#36506f"), 1))
                painter.drawLine(QLineF(x, 0.0, x, float(self.height())))

            first_downbeat = bisect_left(self._downbeat_times, visible_start)
            last_downbeat = bisect_right(self._downbeat_times, visible_end)
            for bar_index in range(first_downbeat, last_downbeat):
                x = self.x_for_time_float(self._downbeat_times[bar_index])
                downbeat_pen = QPen(QColor("#78aee8"))
                downbeat_pen.setWidthF(1.5)
                downbeat_pen.setCosmetic(True)
                painter.setPen(downbeat_pen)
                painter.drawLine(QLineF(x, 0.0, x, float(self.height())))
                painter.setPen(QColor("#a9c8e8"))
                painter.drawText(QPointF(x + 3.0, 11.0), f"B{bar_index + 1}")
        else:
            max_second = int(self._duration) + 2
            first_second = max(
                0,
                int((logical_left - roll_left) / self._pixels_per_second) - 1,
            )
            last_second = min(
                max_second,
                int((logical_right - roll_left) / self._pixels_per_second) + 2,
            )
            for second in range(first_second, last_second + 1):
                x = self.x_for_time(second)
                painter.setPen(QPen(QColor("#36506f"), 1))
                painter.drawLine(x, 0, x, self.height())
                painter.setPen(QColor("#7f94b7"))
                painter.drawText(x + 3, 11, f"{second}s")

        visible_start = max(0.0, (logical_left - roll_left) / self._pixels_per_second)
        visible_end = max(visible_start, (logical_right - roll_left) / self._pixels_per_second)
        first_note = bisect_left(self._note_prefix_max_ends, visible_start)
        last_note = bisect_right(self._note_starts, visible_end)
        for note in self._notes_by_start[first_note:last_note]:
            if not 21 <= note.pitch <= 108:
                continue
            x = self.x_for_time_float(note.start)
            width = max(2.0, self.x_for_time_float(note.end) - x)
            y = (108 - note.pitch) * self._row_height + 1
            color = QColor(self._colors.get(note.instrument, QColor("#4a9eff")))
            if note.instrument in self._muted:
                color.setAlpha(35)
            painter.fillRect(
                QRectF(x, y, width, max(2, self._row_height - 2)),
                color,
            )

    def _note_index_at(self, logical_x: float, y_position: float) -> int | None:
        pitch = 108 - int(float(y_position) // self._row_height)
        for index in range(len(self._notes) - 1, -1, -1):
            note = self._notes[index]
            if note.pitch != pitch:
                continue
            note_left = self.x_for_time_float(note.start)
            note_right = self.x_for_time_float(note.end)
            if note_left - 2 <= logical_x <= max(note_left + 2, note_right) + 2:
                return index
        return None

    def _pitch_for_y(self, y_position: float) -> int:
        return max(21, min(108, 108 - int(float(y_position) // self._row_height)))

    def _replace_selected_note(self, note: MuscriptorRollNote) -> None:
        if self._selected_index is None:
            return
        updated = list(self._notes)
        updated[self._selected_index] = note
        self.set_notes(
            updated,
            duration=self._duration,
            selected_index=self._selected_index,
            selected_indices=self._selected_indices,
        )
        self.selection_changed.emit(note)

    def _replace_selected_notes(
        self,
        replacements: dict[int, MuscriptorRollNote],
    ) -> None:
        if not replacements:
            return
        updated = list(self._notes)
        for index, note in replacements.items():
            updated[index] = note
        self.set_notes(
            updated,
            duration=self._duration,
            selected_index=self._selected_index,
            selected_indices=self._selected_indices,
        )
        self.selection_changed.emit(self.selected_note)

    def _note_rect(self, note: MuscriptorRollNote) -> QRectF:
        x = self.x_for_time_float(note.start)
        return QRectF(
            x,
            (108 - note.pitch) * self._row_height + 1,
            max(2.0, self.x_for_time_float(note.end) - x),
            max(2, self._row_height - 2),
        )

    def _update_marquee_selection(self) -> None:
        if self._marquee_origin is None or self._marquee_current is None:
            return
        marquee = QRectF(self._marquee_origin, self._marquee_current).normalized()
        selected = set(self._marquee_base)
        selected.update(
            index
            for index, note in enumerate(self._notes)
            if self._note_rect(note).intersects(marquee)
        )
        self._set_selected_indices(selected)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.position().x() + self._render_offset_px >= self._keyboard_width
        ):
            logical_x = event.position().x() + self._render_offset_px
            if self._editable:
                self.setFocus(Qt.FocusReason.MouseFocusReason)
                note_index = self._note_index_at(logical_x, event.position().y())
                if note_index is not None:
                    additive = bool(
                        event.modifiers()
                        & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
                    )
                    if additive:
                        selected = set(self._selected_indices)
                        if note_index in selected:
                            selected.remove(note_index)
                            self._set_selected_indices(selected)
                            event.accept()
                            return
                        selected.add(note_index)
                        self._set_selected_indices(selected, primary=note_index)
                    elif note_index not in self._selected_indices:
                        self._set_selected_index(note_index)
                    else:
                        self._set_selected_indices(
                            self._selected_indices,
                            primary=note_index,
                        )
                    note = self._notes[note_index]
                    note_left = self.x_for_time_float(note.start)
                    note_right = self.x_for_time_float(note.end)
                    edge_width = min(6.0, max(3.0, (note_right - note_left) / 3.0))
                    if abs(logical_x - note_left) <= edge_width:
                        self._drag_mode = "resize_start"
                    elif abs(logical_x - note_right) <= edge_width:
                        self._drag_mode = "resize_end"
                    else:
                        self._drag_mode = "move"
                    self._drag_origin_position = event.position()
                    self._drag_origin_note = note
                    self._drag_origin_notes = {
                        index: self._notes[index] for index in self._selected_indices
                    }
                    self._drag_before = self._notes
                    event.accept()
                    return
                additive = bool(
                    event.modifiers()
                    & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
                )
                if not additive:
                    self._set_selected_indices(())
                self._drag_mode = "marquee"
                self._drag_origin_position = event.position()
                self._marquee_origin = QPointF(logical_x, event.position().y())
                self._marquee_current = QPointF(self._marquee_origin)
                self._marquee_base = set(self._selected_indices)
                event.accept()
                return
            seconds = self.time_for_x(logical_x)
            self.seek_requested.emit(max(0.0, seconds))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if (
            self._editable
            and self._drag_mode == "marquee"
            and self._marquee_origin is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self._marquee_current = QPointF(
                event.position().x() + self._render_offset_px,
                event.position().y(),
            )
            self._update_marquee_selection()
            self.update()
            event.accept()
            return
        if (
            self._editable
            and self._drag_mode is not None
            and self._drag_origin_position is not None
            and self._drag_origin_note is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            origin = self._drag_origin_note
            delta_seconds = (
                float(event.position().x() - self._drag_origin_position.x())
                / self._pixels_per_second
            )
            snap = not bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            if self._drag_mode == "move":
                start = origin.start + delta_seconds
                if snap:
                    start = self.snap_time(start)
                delta_seconds = start - origin.start
                delta_seconds = max(
                    -min(note.start for note in self._drag_origin_notes.values()),
                    min(
                        self._duration - max(note.end for note in self._drag_origin_notes.values()),
                        delta_seconds,
                    ),
                )
                pitch_delta = round(
                    (self._drag_origin_position.y() - event.position().y()) / self._row_height
                )
                pitch_delta = max(
                    21 - min(note.pitch for note in self._drag_origin_notes.values()),
                    min(
                        108 - max(note.pitch for note in self._drag_origin_notes.values()),
                        pitch_delta,
                    ),
                )
                replacements = {
                    index: replace(
                        note,
                        start=note.start + delta_seconds,
                        end=note.end + delta_seconds,
                        pitch=note.pitch + pitch_delta,
                    )
                    for index, note in self._drag_origin_notes.items()
                }
            elif self._drag_mode == "resize_start":
                start = origin.start + delta_seconds
                if snap:
                    start = self.snap_time(start)
                delta_seconds = start - origin.start
                delta_seconds = max(
                    -min(note.start for note in self._drag_origin_notes.values()),
                    min(
                        min(
                            note.end - note.start - 0.01
                            for note in self._drag_origin_notes.values()
                        ),
                        delta_seconds,
                    ),
                )
                replacements = {
                    index: replace(note, start=note.start + delta_seconds)
                    for index, note in self._drag_origin_notes.items()
                }
            else:
                end = origin.end + delta_seconds
                if snap:
                    end = self.snap_time(end)
                delta_seconds = end - origin.end
                delta_seconds = max(
                    -min(note.end - note.start - 0.01 for note in self._drag_origin_notes.values()),
                    min(
                        self._duration - max(note.end for note in self._drag_origin_notes.values()),
                        delta_seconds,
                    ),
                )
                replacements = {
                    index: replace(note, end=note.end + delta_seconds)
                    for index, note in self._drag_origin_notes.items()
                }
            self._replace_selected_notes(replacements)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_mode is not None and event.button() == Qt.MouseButton.LeftButton:
            before = self._drag_before
            after = self._notes
            self._drag_mode = None
            self._drag_origin_position = None
            self._drag_origin_note = None
            self._drag_origin_notes.clear()
            self._drag_before = None
            self._marquee_origin = None
            self._marquee_current = None
            self._marquee_base.clear()
            # A click on empty piano-roll space is a selection gesture while
            # edit mode is enabled.  Treating the same release as a transport
            # seek made simple note-selection/deselection enter QMediaPlayer's
            # synchronous seek path while audio was running.
            if before is not None and before != after:
                self.edit_committed.emit(before, after)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if (
            self._editable
            and event.button() == Qt.MouseButton.LeftButton
            and event.position().x() + self._render_offset_px >= self._keyboard_width
        ):
            logical_x = event.position().x() + self._render_offset_px
            if self._note_index_at(logical_x, event.position().y()) is None:
                self.add_note_requested.emit(
                    min(self._duration, self.snap_time(self.time_for_x(logical_x))),
                    self._pitch_for_y(event.position().y()),
                )
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self._editable and event.matches(QKeySequence.StandardKey.Undo):
            self.undo_requested.emit()
            event.accept()
            return
        if self._editable and event.matches(QKeySequence.StandardKey.Redo):
            self.redo_requested.emit()
            event.accept()
            return
        if self._editable:
            command = bool(
                event.modifiers()
                & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
            )
            key = event.key()
            if command and key == Qt.Key.Key_A:
                self.command_requested.emit("select_all")
            elif command and key == Qt.Key.Key_X:
                self.command_requested.emit("cut")
            elif command and key == Qt.Key.Key_C:
                self.command_requested.emit("copy")
            elif command and key == Qt.Key.Key_V:
                self.command_requested.emit("paste")
            elif command and key in {Qt.Key.Key_B, Qt.Key.Key_D}:
                self.command_requested.emit("duplicate")
            elif (key == Qt.Key.Key_Q and event.modifiers() & Qt.KeyboardModifier.AltModifier) or (
                command and key == Qt.Key.Key_U
            ):
                self.command_requested.emit("quantize")
            elif key in {
                Qt.Key.Key_Left,
                Qt.Key.Key_Right,
                Qt.Key.Key_Up,
                Qt.Key.Key_Down,
            }:
                if key in {Qt.Key.Key_Left, Qt.Key.Key_Right}:
                    direction = -1 if key == Qt.Key.Key_Left else 1
                    action = (
                        "resize_time"
                        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                        else "move_time"
                    )
                    self.command_requested.emit(f"{action}:{direction}")
                elif command:
                    direction = 1 if key == Qt.Key.Key_Up else -1
                    self.command_requested.emit(f"velocity:{direction}")
                else:
                    direction = 1 if key == Qt.Key.Key_Up else -1
                    semitones = 12 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
                    self.command_requested.emit(f"pitch:{direction * semitones}")
            else:
                super().keyPressEvent(event)
                return
            event.accept()
            return
        if self._editable and event.key() in {
            Qt.Key.Key_Delete,
            Qt.Key.Key_Backspace,
        }:
            self.delete_requested.emit()
            event.accept()
            return
        if self._editable and event.key() == Qt.Key.Key_Escape:
            self._set_selected_indices(())
            event.accept()
            return
        super().keyPressEvent(event)


class _PianoRollScrollArea(QScrollArea):
    """Piano-roll navigation with explicit horizontal-scroll and zoom gestures."""

    zoom_requested = pyqtSignal(float, float)
    manual_navigation_requested = pyqtSignal()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        modifiers = event.modifiers()
        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()
        vertical_delta = pixel_delta.y() or angle_delta.y()

        zoom_modifiers = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
        if modifiers & zoom_modifiers and vertical_delta:
            if pixel_delta.y():
                steps = pixel_delta.y() / 120.0
            else:
                steps = angle_delta.y() / 120.0
            self.zoom_requested.emit(
                _ROLL_ZOOM_STEP**steps,
                float(event.position().x()),
            )
            event.accept()
            return

        if modifiers & Qt.KeyboardModifier.ShiftModifier and vertical_delta:
            self.manual_navigation_requested.emit()
            if pixel_delta.y():
                distance = pixel_delta.y()
            else:
                distance = angle_delta.y() / 120.0 * _ROLL_WHEEL_STEP_PX
            scrollbar = self.horizontalScrollBar()
            scrollbar.setValue(round(scrollbar.value() - distance))
            event.accept()
            return

        super().wheelEvent(event)


class _AssetWorker(QThread):
    progress = pyqtSignal(float, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        midi_path: str,
        audio_path: str,
        output_dir: str,
        muscriptor_groups: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.midi_path = midi_path
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.muscriptor_groups = bool(muscriptor_groups)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            assets = prepare_midi_playback_assets(
                self.midi_path,
                self.audio_path,
                self.output_dir,
                progress_callback=self.progress.emit,
                cancel_check=lambda: self._cancelled,
                muscriptor_groups=self.muscriptor_groups,
            )
            self.succeeded.emit(assets)
        except Exception as exc:
            self.failed.emit(str(exc))


class _PreviewAssetWorker(QThread):
    succeeded = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        generation: int,
        notes: tuple[MuscriptorRollNote, ...],
        playable_duration: float,
        audio_path: str,
        output_dir: str,
        parent=None,
    ):
        super().__init__(parent)
        self.generation = generation
        self.notes = notes
        self.playable_duration = playable_duration
        self.audio_path = audio_path
        self.output_dir = output_dir
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            assets = prepare_midi_preview_assets(
                self.notes,
                self.playable_duration,
                self.output_dir,
                reference_audio_path=self.audio_path,
                cancel_check=lambda: self._cancelled,
            )
            self.succeeded.emit(self.generation, assets)
        except Exception as exc:
            self.failed.emit(self.generation, str(exc))


class _EditedAssetWorker(QThread):
    """Publish and render one immutable editor snapshot at source tempo."""

    progress = pyqtSignal(int, float, str)
    succeeded = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        generation: int,
        source_midi_path: str,
        notes: tuple[MuscriptorRollNote, ...],
        reference_bpm: float,
        audio_path: str,
        output_dir: str,
        muscriptor_groups: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.generation = int(generation)
        self.source_midi_path = str(source_midi_path)
        self.notes = tuple(notes)
        self.reference_bpm = float(reference_bpm)
        self.audio_path = str(audio_path)
        self.output_dir = str(output_dir)
        self.muscriptor_groups = bool(muscriptor_groups)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            output_dir = Path(self.output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            edited_midi = export_edited_midi(
                self.source_midi_path,
                output_dir / "edited-source-tempo.mid",
                self.notes,
                reference_bpm=self.reference_bpm,
                target_bpm=self.reference_bpm,
            )
            if self._cancelled:
                raise InterruptedError("Edited MIDI playback rendering cancelled")
            assets = prepare_midi_playback_assets(
                edited_midi,
                self.audio_path,
                output_dir,
                progress_callback=lambda value, message: self.progress.emit(
                    self.generation,
                    value,
                    message,
                ),
                cancel_check=lambda: self._cancelled,
                muscriptor_groups=self.muscriptor_groups,
                allow_empty_notes=True,
            )
            self.succeeded.emit(self.generation, assets)
        except Exception as exc:
            self.failed.emit(self.generation, str(exc))


class _InstrumentRow(QFrame):
    mute_toggled = pyqtSignal(str)
    solo_toggled = pyqtSignal(str)

    def __init__(self, instrument: str, *, detected: bool, color: str, parent=None):
        super().__init__(parent)
        self.instrument = instrument
        self.detected = detected
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 4, 5, 4)
        layout.setSpacing(6)
        swatch = QLabel("■")
        swatch.setStyleSheet(f"color: {color if detected else '#4b5157'};")
        self.name_label = QLabel()
        layout.addWidget(swatch)
        layout.addWidget(self.name_label, 1)
        self.not_detected_label = QLabel()
        self.not_detected_label.setStyleSheet("color: #626b73; font-style: italic;")
        self.not_detected_label.setVisible(not detected)
        layout.addWidget(self.not_detected_label)
        self.solo_button = QPushButton("S")
        self.solo_button.setCheckable(True)
        self.solo_button.setFixedWidth(28)
        self.solo_button.setEnabled(detected)
        self.solo_button.clicked.connect(lambda: self.solo_toggled.emit(self.instrument))
        layout.addWidget(self.solo_button)
        self.mute_button = QPushButton()
        self.mute_button.setCheckable(True)
        self.mute_button.setFixedWidth(34)
        self.mute_button.setEnabled(detected)
        self.mute_button.clicked.connect(lambda: self.mute_toggled.emit(self.instrument))
        layout.addWidget(self.mute_button)
        self.set_muted(False)
        self.update_translations()

    def set_muted(self, muted: bool) -> None:
        self.mute_button.setChecked(muted)
        standard_icon = (
            QStyle.StandardPixmap.SP_MediaVolumeMuted
            if muted
            else QStyle.StandardPixmap.SP_MediaVolume
        )
        self.mute_button.setIcon(self.style().standardIcon(standard_icon))
        self.mute_button.setText("")
        self.name_label.setStyleSheet(
            "color: #626b73; text-decoration: line-through;"
            if not self.detected
            else ("color: #6c737a;" if muted else "color: #e5e8ea;")
        )

    def set_soloed(self, soloed: bool) -> None:
        self.solo_button.setChecked(soloed)

    def update_translations(self) -> None:
        self.name_label.setText(_instrument_label(self.instrument))
        self.not_detected_label.setText(t("muscriptor_result.not_detected"))
        self.solo_button.setToolTip(t("muscriptor_result.solo"))
        self.mute_button.setToolTip(t("muscriptor_result.mute"))


class MuscriptorResultWidget(QFrame):
    """One inline result surface whose controls all affect real playback."""

    transcribe_another_requested = pyqtSignal()
    playing_changed = pyqtSignal(bool)

    def __init__(
        self,
        audio_path: str,
        selected_instruments: list[str],
        parent=None,
        *,
        backend_label: str = "MuScriptor-large",
        muscriptor_groups: bool = True,
        source_track_name: str | None = None,
    ):
        super().__init__(parent)
        self.audio_path = str(Path(audio_path).resolve())
        self.selected_instruments = list(selected_instruments)
        self.backend_label = str(backend_label)
        self.muscriptor_groups = bool(muscriptor_groups)
        self.source_track_name = (
            str(source_track_name).strip() if source_track_name is not None else ""
        )
        self._detected: list[str] = []
        self._stream_notes: list[MuscriptorRollNote] = []
        self._assets: MuscriptorPlaybackAssets | None = None
        self._original_assets: MuscriptorPlaybackAssets | None = None
        self._asset_worker: _AssetWorker | None = None
        self._preview_worker: _PreviewAssetWorker | None = None
        self._edit_asset_worker: _EditedAssetWorker | None = None
        self._deferred_preview: tuple[int, MuscriptorPreviewAssets] | None = None
        self._deferred_final_assets: MuscriptorPlaybackAssets | None = None
        self._deferred_editor_assets: tuple[int, MuscriptorPlaybackAssets] | None = None
        self._deferred_apply_scheduled = False
        self._preview_pending: (
            tuple[int, tuple[MuscriptorRollNote, ...], float, int, int] | None
        ) = None
        self._preview_root = Path(tempfile.mkdtemp(prefix="music-to-midi-midi-preview-"))
        self._edit_asset_root = Path(tempfile.mkdtemp(prefix="music-to-midi-midi-editor-audio-"))
        self._edit_asset_generation = 0
        self._edit_asset_applied_generation = 0
        self._edit_asset_pending: tuple[int, tuple[MuscriptorRollNote, ...]] | None = None
        self._active_edit_asset_dir: Path | None = None
        self._edit_asset_debounce = QTimer(self)
        self._edit_asset_debounce.setSingleShot(True)
        self._edit_asset_debounce.setInterval(180)
        self._edit_asset_debounce.timeout.connect(self._start_pending_editor_audio)
        self._preview_generation = 0
        self._preview_ready_generation = 0
        self._preview_applied_generation = 0
        self._preview_last_requested_frontier = 0.0
        self._preview_note_count = 0
        self._preview_duration = 0.0
        self._preview_error: str | None = None
        self._finalizing = False
        self._shutting_down = False
        self._progress_estimator = _ChunkProgressEstimator()
        self._progress_completed = 0
        self._progress_total = 0
        self._runtime_details: dict[str, object] = {}
        self._source_duration_seconds = 0.0
        self._position_ms = 0
        self._last_drift_check_position_ms = 0
        self._playback_clock = _SmoothPlaybackClock()
        self._playing = False
        self._playback_finished = False
        self._transport_scrubbing = False
        self._transport_seek_pending_ms: int | None = None
        self._transport_seek_resume = False
        self._transport_seek_pause_only = False
        self._transport_seek_commit_scheduled = False
        self._transport_seek_commit_players: tuple[QMediaPlayer, ...] = ()
        self._transport_seek_commit_position_ms = 0
        self._transport_seek_commit_pause_only = False
        self._transport_seek_commit_index = 0
        self._transport_seek_commit_phase = ""
        self._transport_commit_timer = QTimer(self)
        self._transport_commit_timer.setSingleShot(True)
        self._transport_commit_timer.timeout.connect(self._commit_next_transport_player)
        self._after_transport_timer = QTimer(self)
        self._after_transport_timer.setSingleShot(True)
        self._after_transport_timer.timeout.connect(self._apply_deferred_after_transport)
        self._deferred_assets_timer = QTimer(self)
        self._deferred_assets_timer.setSingleShot(True)
        self._deferred_assets_timer.timeout.connect(self._try_apply_deferred_assets)
        self._detected_bpm: float | None = None
        self._bpm_user_overridden = False
        self._last_tempo_editor: str | None = None
        self._muted: set[str] = set()
        self._soloed: str | None = None
        self._instrument_rows: dict[str, _InstrumentRow] = {}
        self._players: list[QMediaPlayer] = []
        self._normal_sources: dict[str, Path] = {}
        self._right_sources: dict[str, Path] = {}
        self._normal_players: dict[str, tuple[QMediaPlayer, QAudioOutput]] = {}
        self._right_players: dict[str, tuple[QMediaPlayer, QAudioOutput]] = {}
        self._midi_normal: tuple[QMediaPlayer, QAudioOutput] | None = None
        self._midi_right: tuple[QMediaPlayer, QAudioOutput] | None = None
        self._original_normal: tuple[QMediaPlayer, QAudioOutput] | None = None
        self._original_left: tuple[QMediaPlayer, QAudioOutput] | None = None
        self._active_player_ids: frozenset[int] = frozenset()
        self._startup_sync_pending = False
        self._retired_media: list[tuple[QMediaPlayer, QAudioOutput | None]] = []
        self._retired_media_timer = QTimer(self)
        self._retired_media_timer.setSingleShot(True)
        self._retired_media_timer.timeout.connect(self._release_retired_media)
        self._midi_path = ""
        self._original_edit_notes: tuple[MuscriptorRollNote, ...] = ()
        self._edited_notes: tuple[MuscriptorRollNote, ...] = ()
        self._edit_duration = 0.0
        self._edit_undo: list[tuple[MuscriptorRollNote, ...]] = []
        self._edit_redo: list[tuple[MuscriptorRollNote, ...]] = []
        self._edit_clipboard: tuple[MuscriptorRollNote, ...] = ()
        self._active_edit_instrument = ""
        self._syncing_editor_controls = False
        self._syncing_roll_zoom_control = False
        self.setObjectName("muscriptorResultWorkbench")
        self.setStyleSheet(
            "QFrame#muscriptorResultWorkbench { background: #17243d; border: 1px solid #2c4f7c; "
            "border-radius: 8px; } "
            "QFrame#muscriptorResultWorkbench QPushButton, "
            "QFrame#muscriptorResultWorkbench QToolButton { background: #2a3f5f; color: #e0e0e0; "
            "border: 1px solid #3a4a6a; border-radius: 5px; padding: 5px 9px; } "
            "QFrame#muscriptorResultWorkbench QPushButton:hover, "
            "QFrame#muscriptorResultWorkbench QToolButton:hover { background: #3a5a7c; "
            "border-color: #4a9eff; } "
            "QFrame#muscriptorResultWorkbench QPushButton:checked { background: #203f68; "
            "border-color: #4a9eff; color: #8fc6ff; } "
            "QFrame#muscriptorResultWorkbench QPushButton:disabled, "
            "QFrame#muscriptorResultWorkbench QToolButton:disabled { color: #65738a; "
            "background: #202e49; border-color: #33425d; } "
            "QFrame#muscriptorResultWorkbench QCheckBox { color: #c8d3e6; spacing: 5px; } "
            "QFrame#muscriptorResultWorkbench QSlider::groove:horizontal { background: #283a59; "
            "height: 5px; border-radius: 2px; } "
            "QFrame#muscriptorResultWorkbench QSlider::sub-page:horizontal { background: #4a9eff; "
            "border-radius: 2px; } "
            "QFrame#muscriptorResultWorkbench QSlider::handle:horizontal { background: #dbeaff; "
            "border: 1px solid #4a9eff; width: 13px; margin: -5px 0; border-radius: 6px; } "
            "QFrame#muscriptorResultWorkbench QScrollArea { background: #0f1a2d; "
            "border: 1px solid #365f8d; } "
            "QFrame#muscriptorResultWorkbench QScrollBar:horizontal { background: #101b2d; "
            "height: 15px; margin: 2px; border: 0; border-radius: 6px; } "
            "QFrame#muscriptorResultWorkbench QScrollBar::handle:horizontal { "
            "background: #3d628e; min-width: 52px; margin: 1px; border-radius: 5px; } "
            "QFrame#muscriptorResultWorkbench QScrollBar::handle:horizontal:hover { "
            "background: #4a9eff; } "
            "QFrame#muscriptorResultWorkbench QScrollBar:vertical { background: #101b2d; "
            "width: 15px; margin: 2px; border: 0; border-radius: 6px; } "
            "QFrame#muscriptorResultWorkbench QScrollBar::handle:vertical { "
            "background: #3d628e; min-height: 42px; margin: 1px; border-radius: 5px; } "
            "QFrame#muscriptorResultWorkbench QScrollBar::handle:vertical:hover { "
            "background: #4a9eff; } "
            "QFrame#muscriptorResultWorkbench QScrollBar::add-line, "
            "QFrame#muscriptorResultWorkbench QScrollBar::sub-line { width: 0; height: 0; } "
            "QFrame#muscriptorResultWorkbench QScrollBar::add-page, "
            "QFrame#muscriptorResultWorkbench QScrollBar::sub-page { background: transparent; } "
            "QFrame#muscriptorResultWorkbench QMenu { background: #1f2940; color: #e0e0e0; "
            "border: 1px solid #3a4a6a; } "
            "QFrame#muscriptorResultWorkbench QMenu::item:selected { background: #3a5a7c; }"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.result_controls_panel = QWidget()
        result_controls = QVBoxLayout(self.result_controls_panel)
        result_controls.setContentsMargins(0, 0, 0, 0)
        result_controls.setSpacing(8)
        self.result_controls_panel.setMinimumHeight(0)
        self.result_controls_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        self.source_label.setVisible(bool(self.source_track_name))
        self.source_label.setStyleSheet(
            "color: #8fc6ff; font-weight: 600; background: #122039; "
            "border: 1px solid #2c4f7c; border-radius: 5px; padding: 7px 9px;"
        )
        result_controls.addWidget(self.source_label)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #9fb3d9;")
        result_controls.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background: #223451; border: 0; border-radius: 4px; } "
            "QProgressBar::chunk { background: #4a9eff; border-radius: 4px; }"
        )
        result_controls.addWidget(self.progress_bar)

        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("color: #8da4c9;")
        result_controls.addWidget(self.progress_label)

        self.slow_hint_label = QLabel()
        self.slow_hint_label.setWordWrap(True)
        self.slow_hint_label.setStyleSheet(
            "font-size: 10px; color: #d8b56a; background: #1b2942; "
            "border-left: 3px solid #d8b56a; padding: 5px 8px;"
        )
        self.slow_hint_label.hide()
        result_controls.addWidget(self.slow_hint_label)

        self.playback_status_label = QLabel()
        self.playback_status_label.setWordWrap(True)
        self.playback_status_label.setStyleSheet("color: #73a7ff;")
        result_controls.addWidget(self.playback_status_label)

        controls = QHBoxLayout()
        self.play_button = QPushButton()
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._toggle_playback)
        controls.addWidget(self.play_button)
        self.follow_checkbox = QCheckBox()
        self.follow_checkbox.setChecked(True)
        controls.addWidget(self.follow_checkbox)
        self.clock_label = QLabel("0.0s")
        self.clock_label.setStyleSheet(
            "font-family: Consolas; color: #c8d3e6; background: #16213e; "
            "border: 1px solid #3a4a6a; border-radius: 4px; padding: 4px 7px;"
        )
        controls.addWidget(self.clock_label)
        self.bpm_spin = QDoubleSpinBox()
        self.bpm_spin.setRange(MIN_MIDI_BPM, MAX_MIDI_BPM)
        self.bpm_spin.setDecimals(1)
        self.bpm_spin.setSingleStep(0.1)
        self.bpm_spin.setPrefix("BPM ")
        self.bpm_spin.setKeyboardTracking(False)
        self.bpm_spin.setStyleSheet(
            "font-family: Consolas; color: #c8d3e6; background: #16213e; "
            "border: 1px solid #3a4a6a; border-radius: 4px; padding: 4px 7px;"
        )
        self.bpm_spin.lineEdit().textEdited.connect(
            lambda _text: setattr(self, "_last_tempo_editor", "bpm")
        )
        self.bpm_spin.valueChanged.connect(self._on_result_bpm_changed)
        self.bpm_spin.hide()
        controls.addWidget(self.bpm_spin)
        self.speed_label = QLabel()
        self.speed_label.hide()
        controls.addWidget(self.speed_label)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.05, 20.0)
        self.speed_spin.setDecimals(3)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSuffix("x")
        self.speed_spin.setKeyboardTracking(False)
        self.speed_spin.setMinimumWidth(88)
        self.speed_spin.setStyleSheet(
            "font-family: Consolas; color: #c8d3e6; background: #16213e; "
            "border: 1px solid #3a4a6a; border-radius: 4px; padding: 4px 7px;"
        )
        self.speed_spin.lineEdit().textEdited.connect(
            lambda _text: setattr(self, "_last_tempo_editor", "speed")
        )
        self.speed_spin.valueChanged.connect(self._on_result_speed_changed)
        self.speed_spin.hide()
        controls.addWidget(self.speed_spin)
        controls.addStretch(1)
        self.original_label = QLabel()
        controls.addWidget(self.original_label)
        self.mix_slider = QSlider(Qt.Orientation.Horizontal)
        self.mix_slider.setRange(0, 100)
        self.mix_slider.setValue(75)
        self.mix_slider.setFixedWidth(150)
        self.mix_slider.setEnabled(False)
        self.mix_slider.valueChanged.connect(self._apply_mix)
        controls.addWidget(self.mix_slider)
        self.midi_label = QLabel("MIDI")
        controls.addWidget(self.midi_label)
        self.stereo_checkbox = QCheckBox()
        self.stereo_checkbox.setEnabled(False)
        self.stereo_checkbox.toggled.connect(self._apply_mix)
        controls.addWidget(self.stereo_checkbox)
        result_controls.addLayout(controls)

        transport = QHBoxLayout()
        self.playback_slider = _SeekSlider()
        self.playback_slider.setRange(0, 0)
        self.playback_slider.setSingleStep(100)
        self.playback_slider.setPageStep(5_000)
        self.playback_slider.setEnabled(False)
        self.playback_slider.setToolTip(t("muscriptor_result.playback_progress_tooltip"))
        self.playback_slider.scrub_started.connect(self._on_playback_scrub_started)
        self.playback_slider.seek_requested.connect(self._preview_playback_scrub)
        self.playback_slider.scrub_finished.connect(self._on_playback_scrub_finished)
        transport.addWidget(self.playback_slider, 1)
        self.duration_label = QLabel("/ 0.0s")
        self.duration_label.setStyleSheet("font-family: Consolas; color: #8da4c9;")
        transport.addWidget(self.duration_label)
        result_controls.addLayout(transport)

        self.editor_panel = QWidget()
        editor = QVBoxLayout(self.editor_panel)
        editor.setContentsMargins(0, 0, 0, 0)
        editor.setSpacing(5)
        primary_commands = QHBoxLayout()
        primary_commands.setSpacing(6)
        secondary_commands = QHBoxLayout()
        secondary_commands.setSpacing(6)
        editor_fields = QHBoxLayout()
        editor_fields.setSpacing(6)
        self.edit_toggle = QToolButton()
        self.edit_toggle.setCheckable(True)
        self.edit_toggle.setEnabled(False)
        self.edit_toggle.toggled.connect(self._on_edit_mode_toggled)
        primary_commands.addWidget(self.edit_toggle)
        self.edit_add_button = QPushButton()
        self.edit_add_button.setEnabled(False)
        self.edit_add_button.clicked.connect(self._add_editor_note_at_playhead)
        primary_commands.addWidget(self.edit_add_button)
        self.edit_delete_button = QPushButton()
        self.edit_delete_button.setEnabled(False)
        self.edit_delete_button.clicked.connect(self._delete_selected_editor_note)
        primary_commands.addWidget(self.edit_delete_button)
        self.edit_undo_button = QPushButton()
        self.edit_undo_button.setEnabled(False)
        self.edit_undo_button.clicked.connect(self._undo_editor_notes)
        primary_commands.addWidget(self.edit_undo_button)
        self.edit_redo_button = QPushButton()
        self.edit_redo_button.setEnabled(False)
        self.edit_redo_button.clicked.connect(self._redo_editor_notes)
        primary_commands.addWidget(self.edit_redo_button)
        self.edit_reset_button = QPushButton()
        self.edit_reset_button.setEnabled(False)
        self.edit_reset_button.clicked.connect(self._reset_editor_notes)
        primary_commands.addWidget(self.edit_reset_button)
        primary_commands.addStretch()
        self.edit_select_all_button = QPushButton()
        self.edit_select_all_button.setEnabled(False)
        self.edit_select_all_button.clicked.connect(self._select_all_editor_notes)
        secondary_commands.addWidget(self.edit_select_all_button)
        self.edit_cut_button = QPushButton()
        self.edit_cut_button.setEnabled(False)
        self.edit_cut_button.clicked.connect(self._cut_selected_editor_notes)
        secondary_commands.addWidget(self.edit_cut_button)
        self.edit_copy_button = QPushButton()
        self.edit_copy_button.setEnabled(False)
        self.edit_copy_button.clicked.connect(self._copy_selected_editor_notes)
        secondary_commands.addWidget(self.edit_copy_button)
        self.edit_paste_button = QPushButton()
        self.edit_paste_button.setEnabled(False)
        self.edit_paste_button.clicked.connect(self._paste_editor_notes)
        secondary_commands.addWidget(self.edit_paste_button)
        self.edit_duplicate_button = QPushButton()
        self.edit_duplicate_button.setEnabled(False)
        self.edit_duplicate_button.clicked.connect(self._duplicate_selected_editor_notes)
        secondary_commands.addWidget(self.edit_duplicate_button)
        self.edit_quantize_button = QPushButton()
        self.edit_quantize_button.setEnabled(False)
        self.edit_quantize_button.clicked.connect(self._quantize_selected_editor_notes)
        secondary_commands.addWidget(self.edit_quantize_button)
        secondary_commands.addStretch()
        for command_button in (
            self.edit_toggle,
            self.edit_add_button,
            self.edit_delete_button,
            self.edit_undo_button,
            self.edit_redo_button,
            self.edit_reset_button,
            self.edit_select_all_button,
            self.edit_cut_button,
            self.edit_copy_button,
            self.edit_paste_button,
            self.edit_duplicate_button,
            self.edit_quantize_button,
        ):
            command_button.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
        self.edit_instrument_label = QLabel()
        editor_fields.addWidget(self.edit_instrument_label)
        self.edit_instrument_combo = QComboBox()
        self.edit_instrument_combo.setMinimumWidth(220)
        self.edit_instrument_combo.setEnabled(False)
        self.edit_instrument_combo.currentIndexChanged.connect(self._on_editor_instrument_changed)
        editor_fields.addWidget(self.edit_instrument_combo)
        editor_fields.addSpacing(8)
        self.edit_velocity_label = QLabel()
        editor_fields.addWidget(self.edit_velocity_label)
        self.edit_velocity_spin = QSpinBox()
        self.edit_velocity_spin.setRange(1, 127)
        self.edit_velocity_spin.setValue(100)
        self.edit_velocity_spin.setEnabled(False)
        self.edit_velocity_spin.valueChanged.connect(self._on_editor_velocity_changed)
        editor_fields.addWidget(self.edit_velocity_spin)
        editor_fields.addSpacing(8)
        self.roll_zoom_label = QLabel()
        editor_fields.addWidget(self.roll_zoom_label)
        self.roll_zoom_spin = QDoubleSpinBox()
        self.roll_zoom_spin.setRange(
            _ROLL_MIN_PIXELS_PER_SECOND / _ROLL_BASE_PIXELS_PER_SECOND,
            _ROLL_MAX_PIXELS_PER_SECOND / _ROLL_BASE_PIXELS_PER_SECOND,
        )
        self.roll_zoom_spin.setDecimals(2)
        self.roll_zoom_spin.setSingleStep(0.25)
        self.roll_zoom_spin.setValue(1.0)
        self.roll_zoom_spin.setSuffix("×")
        self.roll_zoom_spin.setKeyboardTracking(False)
        self.roll_zoom_spin.setMinimumWidth(82)
        self.roll_zoom_spin.setStyleSheet(
            "font-family: Consolas; color: #c8d3e6; background: #16213e; "
            "border: 1px solid #3a4a6a; border-radius: 4px; padding: 4px 7px;"
        )
        editor_fields.addWidget(self.roll_zoom_spin)
        self.edit_summary_label = QLabel()
        self.edit_summary_label.setStyleSheet("color: #8da4c9;")
        self.edit_summary_label.setWordWrap(True)
        editor_fields.addWidget(self.edit_summary_label)
        editor_fields.addStretch()
        editor.addLayout(primary_commands)
        editor.addLayout(secondary_commands)
        editor.addLayout(editor_fields)
        result_controls.addWidget(self.editor_panel)

        self.roll_panel = QWidget()
        content = QHBoxLayout(self.roll_panel)
        content.setContentsMargins(0, 0, 0, 0)
        self.roll_scroll = _PianoRollScrollArea()
        self.roll_scroll.setWidgetResizable(False)
        self.roll_scroll.setMinimumHeight(240)
        self.roll_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.roll = _PianoRollCanvas()
        self.roll.seek_requested.connect(self.seek)
        self.roll.edit_committed.connect(self._on_roll_edit_committed)
        self.roll.selection_changed.connect(self._on_roll_selection_changed)
        self.roll.add_note_requested.connect(self._add_editor_note)
        self.roll.delete_requested.connect(self._delete_selected_editor_note)
        self.roll.undo_requested.connect(self._undo_editor_notes)
        self.roll.redo_requested.connect(self._redo_editor_notes)
        self.roll.command_requested.connect(self._on_editor_command)
        self.roll_scroll.setWidget(self.roll)
        self.roll_scroll.horizontalScrollBar().setSingleStep(48)
        self.roll_scroll.zoom_requested.connect(self._on_roll_zoom_requested)
        self.roll_zoom_spin.valueChanged.connect(self._on_roll_zoom_value_changed)
        self.roll_scroll.manual_navigation_requested.connect(self._on_roll_manual_navigation)
        self.roll_scroll.horizontalScrollBar().actionTriggered.connect(self._on_roll_scroll_action)
        self.roll_scroll.horizontalScrollBar().sliderPressed.connect(
            self._on_roll_manual_navigation
        )
        self.follow_checkbox.toggled.connect(self._on_follow_toggled)
        content.addWidget(self.roll_scroll, 4)

        instrument_panel = QFrame()
        instrument_panel.setObjectName("muscriptorInstrumentPanel")
        instrument_panel.setStyleSheet(
            "QFrame#muscriptorInstrumentPanel { background: #16213e; "
            "border: 1px solid #365f8d; border-radius: 6px; }"
        )
        instrument_panel.setMinimumWidth(240)
        instrument_layout = QVBoxLayout(instrument_panel)
        self.instruments_title = QLabel()
        self.instruments_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #dbeaff;")
        instrument_layout.addWidget(self.instruments_title)
        self.instrument_rows_layout = QVBoxLayout()
        self.instrument_rows_layout.setSpacing(2)
        instrument_layout.addLayout(self.instrument_rows_layout)
        instrument_layout.addStretch()
        content.addWidget(instrument_panel, 1)
        self.result_splitter = QSplitter(Qt.Orientation.Vertical)
        self.result_splitter.setObjectName("muscriptorEditorSplitter")
        self.result_splitter.setChildrenCollapsible(True)
        self.result_splitter.setHandleWidth(10)
        self.result_splitter.setStyleSheet(
            "QSplitter#muscriptorEditorSplitter::handle:vertical {"
            "background: #203657; border-top: 1px solid #4a78aa; "
            "border-bottom: 1px solid #10233e; margin: 2px 0; }"
            "QSplitter#muscriptorEditorSplitter::handle:vertical:hover {"
            "background: #365f8d; }"
        )
        self.result_splitter.addWidget(self.result_controls_panel)
        self.result_splitter.addWidget(self.roll_panel)
        self.result_splitter.setCollapsible(0, True)
        self.result_splitter.setCollapsible(1, False)
        self.result_splitter.setStretchFactor(0, 0)
        self.result_splitter.setStretchFactor(1, 1)
        self.result_splitter.setSizes(
            [max(240, self.result_controls_panel.sizeHint().height()), 465]
        )
        self.result_splitter.handle(1).setToolTip(t("muscriptor_result.editor_resize_hint"))
        root.addWidget(self.result_splitter, 1)

        outputs = QHBoxLayout()
        self.download_button = QToolButton()
        self.download_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.download_menu = QMenu(self.download_button)
        self.download_midi_action = self.download_menu.addAction("")
        self.download_transcription_action = self.download_menu.addAction("")
        self.download_stereo_action = self.download_menu.addAction("")
        self.download_transcription_action.setEnabled(False)
        self.download_stereo_action.setEnabled(False)
        self.download_midi_action.triggered.connect(lambda: self._save_asset("midi"))
        self.download_transcription_action.triggered.connect(
            lambda: self._save_asset("transcription")
        )
        self.download_stereo_action.triggered.connect(lambda: self._save_asset("stereo"))
        self.download_button.setMenu(self.download_menu)
        outputs.addWidget(self.download_button)
        outputs.addStretch()
        self.another_button = QPushButton()
        self.another_button.clicked.connect(self.transcribe_another_requested)
        outputs.addWidget(self.another_button)
        root.addLayout(outputs)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(_PLAYHEAD_TIMER_MS)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        audio_source = Path(self.audio_path)
        if audio_source.is_file():
            self._original_normal = self._make_player(audio_source)
            self._original_normal[0].durationChanged.connect(self._on_source_duration_changed)
        self.update_translations()
        self.status_label.setText(t("transcription_result.streaming", backend=self.backend_label))
        self.progress_label.setText(t("muscriptor_result.progress_waiting"))
        self.playback_status_label.setText(t("muscriptor_result.preview_waiting"))

    @property
    def midi_path(self) -> str:
        return self._midi_path

    def add_stream_event(self, payload: object) -> None:
        if not isinstance(payload, dict):
            raise TypeError("Transcription stream event must be a dictionary")
        event_type = payload.get("type")
        if event_type == "runtime":
            required = {
                "model",
                "device",
                "gpu",
                "compute_dtype",
                "batch_size",
                "prelude_forcing",
            }
            missing = sorted(required.difference(payload))
            if missing:
                raise ValueError(f"MuScriptor runtime event is missing fields: {missing!r}")
            self._runtime_details = dict(payload)
        elif event_type in {"note_end", "note_batch"}:
            raw_notes = [payload] if event_type == "note_end" else payload.get("notes")
            if not isinstance(raw_notes, list):
                raise TypeError("MuScriptor note batch must contain a list")
            instruments_changed = False
            for note in raw_notes:
                if not isinstance(note, dict):
                    raise TypeError("MuScriptor note batch entries must be dictionaries")
                instrument = str(note["instrument"])
                if instrument not in self._detected:
                    self._detected.append(instrument)
                    instruments_changed = True
                self._stream_notes.append(
                    MuscriptorRollNote(
                        instrument=instrument,
                        pitch=int(note["pitch"]),
                        velocity=100,
                        start=float(note["start_time"]),
                        end=float(note["end_time"]),
                        program=MUSCRIPTOR_REPRESENTATIVE_PROGRAMS.get(instrument, 0),
                        is_drum=instrument == "drums",
                    )
                )
            if instruments_changed:
                self._rebuild_instrument_rows()
        elif event_type == "snapshot":
            completed = int(payload["completed"])
            total = int(payload["total"])
            duration = float(payload.get("duration_seconds", 0.0))
            frontier = float(payload["frontier_seconds"])
            raw_notes = payload.get("notes")
            if not isinstance(raw_notes, list):
                raise TypeError("Transcription snapshot notes must be a list")
            notes = [
                MuscriptorRollNote(
                    instrument=str(note["instrument"]),
                    pitch=int(note["pitch"]),
                    velocity=int(note["velocity"]),
                    start=float(note["start"]),
                    end=float(note["end"]),
                    program=int(note.get("program", 0)),
                    is_drum=bool(note.get("is_drum", False)),
                )
                for note in raw_notes
            ]
            if any(note.end > frontier + 1e-6 for note in notes):
                raise ValueError("Transcription snapshot contains notes beyond its frontier")
            self._stream_notes = notes
            self._detected = list(dict.fromkeys(note.instrument for note in notes))
            self._source_duration_seconds = max(self._source_duration_seconds, duration)
            self._progress_completed = completed
            self._progress_total = total
            self._progress_estimator.on_anchor(completed, total)
            self._rebuild_instrument_rows()
            self.status_label.setText(
                t(
                    "transcription_result.chunks",
                    backend=self.backend_label,
                    completed=completed,
                    total=total,
                )
            )
            self._queue_preview(completed, total, frontier=frontier)
            self._update_stream_progress()
        elif event_type == "progress":
            completed = int(payload["completed"])
            total = int(payload["total"])
            self._progress_completed = completed
            self._progress_total = total
            self._progress_estimator.on_anchor(completed, total)
            self.status_label.setText(
                t(
                    "progress.muscriptor_chunks",
                    completed=completed,
                    total=total,
                )
            )
            self._queue_preview(completed, total)
            self._update_stream_progress()

    def finalize_result(self, result: ProcessingResult) -> None:
        self._finalizing = True
        self.slow_hint_label.hide()
        self._preview_pending = None
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_worker.cancel()
        self._midi_path = str(Path(result.midi_path).resolve())
        if result.beat_info is not None:
            self.set_bpm_context(
                (
                    result.beat_info.source_bpm
                    if result.beat_info.source_bpm is not None
                    else result.beat_info.bpm
                ),
                result.beat_info.bpm,
            )
        output_dir = Path(self._midi_path).parent / "midi-playback"
        output_dir.mkdir(parents=True, exist_ok=True)
        playback_midi_path = Path(self._midi_path)
        if self._detected_bpm is not None:
            playback_midi_path = _export_midi_with_bpm(
                self._midi_path,
                output_dir / "source-tempo-playback.mid",
                self._detected_bpm,
            )
        if self.muscriptor_groups and result.beat_info is not None:
            from src.core.midi_tempo import read_muscriptor_bar_offset_seconds

            bar_offset = read_muscriptor_bar_offset_seconds(playback_midi_path)
            beats_per_bar = None
            if result.beat_info.time_signature is not None:
                numerator, denominator = result.beat_info.time_signature
                quarter_beats = float(numerator) * 4.0 / float(denominator)
                rounded = round(quarter_beats)
                if math.isclose(quarter_beats, rounded, rel_tol=0.0, abs_tol=1e-9):
                    beats_per_bar = int(rounded)
            self.roll.set_beat_grid(
                (float(value) + bar_offset for value in result.beat_info.beat_times),
                (float(value) + bar_offset for value in (result.beat_info.downbeats or ())),
                beats_per_bar=beats_per_bar,
            )
        else:
            self.roll.set_beat_grid(None)
        read_midi_roll_notes(
            playback_midi_path,
            muscriptor_groups=self.muscriptor_groups,
        )
        if self.muscriptor_groups:
            self.selected_instruments = list(result.selected_instruments)
        self.status_label.setText(t("muscriptor_result.preparing_audio"))
        self.progress_bar.setValue(1000)
        total_duration = self._effective_source_duration()
        if total_duration > 0:
            self.progress_label.setText(
                t(
                    "muscriptor_result.progress_complete",
                    processed=_format_clock(total_duration),
                    total=_format_clock(total_duration),
                    elapsed=_format_clock(self._progress_estimator.elapsed_seconds()),
                )
            )
        self.playback_status_label.setText(t("muscriptor_result.final_audio_preparing"))
        worker = _AssetWorker(
            str(playback_midi_path),
            self.audio_path,
            str(output_dir),
            self.muscriptor_groups,
            self,
        )
        worker.progress.connect(self._on_asset_progress)
        worker.succeeded.connect(self._on_assets_ready)
        worker.failed.connect(self._on_assets_failed)
        worker.finished.connect(lambda worker=worker: self._on_asset_worker_finished(worker))
        worker.finished.connect(worker.deleteLater)
        self._asset_worker = worker
        worker.start()

    def mark_failed(self, error: str) -> None:
        """Stop future snapshots while preserving an already rendered preview."""
        self._finalizing = True
        self.slow_hint_label.hide()
        self._preview_pending = None
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_worker.cancel()
        self.status_label.setText(t("transcription_result.failed", error=error))
        preview_available = self._midi_normal is not None and self._original_normal is not None
        self.play_button.setEnabled(preview_available)
        self.mix_slider.setEnabled(preview_available)

    def mark_cancelled(self) -> None:
        self._finalizing = True
        self.slow_hint_label.hide()
        self._preview_pending = None
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_worker.cancel()
        self.status_label.setText(t("transcription_result.cancelled"))

    def _effective_source_duration(self) -> float:
        if self._source_duration_seconds > 0:
            return self._source_duration_seconds
        if self._progress_total > 0:
            return self._progress_total * _STREAM_CHUNK_SECONDS
        return 0.0

    def _preview_frontier(self, completed: int, total: int) -> float:
        duration = self._source_duration_seconds or total * _STREAM_CHUNK_SECONDS
        frontier = duration * completed / total
        if self._stream_notes:
            frontier = max(frontier, max(note.end for note in self._stream_notes))
        return frontier

    def _queue_preview(
        self,
        completed: int,
        total: int,
        *,
        frontier: float | None = None,
    ) -> None:
        if (
            self._finalizing
            or self._shutting_down
            or self._preview_error is not None
            or not self._stream_notes
        ):
            return
        notes = tuple(self._stream_notes)
        frontier = (
            self._preview_frontier(completed, total)
            if frontier is None
            else max(float(frontier), max(note.end for note in self._stream_notes))
        )
        first_preview = self._preview_generation == 0
        enough_new_audio = (
            frontier - self._preview_last_requested_frontier >= _STREAM_PREVIEW_REFRESH_SECONDS
        )
        playback_position = self._position_ms / 1000.0
        playback_needs_extension = (
            self._preview_duration > 0
            and playback_position
            >= max(
                0.0,
                self._preview_duration - _STREAM_PREVIEW_PLAYBACK_MARGIN_SECONDS,
            )
            and frontier > self._preview_duration + 1e-6
        )
        if not (
            first_preview or completed >= total or enough_new_audio or playback_needs_extension
        ):
            return
        self._preview_generation += 1
        self._preview_last_requested_frontier = frontier
        self._preview_pending = (
            self._preview_generation,
            notes,
            frontier,
            completed,
            total,
        )
        if self._preview_worker is None or not self._preview_worker.isRunning():
            self._start_pending_preview()

    def _start_pending_preview(self) -> None:
        if (
            self._preview_pending is None
            or self._finalizing
            or self._shutting_down
            or self._preview_error is not None
        ):
            return
        generation, notes, frontier, completed, total = self._preview_pending
        self._preview_pending = None
        output_dir = self._preview_root / f"generation-{generation:06d}"
        self.playback_status_label.setText(
            t(
                "muscriptor_result.preview_rendering",
                completed=completed,
                total=total,
            )
        )
        worker = _PreviewAssetWorker(
            generation,
            notes,
            frontier,
            self.audio_path,
            str(output_dir),
            self,
        )
        worker.succeeded.connect(self._on_preview_ready)
        worker.failed.connect(self._on_preview_failed)
        worker.finished.connect(lambda worker=worker: self._on_preview_worker_finished(worker))
        worker.finished.connect(worker.deleteLater)
        self._preview_worker = worker
        # Snapshot rendering grows with the playable frontier. Keep it below
        # inference and the Qt event loop so preview work cannot starve either.
        worker.start(QThread.Priority.LowPriority)

    def _on_preview_ready(self, generation: int, payload: object) -> None:
        if self._finalizing or self._shutting_down or generation <= self._preview_ready_generation:
            return
        if not isinstance(payload, MuscriptorPreviewAssets):
            self._on_preview_failed(
                generation,
                "Invalid MuScriptor preview asset payload",
            )
            return
        self._preview_ready_generation = generation
        if self._playing or self._transport_seek_pending_ms is not None:
            self._deferred_preview = (generation, payload)
            self.playback_status_label.setText(
                t(
                    "muscriptor_result.preview_buffered",
                    time=_format_clock(payload.duration),
                )
            )
            return
        try:
            self._apply_preview_assets(generation, payload)
        except Exception as exc:
            logger.exception(
                "Failed to apply MIDI preview assets | generation=%s",
                generation,
            )
            self._on_preview_failed(generation, str(exc))

    def _apply_preview_assets(
        self,
        generation: int,
        payload: MuscriptorPreviewAssets,
    ) -> None:
        if self._playing or self._transport_seek_pending_ms is not None:
            raise RuntimeError("Cannot replace MIDI preview assets during transport activity")
        if generation <= self._preview_applied_generation:
            return
        self.roll.set_notes(payload.notes, duration=payload.duration)
        position_ms = self._position_ms
        self._dispose_dynamic_players()
        self._ensure_original_player()
        self._midi_normal = self._make_player(payload.transcription_wav)
        self._normal_sources = dict(payload.instrument_wavs)

        self._preview_applied_generation = generation
        self._preview_note_count = len(payload.notes)
        self._preview_duration = payload.duration
        self._detected = list(dict.fromkeys(note.instrument for note in payload.notes))
        self._rebuild_instrument_rows()
        self.play_button.setEnabled(True)
        self._set_playback_duration(self._preview_duration)
        self.mix_slider.setEnabled(True)
        self.stereo_checkbox.setChecked(False)
        self.stereo_checkbox.setEnabled(False)
        self.playback_status_label.setText(
            t(
                "muscriptor_result.preview_ready",
                time=_format_clock(self._preview_duration),
            )
        )
        self._apply_mix()

        self._position_ms = min(position_ms, int(self._preview_duration * 1000))
        self._playback_finished = self._position_ms >= int(self._preview_duration * 1000) - 30
        self._playback_clock.reset(self._position_ms)
        self.playback_slider.setValue(self._position_ms)
        for player in self._all_playback_players():
            player.setPosition(self._position_ms)

    def _apply_deferred_assets(self) -> None:
        if self._playing or self._transport_seek_pending_ms is not None or self._shutting_down:
            return
        if self._deferred_final_assets is not None:
            assets = self._deferred_final_assets
            self._deferred_final_assets = None
            self._deferred_preview = None
            try:
                self._apply_final_assets(assets)
            except Exception as exc:
                logger.exception("Failed to apply deferred final MIDI assets")
                self._on_assets_failed(str(exc))
            return
        if self._deferred_preview is not None:
            generation, payload = self._deferred_preview
            self._deferred_preview = None
            try:
                self._apply_preview_assets(generation, payload)
            except Exception as exc:
                logger.exception(
                    "Failed to apply deferred MIDI preview assets | generation=%s",
                    generation,
                )
                self._on_preview_failed(generation, str(exc))

    def _on_preview_failed(self, generation: int, error: str) -> None:
        if self._finalizing or self._shutting_down or generation < self._preview_ready_generation:
            return
        self._preview_error = error
        self._preview_pending = None
        self.playback_status_label.setText(t("muscriptor_result.preview_failed", error=error))

    def _on_preview_worker_finished(self, worker: _PreviewAssetWorker) -> None:
        if self._preview_worker is worker:
            self._preview_worker = None
        if not self._finalizing and not self._shutting_down and self._preview_error is None:
            self._start_pending_preview()

    def _ensure_original_player(self) -> None:
        if self._original_normal is not None:
            return
        source = Path(self.audio_path)
        if not source.is_file():
            raise FileNotFoundError(f"MuScriptor source audio is missing: {source}")
        self._original_normal = self._make_player(source)
        self._original_normal[0].durationChanged.connect(self._on_source_duration_changed)

    def _dispose_dynamic_players(self) -> None:
        self._startup_sync_pending = False
        original_player = self._original_normal[0] if self._original_normal is not None else None
        retired_players = [player for player in self._players if player is not original_player]
        self._players = [original_player] if original_player is not None else []
        self._active_player_ids = frozenset(
            {id(original_player)}
            if original_player is not None
            and original_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            else set()
        )
        self._normal_sources.clear()
        self._right_sources.clear()
        self._normal_players.clear()
        self._right_players.clear()
        self._midi_normal = None
        self._midi_right = None
        self._original_left = None

        if not retired_players:
            return

        # Silence retired outputs synchronously.  Asset replacement is forbidden
        # while transport is playing, so every retired backend must already be
        # quiescent before its source is unloaded.
        for player in retired_players:
            if player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                raise RuntimeError("Cannot retire a playing MIDI audio backend")
            output = player.audioOutput()
            if output is not None:
                output.setMuted(True)

        # Keep both wrappers owned until a widget-bound timer reaches the next
        # event-loop turn. A contextless singleShot closure can outlive this
        # workbench and dereference an already deleted QMediaPlayer.
        self._retired_media.extend((player, player.audioOutput()) for player in retired_players)
        self._retired_media_timer.start(0)

    def _release_retired_media(self) -> None:
        """Unload retired Qt multimedia objects without leaving stale callbacks."""

        retired_media = self._retired_media
        self._retired_media = []
        for player, output in retired_media:
            if output is not None:
                output.setMuted(True)
            if player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
                player.stop()
            player.setSource(QUrl())
            player.setAudioOutput(None)
            player.deleteLater()
            # Flush only this object's DeferredDelete event. The previous global
            # flush could re-enter deletion of unrelated workers and widgets.
            QCoreApplication.sendPostedEvents(player, QEvent.Type.DeferredDelete)
            if output is not None:
                output.deleteLater()
                QCoreApplication.sendPostedEvents(output, QEvent.Type.DeferredDelete)

    def _on_source_duration_changed(self, duration_ms: int) -> None:
        if duration_ms > 0:
            self._source_duration_seconds = duration_ms / 1000.0

    def _on_asset_progress(self, progress: float, message: str) -> None:
        if self._shutting_down:
            return
        self.playback_status_label.setText(f"{int(progress * 100)}% · {message}")

    def _make_player(self, source: Path) -> tuple[QMediaPlayer, QAudioOutput]:
        player = QMediaPlayer(self)
        output = QAudioOutput(self)
        player.setAudioOutput(output)
        player.setSource(QUrl.fromLocalFile(str(source)))
        player.setPlaybackRate(self._result_playback_rate())
        player.errorOccurred.connect(
            lambda _error, message: self.playback_status_label.setText(
                t("muscriptor_result.player_failed", error=message)
            )
        )
        player.playbackStateChanged.connect(self._on_player_playback_state_changed)
        self._players.append(player)
        return player, output

    def _on_player_playback_state_changed(self, _state) -> None:
        if self._transport_seek_pending_ms is not None:
            return
        if self._deferred_apply_scheduled and not self._playing:
            self._try_apply_deferred_assets()
        if not self._playing or not self._startup_sync_pending:
            return
        active = self._active_playback_players()
        if len(active) < 2 or any(
            player.playbackState() != QMediaPlayer.PlaybackState.PlayingState for player in active
        ):
            return
        master_position = active[0].position()
        for player in active[1:]:
            if abs(player.position() - master_position) > 20:
                self._startup_sync_pending = False
                self.seek(master_position / 1000.0)
                return
        self._startup_sync_pending = False

    def _schedule_transport_seek_commit(self) -> None:
        """Start one serialized stop/position/play transaction for all backends."""

        if (
            self._shutting_down
            or self._transport_seek_pending_ms is None
            or self._transport_seek_commit_scheduled
        ):
            return
        self._transport_seek_commit_scheduled = True
        self._transport_seek_commit_players = tuple(self._all_playback_players())
        self._transport_seek_commit_position_ms = self._transport_seek_pending_ms
        self._transport_seek_commit_pause_only = self._transport_seek_pause_only
        self._transport_seek_commit_index = 0
        self._transport_seek_commit_phase = (
            "pause" if self._transport_seek_commit_pause_only else "stop"
        )
        self._transport_commit_timer.start(0)

    def _finish_transport_seek_commit(self) -> None:
        resume = self._transport_seek_resume and self._playing
        self._transport_seek_pending_ms = None
        self._transport_seek_resume = False
        self._transport_seek_pause_only = False
        self._transport_seek_commit_scheduled = False
        self._transport_seek_commit_players = ()
        self._transport_seek_commit_pause_only = False
        self._transport_seek_commit_index = 0
        self._transport_seek_commit_phase = ""
        self._startup_sync_pending = False
        if not resume and not self._shutting_down:
            self._after_transport_timer.start(0)

    def _apply_deferred_after_transport(self) -> None:
        if self._shutting_down or self._playing or self._transport_seek_pending_ms is not None:
            return
        if self._deferred_editor_assets is not None:
            generation, assets = self._deferred_editor_assets
            self._deferred_editor_assets = None
            self._on_editor_audio_ready(generation, assets)
            return
        if self._deferred_apply_scheduled:
            self._try_apply_deferred_assets()
        else:
            self._schedule_deferred_assets()

    def _commit_next_transport_player(self) -> None:
        """Apply one backend mutation per event-loop turn to avoid Qt re-entry."""

        if self._shutting_down or self._transport_seek_pending_ms is None:
            self._finish_transport_seek_commit()
            return
        if (
            self._transport_seek_pending_ms != self._transport_seek_commit_position_ms
            or self._transport_seek_pause_only != self._transport_seek_commit_pause_only
        ):
            self._transport_seek_commit_players = tuple(self._all_playback_players())
            self._transport_seek_commit_position_ms = self._transport_seek_pending_ms
            self._transport_seek_commit_pause_only = self._transport_seek_pause_only
            self._transport_seek_commit_index = 0
            self._transport_seek_commit_phase = (
                "pause" if self._transport_seek_commit_pause_only else "stop"
            )

        players = self._transport_seek_commit_players
        if self._transport_seek_commit_phase == "pause":
            if self._transport_seek_commit_index < len(players):
                player = players[self._transport_seek_commit_index]
                player.pause()
                self._transport_seek_commit_index += 1
                self._transport_commit_timer.start(0)
                return
            if self._transport_seek_resume and self._playing:
                self._transport_seek_pause_only = False
                self._transport_seek_commit_pause_only = False
                self._transport_seek_commit_players = tuple(self._all_playback_players())
                self._transport_seek_commit_index = 0
                self._transport_seek_commit_phase = "stop"
                self._transport_commit_timer.start(0)
                return
            self._finish_transport_seek_commit()
            return

        if self._transport_seek_commit_phase == "stop":
            if self._transport_seek_commit_index < len(players):
                player = players[self._transport_seek_commit_index]
                if player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
                    player.stop()
                self._transport_seek_commit_index += 1
                self._transport_commit_timer.start(0)
                return
            self._transport_seek_commit_players = tuple(self._all_playback_players())
            self._transport_seek_commit_index = 0
            self._transport_seek_commit_phase = "position"
            self._transport_commit_timer.start(0)
            return

        if self._transport_seek_commit_phase == "position":
            if self._transport_seek_commit_index < len(players):
                player = players[self._transport_seek_commit_index]
                if player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
                    raise RuntimeError("MIDI backend resumed before serialized seek positioning")
                player.setPosition(self._transport_seek_commit_position_ms)
                self._transport_seek_commit_index += 1
                self._transport_commit_timer.start(0)
                return
            if not (self._transport_seek_resume and self._playing):
                self._finish_transport_seek_commit()
                return
            active_players = self._active_playback_players()
            self._active_player_ids = frozenset(id(player) for player in active_players)
            self._transport_seek_commit_players = tuple(active_players)
            self._transport_seek_commit_index = 0
            self._transport_seek_commit_phase = "play"
            self._transport_commit_timer.start(0)
            return

        if self._transport_seek_commit_phase != "play":
            raise RuntimeError(
                f"Invalid serialized MIDI seek phase: {self._transport_seek_commit_phase!r}"
            )
        if not (self._transport_seek_resume and self._playing):
            self._finish_transport_seek_commit()
            return
        if self._transport_seek_commit_index < len(players):
            player = players[self._transport_seek_commit_index]
            player.play()
            self._transport_seek_commit_index += 1
            self._transport_commit_timer.start(0)
            return
        self._finish_transport_seek_commit()

    def _on_assets_ready(self, assets: object) -> None:
        if self._shutting_down:
            return
        if not isinstance(assets, MuscriptorPlaybackAssets):
            self._on_assets_failed("Invalid MuScriptor playback asset payload")
            return
        if self._playing or self._transport_seek_pending_ms is not None:
            self._deferred_final_assets = assets
            self.playback_status_label.setText(t("muscriptor_result.final_audio_buffered"))
            return
        try:
            self._apply_final_assets(assets)
        except Exception as exc:
            logger.exception("Failed to apply final MIDI assets")
            self._on_assets_failed(str(exc))

    def _apply_final_assets(self, assets: MuscriptorPlaybackAssets) -> None:
        if self._playing or self._transport_seek_pending_ms is not None:
            raise RuntimeError("Cannot replace final MIDI assets during transport activity")
        self.roll.set_notes(assets.notes, duration=assets.duration)
        self._original_assets = assets
        self._replace_playback_assets(assets, detected_notes=assets.notes)
        self._begin_editor_session(assets.notes, assets.duration)
        self.status_label.setText(t("muscriptor_result.ready"))
        self.playback_status_label.setText(t("muscriptor_result.final_audio_ready"))
        self._preview_duration = 0.0

    def _replace_playback_assets(
        self,
        assets: MuscriptorPlaybackAssets,
        *,
        detected_notes: Iterable[MuscriptorRollNote],
    ) -> None:
        """Atomically replace every audible MIDI bus while transport is paused."""

        if self._playing or self._transport_seek_pending_ms is not None:
            raise RuntimeError("Cannot replace MIDI playback assets during transport activity")
        position_ms = self._position_ms
        self._dispose_dynamic_players()
        self._assets = assets
        self._ensure_original_player()
        self._original_left = self._make_player(assets.original_left_wav)
        self._midi_normal = self._make_player(assets.live_transcription_wav)
        self._midi_right = self._make_player(assets.transcription_right_wav)
        self._normal_sources = dict(assets.instrument_wavs)
        self._right_sources = dict(assets.instrument_right_wavs)
        self._detected = list(dict.fromkeys(note.instrument for note in detected_notes))
        self._muted.intersection_update(self._detected)
        if self._soloed not in self._detected:
            self._soloed = None
        self._rebuild_instrument_rows()
        self.play_button.setEnabled(True)
        self._set_playback_duration(assets.duration)
        self.mix_slider.setEnabled(bool(self._normal_sources))
        self.stereo_checkbox.setEnabled(True)
        self.download_transcription_action.setEnabled(True)
        self.download_stereo_action.setEnabled(True)
        self._position_ms = min(position_ms, int(round(assets.duration * 1000.0)))
        self._playback_finished = False
        self._playback_clock.reset(self._position_ms)
        self.playback_slider.setValue(self._position_ms)
        for player in self._all_playback_players():
            player.setPosition(self._position_ms)
        self._apply_mix()
        self._update_play_label()

    def _begin_editor_session(
        self,
        notes: Iterable[MuscriptorRollNote],
        duration: float,
    ) -> None:
        snapshot = tuple(notes)
        self._original_edit_notes = snapshot
        self._edited_notes = snapshot
        self._edit_duration = float(duration)
        self._edit_undo.clear()
        self._edit_redo.clear()
        self._edit_clipboard = ()
        self._active_edit_instrument = snapshot[0].instrument if snapshot else ""
        self.roll.set_grid_seconds(self._editor_grid_seconds())
        self._syncing_editor_controls = True
        try:
            self.edit_instrument_combo.clear()
            ordered = list(dict.fromkeys(note.instrument for note in snapshot))
            for instrument in ordered:
                self.edit_instrument_combo.addItem(_instrument_label(instrument), instrument)
            if self._active_edit_instrument:
                index = self.edit_instrument_combo.findData(self._active_edit_instrument)
                if index >= 0:
                    self.edit_instrument_combo.setCurrentIndex(index)
        finally:
            self._syncing_editor_controls = False
        self.edit_toggle.setEnabled(True)
        self.roll.set_editable(self.edit_toggle.isChecked())
        self._sync_editor_controls()

    def _on_edit_mode_toggled(self, enabled: bool) -> None:
        self.roll.set_editable(enabled)
        self._sync_editor_controls()
        if enabled:
            self.roll.setFocus(Qt.FocusReason.OtherFocusReason)

    def _sync_editor_controls(self) -> None:
        editable = self.edit_toggle.isEnabled() and self.edit_toggle.isChecked()
        selected = bool(self.roll.selected_indices)
        dirty = bool(self._original_edit_notes) and self._edited_notes != self._original_edit_notes
        self.edit_add_button.setEnabled(editable and bool(self._active_edit_instrument))
        self.edit_delete_button.setEnabled(editable and selected)
        self.edit_undo_button.setEnabled(editable and bool(self._edit_undo))
        self.edit_redo_button.setEnabled(editable and bool(self._edit_redo))
        self.edit_reset_button.setEnabled(editable and dirty)
        self.edit_select_all_button.setEnabled(editable and bool(self._edited_notes))
        self.edit_cut_button.setEnabled(editable and selected)
        self.edit_copy_button.setEnabled(editable and selected)
        self.edit_paste_button.setEnabled(editable and bool(self._edit_clipboard))
        self.edit_duplicate_button.setEnabled(editable and selected)
        self.edit_quantize_button.setEnabled(editable and selected)
        self.edit_instrument_combo.setEnabled(editable and self.edit_instrument_combo.count() > 0)
        self.edit_velocity_spin.setEnabled(editable and selected)
        self.edit_summary_label.setText(
            t(
                "muscriptor_result.editor_summary",
                count=len(self._edited_notes),
                changes=len(self._edit_undo),
            )
        )

    def _on_roll_selection_changed(self, note: object) -> None:
        selected = note if isinstance(note, MuscriptorRollNote) else None
        self._syncing_editor_controls = True
        try:
            if selected is not None:
                self._active_edit_instrument = selected.instrument
                index = self.edit_instrument_combo.findData(selected.instrument)
                if index >= 0:
                    self.edit_instrument_combo.setCurrentIndex(index)
                self.edit_velocity_spin.setValue(int(selected.velocity))
        finally:
            self._syncing_editor_controls = False
        self._sync_editor_controls()

    def _record_editor_commit(
        self,
        before: tuple[MuscriptorRollNote, ...],
        after: tuple[MuscriptorRollNote, ...],
    ) -> None:
        if before == after:
            return
        if self._playing:
            self.pause()
        self._edit_undo.append(tuple(before))
        if len(self._edit_undo) > 100:
            del self._edit_undo[0]
        self._edit_redo.clear()
        self._edited_notes = tuple(after)
        self._queue_editor_audio_render(before, self._edited_notes)
        self._sync_editor_controls()

    def _on_roll_edit_committed(self, before: object, after: object) -> None:
        if not isinstance(before, tuple) or not isinstance(after, tuple):
            raise TypeError("Piano-roll edit snapshots must be tuples")
        self._record_editor_commit(before, after)

    def _apply_editor_snapshot(
        self,
        snapshot: Iterable[MuscriptorRollNote],
        *,
        selected_index: int | None = None,
        selected_indices: Iterable[int] | None = None,
        record: bool,
    ) -> None:
        before = self._edited_notes
        after = tuple(snapshot)
        self.roll.set_notes(
            after,
            duration=self._edit_duration,
            selected_index=selected_index,
            selected_indices=selected_indices,
        )
        self._edited_notes = after
        self._on_roll_selection_changed(self.roll.selected_note)
        if record:
            self._record_editor_commit(before, after)
        else:
            if before != after:
                self._queue_editor_audio_render(before, after)
            self._sync_editor_controls()

    def _queue_editor_audio_render(
        self,
        before: tuple[MuscriptorRollNote, ...],
        after: tuple[MuscriptorRollNote, ...],
    ) -> None:
        """Invalidate stale audio and queue one render for the latest edit snapshot."""

        if before == after:
            return
        if self._playing:
            self.pause()
        self._edit_asset_generation += 1
        generation = self._edit_asset_generation
        self._edit_asset_pending = (generation, tuple(after))
        self.play_button.setEnabled(False)
        self.playback_slider.setEnabled(False)
        self.playback_status_label.setText(t("muscriptor_result.editor_audio_rendering"))
        if self._edit_asset_worker is not None and self._edit_asset_worker.isRunning():
            self._edit_asset_worker.cancel()

        if tuple(after) == self._original_edit_notes and self._original_assets is not None:
            self._edit_asset_debounce.stop()
            self._edit_asset_pending = None
            self._apply_editor_audio_assets(
                generation,
                self._original_assets,
                output_dir=None,
                restored_original=True,
            )
            return
        self._edit_asset_debounce.start()

    def _start_pending_editor_audio(self) -> None:
        if self._shutting_down or self._edit_asset_pending is None:
            return
        if self._edit_asset_worker is not None and self._edit_asset_worker.isRunning():
            return
        generation, notes = self._edit_asset_pending
        self._edit_asset_pending = None
        if self._original_assets is None or not self._midi_path or self._detected_bpm is None:
            self.playback_status_label.setText(
                t(
                    "muscriptor_result.editor_audio_failed",
                    error="completed MIDI playback context is unavailable",
                )
            )
            return
        output_dir = self._edit_asset_root / f"generation-{generation:06d}"
        worker = _EditedAssetWorker(
            generation,
            self._midi_path,
            notes,
            self._detected_bpm,
            self.audio_path,
            str(output_dir),
            self.muscriptor_groups,
            self,
        )
        worker.progress.connect(self._on_editor_audio_progress)
        worker.succeeded.connect(self._on_editor_audio_ready)
        worker.failed.connect(self._on_editor_audio_failed)
        worker.finished.connect(lambda worker=worker: self._on_editor_audio_worker_finished(worker))
        worker.finished.connect(worker.deleteLater)
        self._edit_asset_worker = worker
        worker.start(QThread.Priority.LowPriority)

    def _on_editor_audio_progress(
        self,
        generation: int,
        progress: float,
        message: str,
    ) -> None:
        if self._shutting_down or generation != self._edit_asset_generation:
            return
        self.playback_status_label.setText(
            t(
                "muscriptor_result.editor_audio_rendering_progress",
                progress=int(round(float(progress) * 100)),
                message=message,
            )
        )

    def _on_editor_audio_ready(self, generation: int, payload: object) -> None:
        if self._shutting_down or generation != self._edit_asset_generation:
            return
        if not isinstance(payload, MuscriptorPlaybackAssets):
            self._on_editor_audio_failed(
                generation,
                "Invalid edited MIDI playback asset payload",
            )
            return
        if self._playing or self._transport_seek_pending_ms is not None:
            self._deferred_editor_assets = (generation, payload)
            return
        try:
            self._apply_editor_audio_assets(
                generation,
                payload,
                output_dir=self._edit_asset_root / f"generation-{generation:06d}",
                restored_original=False,
            )
        except Exception as exc:
            logger.exception(
                "Failed to apply edited MIDI playback assets | generation=%s",
                generation,
            )
            self._on_editor_audio_failed(generation, str(exc))

    def _apply_editor_audio_assets(
        self,
        generation: int,
        assets: MuscriptorPlaybackAssets,
        *,
        output_dir: Path | None,
        restored_original: bool,
    ) -> None:
        if generation != self._edit_asset_generation:
            raise RuntimeError(
                "Cannot apply stale edited MIDI audio: "
                f"generation={generation}, current={self._edit_asset_generation}"
            )
        previous_dir = self._active_edit_asset_dir
        self._replace_playback_assets(assets, detected_notes=self._edited_notes)
        self._edit_asset_applied_generation = generation
        self._active_edit_asset_dir = output_dir
        self.playback_status_label.setText(
            t(
                "muscriptor_result.final_audio_ready"
                if restored_original
                else "muscriptor_result.editor_audio_ready"
            )
        )
        if previous_dir is not None and previous_dir != output_dir:
            QTimer.singleShot(
                1000,
                lambda path=previous_dir: self._remove_editor_audio_directory(path),
            )

    def _on_editor_audio_failed(self, generation: int, error: str) -> None:
        if (
            self._shutting_down
            or generation != self._edit_asset_generation
            or "cancelled" in str(error).lower()
        ):
            return
        logger.error(
            "Edited MIDI playback rendering failed | generation=%s | error=%s",
            generation,
            error,
        )
        self.play_button.setEnabled(False)
        self.playback_slider.setEnabled(False)
        self.playback_status_label.setText(
            t(
                "muscriptor_result.editor_audio_failed",
                error=_compact_editor_error(error),
            )
        )

    def _on_editor_audio_worker_finished(self, worker: _EditedAssetWorker) -> None:
        if self._edit_asset_worker is worker:
            self._edit_asset_worker = None
        worker_output = Path(worker.output_dir).resolve()
        if worker_output != self._active_edit_asset_dir:
            self._remove_editor_audio_directory(worker_output)
        if (
            not self._shutting_down
            and self._edit_asset_pending is not None
            and not self._edit_asset_debounce.isActive()
        ):
            self._start_pending_editor_audio()

    def _remove_editor_audio_directory(self, path: Path) -> None:
        root = self._edit_asset_root.resolve()
        candidate = Path(path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"Refusing to remove edited MIDI assets outside {root}: {candidate}"
            ) from exc
        if not candidate.exists():
            return
        try:
            shutil.rmtree(candidate)
        except OSError as exc:
            logger.warning("Unable to remove edited MIDI assets %s: %s", candidate, exc)

    def _instrument_note_template(self, instrument: str) -> MuscriptorRollNote:
        for note in (*self._edited_notes, *self._original_edit_notes):
            if note.instrument == instrument:
                return note
        if instrument == "drums":
            return MuscriptorRollNote(
                instrument="drums",
                pitch=36,
                velocity=100,
                start=0.0,
                end=0.5,
                program=0,
                is_drum=True,
                track_index=0,
                channel=9,
            )
        if instrument.startswith("gm:"):
            program = int(instrument.split(":", 1)[1])
        else:
            program = MUSCRIPTOR_REPRESENTATIVE_PROGRAMS.get(instrument, 0)
        return MuscriptorRollNote(
            instrument=instrument,
            pitch=60,
            velocity=100,
            start=0.0,
            end=0.5,
            program=program,
            is_drum=False,
            track_index=0,
            channel=0,
        )

    def _add_editor_note_at_playhead(self) -> None:
        self._add_editor_note(self._position_ms / 1000.0, 60)

    def _add_editor_note(self, start: float, pitch: int) -> None:
        if not self.edit_toggle.isChecked() or not self._active_edit_instrument:
            return
        template = self._instrument_note_template(self._active_edit_instrument)
        note_duration = min(0.5, self._edit_duration)
        note_start = max(0.0, min(float(start), self._edit_duration - 0.01))
        note_end = min(self._edit_duration, note_start + note_duration)
        if note_end - note_start < 0.01:
            note_start = max(0.0, self._edit_duration - 0.01)
            note_end = self._edit_duration
        note = replace(
            template,
            pitch=max(21, min(108, int(pitch))),
            velocity=int(self.edit_velocity_spin.value()),
            start=note_start,
            end=note_end,
        )
        updated = (*self._edited_notes, note)
        self._apply_editor_snapshot(
            updated,
            selected_index=len(updated) - 1,
            record=True,
        )

    def _delete_selected_editor_note(self) -> None:
        indices = set(self.roll.selected_indices)
        if not self.edit_toggle.isChecked() or not indices:
            return
        updated = [note for index, note in enumerate(self._edited_notes) if index not in indices]
        first_deleted = min(indices)
        next_index = min(first_deleted, len(updated) - 1) if updated else None
        self._apply_editor_snapshot(updated, selected_index=next_index, record=True)

    def _select_all_editor_notes(self) -> None:
        if not self.edit_toggle.isChecked():
            return
        indices = tuple(range(len(self._edited_notes)))
        self.roll.set_selected_indices(indices, primary=indices[0] if indices else None)
        self._on_roll_selection_changed(self.roll.selected_note)

    def _copy_selected_editor_notes(self) -> None:
        indices = self.roll.selected_indices
        if not self.edit_toggle.isChecked() or not indices:
            return
        self._edit_clipboard = tuple(self._edited_notes[index] for index in indices)
        self._sync_editor_controls()

    def _cut_selected_editor_notes(self) -> None:
        if not self.roll.selected_indices:
            return
        self._copy_selected_editor_notes()
        self._delete_selected_editor_note()

    def _paste_editor_notes(self) -> None:
        if not self.edit_toggle.isChecked() or not self._edit_clipboard:
            return
        source_start = min(note.start for note in self._edit_clipboard)
        source_end = max(note.end for note in self._edit_clipboard)
        span = source_end - source_start
        target_start = max(
            0.0,
            min(self._position_ms / 1000.0, self._edit_duration - span),
        )
        offset = target_start - source_start
        pasted = tuple(
            replace(note, start=note.start + offset, end=note.end + offset)
            for note in self._edit_clipboard
        )
        first_index = len(self._edited_notes)
        selection = tuple(range(first_index, first_index + len(pasted)))
        self._apply_editor_snapshot(
            (*self._edited_notes, *pasted),
            selected_index=selection[0],
            selected_indices=selection,
            record=True,
        )

    def _duplicate_selected_editor_notes(self) -> None:
        indices = self.roll.selected_indices
        if not self.edit_toggle.isChecked() or not indices:
            return
        selected = tuple(self._edited_notes[index] for index in indices)
        selection_start = min(note.start for note in selected)
        selection_end = max(note.end for note in selected)
        offset = max(self._editor_grid_seconds(), selection_end - selection_start)
        offset = min(offset, self._edit_duration - selection_end)
        if offset <= 0:
            return
        duplicated = tuple(
            replace(note, start=note.start + offset, end=note.end + offset) for note in selected
        )
        first_index = len(self._edited_notes)
        selection = tuple(range(first_index, first_index + len(duplicated)))
        self._apply_editor_snapshot(
            (*self._edited_notes, *duplicated),
            selected_index=selection[0],
            selected_indices=selection,
            record=True,
        )

    def _quantize_selected_editor_notes(self) -> None:
        indices = self.roll.selected_indices
        if not self.edit_toggle.isChecked() or not indices:
            return
        grid = self._editor_grid_seconds()
        updated = list(self._edited_notes)
        for index in indices:
            note = updated[index]
            duration = note.end - note.start
            start = round(note.start / grid) * grid
            start = max(0.0, min(self._edit_duration - duration, start))
            updated[index] = replace(note, start=start, end=start + duration)
        self._apply_editor_snapshot(
            updated,
            selected_index=self.roll.selected_index,
            selected_indices=indices,
            record=True,
        )

    def _transform_selected_editor_notes(self, command: str, amount: int) -> None:
        indices = self.roll.selected_indices
        if not self.edit_toggle.isChecked() or not indices:
            return
        updated = list(self._edited_notes)
        selected = [updated[index] for index in indices]
        if command == "move_time":
            delta = self._editor_grid_seconds() * amount
            delta = max(
                -min(note.start for note in selected),
                min(self._edit_duration - max(note.end for note in selected), delta),
            )
            for index in indices:
                note = updated[index]
                updated[index] = replace(
                    note,
                    start=note.start + delta,
                    end=note.end + delta,
                )
        elif command == "resize_time":
            delta = self._editor_grid_seconds() * amount
            delta = max(
                -min(note.end - note.start - 0.01 for note in selected),
                min(self._edit_duration - max(note.end for note in selected), delta),
            )
            for index in indices:
                note = updated[index]
                updated[index] = replace(note, end=note.end + delta)
        elif command == "pitch":
            pitch_delta = max(
                21 - min(note.pitch for note in selected),
                min(108 - max(note.pitch for note in selected), amount),
            )
            for index in indices:
                updated[index] = replace(
                    updated[index],
                    pitch=updated[index].pitch + pitch_delta,
                )
        elif command == "velocity":
            velocity_delta = max(
                1 - min(note.velocity for note in selected),
                min(127 - max(note.velocity for note in selected), amount),
            )
            for index in indices:
                updated[index] = replace(
                    updated[index],
                    velocity=updated[index].velocity + velocity_delta,
                )
        else:
            raise ValueError(f"Unsupported piano-roll transform: {command}")
        self._apply_editor_snapshot(
            updated,
            selected_index=self.roll.selected_index,
            selected_indices=indices,
            record=True,
        )

    def _on_editor_command(self, command: str) -> None:
        handlers = {
            "select_all": self._select_all_editor_notes,
            "cut": self._cut_selected_editor_notes,
            "copy": self._copy_selected_editor_notes,
            "paste": self._paste_editor_notes,
            "duplicate": self._duplicate_selected_editor_notes,
            "quantize": self._quantize_selected_editor_notes,
        }
        handler = handlers.get(command)
        if handler is not None:
            handler()
            return
        name, separator, amount_text = command.partition(":")
        if not separator:
            raise ValueError(f"Unsupported piano-roll command: {command}")
        self._transform_selected_editor_notes(name, int(amount_text))

    def _editor_grid_seconds(self) -> float:
        bpm = float(self.bpm_spin.value())
        if not math.isfinite(bpm) or bpm <= 0:
            raise RuntimeError(f"Cannot derive MIDI editor grid from BPM {bpm!r}")
        return 60.0 / bpm / 4.0

    def _undo_editor_notes(self) -> None:
        if not self._edit_undo:
            return
        self._edit_redo.append(self._edited_notes)
        snapshot = self._edit_undo.pop()
        self._apply_editor_snapshot(snapshot, selected_index=None, record=False)
        self.playback_status_label.setText(
            t("muscriptor_result.editor_audio_notice")
            if snapshot != self._original_edit_notes
            else t("muscriptor_result.final_audio_ready")
        )

    def _redo_editor_notes(self) -> None:
        if not self._edit_redo:
            return
        self._edit_undo.append(self._edited_notes)
        snapshot = self._edit_redo.pop()
        self._apply_editor_snapshot(snapshot, selected_index=None, record=False)
        self.playback_status_label.setText(t("muscriptor_result.editor_audio_notice"))

    def _reset_editor_notes(self) -> None:
        if self._edited_notes == self._original_edit_notes:
            return
        self._apply_editor_snapshot(
            self._original_edit_notes,
            selected_index=None,
            record=True,
        )
        self.playback_status_label.setText(t("muscriptor_result.final_audio_ready"))

    def _on_editor_instrument_changed(self, _index: int) -> None:
        if self._syncing_editor_controls:
            return
        instrument = str(self.edit_instrument_combo.currentData() or "")
        if not instrument:
            return
        self._active_edit_instrument = instrument
        selected_indices = self.roll.selected_indices
        if not selected_indices:
            self._sync_editor_controls()
            return
        template = self._instrument_note_template(instrument)
        updated = list(self._edited_notes)
        for selected_index in selected_indices:
            updated[selected_index] = replace(
                updated[selected_index],
                instrument=instrument,
                program=template.program,
                is_drum=template.is_drum,
                track_index=template.track_index,
                channel=template.channel,
            )
        self._apply_editor_snapshot(
            updated,
            selected_index=self.roll.selected_index,
            selected_indices=selected_indices,
            record=True,
        )

    def _on_editor_velocity_changed(self, velocity: int) -> None:
        if self._syncing_editor_controls:
            return
        selected_indices = self.roll.selected_indices
        if not selected_indices:
            return
        updated = list(self._edited_notes)
        for selected_index in selected_indices:
            updated[selected_index] = replace(
                updated[selected_index],
                velocity=int(velocity),
            )
        self._apply_editor_snapshot(
            updated,
            selected_index=self.roll.selected_index,
            selected_indices=selected_indices,
            record=True,
        )

    def _on_assets_failed(self, error: str) -> None:
        if self._shutting_down:
            return
        self.status_label.setText(t("muscriptor_result.audio_failed", error=error))
        preview_available = self._midi_normal is not None and self._original_normal is not None
        self.playback_status_label.setText(
            t(
                "muscriptor_result.final_audio_failed_preview"
                if preview_available
                else "muscriptor_result.final_audio_failed_unavailable"
            )
        )
        self.play_button.setEnabled(preview_available)
        self.playback_slider.setEnabled(preview_available and self.playback_slider.maximum() > 0)
        self.mix_slider.setEnabled(preview_available)
        self.stereo_checkbox.setEnabled(False)

    def _on_asset_worker_finished(self, worker: _AssetWorker) -> None:
        """Drop the owner reference before Qt deletes the finished QThread."""
        if self._asset_worker is worker:
            self._asset_worker = None

    def _all_playback_players(self) -> list[QMediaPlayer]:
        return list(self._players)

    def _ensure_instrument_players(self, *, stereo: bool) -> None:
        sources = self._right_sources if stereo else self._normal_sources
        players = self._right_players if stereo else self._normal_players
        for instrument, source in sources.items():
            if instrument not in players:
                players[instrument] = self._make_player(source)

    def _active_playback_pairs(self) -> list[tuple[QMediaPlayer, QAudioOutput]]:
        stereo = (
            self.stereo_checkbox.isChecked()
            and self._original_left is not None
            and self._midi_right is not None
        )
        original = self._original_left if stereo else self._original_normal
        midi_mix = self._midi_right if stereo else self._midi_normal
        if self._muted:
            self._ensure_instrument_players(stereo=stereo)
        instrument_players = self._right_players if stereo else self._normal_players
        # The rendered MIDI mix is always the transport master, including when
        # its gain is zero because per-instrument mute/solo playback is active.
        # Keeping it scheduled makes the displayed progress and all seek/drift
        # corrections follow playable MIDI time instead of the source audio.
        pairs: list[tuple[QMediaPlayer, QAudioOutput]] = []
        if midi_mix is not None:
            pairs.append(midi_mix)
        if original is not None:
            pairs.append(original)
        if self._muted:
            pairs.extend(
                pair
                for instrument, pair in instrument_players.items()
                if instrument not in self._muted
            )
        return pairs

    def _active_playback_players(self) -> list[QMediaPlayer]:
        return [player for player, _output in self._active_playback_pairs()]

    def _set_playback_duration(self, seconds: float) -> None:
        duration_ms = max(0, int(round(float(seconds) * 1000.0)))
        self.playback_slider.setRange(0, duration_ms)
        self.playback_slider.setEnabled(duration_ms > 0 and self.play_button.isEnabled())
        self.duration_label.setText(f"/ {_format_seconds(duration_ms / 1000.0)}")
        self.playback_slider.setValue(min(self._position_ms, duration_ms))

    def _sync_active_players(self) -> None:
        active = self._active_playback_players()
        next_ids = frozenset(id(player) for player in active)
        if next_ids == self._active_player_ids:
            return
        if self._transport_seek_pending_ms is not None:
            self._active_player_ids = next_ids
            return
        self._active_player_ids = next_ids
        if self._playing:
            master_position = active[0].position() if active else self._position_ms
            self.seek(master_position / 1000.0)

    def _toggle_playback(self) -> None:
        if self._original_normal is None or self._midi_normal is None:
            raise RuntimeError("MuScriptor playable audio is not ready")
        self._apply_mix()
        if self._transport_seek_pending_ms is not None:
            if self._playing:
                self.pause()
            else:
                self._playing = True
                self._transport_seek_resume = True
                self._schedule_transport_seek_commit()
                self.playing_changed.emit(True)
                self._update_play_label()
            return
        if self._playing:
            self.pause()
        else:
            if self._playback_finished:
                self._position_ms = 0
                self.playback_slider.setValue(0)
                self._playback_finished = False
            elif (
                not self._finalizing
                and self._preview_duration > 0
                and self._position_ms >= int(self._preview_duration * 1000) - 30
            ):
                self._position_ms = 0
            self._playing = True
            self._startup_sync_pending = False
            self._last_drift_check_position_ms = self._position_ms
            self._playback_clock.reset(self._position_ms)
            self._transport_seek_pending_ms = self._position_ms
            self._transport_seek_resume = True
            self._transport_seek_pause_only = False
            self._schedule_transport_seek_commit()
            self.playing_changed.emit(True)
        self._update_play_label()

    def set_detected_bpm(self, bpm_text: str | float) -> None:
        """Seed the result BPM until the user explicitly overrides it."""

        if isinstance(bpm_text, (int, float)):
            bpm = float(bpm_text)
        else:
            match = re.search(r"\d+(?:\.\d+)?", str(bpm_text or ""))
            if match is None:
                return
            bpm = float(match.group(0))
        if not math.isfinite(bpm) or not MIN_MIDI_BPM <= bpm <= MAX_MIDI_BPM:
            return
        self.set_bpm_context(bpm, bpm)

    def set_bpm_context(self, source_bpm: float, target_bpm: float) -> None:
        """Set source/project BPM and expose their real playback-rate ratio."""

        if self._bpm_user_overridden:
            return
        source = float(source_bpm)
        target = float(target_bpm)
        if (
            not math.isfinite(source)
            or not MIN_MIDI_BPM <= source <= MAX_MIDI_BPM
            or not math.isfinite(target)
            or not MIN_MIDI_BPM <= target <= MAX_MIDI_BPM
        ):
            raise ValueError(
                f"Invalid result BPM context: source={source_bpm!r}, target={target_bpm!r}"
            )
        self._detected_bpm = source
        self._last_tempo_editor = None
        bpm_blocker = QSignalBlocker(self.bpm_spin)
        self.bpm_spin.setValue(target)
        del bpm_blocker
        speed_blocker = QSignalBlocker(self.speed_spin)
        self.speed_spin.setRange(MIN_MIDI_BPM / source, MAX_MIDI_BPM / source)
        self.speed_spin.setValue(target / source)
        del speed_blocker
        self.bpm_spin.show()
        self.speed_label.show()
        self.speed_spin.show()
        self.roll.set_grid_seconds(self._editor_grid_seconds())
        self._apply_result_playback_rate()

    def _result_playback_rate(self) -> float:
        if self._detected_bpm is not None:
            source_bpm = float(self._detected_bpm)
            target_bpm = float(self.bpm_spin.value())
            if (
                not math.isfinite(source_bpm)
                or source_bpm <= 0.0
                or not math.isfinite(target_bpm)
                or target_bpm <= 0.0
            ):
                raise RuntimeError(
                    "Invalid result BPM context: "
                    f"source_bpm={source_bpm!r}, target_bpm={target_bpm!r}"
                )
            project_rate = target_bpm / source_bpm
            if not math.isfinite(project_rate) or project_rate <= 0.0:
                raise RuntimeError(f"Invalid project playback rate: {project_rate!r}")
            return project_rate
        speed = float(self.speed_spin.value())
        if not math.isfinite(speed) or speed <= 0.0:
            raise RuntimeError(f"Invalid result playback rate: {speed!r}")
        return speed

    def _on_result_bpm_changed(self, _bpm: float) -> None:
        if self._detected_bpm is None:
            return
        self._last_tempo_editor = "bpm"
        self._bpm_user_overridden = True
        speed_blocker = QSignalBlocker(self.speed_spin)
        self.speed_spin.setValue(float(self.bpm_spin.value()) / float(self._detected_bpm))
        del speed_blocker
        self.roll.set_grid_seconds(self._editor_grid_seconds())
        self._apply_result_playback_rate()

    def _on_result_speed_changed(self, speed: float) -> None:
        if self._detected_bpm is None:
            return
        self._last_tempo_editor = "speed"
        if not math.isfinite(float(speed)) or float(speed) <= 0.0:
            raise ValueError(f"Invalid playback speed: {speed!r}")
        target_bpm = float(self._detected_bpm) * float(speed)
        if not MIN_MIDI_BPM <= target_bpm <= MAX_MIDI_BPM:
            raise ValueError(
                f"Playback speed maps outside the supported BPM range: "
                f"speed={speed!r}, target_bpm={target_bpm!r}"
            )
        self._bpm_user_overridden = True
        bpm_blocker = QSignalBlocker(self.bpm_spin)
        self.bpm_spin.setValue(target_bpm)
        del bpm_blocker
        speed_blocker = QSignalBlocker(self.speed_spin)
        self.speed_spin.setValue(float(self.bpm_spin.value()) / float(self._detected_bpm))
        del speed_blocker
        self.roll.set_grid_seconds(self._editor_grid_seconds())
        self._apply_result_playback_rate()

    def _commit_result_tempo_edit(self) -> float:
        """Commit the project BPM used by both export and linked audition."""

        self.bpm_spin.interpretText()
        target_bpm = float(self.bpm_spin.value())
        if not math.isfinite(target_bpm) or not MIN_MIDI_BPM <= target_bpm <= MAX_MIDI_BPM:
            raise RuntimeError(f"Invalid result BPM at download time: {target_bpm!r}")
        return target_bpm

    def _apply_result_playback_rate(self) -> None:
        rate = self._result_playback_rate()
        active_players = self._active_playback_players()
        master_position = (
            active_players[0].position() if self._playing and active_players else self._position_ms
        )
        for player in self._all_playback_players():
            player.setPlaybackRate(rate)
        if self._playing:
            for player in active_players[1:]:
                if abs(player.position() - master_position) > 20:
                    self.seek(master_position / 1000.0)
                    break
            self._position_ms = master_position
            self._playback_clock.reset(master_position)

    def pause(self) -> None:
        """Pause this workbench without changing its current play position."""
        was_playing = self._playing
        self._transport_seek_resume = False
        if was_playing and self._transport_seek_pending_ms is None:
            active_players = self._active_playback_players()
            if active_players:
                self._position_ms = max(0, active_players[0].position())
            self._transport_seek_pending_ms = self._position_ms
        if self._transport_seek_pending_ms is not None:
            self._transport_seek_pause_only = True
        self._playing = False
        self._startup_sync_pending = False
        self._playback_clock.reset(self._position_ms)
        if self._transport_seek_pending_ms is not None:
            self._schedule_transport_seek_commit()
        else:
            self._schedule_deferred_assets()
        self._update_play_label()
        if was_playing:
            self.playing_changed.emit(False)

    def _on_playback_scrub_started(self) -> None:
        self._transport_scrubbing = True

    def _preview_playback_scrub(self, position_ms: int) -> None:
        """Move only the UI during a pointer gesture; do not churn media backends."""

        duration_ms = self.playback_slider.maximum()
        self._position_ms = max(0, min(int(position_ms), duration_ms))
        position = self._position_ms / 1000.0
        self._playback_finished = False
        self.roll.set_position(position)
        self.clock_label.setText(_format_seconds(position))
        if self.follow_checkbox.isChecked():
            self._follow_roll_to_position(position, allow_backward=True)

    def _on_playback_scrub_finished(self, position_ms: int) -> None:
        try:
            # Commit exactly once on release. Repeated setPosition/pause/play calls
            # during mouse movement can deadlock Qt's Windows FFmpeg backend.
            self.seek(position_ms / 1000.0)
        finally:
            self._transport_scrubbing = False

    def seek(self, seconds: float) -> None:
        position = max(0.0, float(seconds))
        duration_ms = self.playback_slider.maximum()
        if duration_ms > 0:
            position = min(position, duration_ms / 1000.0)
        self._position_ms = int(position * 1000)
        self._last_drift_check_position_ms = self._position_ms
        self._playback_finished = False
        self._playback_clock.reset(self._position_ms)
        self.roll.set_position(position)
        self.clock_label.setText(_format_seconds(position))
        self.playback_slider.setValue(self._position_ms)
        if self.follow_checkbox.isChecked():
            self._follow_roll_to_position(position, allow_backward=True)
        if not self._all_playback_players():
            return

        self._transport_seek_pending_ms = self._position_ms
        self._transport_seek_pause_only = False
        if self._playing:
            self._transport_seek_resume = True
            self._startup_sync_pending = False
        self._schedule_transport_seek_commit()

    def _schedule_deferred_assets(self) -> None:
        if self._deferred_apply_scheduled or self._shutting_down:
            return
        self._deferred_apply_scheduled = True
        self._deferred_assets_timer.start(0)

    def _try_apply_deferred_assets(self) -> None:
        if not self._deferred_apply_scheduled:
            return
        if self._shutting_down:
            self._deferred_apply_scheduled = False
            return
        if self._transport_seek_pending_ms is not None:
            return
        if self._playing or any(
            player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            for player in self._all_playback_players()
        ):
            return
        self._deferred_apply_scheduled = False
        self._apply_deferred_assets()

    def _follow_roll_to_position(self, position: float, *, allow_backward: bool) -> None:
        scrollbar = self.roll_scroll.horizontalScrollBar()
        target = self.roll.x_for_time_float(position) - self.roll_scroll.viewport().width() / 2.0
        target = min(scrollbar.maximum(), max(scrollbar.minimum(), target))
        effective_offset = scrollbar.value() + self.roll.render_offset_px
        if not allow_backward and target < effective_offset:
            return
        base_offset = int(target // _ROLL_FOLLOW_SCROLL_BLOCK_PX) * _ROLL_FOLLOW_SCROLL_BLOCK_PX
        base_offset = min(scrollbar.maximum(), max(scrollbar.minimum(), base_offset))
        if base_offset != scrollbar.value():
            scrollbar.setValue(base_offset)
        self.roll.set_render_offset(target - base_offset)

    def _commit_roll_render_offset(self) -> None:
        scrollbar = self.roll_scroll.horizontalScrollBar()
        effective_offset = scrollbar.value() + self.roll.render_offset_px
        self.roll.set_render_offset(0.0)
        scrollbar.setValue(round(effective_offset))

    def _on_roll_manual_navigation(self) -> None:
        self._commit_roll_render_offset()
        if self.follow_checkbox.isChecked():
            self.follow_checkbox.setChecked(False)

    def _on_roll_scroll_action(self, action: int) -> None:
        if action == QAbstractSlider.SliderAction.SliderMove.value:
            return
        self._on_roll_manual_navigation()

    def _on_follow_toggled(self, checked: bool) -> None:
        if checked:
            self._follow_roll_to_position(
                self._playback_clock.sample(
                    self._position_ms,
                    playback_rate=self._result_playback_rate(),
                )
                / 1000.0,
                allow_backward=True,
            )
        else:
            self._commit_roll_render_offset()

    def _on_roll_zoom_requested(self, factor: float, viewport_x: float) -> None:
        if factor <= 0:
            raise ValueError(f"Piano-roll zoom factor must be positive: {factor}")
        following = self.follow_checkbox.isChecked()
        self._commit_roll_render_offset()
        scrollbar = self.roll_scroll.horizontalScrollBar()
        if following:
            anchor_x = self.roll_scroll.viewport().width() / 2.0
            anchor_time = self.roll.position
        else:
            anchor_x = min(
                float(self.roll_scroll.viewport().width()),
                max(0.0, float(viewport_x)),
            )
            anchor_time = self.roll.time_for_x(scrollbar.value() + anchor_x)
        pixels_per_second = min(
            _ROLL_MAX_PIXELS_PER_SECOND,
            max(
                _ROLL_MIN_PIXELS_PER_SECOND,
                self.roll.pixels_per_second * float(factor),
            ),
        )
        self.roll.set_pixels_per_second(pixels_per_second)
        self._sync_roll_zoom_control()
        if following:
            self._follow_roll_to_position(anchor_time, allow_backward=True)
        else:
            target = self.roll.x_for_time_float(anchor_time) - anchor_x
            target = min(scrollbar.maximum(), max(scrollbar.minimum(), target))
            scrollbar.setValue(round(target))

    def _on_roll_zoom_value_changed(self, ratio: float) -> None:
        if self._syncing_roll_zoom_control:
            return
        desired_pixels_per_second = _ROLL_BASE_PIXELS_PER_SECOND * float(ratio)
        factor = desired_pixels_per_second / self.roll.pixels_per_second
        self._on_roll_zoom_requested(
            factor,
            self.roll_scroll.viewport().width() / 2.0,
        )

    def _sync_roll_zoom_control(self) -> None:
        self._syncing_roll_zoom_control = True
        try:
            blocker = QSignalBlocker(self.roll_zoom_spin)
            self.roll_zoom_spin.setValue(self.roll.pixels_per_second / _ROLL_BASE_PIXELS_PER_SECOND)
            del blocker
        finally:
            self._syncing_roll_zoom_control = False

    def _finish_playback_at(self, position_ms: int) -> None:
        terminal_ms = max(0, int(position_ms))
        was_playing = self._playing
        self._playing = False
        self._playback_finished = True
        self._startup_sync_pending = False
        self._position_ms = terminal_ms
        self._transport_seek_pending_ms = terminal_ms
        self._transport_seek_resume = False
        self._transport_seek_pause_only = True
        self._schedule_transport_seek_commit()
        self._playback_clock.reset(terminal_ms)
        position = terminal_ms / 1000.0
        self.roll.set_position(position)
        self.clock_label.setText(_format_seconds(position))
        self.playback_slider.setValue(min(terminal_ms, self.playback_slider.maximum()))
        if self.follow_checkbox.isChecked():
            self._follow_roll_to_position(position, allow_backward=False)
        self._update_play_label()
        if was_playing:
            self.playing_changed.emit(False)

    def _tick(self) -> None:
        if not self._finalizing:
            self._update_stream_progress()
        if self._transport_scrubbing or self._transport_seek_pending_ms is not None:
            return
        if not self._playing or self._midi_normal is None:
            return
        master_pair = (
            self._midi_right
            if self.stereo_checkbox.isChecked() and self._midi_right is not None
            else self._midi_normal
        )
        master = master_pair[0]
        # Local playback cannot legitimately move backwards between timer ticks. Some
        # Windows multimedia backends briefly report an older clock sample; feeding
        # that value into follow-scroll makes the entire piano roll shake left/right.
        self._position_ms = max(self._position_ms, max(0, master.position()))
        authoritative_position = self._position_ms / 1000.0
        if self._preview_duration > 0 and authoritative_position >= self._preview_duration:
            self._finish_playback_at(int(self._preview_duration * 1000))
            self.playback_status_label.setText(
                t(
                    "muscriptor_result.preview_complete",
                    time=_format_clock(self._preview_duration),
                )
            )
            self._schedule_deferred_assets()
            return
        if master.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
            terminal_ms = max(self._position_ms, max(0, master.duration()))
            self._finish_playback_at(terminal_ms)
            if self._preview_duration > 0:
                self._schedule_deferred_assets()
            return
        if self._position_ms - self._last_drift_check_position_ms >= 500:
            for player in self._active_playback_players()[1:]:
                if abs(player.position() - self._position_ms) > 80:
                    self.seek(self._position_ms / 1000.0)
                    return
            self._last_drift_check_position_ms = self._position_ms
        display_ms = self._playback_clock.sample(
            self._position_ms,
            playback_rate=self._result_playback_rate(),
        )
        if self._preview_duration > 0:
            display_ms = min(display_ms, self._preview_duration * 1000.0)
        position = display_ms / 1000.0
        self.roll.set_position(position)
        self.clock_label.setText(_format_seconds(position))
        self.playback_slider.setValue(min(self._position_ms, self.playback_slider.maximum()))
        if self.follow_checkbox.isChecked():
            self._follow_roll_to_position(position, allow_backward=False)

    def _update_stream_progress(self) -> None:
        if not self._progress_estimator.active or self._progress_total <= 0:
            return
        now = time.monotonic()
        fraction = self._progress_estimator.fraction(now=now)
        self.progress_bar.setValue(int(round(fraction * 1000)))
        total_duration = self._effective_source_duration()
        processed = fraction * total_duration
        elapsed = self._progress_estimator.elapsed_seconds(now=now)
        eta = self._progress_estimator.eta_seconds(now=now)
        if eta is None:
            key = "muscriptor_result.progress_timing_pending"
            values = {
                "processed": _format_clock(processed),
                "total": _format_clock(total_duration),
                "elapsed": _format_clock(elapsed),
            }
        else:
            key = "muscriptor_result.progress_timing"
            values = {
                "processed": _format_clock(processed),
                "total": _format_clock(total_duration),
                "elapsed": _format_clock(elapsed),
                "remaining": _format_clock(eta),
            }
        self.progress_label.setText(t(key, **values))
        self._update_slow_conversion_hint(elapsed=elapsed, processed=processed)

    def _update_slow_conversion_hint(self, *, elapsed: float, processed: float) -> None:
        if self._finalizing or elapsed < _SLOW_CONVERSION_THRESHOLD_SECONDS:
            self.slow_hint_label.hide()
            return

        model = str(self._runtime_details.get("model") or self.backend_label)
        device = str(self._runtime_details.get("device") or "unknown")
        gpu = str(self._runtime_details.get("gpu") or device)
        precision = str(self._runtime_details.get("compute_dtype") or "unknown")
        completed = max(0, int(self._progress_completed))
        total = max(1, int(self._progress_total))
        seconds_per_chunk = self._progress_estimator.ema_chunk_seconds
        if seconds_per_chunk is None:
            seconds_per_chunk = elapsed / max(1, completed)
        realtime_factor = elapsed / max(_STREAM_CHUNK_SECONDS, processed)
        values = {
            "model": model,
            "gpu": gpu,
            "device": device,
            "precision": precision,
            "completed": completed,
            "total": total,
            "seconds_per_chunk": f"{seconds_per_chunk:.1f}",
            "realtime_factor": f"{realtime_factor:.1f}",
        }
        if device.lower().startswith("cpu"):
            key = "muscriptor_result.slow_hint_cpu"
        elif "large" in model.lower():
            key = "muscriptor_result.slow_hint_large"
        else:
            key = "muscriptor_result.slow_hint_check"
        message = t(key, **values)
        if self.slow_hint_label.text() != message:
            self.slow_hint_label.setText(message)
        self.slow_hint_label.show()

    def _apply_mix(self, *_args) -> None:
        if self._original_normal is None:
            return
        stereo_available = (
            self._original_left is not None
            and self._midi_right is not None
            and bool(self._right_sources)
        )
        stereo = self.stereo_checkbox.isChecked() and stereo_available
        mix = self.mix_slider.value() / 100.0
        use_instrument_stems = bool(self._muted)
        if use_instrument_stems:
            self._ensure_instrument_players(stereo=stereo)
        self.mix_slider.setEnabled(bool(self._normal_sources) and not stereo)
        self._original_normal[1].setVolume(0.0 if stereo else 1.0 - mix)
        if self._original_left is not None:
            self._original_left[1].setVolume(1.0 if stereo else 0.0)
        if self._midi_normal is not None:
            self._midi_normal[1].setVolume(mix if not stereo and not use_instrument_stems else 0.0)
        if self._midi_right is not None:
            self._midi_right[1].setVolume(1.0 if stereo and not use_instrument_stems else 0.0)
        for instrument in self._normal_sources:
            audible = instrument not in self._muted
            if instrument in self._normal_players:
                self._normal_players[instrument][1].setVolume(
                    mix if audible and not stereo and use_instrument_stems else 0.0
                )
            if instrument in self._right_sources and instrument in self._right_players:
                self._right_players[instrument][1].setVolume(
                    1.0 if audible and stereo and use_instrument_stems else 0.0
                )
        self._sync_active_players()

    def _toggle_mute(self, instrument: str) -> None:
        self._soloed = None
        if instrument in self._muted:
            self._muted.remove(instrument)
        else:
            self._muted.add(instrument)
        self._sync_instrument_controls()

    def _toggle_solo(self, instrument: str) -> None:
        if self._soloed == instrument:
            self._soloed = None
            self._muted.clear()
        else:
            self._soloed = instrument
            self._muted = set(self._detected) - {instrument}
        self._sync_instrument_controls()

    def _sync_instrument_controls(self) -> None:
        for instrument, row in self._instrument_rows.items():
            muted = instrument in self._muted
            row.set_muted(muted)
            row.set_soloed(self._soloed == instrument)
            self.roll.set_instrument_muted(instrument, muted)
        self._apply_mix()

    def _clear_instrument_rows(self) -> None:
        while self.instrument_rows_layout.count():
            item = self.instrument_rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._instrument_rows.clear()

    def _rebuild_instrument_rows(self) -> None:
        self._clear_instrument_rows()
        ordered = (
            list(self.selected_instruments) if self.selected_instruments else list(self._detected)
        )
        for instrument in self._detected:
            if instrument not in ordered:
                ordered.append(instrument)
        self.roll.set_instrument_order(ordered)
        for index, instrument in enumerate(ordered):
            detected = instrument in self._detected
            row = _InstrumentRow(
                instrument,
                detected=detected,
                color=_INSTRUMENT_COLORS[index % len(_INSTRUMENT_COLORS)],
                parent=self,
            )
            row.mute_toggled.connect(self._toggle_mute)
            row.solo_toggled.connect(self._toggle_solo)
            self.instrument_rows_layout.addWidget(row)
            self._instrument_rows[instrument] = row
        self._sync_instrument_controls()

    def _save_asset(self, kind: str) -> None:
        target_bpm: float | None = None
        if kind == "midi":
            source = Path(self._midi_path)
            target_bpm = self._commit_result_tempo_edit()
            edited = bool(self._original_edit_notes) and (
                self._edited_notes != self._original_edit_notes
            )
            edit_suffix = "_edited" if edited else ""
            filter_text = "MIDI (*.mid)"
            suggested_name = f"{source.stem}{edit_suffix}_{target_bpm:.1f}BPM{source.suffix}"
        elif kind == "transcription" and self._assets is not None:
            source = self._assets.transcription_wav
            filter_text = "WAV (*.wav)"
            suggested_name = source.name
        elif kind == "stereo" and self._assets is not None:
            source = self._assets.stereo_mix_wav
            filter_text = "WAV (*.wav)"
            suggested_name = source.name
        else:
            raise RuntimeError(f"MuScriptor download asset is not ready: {kind}")
        destination, _selected_filter = QFileDialog.getSaveFileName(
            self,
            t("muscriptor_result.download"),
            suggested_name,
            filter_text,
        )
        if destination:
            if kind == "midi":
                if target_bpm is None:
                    raise RuntimeError("Result BPM was not committed before MIDI download")
                try:
                    if edited:
                        if self._detected_bpm is None:
                            raise RuntimeError(
                                "Detected/reference BPM is unavailable for edited MIDI export"
                            )
                        published_path = export_edited_midi(
                            source,
                            destination,
                            self._edited_notes,
                            reference_bpm=self._detected_bpm,
                            target_bpm=target_bpm,
                        )
                    else:
                        if self._detected_bpm is None:
                            raise RuntimeError(
                                "Detected/reference BPM is unavailable for MIDI export"
                            )
                        published_path = _export_midi_at_project_speed(
                            source,
                            destination,
                            self._detected_bpm,
                            target_bpm,
                        )
                    if self.muscriptor_groups:
                        from src.core.midi_tempo import repeat_tempo_events_on_note_tracks

                        published_path = repeat_tempo_events_on_note_tracks(
                            published_path,
                            label="MuScriptor exported MuseScore tempo metadata",
                        )
                except Exception as exc:
                    self.playback_status_label.setText(
                        t(
                            "muscriptor_result.editor_export_failed",
                            error=_compact_editor_error(exc),
                        )
                    )
                    raise
                self.playback_status_label.setText(
                    t(
                        (
                            "muscriptor_result.edited_midi_saved"
                            if edited
                            else "muscriptor_result.midi_saved"
                        ),
                        bpm=f"{target_bpm:.1f}",
                        path=str(published_path),
                    )
                )
            else:
                shutil.copy2(source, destination)

    def _update_play_label(self) -> None:
        standard_icon = (
            QStyle.StandardPixmap.SP_MediaPause
            if self._playing
            else QStyle.StandardPixmap.SP_MediaPlay
        )
        self.play_button.setIcon(self.style().standardIcon(standard_icon))
        self.play_button.setText(
            t("muscriptor_result.pause") if self._playing else t("muscriptor_result.play")
        )

    def update_translations(self) -> None:
        self.source_label.setText(
            t(
                "muscriptor_result.linked_source",
                track=self.source_track_name,
                backend=self.backend_label,
            )
            if self.source_track_name
            else ""
        )
        self._update_play_label()
        self.follow_checkbox.setText(t("muscriptor_result.follow"))
        self.playback_slider.setToolTip(t("muscriptor_result.playback_progress_tooltip"))
        self.bpm_spin.setToolTip(t("muscriptor_result.export_bpm_tooltip"))
        self.speed_label.setText(t("muscriptor_result.playback_speed_label"))
        self.speed_spin.setToolTip(t("muscriptor_result.playback_speed_tooltip"))
        self.edit_toggle.setText(t("muscriptor_result.editor_toggle"))
        self.edit_toggle.setToolTip(t("muscriptor_result.editor_help"))
        self.edit_add_button.setText(t("muscriptor_result.editor_add"))
        self.edit_delete_button.setText(t("muscriptor_result.editor_delete"))
        self.edit_undo_button.setText(t("muscriptor_result.editor_undo"))
        self.edit_redo_button.setText(t("muscriptor_result.editor_redo"))
        self.edit_reset_button.setText(t("muscriptor_result.editor_reset"))
        self.edit_select_all_button.setText(t("muscriptor_result.editor_select_all"))
        self.edit_cut_button.setText(t("muscriptor_result.editor_cut"))
        self.edit_copy_button.setText(t("muscriptor_result.editor_copy"))
        self.edit_paste_button.setText(t("muscriptor_result.editor_paste"))
        self.edit_duplicate_button.setText(t("muscriptor_result.editor_duplicate"))
        self.edit_quantize_button.setText(t("muscriptor_result.editor_quantize"))
        self.edit_instrument_label.setText(t("muscriptor_result.editor_instrument"))
        self.edit_velocity_label.setText(t("muscriptor_result.editor_velocity"))
        self.roll_zoom_label.setText(t("muscriptor_result.editor_view_zoom"))
        self.roll_zoom_spin.setToolTip(t("muscriptor_result.editor_view_zoom_tooltip"))
        self.result_splitter.handle(1).setToolTip(t("muscriptor_result.editor_resize_hint"))
        self.original_label.setText(t("muscriptor_result.original"))
        self.stereo_checkbox.setText(t("muscriptor_result.stereo"))
        self.instruments_title.setText(t("muscriptor_result.instruments"))
        self.download_button.setText(t("muscriptor_result.download"))
        self.download_midi_action.setText(t("muscriptor_result.export_edited_midi"))
        self.download_transcription_action.setText(t("muscriptor_result.download_transcription"))
        self.download_stereo_action.setText(t("muscriptor_result.download_stereo"))
        self.another_button.setText(
            t("muscriptor_result.close_detail")
            if self.source_track_name
            else t("muscriptor_result.another")
        )
        for row in self._instrument_rows.values():
            row.update_translations()
        self._sync_editor_controls()

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self._preview_pending = None
        self._edit_asset_pending = None
        self._deferred_preview = None
        self._deferred_final_assets = None
        self._deferred_editor_assets = None
        self._deferred_apply_scheduled = False
        self.timer.stop()
        self._edit_asset_debounce.stop()
        self._transport_commit_timer.stop()
        self._after_transport_timer.stop()
        self._deferred_assets_timer.stop()
        self._retired_media_timer.stop()
        self._playing = False
        self._transport_scrubbing = False
        self._transport_seek_pending_ms = None
        self._transport_seek_resume = False
        self._transport_seek_pause_only = False
        self._transport_seek_commit_scheduled = False
        self._transport_seek_commit_players = ()
        self._transport_seek_commit_index = 0
        self._transport_seek_commit_phase = ""

        # QThread objects must finish before their parent and the media objects
        # they publish into are destroyed. Waiting after media teardown leaves
        # queued worker signals targeting a half-destroyed workbench.
        preview_worker = self._preview_worker
        self._preview_worker = None
        if preview_worker is not None and preview_worker.isRunning():
            preview_worker.cancel()
            preview_worker.wait()
        asset_worker = self._asset_worker
        self._asset_worker = None
        if asset_worker is not None and asset_worker.isRunning():
            asset_worker.cancel()
            # QThread must outlive its run() method.  The SoundFont download is
            # not interruptible inside huggingface_hub, so a timed wait could
            # destroy a still-running worker and crash Qt during result reset.
            asset_worker.wait()
        edit_asset_worker = self._edit_asset_worker
        self._edit_asset_worker = None
        if edit_asset_worker is not None and edit_asset_worker.isRunning():
            edit_asset_worker.cancel()
            edit_asset_worker.wait()

        active_media = [(player, player.audioOutput()) for player in self._all_playback_players()]
        self._players.clear()
        self._retired_media.extend(active_media)
        self._release_retired_media()
        self._active_player_ids = frozenset()
        self._startup_sync_pending = False
        self._normal_sources.clear()
        self._right_sources.clear()
        self._normal_players.clear()
        self._right_players.clear()
        self._midi_normal = None
        self._midi_right = None
        self._original_normal = None
        self._original_left = None
        if self._preview_root.exists():
            try:
                shutil.rmtree(self._preview_root)
            except OSError as exc:
                logger.warning(
                    "Unable to remove MuScriptor preview directory %s: %s",
                    self._preview_root,
                    exc,
                )
        if self._edit_asset_root.exists():
            try:
                shutil.rmtree(self._edit_asset_root)
            except OSError as exc:
                logger.warning(
                    "Unable to remove edited MIDI audio directory %s: %s",
                    self._edit_asset_root,
                    exc,
                )
