"""Inline project-native MIDI piano roll, mixer, and exports."""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
import time
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

from PyQt6.QtCore import QLineF, QRectF, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QCheckBox,
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
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.core.muscriptor_result_assets import (
    MuscriptorPlaybackAssets,
    MuscriptorPreviewAssets,
    MuscriptorRollNote,
    prepare_midi_playback_assets,
    prepare_midi_preview_assets,
    read_midi_roll_notes,
)
from src.gui.theme import DARK_FILE_DIALOG_OPTIONS
from src.i18n.translator import get_translator, t
from src.models.data_models import ProcessingResult
from src.models.gm_instruments import get_instrument_name
from src.models.muscriptor_instruments import (
    MUSCRIPTOR_REPRESENTATIVE_PROGRAMS,
    muscriptor_instrument_label,
)

_INSTRUMENT_COLORS = (
    "#4a9eff",
    "#ff8d66",
    "#73a7ff",
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
_PLAYHEAD_TIMER_MS = 16
_PLAYHEAD_MAX_LEAD_MS = 120.0
_ROLL_FOLLOW_SCROLL_BLOCK_PX = 12
_ROLL_MIN_PIXELS_PER_SECOND = 46.0
_ROLL_MAX_PIXELS_PER_SECOND = 368.0
_ROLL_ZOOM_STEP = 1.15
_ROLL_WHEEL_STEP_PX = 96
_ROLL_TILE_WIDTH = 512
_ROLL_TILE_CACHE_LIMIT = 8

logger = logging.getLogger(__name__)


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
            if sample <= 0:
                raise ValueError(f"Invalid MuScriptor chunk duration sample: {sample}")
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

    def sample(self, reported_ms: float, *, now: float | None = None) -> float:
        current = time.monotonic() if now is None else float(now)
        reported = max(0.0, float(reported_ms))
        elapsed_ms = max(0.0, (current - self._sampled_at) * 1000.0)
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


class _PianoRollCanvas(QWidget):
    seek_requested = pyqtSignal(float)

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
        self._tile_cache: OrderedDict[tuple[int, int], QPixmap] = OrderedDict()
        self._pixels_per_second = 92.0
        self._keyboard_width = 72
        self._row_height = 7
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setMinimumHeight(88 * self._row_height)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._update_size()

    def set_notes(
        self,
        notes: Iterable[MuscriptorRollNote],
        *,
        duration: float | None = None,
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
        instruments = list(dict.fromkeys(note.instrument for note in normalized))
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
        self._tile_cache.clear()
        self._update_size()
        self.update()

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

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.position().x() + self._render_offset_px >= self._keyboard_width
        ):
            seconds = self.time_for_x(event.position().x() + self._render_offset_px)
            self.seek_requested.emit(max(0.0, seconds))
        super().mousePressEvent(event)


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
        self._asset_worker: _AssetWorker | None = None
        self._preview_worker: _PreviewAssetWorker | None = None
        self._deferred_preview: tuple[int, MuscriptorPreviewAssets] | None = None
        self._deferred_final_assets: MuscriptorPlaybackAssets | None = None
        self._deferred_apply_scheduled = False
        self._preview_pending: (
            tuple[int, tuple[MuscriptorRollNote, ...], float, int, int] | None
        ) = None
        self._preview_root = Path(tempfile.mkdtemp(prefix="music-to-midi-midi-preview-"))
        self._preview_generation = 0
        self._preview_ready_generation = 0
        self._preview_applied_generation = 0
        self._preview_note_count = 0
        self._preview_duration = 0.0
        self._preview_error: str | None = None
        self._finalizing = False
        self._shutting_down = False
        self._progress_estimator = _ChunkProgressEstimator()
        self._progress_completed = 0
        self._progress_total = 0
        self._source_duration_seconds = 0.0
        self._position_ms = 0
        self._playback_clock = _SmoothPlaybackClock()
        self._playing = False
        self._playback_finished = False
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
        self._midi_path = ""
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

        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        self.source_label.setVisible(bool(self.source_track_name))
        self.source_label.setStyleSheet(
            "color: #8fc6ff; font-weight: 600; background: #122039; "
            "border: 1px solid #2c4f7c; border-radius: 5px; padding: 7px 9px;"
        )
        root.addWidget(self.source_label)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #9fb3d9;")
        root.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background: #223451; border: 0; border-radius: 4px; } "
            "QProgressBar::chunk { background: #4a9eff; border-radius: 4px; }"
        )
        root.addWidget(self.progress_bar)

        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("color: #8da4c9;")
        root.addWidget(self.progress_label)

        self.playback_status_label = QLabel()
        self.playback_status_label.setWordWrap(True)
        self.playback_status_label.setStyleSheet("color: #73a7ff;")
        root.addWidget(self.playback_status_label)

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
        controls.addStretch()
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
        root.addLayout(controls)

        content = QHBoxLayout()
        self.roll_scroll = _PianoRollScrollArea()
        self.roll_scroll.setWidgetResizable(False)
        self.roll_scroll.setMinimumHeight(390)
        self.roll_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.roll = _PianoRollCanvas()
        self.roll.seek_requested.connect(self.seek)
        self.roll_scroll.setWidget(self.roll)
        self.roll_scroll.horizontalScrollBar().setSingleStep(48)
        self.roll_scroll.zoom_requested.connect(self._on_roll_zoom_requested)
        self.roll_scroll.manual_navigation_requested.connect(self._on_roll_manual_navigation)
        self.roll_scroll.horizontalScrollBar().actionTriggered.connect(
            lambda _action: self._on_roll_manual_navigation()
        )
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
        root.addLayout(content)

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
        if event_type in {"note_end", "note_batch"}:
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
        self._preview_pending = None
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_worker.cancel()
        self._midi_path = str(Path(result.midi_path).resolve())
        read_midi_roll_notes(
            self._midi_path,
            muscriptor_groups=self.muscriptor_groups,
        )
        if self.muscriptor_groups:
            self.selected_instruments = list(result.selected_instruments)
        output_dir = Path(self._midi_path).parent / "midi-playback"
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
            self._midi_path,
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
        self._preview_pending = None
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_worker.cancel()
        self.status_label.setText(t("transcription_result.failed", error=error))
        preview_available = self._midi_normal is not None and self._original_normal is not None
        self.play_button.setEnabled(preview_available)
        self.mix_slider.setEnabled(preview_available)

    def mark_cancelled(self) -> None:
        self._finalizing = True
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
        self._preview_generation += 1
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
        if self._playing:
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
        if self._playing:
            raise RuntimeError("Cannot replace MIDI preview assets during playback")
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
        for player in self._all_playback_players():
            player.setPosition(self._position_ms)

    def _apply_deferred_assets(self) -> None:
        if self._playing or self._shutting_down:
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

        # Silence retired outputs synchronously. Their FFmpeg backends are released
        # on the next event-loop turn, but no stale MIDI tail may remain audible in
        # that interval. Do not issue another pause here: preview rollover already
        # paused the active set, and an overlapping pause/stop/source-clear sequence
        # can crash Qt's Windows FFmpeg backend.
        for player in retired_players:
            output = player.audioOutput()
            if output is not None:
                output.setMuted(True)

        # Qt FFmpeg can deadlock when QMediaPlayer.stop() is called from the
        # same timer callback that observed EndOfMedia and committed the next
        # preview. Remove the old players from all live routing immediately,
        # then release their backends on the next event-loop turn. Let QObject
        # destruction detach the source/output in its normal order instead of
        # manually changing both relationships while stop notifications remain
        # queued.
        def release_retired_players() -> None:
            for player in retired_players:
                player.stop()
                output = player.audioOutput()
                if output is not None:
                    output.deleteLater()
                player.deleteLater()

        QTimer.singleShot(0, release_retired_players)

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
        player.errorOccurred.connect(
            lambda _error, message: self.playback_status_label.setText(
                t("muscriptor_result.player_failed", error=message)
            )
        )
        player.playbackStateChanged.connect(self._on_player_playback_state_changed)
        self._players.append(player)
        return player, output

    def _on_player_playback_state_changed(self, _state) -> None:
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
                player.setPosition(master_position)
        self._startup_sync_pending = False

    def _on_assets_ready(self, assets: object) -> None:
        if self._shutting_down:
            return
        if not isinstance(assets, MuscriptorPlaybackAssets):
            self._on_assets_failed("Invalid MuScriptor playback asset payload")
            return
        if self._playing:
            self._deferred_final_assets = assets
            self.playback_status_label.setText(t("muscriptor_result.final_audio_buffered"))
            return
        try:
            self._apply_final_assets(assets)
        except Exception as exc:
            logger.exception("Failed to apply final MIDI assets")
            self._on_assets_failed(str(exc))

    def _apply_final_assets(self, assets: MuscriptorPlaybackAssets) -> None:
        if self._playing:
            raise RuntimeError("Cannot replace final MIDI assets during playback")
        self.roll.set_notes(assets.notes, duration=assets.duration)
        position_ms = self._position_ms
        self._dispose_dynamic_players()
        self._assets = assets
        self._ensure_original_player()
        self._original_left = self._make_player(assets.original_left_wav)
        self._midi_normal = self._make_player(assets.live_transcription_wav)
        self._midi_right = self._make_player(assets.transcription_right_wav)
        self._normal_sources = dict(assets.instrument_wavs)
        self._right_sources = dict(assets.instrument_right_wavs)
        self._detected = list(dict.fromkeys(note.instrument for note in assets.notes))
        self._rebuild_instrument_rows()
        self.play_button.setEnabled(True)
        self.mix_slider.setEnabled(True)
        self.stereo_checkbox.setEnabled(True)
        self.download_transcription_action.setEnabled(True)
        self.download_stereo_action.setEnabled(True)
        self.status_label.setText(t("muscriptor_result.ready"))
        self.playback_status_label.setText(t("muscriptor_result.final_audio_ready"))
        self._preview_duration = 0.0
        self._position_ms = position_ms
        self._playback_finished = False
        self._playback_clock.reset(self._position_ms)
        self._apply_mix()
        self._update_play_label()

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
        pairs: list[tuple[QMediaPlayer, QAudioOutput]] = []
        if original is not None:
            pairs.append(original)
        if self._muted:
            pairs.extend(
                pair
                for instrument, pair in instrument_players.items()
                if instrument not in self._muted
            )
        elif midi_mix is not None:
            pairs.append(midi_mix)
        return pairs

    def _active_playback_players(self) -> list[QMediaPlayer]:
        return [player for player, _output in self._active_playback_pairs()]

    def _sync_active_players(self) -> None:
        active = self._active_playback_players()
        next_ids = frozenset(id(player) for player in active)
        if next_ids == self._active_player_ids:
            return
        for player in self._all_playback_players():
            if id(player) not in next_ids:
                player.pause()
        if self._playing and len(active) > 1:
            self._startup_sync_pending = True
        for player in active:
            if id(player) in self._active_player_ids:
                continue
            player.setPosition(self._position_ms)
            if self._playing:
                player.play()
        self._active_player_ids = next_ids

    def _toggle_playback(self) -> None:
        if self._original_normal is None or self._midi_normal is None:
            raise RuntimeError("MuScriptor playable audio is not ready")
        self._apply_mix()
        active_players = self._active_playback_players()
        if self._playing:
            self.pause()
        else:
            if self._playback_finished:
                self._position_ms = 0
                self._playback_finished = False
            elif (
                not self._finalizing
                and self._preview_duration > 0
                and self._position_ms >= int(self._preview_duration * 1000) - 30
            ):
                self._position_ms = 0
            self._playing = True
            self._startup_sync_pending = len(active_players) > 1
            self._playback_clock.reset(self._position_ms)
            for player in active_players:
                player.setPosition(self._position_ms)
                player.play()
            self.playing_changed.emit(True)
        self._update_play_label()

    def pause(self) -> None:
        """Pause this workbench without changing its current play position."""
        was_playing = self._playing
        for player in self._all_playback_players():
            player.pause()
        self._playing = False
        self._startup_sync_pending = False
        self._playback_clock.reset(self._position_ms)
        self._schedule_deferred_assets()
        self._update_play_label()
        if was_playing:
            self.playing_changed.emit(False)

    def seek(self, seconds: float) -> None:
        position = max(0.0, float(seconds))
        if not self._finalizing and self._preview_duration > 0:
            position = min(position, self._preview_duration)
        self._position_ms = int(position * 1000)
        self._playback_finished = False
        self._playback_clock.reset(self._position_ms)
        for player in self._all_playback_players():
            player.setPosition(self._position_ms)
        self.roll.set_position(position)
        self.clock_label.setText(_format_seconds(position))
        if self.follow_checkbox.isChecked():
            self._follow_roll_to_position(position, allow_backward=True)

    def _schedule_deferred_assets(self) -> None:
        if self._deferred_apply_scheduled or self._shutting_down:
            return
        self._deferred_apply_scheduled = True

        def apply_after_players_settle() -> None:
            self._deferred_apply_scheduled = False
            self._apply_deferred_assets()

        # QMediaPlayer pause is asynchronous on Qt's Windows FFmpeg backend.
        # Commit a buffered preview/final asset on the following event-loop turn so
        # its backend is never cleared while that pause transition is still active.
        QTimer.singleShot(0, apply_after_players_settle)

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

    def _on_follow_toggled(self, checked: bool) -> None:
        if checked:
            self._follow_roll_to_position(
                self._playback_clock.sample(self._position_ms) / 1000.0,
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
        if following:
            self._follow_roll_to_position(anchor_time, allow_backward=True)
        else:
            target = self.roll.x_for_time_float(anchor_time) - anchor_x
            target = min(scrollbar.maximum(), max(scrollbar.minimum(), target))
            scrollbar.setValue(round(target))

    def _finish_playback_at(self, position_ms: int) -> None:
        terminal_ms = max(0, int(position_ms))
        was_playing = self._playing
        for player in self._all_playback_players():
            player.pause()
        self._playing = False
        self._playback_finished = True
        self._startup_sync_pending = False
        self._position_ms = terminal_ms
        self._playback_clock.reset(terminal_ms)
        position = terminal_ms / 1000.0
        self.roll.set_position(position)
        self.clock_label.setText(_format_seconds(position))
        if self.follow_checkbox.isChecked():
            self._follow_roll_to_position(position, allow_backward=False)
        self._update_play_label()
        if was_playing:
            self.playing_changed.emit(False)

    def _tick(self) -> None:
        if not self._finalizing:
            self._update_stream_progress()
        if not self._playing or self._original_normal is None:
            return
        master_pair = (
            self._original_left
            if self.stereo_checkbox.isChecked() and self._original_left is not None
            else self._original_normal
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
        display_ms = self._playback_clock.sample(self._position_ms)
        if self._preview_duration > 0:
            display_ms = min(display_ms, self._preview_duration * 1000.0)
        position = display_ms / 1000.0
        self.roll.set_position(position)
        self.clock_label.setText(_format_seconds(position))
        if self.follow_checkbox.isChecked():
            self._follow_roll_to_position(position, allow_backward=False)
        if master.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
            terminal_ms = max(self._position_ms, max(0, master.duration()))
            self._finish_playback_at(terminal_ms)

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
        if kind == "midi":
            source = Path(self._midi_path)
            filter_text = "MIDI (*.mid)"
        elif kind == "transcription" and self._assets is not None:
            source = self._assets.transcription_wav
            filter_text = "WAV (*.wav)"
        elif kind == "stereo" and self._assets is not None:
            source = self._assets.stereo_mix_wav
            filter_text = "WAV (*.wav)"
        else:
            raise RuntimeError(f"MuScriptor download asset is not ready: {kind}")
        destination, _selected_filter = QFileDialog.getSaveFileName(
            self,
            t("muscriptor_result.download"),
            str(source.name),
            filter_text,
            options=DARK_FILE_DIALOG_OPTIONS,
        )
        if destination:
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
        self.original_label.setText(t("muscriptor_result.original"))
        self.stereo_checkbox.setText(t("muscriptor_result.stereo"))
        self.instruments_title.setText(t("muscriptor_result.instruments"))
        self.download_button.setText(t("muscriptor_result.download"))
        self.download_midi_action.setText(t("muscriptor_result.download_midi"))
        self.download_transcription_action.setText(t("muscriptor_result.download_transcription"))
        self.download_stereo_action.setText(t("muscriptor_result.download_stereo"))
        self.another_button.setText(
            t("muscriptor_result.close_detail")
            if self.source_track_name
            else t("muscriptor_result.another")
        )
        for row in self._instrument_rows.values():
            row.update_translations()

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self._preview_pending = None
        self._deferred_preview = None
        self._deferred_final_assets = None
        self.timer.stop()
        for player in self._all_playback_players():
            player.stop()
            player.setSource(QUrl())
            output = player.audioOutput()
            player.setAudioOutput(None)
            if output is not None:
                output.deleteLater()
            player.deleteLater()
        self._players.clear()
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
        if self._preview_root.exists():
            try:
                shutil.rmtree(self._preview_root)
            except OSError as exc:
                logger.warning(
                    "Unable to remove MuScriptor preview directory %s: %s",
                    self._preview_root,
                    exc,
                )
