"""True PCM waveform decoding and interactive timeline widgets."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt, QThread, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygon,
    QWheelEvent,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from src.i18n.translator import t
from src.utils.runtime_paths import get_ffmpeg_executable

_PCM_SAMPLE_RATE = 8_000
_DEFAULT_SAMPLES_PER_BUCKET = 256
_READ_SIZE_BYTES = 64 * 1024
_PCM_CHANNELS = 2
_PCM_FRAME_SIZE = np.dtype("<f4").itemsize * _PCM_CHANNELS
_TERMINATE_WAIT_MS = 1_500
_KILL_WAIT_MS = 3_500
_BACKGROUND_COLOR = QColor("#101a32")
_RULER_BACKGROUND_COLOR = QColor("#0e1729")
_GRID_COLOR = QColor("#2b3f5f")
_TEXT_COLOR = QColor("#aebbd0")
_ERROR_COLOR = QColor("#ff8fa8")
_PLAYHEAD_COLOR = QColor("#ff6b6b")


@dataclass(frozen=True)
class WaveformEnvelope:
    """Min/max PCM buckets decoded from one real audio stream."""

    minimums: np.ndarray
    maximums: np.ndarray
    sample_rate: int
    frame_count: int
    samples_per_bucket: int

    def __post_init__(self) -> None:
        minimums = np.asarray(self.minimums, dtype=np.float32).reshape(-1).copy()
        maximums = np.asarray(self.maximums, dtype=np.float32).reshape(-1).copy()
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if self.samples_per_bucket <= 0:
            raise ValueError("samples_per_bucket must be positive")
        if minimums.size == 0 or maximums.size == 0:
            raise ValueError("waveform envelope cannot be empty")
        if minimums.shape != maximums.shape:
            raise ValueError("minimums and maximums must have the same shape")
        if not np.isfinite(minimums).all() or not np.isfinite(maximums).all():
            raise ValueError("waveform envelope contains non-finite PCM values")
        if np.any(minimums > maximums):
            raise ValueError("waveform envelope minimum exceeds maximum")

        minimums.setflags(write=False)
        maximums.setflags(write=False)
        object.__setattr__(self, "minimums", minimums)
        object.__setattr__(self, "maximums", maximums)

    @property
    def bucket_count(self) -> int:
        return int(self.minimums.size)

    @property
    def duration_ms(self) -> int:
        return max(1, round(self.frame_count * 1000.0 / self.sample_rate))

    @property
    def bucket_duration_ms(self) -> float:
        return self.samples_per_bucket * 1000.0 / self.sample_rate


class FfmpegWaveformLoader(QThread):
    """Decode an audio file into a bounded real-PCM envelope off the GUI thread."""

    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        parent=None,
        *,
        samples_per_bucket: int = _DEFAULT_SAMPLES_PER_BUCKET,
    ) -> None:
        super().__init__(parent)
        if samples_per_bucket <= 0:
            raise ValueError("samples_per_bucket must be positive")
        self._samples_per_bucket = int(samples_per_bucket)
        self._path: Path | None = None
        self._cancel_requested = threading.Event()
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._terminal_lock = threading.Lock()
        self._terminal_emitted = False

    def start(self, path: Path) -> None:  # type: ignore[override]
        """Begin decoding ``path``; completion is reported only by signals."""
        if self.isRunning():
            raise RuntimeError("waveform decoding is already running")
        self._path = Path(path).expanduser()
        with self._terminal_lock:
            self._cancel_requested.clear()
            self._terminal_emitted = False
        super().start()

    def cancel(self) -> None:
        """Cancel decoding and synchronously make the thread safe to delete.

        FFmpeg first receives a terminate request.  The call can wait up to five
        seconds in the exceptional case where FFmpeg must then be force-killed.
        """
        with self._terminal_lock:
            self._cancel_requested.set()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

        if not self.isRunning() or QThread.currentThread() is self:
            return
        if self.wait(_TERMINATE_WAIT_MS):
            return

        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        if self.wait(_KILL_WAIT_MS):
            return

        message = "FFmpeg waveform decoding thread did not stop after force termination"
        self._emit_failed(message)
        raise RuntimeError(message)

    def run(self) -> None:
        path = self._path
        if path is None:
            self._emit_failed("No audio path was provided for waveform decoding")
            return
        if self._cancel_requested.is_set():
            return
        if not path.is_file():
            self._emit_failed(f"Audio file does not exist: {path}")
            return

        try:
            envelope = self._decode(path.resolve())
        except Exception as exc:
            if not self._cancel_requested.is_set():
                self._emit_failed(f"Unable to decode waveform for {path}: {exc}")
            return

        if envelope is not None and not self._cancel_requested.is_set():
            self._emit_loaded(envelope)

    def _emit_loaded(self, envelope: WaveformEnvelope) -> None:
        with self._terminal_lock:
            if self._terminal_emitted or self._cancel_requested.is_set():
                return
            self._terminal_emitted = True
        self.loaded.emit(envelope)

    def _emit_failed(self, message: str) -> None:
        with self._terminal_lock:
            if self._terminal_emitted:
                return
            self._terminal_emitted = True
        self.failed.emit(message)

    def _decode(self, path: Path) -> WaveformEnvelope | None:
        command = [
            get_ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            str(_PCM_CHANNELS),
            "-ar",
            str(_PCM_SAMPLE_RATE),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ]
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        minimum_parts: list[np.ndarray] = []
        maximum_parts: list[np.ndarray] = []
        frame_carry = np.empty((0, _PCM_CHANNELS), dtype=np.float32)
        byte_carry = b""
        frame_count = 0

        with tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                bufsize=0,
                creationflags=creation_flags,
            )
            with self._process_lock:
                self._process = process

            try:
                if process.stdout is None:
                    raise RuntimeError("FFmpeg did not expose its PCM output stream")
                while not self._cancel_requested.is_set():
                    chunk = process.stdout.read(_READ_SIZE_BYTES)
                    if not chunk:
                        break
                    raw = byte_carry + chunk
                    usable_size = len(raw) - (len(raw) % _PCM_FRAME_SIZE)
                    byte_carry = raw[usable_size:]
                    if usable_size == 0:
                        continue

                    frames = np.frombuffer(raw[:usable_size], dtype="<f4").reshape(
                        -1,
                        _PCM_CHANNELS,
                    )
                    if not np.isfinite(frames).all():
                        raise RuntimeError("FFmpeg produced non-finite PCM samples")
                    frame_count += int(frames.shape[0])
                    if frame_carry.size:
                        frames = np.concatenate((frame_carry, frames), axis=0)

                    complete_count = (
                        frames.shape[0] // self._samples_per_bucket
                    ) * self._samples_per_bucket
                    if complete_count:
                        buckets = frames[:complete_count].reshape(
                            -1,
                            self._samples_per_bucket,
                            _PCM_CHANNELS,
                        )
                        minimum_parts.append(
                            buckets.min(axis=(1, 2)).astype(np.float32)
                        )
                        maximum_parts.append(
                            buckets.max(axis=(1, 2)).astype(np.float32)
                        )
                    frame_carry = frames[complete_count:].copy()

                if self._cancel_requested.is_set():
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=5)
                    return None

                if byte_carry:
                    raise RuntimeError("FFmpeg returned an incomplete stereo PCM frame")
                return_code = process.wait()
                stderr_file.seek(0)
                stderr_text = (
                    stderr_file.read().decode("utf-8", errors="replace").strip()
                )
                if return_code != 0:
                    detail = stderr_text or f"FFmpeg exited with code {return_code}"
                    raise RuntimeError(detail)
                if frame_count == 0:
                    raise RuntimeError("FFmpeg decoded zero PCM frames")

                if frame_carry.size:
                    minimum_parts.append(
                        np.asarray([frame_carry.min()], dtype=np.float32)
                    )
                    maximum_parts.append(
                        np.asarray([frame_carry.max()], dtype=np.float32)
                    )

                minimums = np.concatenate(minimum_parts)
                maximums = np.concatenate(maximum_parts)
                return WaveformEnvelope(
                    minimums=minimums,
                    maximums=maximums,
                    sample_rate=_PCM_SAMPLE_RATE,
                    frame_count=frame_count,
                    samples_per_bucket=self._samples_per_bucket,
                )
            finally:
                if process.stdout is not None:
                    process.stdout.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
                with self._process_lock:
                    self._process = None


class _InteractiveTimelineWidget(QWidget):
    seek_requested = pyqtSignal(int)
    zoom_requested = pyqtSignal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._duration_ms = 0
        self._view_start_ms = 0
        self._view_end_ms = 1
        self._position_ms = 0
        self._seeking = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.SizeHorCursor)

    def _set_timeline_values(
        self,
        duration_ms: int,
        view_start_ms: int,
        view_end_ms: int,
        position_ms: int,
    ) -> None:
        duration_ms = int(duration_ms)
        view_start_ms = int(view_start_ms)
        view_end_ms = int(view_end_ms)
        position_ms = int(position_ms)
        if duration_ms < 0:
            raise ValueError("duration_ms cannot be negative")
        if view_start_ms < 0 or view_end_ms <= view_start_ms:
            raise ValueError("timeline view must have a positive non-negative range")
        if position_ms < 0 or (duration_ms > 0 and position_ms > duration_ms):
            raise ValueError("position_ms must be inside the timeline duration")

        self._duration_ms = duration_ms
        self._view_start_ms = view_start_ms
        self._view_end_ms = view_end_ms
        self._position_ms = position_ms
        self.update()

    def _time_at_x(self, x_position: float) -> int:
        drawable_width = max(1, self.width() - 1)
        ratio = min(1.0, max(0.0, float(x_position) / drawable_width))
        time_ms = round(
            self._view_start_ms + ratio * (self._view_end_ms - self._view_start_ms)
        )
        if self._duration_ms > 0:
            time_ms = min(time_ms, self._duration_ms)
        return max(0, time_ms)

    def _x_at_time(self, time_ms: float) -> float:
        view_span = self._view_end_ms - self._view_start_ms
        return (
            (float(time_ms) - self._view_start_ms)
            * max(1, self.width() - 1)
            / view_span
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._seeking = True
            self.seek_requested.emit(self._time_at_x(event.position().x()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._seeking and event.buttons() & Qt.MouseButton.LeftButton:
            self.seek_requested.emit(self._time_at_x(event.position().x()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._seeking and event.button() == Qt.MouseButton.LeftButton:
            self._seeking = False
            self.seek_requested.emit(self._time_at_x(event.position().x()))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        # Timeline zoom and seeking require an explicit click/drag or button.
        # Ignoring the event lets the outer result page keep scrolling.
        event.ignore()


class WaveformLane(_InteractiveTimelineWidget):
    """Paint one real waveform envelope and a shared logical playhead."""

    def __init__(
        self,
        path: Path,
        color,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.path = Path(path)
        self._color = QColor(color)
        if not self._color.isValid():
            raise ValueError(f"Invalid waveform color: {color}")
        self._envelope: WaveformEnvelope | None = None
        self._error = ""
        self._offset_ms = 0
        self._waveform_cache: QPixmap | None = None
        self._waveform_cache_key: tuple[int, int, int, int, int, int] | None = None
        self.setObjectName("waveformLane")
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setMinimumHeight(64)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.update_translations()

    def set_envelope(self, envelope: WaveformEnvelope) -> None:
        if not isinstance(envelope, WaveformEnvelope):
            raise TypeError("envelope must be a WaveformEnvelope")
        self._envelope = envelope
        self._error = ""
        self._invalidate_waveform_cache()
        if self._duration_ms == 0:
            self._duration_ms = envelope.duration_ms
            self._view_start_ms = 0
            self._view_end_ms = envelope.duration_ms
        self.update_translations()
        self.update()

    def set_error(self, message: str) -> None:
        self._error = str(message).strip()
        self._invalidate_waveform_cache()
        self.update_translations()
        self.update()

    def set_timeline(
        self,
        duration_ms: int,
        view_start_ms: int,
        view_end_ms: int,
        position_ms: int,
        offset_ms: int,
    ) -> None:
        previous_view = (
            self._duration_ms,
            self._view_start_ms,
            self._view_end_ms,
            self._offset_ms,
        )
        self._offset_ms = int(offset_ms)
        self._set_timeline_values(
            duration_ms,
            view_start_ms,
            view_end_ms,
            position_ms,
        )
        current_view = (
            self._duration_ms,
            self._view_start_ms,
            self._view_end_ms,
            self._offset_ms,
        )
        if current_view != previous_view:
            self._invalidate_waveform_cache()

    def update_translations(self) -> None:
        label = t("dialogs.complete.audio_tracks.timeline")
        self.setAccessibleName(f"{label}: {self.path.name}")
        tooltip = str(self.path)
        if self._error:
            tooltip = f"{tooltip}\n{self._error}"
        self.setToolTip(tooltip)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKGROUND_COLOR)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        center_y = self.height() // 2
        painter.setPen(QPen(_GRID_COLOR, 1))
        painter.drawLine(0, center_y, self.width(), center_y)

        if self._error:
            painter.setPen(_ERROR_COLOR)
            painter.drawText(
                self.rect().adjusted(8, 4, -8, -4),
                Qt.AlignmentFlag.AlignLeft
                | Qt.AlignmentFlag.AlignVCenter
                | Qt.TextFlag.TextWordWrap,
                self._error,
            )
        elif self._envelope is None:
            painter.setPen(_TEXT_COLOR)
            painter.drawText(
                self.rect().adjusted(8, 0, -8, 0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                t("dialogs.complete.audio_tracks.waveform_loading"),
            )
        else:
            painter.drawPixmap(0, 0, self._cached_waveform(self._envelope))

        if self._view_start_ms <= self._position_ms <= self._view_end_ms:
            playhead_x = round(self._x_at_time(self._position_ms))
            painter.setPen(QPen(_PLAYHEAD_COLOR, 2))
            painter.drawLine(playhead_x, 0, playhead_x, self.height())

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._invalidate_waveform_cache()
        super().resizeEvent(event)

    def _invalidate_waveform_cache(self) -> None:
        self._waveform_cache = None
        self._waveform_cache_key = None

    def _cached_waveform(self, envelope: WaveformEnvelope) -> QPixmap:
        cache_key = (
            self.width(),
            self.height(),
            id(envelope),
            self._view_start_ms,
            self._view_end_ms,
            self._offset_ms,
        )
        if self._waveform_cache is not None and self._waveform_cache_key == cache_key:
            return self._waveform_cache

        pixmap = QPixmap(max(1, self.width()), max(1, self.height()))
        pixmap.fill(Qt.GlobalColor.transparent)
        cache_painter = QPainter(pixmap)
        self._paint_waveform(cache_painter, envelope)
        cache_painter.end()
        self._waveform_cache = pixmap
        self._waveform_cache_key = cache_key
        return pixmap

    def _paint_waveform(
        self,
        painter: QPainter,
        envelope: WaveformEnvelope,
    ) -> None:
        if self.width() <= 0 or self.height() <= 0:
            return
        clip_start_ms = self._offset_ms
        clip_end_ms = self._offset_ms + envelope.duration_ms
        visible_start_ms = max(self._view_start_ms, clip_start_ms)
        visible_end_ms = min(self._view_end_ms, clip_end_ms)
        if visible_end_ms <= visible_start_ms:
            return

        clip_left = max(0, math.floor(self._x_at_time(visible_start_ms)))
        clip_right = min(
            self.width() - 1,
            math.ceil(self._x_at_time(visible_end_ms)),
        )
        clip_color = QColor(self._color)
        clip_color.setAlpha(24)
        painter.fillRect(
            QRect(clip_left, 0, max(1, clip_right - clip_left + 1), self.height()),
            clip_color,
        )

        view_span_ms = self._view_end_ms - self._view_start_ms
        bucket_ms = envelope.bucket_duration_ms
        upper_margin = 4
        amplitude_height = max(1, self.height() - 2 * upper_margin)
        lines = []
        for x_position in range(clip_left, clip_right + 1):
            global_left = self._view_start_ms + x_position * view_span_ms / max(
                1, self.width() - 1
            )
            global_right = self._view_start_ms + (x_position + 1) * view_span_ms / max(
                1, self.width() - 1
            )
            source_left = max(0.0, global_left - self._offset_ms)
            source_right = min(
                float(envelope.duration_ms),
                global_right - self._offset_ms,
            )
            if source_right <= source_left:
                continue
            first_bucket = max(0, int(math.floor(source_left / bucket_ms)))
            last_bucket = min(
                envelope.bucket_count - 1,
                max(first_bucket, int(math.ceil(source_right / bucket_ms) - 1)),
            )
            minimum = float(envelope.minimums[first_bucket : last_bucket + 1].min())
            maximum = float(envelope.maximums[first_bucket : last_bucket + 1].max())
            minimum = max(-1.0, min(1.0, minimum))
            maximum = max(-1.0, min(1.0, maximum))
            top = upper_margin + round((1.0 - maximum) * amplitude_height / 2.0)
            bottom = upper_margin + round((1.0 - minimum) * amplitude_height / 2.0)
            lines.append((QPoint(x_position, top), QPoint(x_position, bottom)))

        painter.setPen(QPen(self._color, 1))
        for start, end in lines:
            painter.drawLine(start, end)


class TimelineRuler(_InteractiveTimelineWidget):
    """Interactive time ruler sharing seek/zoom semantics with waveform lanes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("audioTimelineRuler")
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setMinimumHeight(30)
        self.setMaximumHeight(36)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.update_translations()

    def set_timeline(
        self,
        duration_ms: int,
        view_start_ms: int,
        view_end_ms: int,
        position_ms: int,
    ) -> None:
        self._set_timeline_values(
            duration_ms,
            view_start_ms,
            view_end_ms,
            position_ms,
        )

    def update_translations(self) -> None:
        label = t("dialogs.complete.audio_tracks.timeline")
        self.setAccessibleName(label)
        self.setToolTip(label)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), _RULER_BACKGROUND_COLOR)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QPen(_GRID_COLOR, 1))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)

        major_interval = self._major_tick_interval_ms()
        minor_interval = max(1, major_interval // 5)
        first_minor = math.floor(self._view_start_ms / minor_interval) * minor_interval
        painter.setFont(self.font())
        tick_ms = first_minor
        while tick_ms <= self._view_end_ms:
            if tick_ms >= self._view_start_ms:
                x_position = round(self._x_at_time(tick_ms))
                is_major = tick_ms % major_interval == 0
                tick_height = 10 if is_major else 5
                painter.setPen(QPen(_TEXT_COLOR if is_major else _GRID_COLOR, 1))
                painter.drawLine(
                    x_position,
                    self.height() - tick_height,
                    x_position,
                    self.height() - 1,
                )
                if is_major:
                    painter.drawText(
                        QRect(x_position + 4, 1, 90, self.height() - 12),
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                        _format_ruler_time(tick_ms, major_interval),
                    )
            tick_ms += minor_interval

        if self._view_start_ms <= self._position_ms <= self._view_end_ms:
            playhead_x = round(self._x_at_time(self._position_ms))
            painter.setPen(QPen(_PLAYHEAD_COLOR, 2))
            painter.drawLine(playhead_x, 0, playhead_x, self.height())
            painter.setBrush(_PLAYHEAD_COLOR)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(
                QPolygon(
                    [
                        QPoint(playhead_x - 5, 0),
                        QPoint(playhead_x + 5, 0),
                        QPoint(playhead_x, 7),
                    ]
                )
            )

    def _major_tick_interval_ms(self) -> int:
        view_span = self._view_end_ms - self._view_start_ms
        target_tick_count = max(1.0, self.width() / 90.0)
        raw_interval = max(1.0, view_span / target_tick_count)
        power = 10.0 ** math.floor(math.log10(raw_interval))
        normalized = raw_interval / power
        if normalized <= 1.0:
            multiplier = 1.0
        elif normalized <= 2.0:
            multiplier = 2.0
        elif normalized <= 5.0:
            multiplier = 5.0
        else:
            multiplier = 10.0
        return max(1, round(multiplier * power))


def _format_ruler_time(position_ms: int, interval_ms: int) -> str:
    total_ms = max(0, int(position_ms))
    total_seconds, milliseconds = divmod(total_ms, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if interval_ms < 1000:
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        return f"{minutes:d}:{seconds:02d}.{milliseconds:03d}"
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


__all__ = [
    "FfmpegWaveformLoader",
    "TimelineRuler",
    "WaveformEnvelope",
    "WaveformLane",
]
