"""A pinned chord ruler and cancellable SoundFont audition for the desktop roll."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import mido
from PyQt6 import sip
from PyQt6.QtCore import (
    QLineF,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    Qt,
    QThread,
    QTimer,
    QUrl,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QCursor, QFont, QFontMetrics, QPainter, QPen, QPolygonF
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import QWidget

from src.core.chord_analysis import analyze_audio_chords
from src.core.midi_chords import ROOTS, MidiChord, read_chord_record, split_chords_at_bars
from src.core.muscriptor_result_assets import render_midi_audio_export
from src.i18n.translator import t

_CHORD_COLORS = (
    "#67c7ed",
    "#a99af4",
    "#f5c16c",
    "#e38b9c",
    "#6bd4bd",
    "#a6cf72",
    "#70c8bc",
    "#d58abb",
    "#a493e0",
    "#74aedb",
    "#b39aeb",
    "#83c3dd",
)
_CHORD_SUFFIXES = {
    "maj": "",
    "min": "m",
    "maj7": "maj7",
    "min7": "m7",
    "7": "7",
    "dim": "dim",
    "aug": "+",
    "dim7": "dim7",
    "hdim7": "m7♭5",
    "minmaj7": "m(maj7)",
    "maj6": "6",
    "min6": "m6",
    "sus2": "sus2",
    "sus4": "sus4",
}


@lru_cache(maxsize=512)
def _chord_presentation(label: str) -> tuple[str, str]:
    """Compact notation and stable root colors; never rewrite model records."""
    if label in {"N", "X"}:
        return label, "#78869c"
    name, _, bass = label.partition("/")
    root, _, quality = name.partition(":")
    pitch_class = (ROOTS[root[0]] + root.count("#") - root.count("b")) % 12
    symbol = root.replace("#", "♯").replace("b", "♭") + _CHORD_SUFFIXES[quality or "maj"]
    if bass:
        # Harte slash labels encode a scale degree relative to the root.
        # Display its spelled note name, retaining the exact label in the tooltip.
        degree = int(bass[-1]) - 1
        letter = "CDEFGAB"[("CDEFGAB".index(root[0]) + degree) % 7]
        bass_pitch = pitch_class + (0, 2, 4, 5, 7, 9, 11)[degree]
        bass_pitch += bass.count("#") - bass.count("b")
        accidental = (bass_pitch - ROOTS[letter] + 6) % 12 - 6
        symbol += "/" + letter + "♯" * max(0, accidental) + "♭" * max(0, -accidental)
    return symbol, _CHORD_COLORS[pitch_class]


class _ChordAnalysisWorker(QThread):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, audio_path, gpu_device, parent):
        super().__init__(parent)
        self.audio_path, self.gpu_device = audio_path, gpu_device

    def run(self):
        try:
            record = analyze_audio_chords(
                self.audio_path,
                gpu_device=self.gpu_device,
                cancel_check=self.isInterruptionRequested,
            )
            if not self.isInterruptionRequested():
                self.ready.emit(record)
        except InterruptedError:
            pass
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))


class MidiChordLane(QWidget):
    audition_requested = pyqtSignal(object)

    def __init__(self, roll, scroll, parent=None):
        super().__init__(parent)
        self.roll, self.scroll = roll, scroll
        self.chords: tuple[MidiChord, ...] = ()
        self._source_chords: tuple[MidiChord, ...] = ()
        self._starts: tuple[float, ...] = ()
        self._ends: tuple[float, ...] = ()
        self._callouts = ()
        self._label_rows = 0
        self._layout_scale = None
        self.record = None
        self.worker = None
        self.closed = False
        self.error = ""
        self.audio_path = None
        self.gpu_device = 0
        self.source_offset = 0.0
        self.state = "chord_waiting"
        self.setFixedHeight(60)
        self.setMinimumWidth(0)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(80)
        self._refresh_timer.timeout.connect(self.refresh)
        roll.content_changed.connect(self.schedule_refresh)
        # Follow playback moves by fractional pixels between coarse scrollbar
        # steps. Repaint in the same frame as the roll; restarting the content
        # debounce here starves refreshes until the scrollbar jumps again.
        roll.view_changed.connect(self._relayout)
        roll.position_changed.connect(self.update)
        scroll.horizontalScrollBar().valueChanged.connect(self._relayout)
        self.update_translations()

    def schedule_refresh(self):
        self._refresh_timer.start()

    def refresh(self):
        self.chords = (
            split_chords_at_bars(
                (
                    replace(
                        chord,
                        start=chord.start + self.source_offset,
                        end=chord.end + self.source_offset,
                    )
                    for chord in self._source_chords
                ),
                self.roll._daw_reference_bpm,
                self.roll._daw_time_signature,
            )
            if self.roll._daw_reference_bpm is not None
            else ()
        )
        self._starts = tuple(chord.start for chord in self.chords)
        self._ends = tuple(chord.end for chord in self.chords)
        self._label_rows = 0
        self._relayout()

    def analyze_audio(self, audio_path, gpu_device=0):
        if self.closed or self.worker is not None:
            return
        self.audio_path, self.gpu_device = audio_path, gpu_device
        self.state, self.error = "chord_analyzing", ""
        self.record = None
        self._source_chords = self.chords = ()
        worker = _ChordAnalysisWorker(audio_path, gpu_device, self)
        self.worker = worker
        worker.ready.connect(self.set_record)
        worker.failed.connect(self._analysis_failed)
        worker.finished.connect(self._analysis_finished)
        worker.start()
        self.update()

    def set_record(self, record):
        if self.closed:
            return
        try:
            self._source_chords = read_chord_record(record)
        except Exception as exc:
            self._analysis_failed(str(exc))
            return
        self.record = record
        self.state = "chord_empty"
        self.refresh()

    def _analysis_failed(self, error):
        if not self.closed:
            self.state, self.error = "chord_analysis_failed", error
            self.update()

    def _analysis_finished(self):
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.deleteLater()

    def shutdown(self):
        self.closed = True
        self._refresh_timer.stop()
        if self.worker is not None:
            self.worker.requestInterruption()
            self.worker.wait()

    def update_translations(self):
        self.setToolTip(t("muscriptor_result.chord_hint"))
        self.setAccessibleName(t("muscriptor_result.chords"))
        self.update()

    def x_for_time(self, seconds):
        origin = self.mapFromGlobal(self.scroll.viewport().mapToGlobal(QPoint(0, 0))).x()
        return (
            origin
            + self.roll.x_for_time_float(seconds)
            - self.scroll.horizontalScrollBar().value()
            - self.roll.render_offset_px
        )

    def chord_rect(self, chord):
        left, right = self.x_for_time(chord.start), self.x_for_time(chord.end)
        width = max(0.0, right - left)
        return QRectF(
            left, 23 + self._label_rows * 24, max(0.0, width - min(3.0, width * 0.15)), 31
        )

    def _relayout(self):
        """Give every narrow interval a full label without changing its timing."""
        scale = (self.width(), self.roll.pixels_per_second)
        if self._layout_scale != scale:
            self._label_rows = 0
            self._layout_scale = scale
        font = QFont(self.font())
        font.setWeight(QFont.Weight.DemiBold)
        metrics = QFontMetrics(font)
        row_ends, callouts = [], []
        for chord in self._visible_chords():
            rect = self.chord_rect(chord).intersected(QRectF(self.rect()))
            if rect.width() <= 0:
                continue
            symbol = _chord_presentation(chord.label)[0]
            width = metrics.horizontalAdvance(symbol) + 16
            if rect.width() >= width:
                continue
            left = max(3.0, min(self.width() - width - 3.0, rect.center().x() - width / 2))
            row = len(row_ends)
            for i, end in enumerate(row_ends):
                candidate = max(left, end + 5)
                if candidate + width <= self.width() - 3 and candidate - left <= width:
                    row, left = i, candidate
                    break
            if row == len(row_ends):
                row_ends.append(0.0)
            row_ends[row] = left + width
            callouts.append((chord, QRectF(left, 22 + row * 24, width, 20)))
        # Reserve rows during scrolling so the piano roll does not bounce vertically.
        self._label_rows = max(self._label_rows, len(row_ends)) if self.chords else 0
        self._callouts = tuple(
            (
                chord,
                QRectF(
                    rect.x(),
                    22 + (self._label_rows - 1) * 24 - (rect.y() - 22),
                    rect.width(),
                    rect.height(),
                ),
            )
            for chord, rect in callouts
        )
        self.setFixedHeight(60 + self._label_rows * 24)
        self.update()

    def resizeEvent(self, event):  # noqa: N802
        self._relayout()
        super().resizeEvent(event)

    def current_chord(self):
        if not self.chords:
            return None
        index = bisect_right(self._starts, self.roll.position) - 1
        if index >= 0 and self.roll.position < self.chords[index].end:
            return self.chords[index]
        return None

    def _visible_chords(self):
        if not self.chords:
            return ()
        origin = self.x_for_time(0)
        left = -origin / self.roll.pixels_per_second
        right = (self.width() - origin) / self.roll.pixels_per_second
        return self.chords[bisect_left(self._ends, left) : bisect_right(self._starts, right)]

    def _paint_chord(self, painter, chord, current, hovered_chord, rect=None, show_label=True):
        callout = rect is not None
        rect = self.chord_rect(chord) if rect is None else rect
        if rect.width() <= 0:
            return
        symbol, color = _chord_presentation(chord.label)
        accent = QColor(color)
        active = chord == current
        hovered = chord == hovered_chord and bool(chord.pitches)
        fill = QColor(accent)
        fill.setAlpha(100 if active else 62 if hovered else 30 if chord.pitches else 12)
        border = QColor(accent)
        border.setAlpha(240 if active else 180 if hovered else 95 if chord.pitches else 50)
        painter.save()
        painter.setClipRect(rect)
        if callout:
            painter.fillRect(rect, QColor("#101c30"))
        if rect.width() < 2:
            painter.fillRect(rect, border)
            painter.restore()
            return
        painter.setBrush(fill)
        painter.setPen(QPen(border, 1.5 if active or hovered else 1.0))
        radius = min(4.0, rect.width() / 4.0)
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        if active:
            painter.setPen(QPen(accent, 2.0))
            painter.drawLine(
                QLineF(rect.left() + radius, rect.top() + 1, rect.right() - radius, rect.top() + 1)
            )
        # Keep the label visible when the beginning of a long chord scrolls offscreen.
        visible = rect.intersected(QRectF(self.rect()))
        text_rect = visible.adjusted(7, 0, -5, 0)
        font = QFont(self.font())
        font.setWeight(QFont.Weight.DemiBold if chord.pitches else QFont.Weight.Normal)
        painter.setFont(font)
        painter.setPen(
            QColor("#ffffff")
            if active
            else QColor("#e2eaf6") if chord.pitches else QColor("#8797ae")
        )
        metrics = painter.fontMetrics()
        if show_label and text_rect.width() >= metrics.horizontalAdvance(symbol):
            painter.drawText(
                text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, symbol
            )
        painter.restore()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101c30"))
        painter.fillRect(QRectF(0, 0, self.width(), 20), QColor("#142239"))
        painter.setPen(QColor("#293b55"))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        title = t("muscriptor_result.chords")
        title_width = min(self.width(), painter.fontMetrics().horizontalAdvance(title) + 16)
        bpm = self.roll._daw_reference_bpm
        if bpm:
            numerator, denominator = self.roll._daw_time_signature
            bar = 60.0 / bpm * 4.0 / denominator * numerator
            left_seconds = max(0, -self.x_for_time(0) / self.roll.pixels_per_second)
            first = int(left_seconds / bar)
            last = int((left_seconds + self.width() / self.roll.pixels_per_second) / bar) + 1
            label_right = title_width
            for index in range(first, last + 1):
                x = self.x_for_time(index * bar)
                painter.setPen(QPen(QColor("#2b3e57")))
                painter.drawLine(QLineF(x, 20, x, self.height() - 1))
                label = f"{index + 1}.1"
                if x >= label_right:
                    painter.setPen(QColor("#87a3c4"))
                    painter.drawText(QPointF(x + 5, 15), label)
                    label_right = x + painter.fontMetrics().horizontalAdvance(label) + 14
        # A solid title backing prevents moving bar numbers from crossing the title.
        painter.fillRect(QRectF(0, 0, title_width, 20), QColor("#142239"))
        painter.setPen(QColor("#a9bdd6"))
        painter.drawText(7, 15, title)
        current = self.current_chord()
        pointer = QPointF(self.mapFromGlobal(QCursor.pos())) if self.underMouse() else None
        hovered = self.chord_at(pointer) if pointer is not None else None
        callout_chords = {chord for chord, _ in self._callouts}
        for chord in self._visible_chords():
            self._paint_chord(
                painter, chord, current, hovered, show_label=chord not in callout_chords
            )
        for chord, rect in self._callouts:
            accent = QColor(_chord_presentation(chord.label)[1])
            accent.setAlpha(180 if chord == current or chord == hovered else 95)
            painter.setPen(QPen(accent, 1.0))
            anchor = self.chord_rect(chord).intersected(QRectF(self.rect())).center().x()
            painter.drawLine(
                QLineF(rect.center().x(), rect.bottom(), anchor, self.chord_rect(chord).top())
            )
        for chord, rect in self._callouts:
            self._paint_chord(painter, chord, current, hovered, rect=rect)
        if not self.chords:
            painter.setPen(QColor("#93abc9"))
            text = painter.fontMetrics().elidedText(
                t(f"muscriptor_result.{self.state}"),
                Qt.TextElideMode.ElideRight,
                max(0, self.width() - 14),
            )
            painter.drawText(
                self.rect().adjusted(7, 20, -7, 0),
                Qt.AlignmentFlag.AlignVCenter,
                text,
            )
        else:
            x = self.x_for_time(self.roll.position)
            if 0 <= x <= self.width():
                painter.setPen(QPen(QColor(91, 220, 245, 35), 7.0))
                painter.drawLine(QLineF(x, 20, x, self.height()))
                painter.setPen(QPen(QColor("#75e5f5"), 1.5))
                painter.drawLine(QLineF(x, 20, x, self.height()))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#75e5f5"))
                painter.drawPolygon(
                    QPolygonF([QPointF(x - 4, 20), QPointF(x + 4, 20), QPointF(x, 24)])
                )

    def chord_at(self, position):
        for chord, rect in self._callouts:
            if rect.contains(position):
                return chord
        return next(
            (
                chord
                for chord in self._visible_chords()
                if self.chord_rect(chord).contains(position)
            ),
            None,
        )

    def mouseMoveEvent(self, event):  # noqa: N802
        chord = self.chord_at(event.position())
        clickable = (chord and chord.pitches) or self.state == "chord_analysis_failed"
        self.setCursor(
            Qt.CursorShape.PointingHandCursor if clickable else Qt.CursorShape.ArrowCursor
        )
        self.setToolTip(
            self.error
            if self.error
            else (self._chord_tooltip(chord) if chord else t("muscriptor_result.chord_hint"))
        )
        self.update()

    def _chord_tooltip(self, chord):
        description = (
            t("muscriptor_result.chord_audition", chord=_chord_presentation(chord.label)[0])
            if chord.pitches
            else t(
                "muscriptor_result.chord_none"
                if chord.label == "N"
                else "muscriptor_result.chord_unknown"
            )
        )
        return (
            description
            + "\n"
            + t(
                "muscriptor_result.chord_interval",
                label=chord.label,
                start=f"{chord.start:.2f}",
                end=f"{chord.end:.2f}",
            )
        )

    def leaveEvent(self, event):  # noqa: N802
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        chord = self.chord_at(event.position())
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.state == "chord_analysis_failed"
            and self.audio_path
        ):
            self.analyze_audio(self.audio_path, self.gpu_device)
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton and chord and chord.pitches:
            self.audition_requested.emit(chord)
            event.accept()
        else:
            super().mousePressEvent(event)


class _ChordRenderWorker(QThread):
    ready = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, chord, duration, destination, selection, parent):
        super().__init__(parent)
        self.chord, self.duration, self.destination = chord, duration, destination
        self.selection = selection

    def run(self):
        try:
            source = self.destination.with_suffix(".mid")
            midi = mido.MidiFile()
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.append(mido.MetaMessage("set_tempo", tempo=500_000))
            track.append(mido.Message("program_change", program=0))
            for pitch in self.chord.pitches:
                track.append(mido.Message("note_on", note=pitch, velocity=82))
            for index, pitch in enumerate(self.chord.pitches):
                track.append(
                    mido.Message(
                        "note_off",
                        note=pitch,
                        velocity=0,
                        time=round(self.duration * 960) if index == 0 else 0,
                    )
                )
            midi.save(source)
            render_midi_audio_export(
                source,
                self.destination,
                "pcm16_44100",
                cancel_check=self.isInterruptionRequested,
                soundfont_selection=self.selection,
            )
            if not self.isInterruptionRequested():
                self.ready.emit(str(self.destination))
        except InterruptedError:
            pass
        except Exception as exc:
            self.failed.emit(str(exc))


class ChordAuditioner(QObject):
    failed = pyqtSignal(str)
    ready = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, cache: Path, parent=None):
        super().__init__(parent)
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.output)
        self.player.errorOccurred.connect(lambda *_: self.failed.emit(self.player.errorString()))
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.worker = None
        self.pending = None
        self.generation = 0
        self.closed = False
        self._cache_key = None
        self._cache_path = None

    def _on_media_status(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.finished.emit()

    def play(self, chord: MidiChord, duration: float, selection=None):
        self.stop()
        self.generation += 1
        key = (chord.pitches, round(duration, 3), repr(selection))
        if key == self._cache_key and self._cache_path and self._cache_path.is_file():
            self.player.setSource(QUrl.fromLocalFile(str(self._cache_path)))
            self.player.play()
            self.ready.emit()
            return
        self.pending = (chord, duration, selection, self.generation, key)
        self._start_pending()

    def _start_pending(self):
        if self.worker is not None or self.pending is None or self.closed:
            return
        chord, duration, selection, generation, key = self.pending
        self.pending = None
        destination = self.cache / f"chord-{generation}.wav"
        worker = _ChordRenderWorker(chord, duration, destination, selection, self)
        self.worker = worker

        def ready(path):
            if generation == self.generation and not self.closed:
                self._cache_key, self._cache_path = key, Path(path)
                self.player.setSource(QUrl.fromLocalFile(path))
                self.player.play()
                self.ready.emit()

        def failed(error):
            if generation == self.generation and not self.closed:
                self.failed.emit(error)

        def finished():
            self.worker = None
            worker.deleteLater()
            self._start_pending()

        worker.ready.connect(ready)
        worker.failed.connect(failed)
        worker.finished.connect(finished)
        worker.start()

    def stop(self):
        self.generation += 1
        self.pending = None
        self.player.stop()
        if self.worker is not None:
            self.worker.requestInterruption()

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        self.stop()
        if self.worker is not None:
            self.worker.wait()
        self.player.setSource(QUrl())
        # The Windows media backend can release the WAV asynchronously after
        # setSource(). Destroy our private player before its cache is removed.
        sip.delete(self.player)
