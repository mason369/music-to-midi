"""A pinned chord ruler and cancellable SoundFont audition for the desktop roll."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mido
from PyQt6 import sip
from PyQt6.QtCore import QObject, QPoint, QRectF, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import QWidget

from src.core.chord_analysis import analyze_audio_chords
from src.core.midi_chords import MidiChord, read_chord_record, split_chords_at_bars
from src.core.muscriptor_result_assets import render_midi_audio_export
from src.i18n.translator import t


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
        self.record = None
        self.worker = None
        self.closed = False
        self.error = ""
        self.audio_path = None
        self.gpu_device = 0
        self.source_offset = 0.0
        self.state = "chord_waiting"
        self.setFixedHeight(56)
        self.setMinimumWidth(0)
        self.setMouseTracking(True)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(80)
        self._refresh_timer.timeout.connect(self.refresh)
        roll.content_changed.connect(self.schedule_refresh)
        # Follow playback moves by fractional pixels between coarse scrollbar
        # steps. Repaint in the same frame as the roll; restarting the content
        # debounce here starves refreshes until the scrollbar jumps again.
        roll.view_changed.connect(self.update)
        scroll.horizontalScrollBar().valueChanged.connect(self.update)
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
        self.update()

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
        return QRectF(left, 21, max(1, right - left - 2), 30)

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#132139"))
        painter.setPen(QColor("#a9c8e8"))
        painter.drawText(6, 15, t("muscriptor_result.chords"))
        bpm = self.roll._daw_reference_bpm
        if bpm:
            numerator, denominator = self.roll._daw_time_signature
            bar = 60.0 / bpm * 4.0 / denominator * numerator
            left_seconds = max(0, -self.x_for_time(0) / self.roll.pixels_per_second)
            first = int(left_seconds / bar)
            last = int((left_seconds + self.width() / self.roll.pixels_per_second) / bar) + 1
            for index in range(first, last + 1):
                x = self.x_for_time(index * bar)
                painter.setPen(QPen(QColor("#78aee8")))
                painter.drawLine(int(x), 18, int(x), self.height())
                if x > painter.fontMetrics().horizontalAdvance(t("muscriptor_result.chords")) + 14:
                    painter.drawText(int(x) + 3, 15, f"{index + 1}.1")
        for chord in self.chords:
            rect = self.chord_rect(chord)
            if rect.right() < 0 or rect.left() > self.width():
                continue
            painter.fillRect(rect, QColor("#284c70"))
            painter.setPen(QColor("#e4efff"))
            painter.save()
            painter.setClipRect(rect)
            painter.drawText(rect.adjusted(5, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter, chord.label)
            painter.restore()
        if not self.chords:
            painter.setPen(QColor("#93abc9"))
            painter.drawText(
                self.rect().adjusted(6, 20, -4, 0),
                Qt.AlignmentFlag.AlignVCenter,
                t(f"muscriptor_result.{self.state}"),
            )

    def chord_at(self, position):
        return next(
            (chord for chord in self.chords if self.chord_rect(chord).contains(position)), None
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
            else (
                t("muscriptor_result.chord_audition", chord=chord.label)
                if chord and chord.pitches
                else t("muscriptor_result.chord_hint")
            )
        )

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
