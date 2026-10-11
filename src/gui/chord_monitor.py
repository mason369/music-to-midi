"""Cancellable, generation-checked preparation of the transport's chord bus."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from src.core.chord_audio import CHORD_SAMPLE_RATE, render_chord_pcm


class _ChordMonitorWorker(QThread):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, chords, frames, offset, selection, cache, parent):
        super().__init__(parent)
        self.chords, self.frames, self.offset = chords, frames, offset
        self.selection, self.cache = selection, cache

    def run(self):
        try:
            samples = render_chord_pcm(
                self.chords,
                self.frames,
                self.cache,
                offset=self.offset,
                selection=self.selection,
                cancel_check=self.isInterruptionRequested,
            )
            if not self.isInterruptionRequested():
                self.ready.emit(samples)
        except InterruptedError:
            pass
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))


class ChordMonitor(QObject):
    state_changed = pyqtSignal(str, str)

    def __init__(self, player, cache: Path, parent=None):
        super().__init__(parent)
        self.player, self.cache = player, cache
        self.enabled, self.volume = False, 100
        self.context = None
        self.key = None
        self.samples = None
        self.worker = None
        self.pending = None
        self.generation = 0
        self.closed = False
        self.error = ""

    def configure(self, chords, offset, selection):
        frames = round(self.player.duration_seconds * CHORD_SAMPLE_RATE)
        identity = json.dumps(selection.cache_identity(), sort_keys=True) if selection else None
        key = (tuple(chords), frames, offset, identity)
        if key != self.key:
            self._cancel()
            self.key, self.samples, self.error = key, None, ""
            self.context = (tuple(chords), frames, offset, selection)
        self.player.set_chord_bus(self.samples)
        self._update()

    def set_controls(self, enabled: bool, volume: int):
        self.enabled, self.volume = bool(enabled), int(volume)
        if not self.enabled or not self.volume:
            self._cancel()
        self._update()

    def retry(self):
        self._cancel()
        self.error = ""
        self._update()

    def _update(self):
        if self.closed:
            return
        audible = self.enabled and self.volume > 0
        self.player.set_chord_volume(self.volume if audible and self.samples is not None else 0)
        if not self.volume:
            state = "muted"
        elif not self.enabled:
            state = "off"
        elif not self.context or not self.context[0] or self.context[1] <= 0:
            state = "waiting"
        elif not any(chord.pitches for chord in self.context[0]):
            state = "empty"
        elif self.error:
            state = "failed"
        elif self.samples is not None:
            state = "ready"
        else:
            state = "loading"
            if self.pending is None and self.worker is None:
                self.pending = (self.generation, self.context)
                self._start_pending()
            elif self.worker is not None and self.worker.isInterruptionRequested():
                self.pending = (self.generation, self.context)
        self.state_changed.emit(state, self.error)

    def _cancel(self):
        self.generation += 1
        self.pending = None
        if self.worker is not None:
            self.worker.requestInterruption()

    def _start_pending(self):
        if self.closed or self.worker is not None or self.pending is None:
            return
        generation, context = self.pending
        self.pending = None
        worker = _ChordMonitorWorker(*context, self.cache, self)
        self.worker = worker

        def ready(samples):
            if not self.closed and generation == self.generation:
                self.samples = samples
                self.player.set_chord_bus(samples)
                self._update()

        def failed(error):
            if not self.closed and generation == self.generation:
                self.error = error
                self._update()

        def finished():
            self.worker = None
            worker.deleteLater()
            self._start_pending()

        worker.ready.connect(ready)
        worker.failed.connect(failed)
        worker.finished.connect(finished)
        worker.start()

    def shutdown(self):
        self.closed = True
        self._cancel()
        self.player.set_chord_volume(0)
        if self.worker is not None:
            self.worker.wait()
        self.samples = None
