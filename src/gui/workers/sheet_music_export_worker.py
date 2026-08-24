"""Background MuseScore engraving worker for the desktop result workbench."""

from __future__ import annotations

import threading

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.sheet_music import export_sheet_music_zip


class SheetMusicExportWorker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        *,
        midi_path: str,
        destination: str,
        quantize_grid: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.midi_path = str(midi_path)
        self.destination = str(destination)
        self.quantize_grid = str(quantize_grid)
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            result = export_sheet_music_zip(
                self.midi_path,
                self.destination,
                quantize_grid=self.quantize_grid,
                cancel_check=self._cancel_event.is_set,
            )
            self.succeeded.emit(result)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


__all__ = ["SheetMusicExportWorker"]
