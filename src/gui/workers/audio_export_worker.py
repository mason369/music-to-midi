"""Strict background copying for exported desktop audio tracks."""

from __future__ import annotations

import os
import shutil
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from uuid import uuid4

from PyQt6.QtCore import QThread, pyqtSignal

_COPY_CHUNK_BYTES = 8 * 1024 * 1024
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioExportItem:
    """One source track and its explicit user-selected destination."""

    label: str
    source: Path
    destination: Path


class AudioExportWorker(QThread):
    """Copy audio files off the GUI thread and publish only complete files."""

    progress_updated = pyqtSignal(int, int, str)
    export_succeeded = pyqtSignal(object)
    export_failed = pyqtSignal(str)
    export_cancelled = pyqtSignal()

    def __init__(
        self,
        items: tuple[AudioExportItem, ...],
        parent=None,
    ) -> None:
        super().__init__(parent)
        if not items:
            raise ValueError("At least one audio export item is required")
        self._items = tuple(items)
        self._cancel_requested = Event()
        self._staged_paths: set[Path] = set()

    def cancel(self) -> None:
        """Request cancellation before any staged files are published."""

        self._cancel_requested.set()

    @staticmethod
    def _is_same_file(source: Path, destination: Path) -> bool:
        return destination.exists() and os.path.samefile(source, destination)

    def _validated_items(self) -> tuple[tuple[AudioExportItem, int, bool], ...]:
        destinations: set[Path] = set()
        validated = []
        for item in self._items:
            source = Path(item.source).expanduser().resolve(strict=True)
            if not source.is_file():
                raise FileNotFoundError(f"Audio export source is not a file: {source}")
            source_size = source.stat().st_size
            if source_size <= 0:
                raise ValueError(f"Audio export source is empty: {source}")

            destination = Path(item.destination).expanduser().resolve(strict=False)
            parent = destination.parent.resolve(strict=True)
            if not parent.is_dir():
                raise NotADirectoryError(f"Audio export destination is not a directory: {parent}")
            destination = parent / destination.name
            if destination in destinations:
                raise ValueError(f"Duplicate audio export destination: {destination}")
            if destination.exists() and destination.is_dir():
                raise IsADirectoryError(f"Audio export destination is a directory: {destination}")
            destinations.add(destination)
            normalized = AudioExportItem(str(item.label), source, destination)
            validated.append((normalized, source_size, self._is_same_file(source, destination)))
        return tuple(validated)

    def _copy_to_stage(
        self,
        item: AudioExportItem,
        source_size: int,
        completed_bytes: int,
        total_bytes: int,
    ) -> tuple[Path, int]:
        part_path = item.destination.with_name(
            f".{item.destination.name}.{uuid4().hex}.music-to-midi.part"
        )
        self._staged_paths.add(part_path)
        with item.source.open("rb") as source_handle, part_path.open("xb") as output_handle:
            while True:
                if self._cancel_requested.is_set():
                    raise InterruptedError("Audio export was cancelled")
                chunk = source_handle.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                output_handle.write(chunk)
                completed_bytes += len(chunk)
                self.progress_updated.emit(completed_bytes, total_bytes, item.label)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        shutil.copystat(item.source, part_path)
        if part_path.stat().st_size != source_size:
            raise OSError(
                "Staged audio export size does not match its source: "
                f"source={item.source}, staged={part_path}"
            )
        return part_path, completed_bytes

    def _cleanup_staged_files(self) -> tuple[str, ...]:
        failures = []
        for path in tuple(self._staged_paths):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                detail = f"{path}: {type(exc).__name__}: {exc}"
                failures.append(detail)
                logger.error("Audio export staging cleanup failed: %s", detail)
            self._staged_paths.discard(path)
        return tuple(failures)

    def run(self) -> None:
        staged: list[tuple[AudioExportItem, Path, int]] = []
        try:
            validated = self._validated_items()
            total_bytes = sum(source_size for _item, source_size, _same in validated)
            completed_bytes = 0
            destinations: list[str] = []
            for item, source_size, same_file in validated:
                if self._cancel_requested.is_set():
                    raise InterruptedError("Audio export was cancelled")
                if same_file:
                    completed_bytes += source_size
                    self.progress_updated.emit(completed_bytes, total_bytes, item.label)
                    destinations.append(str(item.destination))
                    continue
                part_path, completed_bytes = self._copy_to_stage(
                    item,
                    source_size,
                    completed_bytes,
                    total_bytes,
                )
                staged.append((item, part_path, source_size))
                destinations.append(str(item.destination))

            if self._cancel_requested.is_set():
                raise InterruptedError("Audio export was cancelled")

            # All large copies finish before publication, so a copy failure never
            # exposes a truncated destination as a completed WAV.
            for item, part_path, source_size in staged:
                os.replace(part_path, item.destination)
                self._staged_paths.discard(part_path)
                if item.destination.stat().st_size != source_size:
                    raise OSError(
                        "Published audio export size does not match its source: "
                        f"source={item.source}, destination={item.destination}"
                    )
            self.export_succeeded.emit(tuple(destinations))
        except InterruptedError:
            cleanup_failures = self._cleanup_staged_files()
            if cleanup_failures:
                self.export_failed.emit(
                    "Audio export cancellation cleanup failed: " + "; ".join(cleanup_failures)
                )
            else:
                self.export_cancelled.emit()
        except Exception as exc:
            cleanup_failures = self._cleanup_staged_files()
            detail = f"{type(exc).__name__}: {exc}"
            if cleanup_failures:
                detail += "; staging cleanup failed: " + "; ".join(cleanup_failures)
            self.export_failed.emit(detail)


__all__ = ["AudioExportItem", "AudioExportWorker"]
