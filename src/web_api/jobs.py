"""Durable job state and a single-owner GPU execution queue."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.models.data_models import ProcessingProgress
from src.web_api.engine import ArtifactSpec, ExecutionResult, InferenceEngine
from src.web_api.schemas import JobStatus

LOGGER = logging.getLogger(__name__)


class QueueCapacityError(RuntimeError):
    """Raised when the configured waiting-job limit has been reached."""


class InsufficientStorageError(RuntimeError):
    """Raised before a job would violate the persistent-volume free-space floor."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class ArtifactRecord:
    id: str
    kind: str
    name: str
    media_type: str
    size: int
    relative_path: str
    track_id: str | None = None
    sha256: str = ""


@dataclass
class JobRecord:
    id: str
    kind: str
    status: JobStatus
    created_at: str
    updated_at: str
    original_filename: str
    options: dict[str, Any]
    source_path: Path
    output_dir: Path
    tempo_source_path: Path | None = None
    parent_job_id: str | None = None
    retry_of_job_id: str | None = None
    track_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    revision: int = 1
    progress: dict[str, Any] = field(
        default_factory=lambda: {
            "stage": "queued",
            "stage_progress": 0.0,
            "overall_progress": 0.0,
            "message": "任务已进入推理队列",
            "bpm_display": None,
            "source_bpm": None,
            "target_bpm": None,
        }
    )
    error: str | None = None
    result: dict[str, Any] | None = None
    artifacts: dict[str, ArtifactRecord] = field(default_factory=dict)
    processor: object | None = field(default=None, repr=False)
    cancel_requested: bool = False


class JobManager:
    """Own job files, state transitions, cancellation, and one GPU worker."""

    MANIFEST_NAME = "job.json"

    def __init__(
        self,
        root: str | Path,
        *,
        engine: InferenceEngine | None = None,
        start_worker: bool = True,
        retention_days: int = 30,
        retention_max_jobs: int = 200,
        retention_max_bytes: int = 0,
        max_queued_jobs: int = 0,
        min_free_bytes: int = 0,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._engine = engine or InferenceEngine()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._quarantine_cleanup_lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._pending: list[str] = []
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._closed = False
        if isinstance(retention_days, bool) or not 0 <= int(retention_days) <= 3650:
            raise ValueError("retention_days must be an integer between 0 and 3650")
        if isinstance(retention_max_jobs, bool) or not 0 <= int(retention_max_jobs) <= 100000:
            raise ValueError("retention_max_jobs must be an integer between 0 and 100000")
        if isinstance(retention_max_bytes, bool) or int(retention_max_bytes) < 0:
            raise ValueError("retention_max_bytes must be a non-negative integer")
        if isinstance(max_queued_jobs, bool) or not 0 <= int(max_queued_jobs) <= 100000:
            raise ValueError("max_queued_jobs must be an integer between 0 and 100000")
        if isinstance(min_free_bytes, bool) or int(min_free_bytes) < 0:
            raise ValueError("min_free_bytes must be a non-negative integer")
        self.retention_days = int(retention_days)
        self.retention_max_jobs = int(retention_max_jobs)
        self.retention_max_bytes = int(retention_max_bytes)
        self.max_queued_jobs = int(max_queued_jobs)
        self.min_free_bytes = int(min_free_bytes)
        self._last_retention_run: str | None = None
        self._last_retention_deleted_jobs = 0
        self._last_retention_deleted_bytes = 0
        self._last_retention_error: str | None = None
        self._load_manifests()
        self.prune_retained_jobs()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="music-to-midi-inference",
            daemon=True,
        )
        if start_worker:
            self._worker.start()

    def require_submission_capacity(self, *, additional_bytes: int = 0) -> None:
        if isinstance(additional_bytes, bool) or int(additional_bytes) < 0:
            raise ValueError("additional_bytes must be a non-negative integer")
        free_bytes = shutil.disk_usage(self.root).free
        if free_bytes - int(additional_bytes) < self.min_free_bytes:
            raise InsufficientStorageError(
                "job storage does not have enough free space: "
                f"free={free_bytes}, incoming={int(additional_bytes)}, "
                f"required_reserve={self.min_free_bytes}"
            )
        with self._lock:
            if self._closed:
                raise RuntimeError("job manager is closed")
            if self.max_queued_jobs and len(self._pending) >= self.max_queued_jobs:
                raise QueueCapacityError(
                    "inference queue is full: "
                    f"queued={len(self._pending)}, limit={self.max_queued_jobs}"
                )

    def reserve_job_dir(self) -> tuple[str, Path]:
        with self._lock:
            if self._closed:
                raise RuntimeError("job manager is closed")
            while True:
                job_id = uuid.uuid4().hex
                job_dir = self.root / job_id
                try:
                    job_dir.mkdir(parents=False, exist_ok=False)
                    (job_dir / "input").mkdir()
                    (job_dir / "output").mkdir()
                    return job_id, job_dir
                except FileExistsError:
                    continue

    def discard_reserved(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        with self._lock:
            if job_id in self._jobs:
                raise RuntimeError(f"cannot discard submitted job {job_id}")
        shutil.rmtree(job_dir, ignore_errors=False)

    def submit_reserved(
        self,
        *,
        job_id: str,
        kind: str,
        source_path: str | Path,
        tempo_source_path: str | Path | None = None,
        original_filename: str,
        options: dict[str, Any],
        parent_job_id: str | None = None,
        retry_of_job_id: str | None = None,
        track_id: str | None = None,
    ) -> dict[str, Any]:
        self.require_submission_capacity()
        source = Path(source_path).resolve()
        tempo_source = Path(tempo_source_path).resolve() if tempo_source_path else None
        job_dir = self._job_dir(job_id)
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"job input does not exist or is empty: {source}")
        if not job_dir.is_dir():
            raise FileNotFoundError(f"reserved job directory is missing: {job_dir}")
        if tempo_source is not None and (
            not tempo_source.is_file() or tempo_source.stat().st_size <= 0
        ):
            raise FileNotFoundError(f"job tempo source does not exist or is empty: {tempo_source}")
        now = _utc_now()
        record = JobRecord(
            id=job_id,
            kind=kind,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            original_filename=original_filename,
            options=dict(options),
            source_path=source,
            tempo_source_path=tempo_source,
            output_dir=(job_dir / "output").resolve(),
            parent_job_id=parent_job_id,
            retry_of_job_id=retry_of_job_id,
            track_id=track_id,
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("job manager is closed")
            if self.max_queued_jobs and len(self._pending) >= self.max_queued_jobs:
                raise QueueCapacityError(
                    "inference queue is full: "
                    f"queued={len(self._pending)}, limit={self.max_queued_jobs}"
                )
            if job_id in self._jobs:
                raise RuntimeError(f"job already exists: {job_id}")
            self._jobs[job_id] = record
            self._pending.append(job_id)
            self._persist_locked(record)
            self._queue.put(job_id)
            self._condition.notify_all()
            return self._snapshot_locked(record)

    def submit_parent_track(
        self,
        parent_job_id: str,
        track_id: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            parent = self._require_job_locked(parent_job_id)
            if parent.status is not JobStatus.SUCCEEDED:
                raise RuntimeError("separation job has not completed successfully")
            if not parent.result or not parent.result.get("manual_midi_required"):
                raise RuntimeError("job is not a WAV-only separation result")
            track = next(
                (item for item in parent.result.get("tracks", []) if item.get("id") == track_id),
                None,
            )
            if track is None:
                raise KeyError(f"unknown separated track: {track_id}")
            artifact = parent.artifacts.get(str(track.get("artifact_id")))
            if artifact is None:
                raise RuntimeError(f"separated track artifact is missing: {track_id}")
            source_path = self._artifact_path_locked(parent, artifact)
            original_filename = source_path.name
        job_id, _job_dir = self.reserve_job_dir()
        try:
            return self.submit_reserved(
                job_id=job_id,
                kind="manual_midi",
                source_path=source_path,
                tempo_source_path=parent.source_path,
                original_filename=original_filename,
                options=options,
                parent_job_id=parent_job_id,
                track_id=track_id,
            )
        except Exception:
            self.discard_reserved(job_id)
            raise

    def retry(self, job_id: str) -> dict[str, Any]:
        """Create a new explicit job from a terminal job's retained input and options."""
        with self._lock:
            source_record = self._require_job_locked(job_id)
            if not source_record.status.terminal:
                raise RuntimeError("running or queued jobs cannot be retried")
            source_path = source_record.source_path
            tempo_source_path = source_record.tempo_source_path
            if not source_path.is_file() or source_path.stat().st_size <= 0:
                raise RuntimeError("the retained job input is missing or empty")
            source_size = source_path.stat().st_size
            kind = source_record.kind
            original_filename = source_record.original_filename
            options = dict(source_record.options)
            parent_job_id = source_record.parent_job_id
            track_id = source_record.track_id

        self.require_submission_capacity(additional_bytes=source_size)
        retry_id, retry_dir = self.reserve_job_dir()
        destination = retry_dir / "input" / f"source{source_path.suffix.lower()}"
        try:
            with source_path.open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            if destination.stat().st_size != source_size:
                raise RuntimeError("retained job input changed while it was copied for retry")
            return self.submit_reserved(
                job_id=retry_id,
                kind=kind,
                source_path=destination,
                tempo_source_path=(
                    destination
                    if tempo_source_path is not None
                    and tempo_source_path.resolve() == source_path.resolve()
                    else tempo_source_path
                ),
                original_filename=original_filename,
                options=options,
                parent_job_id=parent_job_id,
                retry_of_job_id=job_id,
                track_id=track_id,
            )
        except Exception:
            self.discard_reserved(retry_id)
            raise

    def list_snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            ordered = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [self._snapshot_locked(record) for record in ordered]

    def snapshot(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked(self._require_job_locked(job_id))

    def wait_for_revision(
        self,
        job_id: str,
        after_revision: int,
        timeout: float,
    ) -> dict[str, Any] | None:
        with self._condition:
            record = self._require_job_locked(job_id)
            if record.revision > after_revision:
                return self._snapshot_locked(record)
            self._condition.wait_for(
                lambda: self._require_job_locked(job_id).revision > after_revision,
                timeout=max(0.0, float(timeout)),
            )
            record = self._require_job_locked(job_id)
            if record.revision <= after_revision:
                return None
            return self._snapshot_locked(record)

    def cancel(self, job_id: str) -> dict[str, Any]:
        processor = None
        with self._condition:
            record = self._require_job_locked(job_id)
            if record.status.terminal:
                return self._snapshot_locked(record)
            record.cancel_requested = True
            now = _utc_now()
            if record.status is JobStatus.QUEUED:
                record.status = JobStatus.CANCELLED
                record.finished_at = now
                record.progress = {
                    **record.progress,
                    "stage": "cancelled",
                    "message": "任务在开始推理前已取消",
                }
                if job_id in self._pending:
                    self._pending.remove(job_id)
            else:
                record.status = JobStatus.CANCELLING
                record.progress = {
                    **record.progress,
                    "message": "正在停止当前推理任务",
                }
                processor = record.processor
            self._touch_locked(record)
            snapshot = self._snapshot_locked(record)
        if processor is not None:
            cancel = getattr(processor, "cancel", None)
            if not callable(cancel):
                raise RuntimeError("active inference processor does not support cancellation")
            cancel()
        return snapshot

    def resolve_artifact(self, job_id: str, artifact_id: str) -> tuple[Path, ArtifactRecord]:
        with self._lock:
            record = self._require_job_locked(job_id)
            if record.status is not JobStatus.SUCCEEDED:
                raise RuntimeError("job artifacts are unavailable before successful completion")
            artifact = record.artifacts.get(artifact_id)
            if artifact is None:
                raise KeyError(f"unknown artifact: {artifact_id}")
            return self._artifact_path_locked(record, artifact), artifact

    def delete(self, job_id: str, *, cascade: bool = False) -> None:
        with self._condition:
            record = self._require_job_locked(job_id)
            if not record.status.terminal:
                raise RuntimeError("running or queued jobs must be cancelled before deletion")
            descendants = self._descendant_ids_locked(job_id)
            if descendants and not cascade:
                raise RuntimeError(
                    "job owns manual MIDI children; repeat deletion with cascade=true"
                )
            delete_ids = [*descendants, job_id]
            active = [
                candidate
                for candidate in delete_ids
                if not self._require_job_locked(candidate).status.terminal
            ]
            if active:
                raise RuntimeError("job family still contains active tasks: " + ", ".join(active))
            quarantined, _deleted_bytes = self._quarantine_jobs_locked(delete_ids)
        self._remove_quarantined(quarantined)

    def prune_retained_jobs(self) -> dict[str, Any]:
        """Apply age/count retention atomically to complete parent/child families."""
        trash_root = self.root / ".trash"
        if trash_root.is_dir():
            self._remove_quarantined([path for path in trash_root.iterdir() if path.is_dir()])

        with self._condition:
            families = self._job_families_locked()
            terminal_families: list[tuple[str, list[str], datetime, int]] = []
            for root_id, family_ids in families.items():
                records = [self._require_job_locked(item) for item in family_ids]
                if not all(record.status.terminal for record in records):
                    continue
                newest = max(
                    _parse_utc(record.finished_at or record.updated_at or record.created_at)
                    for record in records
                )
                family_bytes = sum(self._job_dir_size(job_id) for job_id in family_ids)
                terminal_families.append((root_id, family_ids, newest, family_bytes))

            selected_roots: set[str] = set()
            if self.retention_days:
                cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
                selected_roots.update(
                    root_id
                    for root_id, _family_ids, newest, _family_bytes in terminal_families
                    if newest < cutoff
                )

            remaining = [
                family
                for family in sorted(terminal_families, key=lambda item: item[2])
                if family[0] not in selected_roots
            ]
            remaining_job_count = sum(len(family_ids) for _, family_ids, _, _ in remaining)
            if self.retention_max_jobs:
                while remaining_job_count > self.retention_max_jobs and remaining:
                    root_id, family_ids, _newest, _family_bytes = remaining.pop(0)
                    selected_roots.add(root_id)
                    remaining_job_count -= len(family_ids)

            remaining_bytes = sum(family_bytes for _, _, _, family_bytes in remaining)
            if self.retention_max_bytes:
                while remaining_bytes > self.retention_max_bytes and remaining:
                    root_id, _family_ids, _newest, family_bytes = remaining.pop(0)
                    selected_roots.add(root_id)
                    remaining_bytes -= family_bytes

            delete_ids = [
                job_id
                for root_id, family_ids, _newest, _family_bytes in terminal_families
                if root_id in selected_roots
                for job_id in family_ids
            ]
            if delete_ids:
                quarantined, deleted_bytes = self._quarantine_jobs_locked(delete_ids)
            else:
                quarantined, deleted_bytes = [], 0

        try:
            self._remove_quarantined(quarantined)
        except Exception as exc:
            with self._lock:
                self._last_retention_error = str(exc)
            raise

        with self._lock:
            self._last_retention_run = _utc_now()
            self._last_retention_deleted_jobs = len(delete_ids)
            self._last_retention_deleted_bytes = deleted_bytes
            self._last_retention_error = None
        if delete_ids:
            LOGGER.info(
                "Retention removed %d terminal jobs (%d bytes) across %d families",
                len(delete_ids),
                deleted_bytes,
                len(selected_roots),
            )
        return {
            "deleted_jobs": len(delete_ids),
            "deleted_bytes": deleted_bytes,
            "deleted_families": len(selected_roots),
        }

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "worker_alive": self._worker.is_alive(),
                "queued_jobs": len(self._pending),
                "max_queued_jobs": self.max_queued_jobs,
                "known_jobs": len(self._jobs),
                "data_root": str(self.root),
                "storage": {
                    "free_bytes": shutil.disk_usage(self.root).free,
                    "min_free_bytes": self.min_free_bytes,
                },
                "retention": {
                    "days": self.retention_days,
                    "max_jobs": self.retention_max_jobs,
                    "max_bytes": self.retention_max_bytes,
                    "last_run": self._last_retention_run,
                    "last_deleted_jobs": self._last_retention_deleted_jobs,
                    "last_deleted_bytes": self._last_retention_deleted_bytes,
                    "error": self._last_retention_error,
                },
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            active = [
                record
                for record in self._jobs.values()
                if record.status in {JobStatus.RUNNING, JobStatus.CANCELLING}
            ]
        for record in active:
            processor = record.processor
            cancel = getattr(processor, "cancel", None)
            if callable(cancel):
                cancel()
        self._queue.put(None)
        if self._worker.is_alive():
            self._worker.join(timeout=30.0)

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            with self._condition:
                record = self._jobs.get(job_id)
                if record is None or record.status is JobStatus.CANCELLED:
                    continue
                if record.status is not JobStatus.QUEUED:
                    continue
                if job_id in self._pending:
                    self._pending.remove(job_id)
                record.status = JobStatus.RUNNING
                record.started_at = _utc_now()
                record.progress = {
                    **record.progress,
                    "stage": "starting",
                    "message": "后端已取得独占推理槽，正在初始化",
                }
                self._touch_locked(record)

            def progress_callback(progress: ProcessingProgress) -> None:
                self._update_progress(job_id, progress)

            def processor_callback(processor: object | None) -> None:
                self._bind_processor(job_id, processor)

            try:
                run_kwargs = dict(
                    kind=record.kind,
                    source_path=record.source_path,
                    output_dir=record.output_dir,
                    options=record.options,
                    progress_callback=progress_callback,
                    processor_callback=processor_callback,
                    track_id=record.track_id,
                )
                if record.tempo_source_path is not None:
                    run_kwargs["tempo_source_path"] = record.tempo_source_path
                execution = self._engine.run(**run_kwargs)
                self._finish_success(job_id, execution)
            except InterruptedError:
                self._finish_cancelled(job_id, "任务已按用户请求停止")
            except Exception as exc:
                self._finish_failed(job_id, str(exc))
            finally:
                try:
                    self.prune_retained_jobs()
                except Exception as exc:
                    with self._lock:
                        self._last_retention_error = str(exc)
                    LOGGER.exception("Retention maintenance failed after job %s", job_id)

    def _bind_processor(self, job_id: str, processor: object | None) -> None:
        cancel_immediately = False
        with self._lock:
            record = self._require_job_locked(job_id)
            record.processor = processor
            cancel_immediately = bool(processor is not None and record.cancel_requested)
        if cancel_immediately:
            cancel = getattr(processor, "cancel", None)
            if not callable(cancel):
                raise RuntimeError("active inference processor does not support cancellation")
            cancel()

    def _update_progress(self, job_id: str, progress: ProcessingProgress) -> None:
        with self._condition:
            record = self._require_job_locked(job_id)
            if record.status.terminal:
                return
            payload = progress.to_dict()
            payload["stage_progress"] = max(0.0, min(1.0, float(payload["stage_progress"])))
            payload["overall_progress"] = max(0.0, min(1.0, float(payload["overall_progress"])))
            record.progress = payload
            self._touch_locked(record)

    def _finish_success(self, job_id: str, execution: ExecutionResult) -> None:
        with self._condition:
            record = self._require_job_locked(job_id)
            if record.cancel_requested:
                record.status = JobStatus.CANCELLED
                record.finished_at = _utc_now()
                record.progress = {
                    **record.progress,
                    "stage": "cancelled",
                    "message": "任务在产物发布前已取消",
                }
                self._touch_locked(record)
                return
            artifact_records: dict[str, ArtifactRecord] = {}
            for spec in execution.artifacts:
                artifact = self._artifact_record_locked(record, spec)
                if artifact.id in artifact_records:
                    raise RuntimeError(f"duplicate artifact id: {artifact.id}")
                artifact_records[artifact.id] = artifact
            record.artifacts = artifact_records
            record.result = dict(execution.result)
            record.status = JobStatus.SUCCEEDED
            record.finished_at = _utc_now()
            record.error = None
            record.progress = {
                **record.progress,
                "stage": "complete",
                "stage_progress": 1.0,
                "overall_progress": 1.0,
                "message": "推理与产物校验已完成",
            }
            self._touch_locked(record)

    def _finish_cancelled(self, job_id: str, message: str) -> None:
        with self._condition:
            record = self._require_job_locked(job_id)
            record.status = JobStatus.CANCELLED
            record.finished_at = _utc_now()
            record.processor = None
            record.progress = {**record.progress, "stage": "cancelled", "message": message}
            self._touch_locked(record)

    def _finish_failed(self, job_id: str, error: str) -> None:
        with self._condition:
            record = self._require_job_locked(job_id)
            record.status = JobStatus.FAILED
            record.finished_at = _utc_now()
            record.processor = None
            record.error = error or "backend inference failed without an error message"
            record.progress = {
                **record.progress,
                "stage": "failed",
                "message": "推理失败，后端已停止该任务",
            }
            self._touch_locked(record)

    def _artifact_record_locked(
        self,
        record: JobRecord,
        spec: ArtifactSpec,
    ) -> ArtifactRecord:
        path = spec.path.resolve()
        job_dir = self._job_dir(record.id)
        try:
            relative = path.relative_to(job_dir)
        except ValueError as exc:
            raise RuntimeError(f"artifact escaped its job directory: {path}") from exc
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"artifact does not exist or is empty: {path}")
        return ArtifactRecord(
            id=spec.id,
            kind=spec.kind,
            name=path.name,
            media_type=spec.media_type,
            size=path.stat().st_size,
            relative_path=relative.as_posix(),
            track_id=spec.track_id,
            sha256=self._sha256(path),
        )

    def _artifact_path_locked(self, record: JobRecord, artifact: ArtifactRecord) -> Path:
        job_dir = self._job_dir(record.id)
        path = (job_dir / artifact.relative_path).resolve()
        try:
            path.relative_to(job_dir)
        except ValueError as exc:
            raise RuntimeError("artifact path escaped its owning job") from exc
        if not path.is_file() or path.stat().st_size != artifact.size:
            raise RuntimeError(f"artifact is missing or changed after publication: {path}")
        if artifact.sha256 and self._sha256(path) != artifact.sha256:
            raise RuntimeError(f"artifact checksum changed after publication: {path}")
        return path

    def _snapshot_locked(self, record: JobRecord) -> dict[str, Any]:
        queue_position = None
        if record.status is JobStatus.QUEUED and record.id in self._pending:
            queue_position = self._pending.index(record.id) + 1
        artifacts = [
            {
                "id": artifact.id,
                "kind": artifact.kind,
                "name": artifact.name,
                "media_type": artifact.media_type,
                "size": artifact.size,
                "track_id": artifact.track_id,
                "sha256": artifact.sha256,
                "download_url": f"/api/v1/jobs/{record.id}/files/{artifact.id}",
            }
            for artifact in record.artifacts.values()
        ]
        return {
            "id": record.id,
            "kind": record.kind,
            "status": record.status.value,
            "revision": record.revision,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "original_filename": record.original_filename,
            "parent_job_id": record.parent_job_id,
            "retry_of_job_id": record.retry_of_job_id,
            "track_id": record.track_id,
            "queue_position": queue_position,
            "request": dict(record.options),
            "progress": dict(record.progress),
            "error": record.error,
            "result": dict(record.result) if record.result is not None else None,
            "artifacts": artifacts,
        }

    def _touch_locked(self, record: JobRecord) -> None:
        record.updated_at = _utc_now()
        record.revision += 1
        self._persist_locked(record)
        self._condition.notify_all()

    def _persist_locked(self, record: JobRecord) -> None:
        job_dir = self._job_dir(record.id)
        manifest = {
            **self._snapshot_locked(record),
            "options": record.options,
            "source_path": self._path_for_manifest(record.source_path),
            "tempo_source_path": (
                self._path_for_manifest(record.tempo_source_path)
                if record.tempo_source_path is not None
                else None
            ),
            "output_dir": self._path_for_manifest(record.output_dir),
            "artifacts": [artifact.__dict__ for artifact in record.artifacts.values()],
        }
        target = job_dir / self.MANIFEST_NAME
        temporary = job_dir / f".{self.MANIFEST_NAME}.{uuid.uuid4().hex}.tmp"
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)

    def _load_manifests(self) -> None:
        for manifest_path in sorted(self.root.glob(f"*/{self.MANIFEST_NAME}")):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                status = JobStatus(payload["status"])
                if not status.terminal:
                    status = JobStatus.FAILED
                    payload["error"] = (
                        "处理服务在任务完成前退出，本次处理无法继续；原始音频仍可重新提交。"
                    )
                    payload["finished_at"] = _utc_now()
                artifacts = {
                    item["id"]: ArtifactRecord(**item) for item in payload.get("artifacts", [])
                }
                manifest_revision = int(payload.get("revision", 1))
                if payload["status"] != status.value:
                    manifest_revision += 1
                record = JobRecord(
                    id=payload["id"],
                    kind=payload["kind"],
                    status=status,
                    created_at=payload["created_at"],
                    updated_at=payload.get("updated_at", payload["created_at"]),
                    original_filename=payload.get("original_filename", "input"),
                    options=dict(payload.get("options") or {}),
                    source_path=self._path_from_manifest(payload["source_path"]),
                    tempo_source_path=(
                        self._path_from_manifest(payload["tempo_source_path"])
                        if payload.get("tempo_source_path")
                        else None
                    ),
                    output_dir=self._path_from_manifest(payload["output_dir"]),
                    parent_job_id=payload.get("parent_job_id"),
                    retry_of_job_id=payload.get("retry_of_job_id"),
                    track_id=payload.get("track_id"),
                    started_at=payload.get("started_at"),
                    finished_at=payload.get("finished_at"),
                    revision=manifest_revision,
                    progress=dict(payload.get("progress") or {}),
                    error=payload.get("error"),
                    result=payload.get("result"),
                    artifacts=artifacts,
                )
                migrated_artifact_hash = False
                for artifact in record.artifacts.values():
                    if not artifact.sha256:
                        artifact.sha256 = self._sha256(self._artifact_path_locked(record, artifact))
                        migrated_artifact_hash = True
                self._jobs[record.id] = record
                interrupted_job = status is JobStatus.FAILED and payload["status"] != status.value
                if interrupted_job:
                    record.progress = {
                        **record.progress,
                        "stage": "failed",
                        "message": "服务异常退出，任务已明确标记失败",
                    }
                if interrupted_job or migrated_artifact_hash:
                    self._persist_locked(record)
            except Exception as exc:
                raise RuntimeError(f"cannot load job manifest {manifest_path}: {exc}") from exc

    def _path_for_manifest(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError:
            return str(resolved)

    def _path_from_manifest(self, value: str) -> Path:
        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()

    def _descendant_ids_locked(self, job_id: str) -> list[str]:
        descendants: list[str] = []
        pending = [job_id]
        while pending:
            parent_id = pending.pop()
            children = sorted(
                record.id for record in self._jobs.values() if record.parent_job_id == parent_id
            )
            for child_id in children:
                if child_id in descendants:
                    raise RuntimeError("job parent graph contains a cycle")
                descendants.append(child_id)
                pending.append(child_id)
        return descendants

    def _job_families_locked(self) -> dict[str, list[str]]:
        families: dict[str, list[str]] = {}
        for record in self._jobs.values():
            root_id = record.id
            parent_id = record.parent_job_id
            visited = {record.id}
            while parent_id is not None and parent_id in self._jobs:
                if parent_id in visited:
                    raise RuntimeError("job parent graph contains a cycle")
                visited.add(parent_id)
                root_id = parent_id
                parent_id = self._jobs[parent_id].parent_job_id
            families.setdefault(root_id, []).append(record.id)
        for family_ids in families.values():
            family_ids.sort()
        return families

    def _quarantine_jobs_locked(self, job_ids: list[str]) -> tuple[list[Path], int]:
        unique_ids = list(dict.fromkeys(job_ids))
        trash_root = self.root / ".trash"
        trash_root.mkdir(parents=True, exist_ok=True)
        moved: list[tuple[Path, Path]] = []
        deleted_bytes = 0
        try:
            for job_id in unique_ids:
                self._require_job_locked(job_id)
                source = self._job_dir(job_id)
                if not source.is_dir():
                    raise RuntimeError(f"job directory is missing: {source}")
                deleted_bytes += sum(
                    path.stat().st_size for path in source.rglob("*") if path.is_file()
                )
                quarantined = trash_root / f"{job_id}-{uuid.uuid4().hex}"
                os.replace(source, quarantined)
                moved.append((source, quarantined))
        except Exception:
            for source, quarantined in reversed(moved):
                if quarantined.exists() and not source.exists():
                    os.replace(quarantined, source)
            raise

        for job_id in unique_ids:
            del self._jobs[job_id]
            if job_id in self._pending:
                self._pending.remove(job_id)
        self._condition.notify_all()
        return [quarantined for _source, quarantined in moved], deleted_bytes

    def _remove_quarantined(self, paths: list[Path]) -> None:
        with self._quarantine_cleanup_lock:
            for path in paths:
                if not path.exists():
                    continue
                try:
                    shutil.rmtree(path, ignore_errors=False)
                except OSError as exc:
                    raise RuntimeError(
                        f"failed to remove quarantined job data {path}: {exc}"
                    ) from exc

    def _require_job_locked(self, job_id: str) -> JobRecord:
        record = self._jobs.get(job_id)
        if record is None:
            raise KeyError(f"unknown job: {job_id}")
        return record

    def _job_dir(self, job_id: str) -> Path:
        if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
            raise ValueError("invalid job id")
        return (self.root / job_id).resolve()

    def _job_dir_size(self, job_id: str) -> int:
        job_dir = self._job_dir(job_id)
        if not job_dir.is_dir():
            return 0
        return sum(path.stat().st_size for path in job_dir.rglob("*") if path.is_file())

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
