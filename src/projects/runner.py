"""Sequential execution with independently reusable separation and MIDI steps."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Callable

from src.projects.store import (
    Checkpoint,
    ProjectStore,
    SPLIT_STEMS,
    exclusive,
    now,
    step_signature,
    plan_complete,
)

# Bump only when the output contract changes, not when UI/packaging changes.
EXECUTION_CONTRACT = 1


class ProjectRunner:
    def __init__(
        self,
        store: ProjectStore,
        engine_factory: Callable | None = None,
        on_update: Callable[[dict], None] | None = None,
    ):
        self.store = store
        self.engine_factory = engine_factory
        self.on_update = on_update
        self.cancelled = threading.Event()
        self._processor = None
        self._processor_lock = threading.Lock()

    def cancel(self):
        self.cancelled.set()
        with self._processor_lock:
            if self._processor is not None:
                self._processor.cancel()

    def _bind(self, processor):
        with self._processor_lock:
            self._processor = processor
            if processor is not None and self.cancelled.is_set():
                processor.cancel()

    def _check_cancel(self):
        if self.cancelled.is_set():
            raise InterruptedError("用户已停止项目处理")

    def _publish(self, document):
        self.store.save(document)
        if self.on_update:
            self.on_update(self.store.snapshot())

    def _step(self, document, song, *, kind, source, options, track_id=None):
        self._check_cancel()
        source_path = self.store.verify(source)
        signature = step_signature(kind, source.sha256, song.source.sha256, options)
        if kind == "primary":
            song.primary_key = signature
        else:
            song.track_keys[track_id] = signature
        saved = song.checkpoints.get(signature)
        if saved and saved.status == "succeeded":
            if step_signature(kind, source.sha256, song.source.sha256, saved.options) != signature:
                raise RuntimeError("保存的模型参数与断点签名不一致")
            if not saved.artifacts or saved.result is None:
                raise RuntimeError("成功断点缺少产物或结果，不能复用")
            for asset in saved.artifacts:
                self.store.verify(asset)
            document.progress.update(
                stage="reused", track_id=track_id, percent=100, message="已校验并复用完成结果"
            )
            self._publish(document)
            return saved
        if saved and saved.status == "running":
            saved.status = "interrupted"
            saved.error = "上次执行中断；本次从该步骤重新开始"
            self._publish(document)
        directory = self.store.root / "songs" / song.id / "outputs" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        step = Checkpoint(
            signature=signature,
            kind=kind,
            options=options,
            status="running",
            output_dir=directory.relative_to(self.store.root).as_posix(),
        )
        song.checkpoints[signature] = step
        document.progress.update(stage=kind, track_id=track_id, percent=0, message="开始处理")
        self._publish(document)

        def progress(update):
            self._check_cancel()
            document.progress.update(
                stage=update.stage.value,
                track_id=track_id,
                percent=round(update.overall_progress * 100),
                message=update.message,
            )
            self._publish(document)

        try:
            if self.engine_factory is None:
                from src.web_api.engine import InferenceEngine

                engine = InferenceEngine()
            else:
                engine = self.engine_factory()
            execution = engine.run(
                kind=kind,
                source_path=source_path,
                output_dir=directory,
                options=options,
                progress_callback=progress,
                processor_callback=self._bind,
                track_id=track_id,
                tempo_source_path=self.store.verify(song.source) if kind == "manual_midi" else None,
            )
            self._check_cancel()
            self.store.verify(source)
            if not execution.artifacts or not isinstance(execution.result, dict):
                raise RuntimeError("推理未返回有效产物及结果")
            assets = []
            for artifact in execution.artifacts:
                path = Path(artifact.path).resolve()
                if not path.is_relative_to(directory):
                    raise RuntimeError("推理产物不在本次步骤输出目录中")
                assets.append(
                    self.store.asset(
                        path,
                        id=artifact.id,
                        kind=artifact.kind,
                        track_id=artifact.track_id,
                        media_type=artifact.media_type,
                    )
                )
            if kind == "primary" and options["processing_mode"] in SPLIT_STEMS:
                actual = [a.track_id for a in assets if a.kind == "audio_track"]
                if sorted(actual) != sorted(SPLIT_STEMS[options["processing_mode"]]):
                    raise RuntimeError("分离声部不完整，无法保存为成功断点")
            elif not any(a.kind == "midi" for a in assets):
                raise RuntimeError("转写没有生成 MIDI")
            step.artifacts = assets
            step.result = execution.result
            step.status = "succeeded"
            step.finished_at = now()
            document.progress.update(percent=100, message="处理完成")
        except (InterruptedError, KeyboardInterrupt) as exc:
            step.status, step.error, step.finished_at = "cancelled", str(exc), now()
            self._publish(document)
            raise
        except Exception as exc:
            step.status, step.error, step.finished_at = (
                "failed",
                f"{type(exc).__name__}: {exc}",
                now(),
            )
            self._publish(document)
            raise
        finally:
            self._bind(None)
        self._publish(document)
        return step

    def run(
        self,
        song_ids: list[str] | None = None,
        *,
        fail_fast: bool = False,
        track_ids: list[str] | None = None,
        primary_only: bool = False,
    ) -> dict:
        with exclusive(self.store.run_lock):
            document = self.store.load()
            ids = set(song_ids if song_ids is not None else (s.id for s in document.songs))
            if not ids or ids - {s.id for s in document.songs}:
                raise ValueError("没有选择有效歌曲")
            selected = [s for s in document.songs if s.id in ids]
            if track_ids is not None and (
                not track_ids or any(set(track_ids) - set(s.workflow.stems) for s in selected)
            ):
                raise ValueError("所选声部没有已保存的转写计划")
            document.status, document.error = "running", None
            self._publish(document)
            failed = False
            for index, song in enumerate(selected, 1):
                if self.cancelled.is_set():
                    break
                song.status, song.error = "running", None
                document.progress = {
                    "index": index,
                    "total": len(selected),
                    "song_id": song.id,
                    "song_name": song.name,
                    "percent": 0,
                    "track_id": None,
                }
                self._publish(document)
                try:
                    primary = self._step(
                        document,
                        song,
                        kind="primary",
                        source=song.source,
                        options=song.workflow.primary.model_dump(mode="json"),
                    )
                    for track_id, options in song.workflow.stems.items():
                        if primary_only:
                            break
                        if track_ids is not None and track_id not in track_ids:
                            continue
                        source = song.extra_tracks.get(track_id) or next(
                            (
                                a
                                for a in primary.artifacts
                                if a.kind == "audio_track" and a.track_id == track_id
                            ),
                            None,
                        )
                        if source is None:
                            raise RuntimeError(f"找不到分离声部：{track_id}")
                        self._step(
                            document,
                            song,
                            kind="manual_midi",
                            source=source,
                            options=options.model_dump(mode="json"),
                            track_id=track_id,
                        )
                    song.status = "succeeded" if plan_complete(song) else "pending"
                except (InterruptedError, KeyboardInterrupt) as exc:
                    self.cancelled.set()
                    song.status, song.error = "cancelled", str(exc)
                    self._publish(document)
                    break
                except Exception as exc:
                    failed = True
                    song.status, song.error = "failed", f"{type(exc).__name__}: {exc}"
                    document.error = song.error
                    self._publish(document)
                    if fail_fast:
                        break
                self._publish(document)
            document.status = (
                "cancelled" if self.cancelled.is_set() else "failed" if failed else "succeeded"
            )
            document.last_run_status = document.status
            if document.status == "succeeded" and any(
                s.status != "succeeded" for s in document.songs
            ):
                document.status = "pending"
            self._publish(document)
            return self.store.snapshot()
