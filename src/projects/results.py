"""Resolve saved project results for the existing MIDI presentation tools."""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

from src.projects.store import exclusive


class ProjectResults:
    def __init__(self, service):
        self.service = service
        self._lock = threading.RLock()

    def context(self, result_id):
        match = re.fullmatch(r"project-([0-9a-f]{32})-([0-9a-f]{32})-([0-9a-f]{64})", result_id)
        if not match:
            raise ValueError("项目结果 ID 无效")
        project_id, song_id, signature = match.groups()
        store = self.service.store(project_id)
        document = store.load()
        song = next((s for s in document.songs if s.id == song_id), None)
        step = song.checkpoints.get(signature) if song else None
        if step is None:
            raise KeyError("项目中没有此结果")
        if step.status != "succeeded":
            raise RuntimeError("此步骤尚未成功完成")
        return store, document, song, step, project_id

    def resolve_artifact(self, result_id, artifact_id):
        store, _, _, step, _ = self.context(result_id)
        artifact = next((a for a in step.artifacts if a.id == artifact_id), None)
        if not artifact:
            raise KeyError("项目中没有此产物")
        return store.verify(artifact), artifact

    def snapshot(self, result_id):
        _, document, song, step, project_id = self.context(result_id)
        return dict(
            id=result_id,
            kind=step.kind,
            status=step.status,
            revision=document.revision,
            created_at=step.started_at,
            updated_at=document.updated_at,
            started_at=step.started_at,
            finished_at=step.finished_at,
            original_filename=song.name,
            request=step.options,
            result=step.result,
            progress={},
            artifacts=[
                dict(
                    a.model_dump(),
                    name=Path(a.path).name,
                    download_url=f"/api/v1/projects/{project_id}/files/{quote(a.path, safe='/')}",
                )
                for a in step.artifacts
            ],
        )

    def register_soundfont(self, result_id, paths):
        store, _, _, _, _ = self.context(result_id)
        with exclusive(store.run_lock):
            store, document, _, step, _ = self.context(result_id)
            for path in paths:
                asset = store.asset(
                    Path(path), id="soundfont_" + uuid.uuid4().hex, kind="soundfont"
                )
                step.artifacts = [a for a in step.artifacts if a.path != asset.path]
                step.artifacts.append(asset)
            store.save(document)

    def generate_sheet_music(self, result_id, *, artifact_id, quantize_grid="1/32"):
        from src.core.sheet_music import export_sheet_music_zip
        from src.core.midi_quantization import MIDI_QUANTIZE_GRIDS

        if quantize_grid not in MIDI_QUANTIZE_GRIDS:
            raise ValueError("乐谱量化网格无效")
        store, _, _, _, _ = self.context(result_id)
        with exclusive(store.run_lock):
            store, document, _, step, _ = self.context(result_id)
            source, original = self.resolve_artifact(result_id, artifact_id)
            if original.kind != "midi":
                raise ValueError("乐谱输入必须是 MIDI")
            identity = f"sheet_{original.sha256[:16]}_{quantize_grid.split('/')[1]}"
            existing = next((a for a in step.artifacts if a.id == identity), None)
            if existing:
                store.verify(existing)
            else:
                target = source.parent / "sheet_music" / (identity + ".zip")
                target.parent.mkdir(parents=True, exist_ok=True)
                export_sheet_music_zip(source, target, quantize_grid=quantize_grid)
                step.artifacts.append(
                    store.asset(
                        target,
                        id=identity,
                        kind="sheet_music",
                        media_type="application/zip",
                        track_id=original.track_id,
                        source_artifact_id=original.id,
                    )
                )
                store.save(document)
        return self.snapshot(result_id)
