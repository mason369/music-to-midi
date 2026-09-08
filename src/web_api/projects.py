"""Project API with explicit runs and verified, portable file transfers."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import Field

from src.projects import ProjectStore, Workflow
from src.projects.labels import LABELS
from src.projects.service import ProjectService
from src.projects.store import StrictModel, TrackView
from src.web_api.schemas import InferenceOptions, ManualMidiOptions


class CreateProject(StrictModel):
    name: str = Field(default="音乐项目", min_length=1, max_length=200)
    workflow: Workflow = Field(default_factory=Workflow)


class ConfigureProject(StrictModel):
    workflow: Workflow
    song_ids: list[str] | None = None
    revision: int = Field(ge=0)


class RunProject(StrictModel):
    song_ids: list[str] | None = None
    track_ids: list[str] | None = None
    primary_only: bool = False


class UpdateSongState(StrictModel):
    revision: int = Field(ge=0)
    mixer: dict[str, TrackView]
    stems: dict[str, ManualMidiOptions]


def install_project_routes(
    app, *, job_manager, write_upload, max_upload_bytes, require_primary, require_manual
):
    class CheckedEngine:
        def run(self, **kwargs):
            if kwargs["kind"] == "primary":
                require_primary(InferenceOptions.model_validate(kwargs["options"]))
            else:
                require_manual(ManualMidiOptions.model_validate(kwargs["options"]))
            return job_manager._engine.run(**kwargs)

    service = ProjectService(job_manager.root / "projects", CheckedEngine)
    app.state.project_service = service
    router = APIRouter(prefix="/api/v1/projects", tags=["projects"])

    def get_store(project_id):
        try:
            return service.store(project_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    def call(operation):
        try:
            return operation()
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/capabilities")
    def project_capabilities():
        return {"schema_version": 1, "labels": LABELS, "archive_extension": ".mtmproject"}

    @router.get("")
    def list_projects():
        return call(service.list)

    @router.post("", status_code=201)
    def create_project(request: CreateProject):
        return call(lambda: service.create(request.name, request.workflow))

    @router.post("/import", status_code=201)
    async def import_project(file: UploadFile):
        if Path(file.filename or "").suffix.lower() != ".mtmproject":
            await file.close()
            raise HTTPException(415, "需要 .mtmproject 项目包")
        incoming = service.root / f".incoming-{uuid.uuid4().hex}.mtmproject"
        try:
            job_manager.require_submission_capacity()
            service.root.mkdir(parents=True, exist_ok=True)
            await write_upload(file, incoming, max_upload_bytes, job_manager.min_free_bytes)
            project_id = uuid.uuid4().hex
            call(
                lambda: ProjectStore.import_archive(
                    incoming, service.root / project_id, max_bytes=max_upload_bytes
                )
            )
            return service.snapshot(project_id)
        finally:
            incoming.unlink(missing_ok=True)

    @router.get("/{project_id}")
    def read_project(project_id: str):
        get_store(project_id)
        return call(lambda: service.snapshot(project_id))

    @router.post("/{project_id}/songs")
    @router.post("/{project_id}/songs/{song_id}/tracks")
    async def add_song(project_id: str, file: UploadFile, song_id: str | None = None):
        store = get_store(project_id)
        suffix = Path(file.filename or "").suffix.lower()
        from src.projects.store import AUDIO_SUFFIXES

        if suffix not in AUDIO_SUFFIXES:
            await file.close()
            raise HTTPException(415, "不支持的音频格式")
        incoming_dir = service.root / f".upload-{uuid.uuid4().hex}"
        incoming_dir.mkdir()
        # Preserve only the basename; imported paths always come from this upload.
        filename = (file.filename or "song.wav").replace("\\", "/").rsplit("/", 1)[-1]
        if not filename or ":" in filename:
            await file.close()
            incoming_dir.rmdir()
            raise HTTPException(422, "文件名无效")
        incoming = incoming_dir / filename
        try:
            await write_upload(file, incoming, max_upload_bytes, job_manager.min_free_bytes)
            if song_id:
                call(lambda: store.add_track(song_id, incoming))
            else:
                call(lambda: store.add([incoming]))
            return service.snapshot(project_id)
        finally:
            incoming.unlink(missing_ok=True)
            incoming_dir.rmdir()

    @router.put("/{project_id}/workflow")
    def configure_project(project_id: str, request: ConfigureProject):
        store = get_store(project_id)
        call(lambda: store.configure(request.workflow, request.song_ids, request.revision))
        return service.snapshot(project_id)

    @router.post("/{project_id}/run", status_code=202)
    def run_project(project_id: str, request: RunProject):
        get_store(project_id)
        job_manager.require_submission_capacity()
        return call(
            lambda: service.start(
                project_id, request.song_ids, request.track_ids, request.primary_only
            )
        )

    @router.put("/{project_id}/songs/{song_id}/state")
    def update_song_state(project_id: str, song_id: str, request: UpdateSongState):
        store = get_store(project_id)
        call(
            lambda: store.update_song_state(song_id, request.mixer, request.stems, request.revision)
        )
        return service.snapshot(project_id)

    @router.post("/{project_id}/cancel")
    def cancel_project(project_id: str):
        get_store(project_id)
        return call(lambda: service.cancel(project_id))

    @router.get("/{project_id}/archive")
    def export_project(project_id: str):
        store = get_store(project_id)
        path = call(lambda: store.export_archive(store.root / "project.mtmproject"))
        return FileResponse(path, media_type="application/zip", filename=f"{project_id}.mtmproject")

    @router.get("/{project_id}/files/{asset_path:path}")
    def project_file(project_id: str, asset_path: str):
        store = get_store(project_id)
        document = store.load()
        assets = [song.source for song in document.songs]
        for song in document.songs:
            assets.extend(song.extra_tracks.values())
            for step in song.checkpoints.values():
                if step.status == "succeeded":
                    assets.extend(step.artifacts)
        asset = next((value for value in assets if value.path == asset_path), None)
        if asset is None:
            raise HTTPException(404, "项目中没有此资源")
        path = call(lambda: store.verify(asset))
        return FileResponse(path, media_type=asset.media_type, filename=path.name)

    app.include_router(router)
    from src.projects.results import ProjectResults
    from src.web_api.soundfonts import install_soundfont_routes
    from src.core.sheet_music import SheetMusicExportError

    results = ProjectResults(service)
    install_soundfont_routes(
        app,
        results,
        prefix="/api/v1/project-results",
        include_runtime=False,
        on_import=results.register_soundfont,
    )

    @app.post("/api/v1/project-results/{result_id}/sheet-music")
    def project_sheet_music(result_id: str, artifact_id: str, quantize_grid: str = "1/32"):
        try:
            return call(
                lambda: results.generate_sheet_music(
                    result_id, artifact_id=artifact_id, quantize_grid=quantize_grid
                )
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except SheetMusicExportError as exc:
            raise HTTPException(422, str(exc)) from exc

    return service
