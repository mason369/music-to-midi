"""FastAPI application exposing inference independently from every frontend."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from src.core.manual_midi import MANUAL_MIDI_ROUTES
from src.models.data_models import (
    MAX_MIDI_BPM,
    MIN_MIDI_BPM,
    MidiTrackMode,
    MultiInstrumentModel,
    MuscriptorModel,
    ProcessingMode,
    YourMT3Model,
)
from src.web_api.jobs import JobManager
from src.web_api.schemas import InferenceOptions, JobSnapshot, ManualMidiOptions

API_VERSION = "1.0"
DEFAULT_MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


def _default_data_root() -> Path:
    configured = os.environ.get("MUSIC_TO_MIDI_API_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "MidiOutput" / "WebAPI").resolve()


def _default_origins() -> list[str]:
    configured = os.environ.get("MUSIC_TO_MIDI_ALLOWED_ORIGINS", "").strip()
    if configured:
        origins = [item.strip() for item in configured.split(",") if item.strip()]
        if not origins:
            raise RuntimeError("MUSIC_TO_MIDI_ALLOWED_ORIGINS contains no valid origins")
        return origins
    return [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8765",
        "http://localhost:8765",
    ]


def _max_upload_bytes() -> int:
    raw = os.environ.get("MUSIC_TO_MIDI_MAX_UPLOAD_BYTES", "").strip()
    if not raw:
        return DEFAULT_MAX_UPLOAD_BYTES
    value = int(raw)
    if value <= 0:
        raise RuntimeError("MUSIC_TO_MIDI_MAX_UPLOAD_BYTES must be positive")
    return value


def _parse_json_model(raw: str, model_type):
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422, detail=f"options is not valid JSON: {exc.msg}"
        ) from exc
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False, include_context=False),
        ) from exc


def _safe_upload_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"unsupported audio extension {suffix or '<none>'}; "
                f"expected one of {sorted(SUPPORTED_AUDIO_SUFFIXES)}"
            ),
        )
    return suffix


async def _write_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    total = 0
    try:
        with destination.open("xb") as stream:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"audio upload exceeds the configured {max_bytes}-byte limit",
                    )
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        await upload.close()
    if total <= 0:
        raise HTTPException(status_code=400, detail="uploaded audio file is empty")
    return total


def _job_or_http(manager: JobManager, job_id: str) -> dict:
    try:
        return manager.snapshot(job_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _capabilities() -> dict:
    modes = [
        {
            "id": ProcessingMode.SMART.value,
            "label_zh": "多乐器 MIDI 转写",
            "label_en": "Multi-instrument transcription",
            "kind": "direct",
        },
        {
            "id": ProcessingMode.VOCAL_SPLIT.value,
            "label_zh": "人声 / 伴奏分离",
            "label_en": "Vocal / accompaniment split",
            "kind": "separation",
        },
        {
            "id": ProcessingMode.SIX_STEM_SPLIT.value,
            "label_zh": "六声部分离",
            "label_en": "Six-stem separation",
            "kind": "separation",
        },
        {
            "id": ProcessingMode.PIANO_TRANSKUN.value,
            "label_zh": "钢琴 · TransKun V2",
            "label_en": "Piano · TransKun V2",
            "kind": "direct",
        },
        {
            "id": ProcessingMode.PIANO_TRANSKUN_V2_AUG.value,
            "label_zh": "钢琴 · TransKun V2 Aug",
            "label_en": "Piano · TransKun V2 Aug",
            "kind": "direct",
        },
        {
            "id": ProcessingMode.PIANO_ARIA_AMT.value,
            "label_zh": "钢琴 · Aria-AMT",
            "label_en": "Piano · Aria-AMT",
            "kind": "direct",
        },
        {
            "id": ProcessingMode.PIANO_BYTEDANCE_PEDAL.value,
            "label_zh": "钢琴 · ByteDance 踏板",
            "label_en": "Piano · ByteDance Pedal",
            "kind": "direct",
        },
    ]
    manual_labels = {
        "miros": "MIROS",
        "muscriptor": "MuScriptor Large",
        "muscriptor:medium": "MuScriptor Medium",
        "muscriptor:small": "MuScriptor Small",
        ProcessingMode.PIANO_TRANSKUN.value: "TransKun V2",
        ProcessingMode.PIANO_TRANSKUN_V2_AUG.value: "TransKun V2 Aug",
        ProcessingMode.PIANO_ARIA_AMT.value: "Aria-AMT",
        ProcessingMode.PIANO_BYTEDANCE_PEDAL.value: "ByteDance Piano Pedal",
    }
    for model in YourMT3Model:
        if model is not YourMT3Model.LEGACY_MC13:
            manual_labels[f"yourmt3:{model.value}"] = f"YourMT3+ · {model.value}"
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        torch_version = str(torch.__version__)
    except Exception as exc:
        cuda_available = False
        devices = []
        torch_version = f"unavailable: {exc}"
    return {
        "api_version": API_VERSION,
        "modes": modes,
        "backends": [model.value for model in MultiInstrumentModel],
        "yourmt3_models": [
            model.value for model in YourMT3Model if model is not YourMT3Model.LEGACY_MC13
        ],
        "muscriptor_models": [model.value for model in MuscriptorModel],
        "midi_track_modes": [mode.value for mode in MidiTrackMode],
        "manual_midi_routes": [
            {"id": route, "label": manual_labels[route]} for route in MANUAL_MIDI_ROUTES
        ],
        "limits": {
            "custom_bpm_min": MIN_MIDI_BPM,
            "custom_bpm_max": MAX_MIDI_BPM,
            "max_upload_bytes": _max_upload_bytes(),
            "audio_extensions": sorted(SUPPORTED_AUDIO_SUFFIXES),
        },
        "runtime": {
            "torch": torch_version,
            "cuda_available": cuda_available,
            "cuda_devices": devices,
            "gpu_queue_concurrency": 1,
        },
    }


def create_app(
    *,
    manager: JobManager | None = None,
    data_root: str | Path | None = None,
    cors_origins: list[str] | None = None,
    frontend_root: str | Path | None = None,
) -> FastAPI:
    owned_manager = manager is None
    job_manager = manager or JobManager(data_root or _default_data_root())

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if owned_manager:
            job_manager.close()

    app = FastAPI(
        title="Music to MIDI Inference API",
        version=API_VERSION,
        description=(
            "Standalone, versioned inference backend for the desktop-equivalent web client. "
            "GPU jobs are serialized; progress and cancellation remain responsive over HTTP."
        ),
        lifespan=lifespan,
    )
    app.state.job_manager = job_manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or _default_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Last-Event-ID", "Range"],
        expose_headers=[
            "Content-Length",
            "Content-Range",
            "Accept-Ranges",
            "ETag",
            "X-Content-SHA256",
        ],
    )

    @app.get("/api/v1/health")
    def health() -> dict:
        state = job_manager.health()
        if not state["worker_alive"]:
            raise HTTPException(status_code=503, detail="inference worker is not alive")
        return {"status": "ok", "api_version": API_VERSION, **state}

    @app.get("/api/v1/capabilities")
    def capabilities() -> dict:
        return _capabilities()

    @app.get("/api/v1/jobs", response_model=list[JobSnapshot])
    def list_jobs() -> list[dict]:
        return job_manager.list_snapshots()

    @app.post("/api/v1/jobs", response_model=JobSnapshot, status_code=202)
    async def create_primary_job(
        audio: Annotated[UploadFile, File(...)],
        options: Annotated[str, Form(...)],
    ) -> dict:
        parsed = _parse_json_model(options, InferenceOptions)
        suffix = _safe_upload_suffix(audio.filename)
        job_id, job_dir = job_manager.reserve_job_dir()
        destination = job_dir / "input" / f"source{suffix}"
        try:
            await _write_upload(audio, destination, _max_upload_bytes())
            return job_manager.submit_reserved(
                job_id=job_id,
                kind="primary",
                source_path=destination,
                original_filename=Path(audio.filename or destination.name).name,
                options=parsed.model_dump(),
            )
        except Exception:
            job_manager.discard_reserved(job_id)
            raise

    @app.post("/api/v1/manual-jobs", response_model=JobSnapshot, status_code=202)
    async def create_uploaded_manual_job(
        audio: Annotated[UploadFile, File(...)],
        options: Annotated[str, Form(...)],
    ) -> dict:
        parsed = _parse_json_model(options, ManualMidiOptions)
        suffix = _safe_upload_suffix(audio.filename)
        job_id, job_dir = job_manager.reserve_job_dir()
        destination = job_dir / "input" / f"source{suffix}"
        try:
            await _write_upload(audio, destination, _max_upload_bytes())
            return job_manager.submit_reserved(
                job_id=job_id,
                kind="manual_midi",
                source_path=destination,
                original_filename=Path(audio.filename or destination.name).name,
                options=parsed.model_dump(),
                track_id=Path(audio.filename or "local-track").stem,
            )
        except Exception:
            job_manager.discard_reserved(job_id)
            raise

    @app.post(
        "/api/v1/jobs/{job_id}/tracks/{track_id}/midi",
        response_model=JobSnapshot,
        status_code=202,
    )
    def create_parent_track_job(
        job_id: str,
        track_id: str,
        options: ManualMidiOptions,
    ) -> dict:
        try:
            return job_manager.submit_parent_track(job_id, track_id, options.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/v1/jobs/{job_id}", response_model=JobSnapshot)
    def get_job(job_id: str) -> dict:
        return _job_or_http(job_manager, job_id)

    @app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobSnapshot, status_code=202)
    def cancel_job(job_id: str) -> dict:
        try:
            return job_manager.cancel(job_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/jobs/{job_id}/retry", response_model=JobSnapshot, status_code=202)
    def retry_job(job_id: str) -> dict:
        try:
            return job_manager.retry(job_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.delete("/api/v1/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> Response:
        try:
            job_manager.delete(job_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(status_code=204)

    @app.get("/api/v1/jobs/{job_id}/files/{artifact_id}")
    def download_artifact(job_id: str, artifact_id: str) -> FileResponse:
        try:
            path, artifact = job_manager.resolve_artifact(job_id, artifact_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return FileResponse(
            path,
            media_type=artifact.media_type,
            filename=artifact.name,
            headers={
                "ETag": f'"sha256-{artifact.sha256}"',
                "X-Content-SHA256": artifact.sha256,
            },
        )

    @app.get("/api/v1/jobs/{job_id}/events")
    async def job_events(
        request: Request,
        job_id: str,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        first = _job_or_http(job_manager, job_id)
        try:
            after_revision = int(last_event_id or request.query_params.get("after", "0"))
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="event revision must be an integer"
            ) from exc

        async def stream() -> AsyncIterator[str]:
            nonlocal after_revision
            if first["revision"] > after_revision:
                after_revision = int(first["revision"])
                yield (
                    f"id: {after_revision}\n"
                    "event: job\n"
                    f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
                )
                if first["status"] in {"succeeded", "failed", "cancelled"}:
                    return
            while not await request.is_disconnected():
                update = await asyncio.to_thread(
                    job_manager.wait_for_revision,
                    job_id,
                    after_revision,
                    15.0,
                )
                if update is None:
                    yield ": keep-alive\n\n"
                    continue
                after_revision = int(update["revision"])
                yield (
                    f"id: {after_revision}\n"
                    "event: job\n"
                    f"data: {json.dumps(update, ensure_ascii=False)}\n\n"
                )
                if update["status"] in {"succeeded", "failed", "cancelled"}:
                    return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    if frontend_root is not None:
        resolved_frontend = Path(frontend_root).resolve()
        if not (resolved_frontend / "index.html").is_file():
            raise RuntimeError(f"frontend index is missing: {resolved_frontend / 'index.html'}")
        app.mount("/", StaticFiles(directory=resolved_frontend, html=True), name="frontend")
    else:

        @app.get("/")
        def api_root() -> JSONResponse:
            return JSONResponse(
                {
                    "service": "Music to MIDI Inference API",
                    "version": API_VERSION,
                    "health": "/api/v1/health",
                    "docs": "/docs",
                }
            )

    return app
