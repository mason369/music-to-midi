"""FastAPI application exposing inference independently from every frontend."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from src.core.manual_midi import MANUAL_MIDI_ROUTES
from src.core.midi_quantization import (
    DEFAULT_MIDI_QUANTIZE_GRID,
    DEFAULT_MIDI_QUANTIZE_SCOPE,
    MIDI_QUANTIZE_GRIDS,
)
from src.core.sheet_music import SheetMusicExportError
from src.i18n.translator import Translator
from src.model_profiles import (
    MUSCRIPTOR_PROFILE_IDS,
    YOURMT3_PROFILE_IDS,
    inspect_model_profiles,
    primary_profile_id,
    require_profile_available,
)
from src.models.data_models import (
    MAX_TEMPO_BPM,
    MIN_TEMPO_BPM,
    MidiTrackMode,
    MultiInstrumentModel,
    MuscriptorModel,
    MuscriptorProcessingChain,
    ProcessingMode,
    TempoMode,
    YourMT3Model,
)
from src.models.muscriptor_instruments import (
    MUSCRIPTOR_INSTRUMENTS,
    muscriptor_instrument_label,
)
from src.utils.musescore_runtime import MuseScoreRuntimeError
from src.utils.yourmt3_downloader import YOURMT3_MODELS
from src.web_api.jobs import InsufficientStorageError, JobManager, QueueCapacityError
from src.web_api.schemas import InferenceOptions, JobSnapshot, ManualMidiOptions
from src.web_contract import API_VERSION

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


def _environment_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be one of 1/0, true/false, yes/no or on/off")


def _deployment_capabilities() -> dict[str, object]:
    public = _environment_flag("MUSIC_TO_MIDI_PUBLIC_DEPLOYMENT")
    edge_auth = os.environ.get("MUSIC_TO_MIDI_EDGE_AUTH", "").strip().lower()
    tls_at_edge = _environment_flag("MUSIC_TO_MIDI_TLS_TERMINATED_AT_EDGE")
    errors: list[str] = []
    if public and edge_auth != "basic":
        errors.append("public deployment requires MUSIC_TO_MIDI_EDGE_AUTH=basic")
    if public and not tls_at_edge:
        errors.append("public deployment requires TLS termination at the edge")
    return {
        "trust_boundary": "authenticated_single_owner" if public else "trusted_lan",
        "authentication": edge_auth == "basic",
        "authorization": False,
        "tls_terminated_here": False,
        "tls_terminated_at_edge": tls_at_edge,
        "configuration_ready": not errors,
        "configuration_error": "; ".join(errors) or None,
    }


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


async def _write_upload(
    upload: UploadFile,
    destination: Path,
    max_bytes: int,
    min_free_bytes: int = 0,
) -> int:
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
                free_bytes = shutil.disk_usage(destination.parent).free
                if free_bytes - len(chunk) < min_free_bytes:
                    raise HTTPException(
                        status_code=507,
                        detail=(
                            "job storage free-space floor would be violated: "
                            f"free={free_bytes}, incoming_chunk={len(chunk)}, "
                            f"required_reserve={min_free_bytes}"
                        ),
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


def _capabilities(
    *,
    max_upload_bytes: int | None = None,
    max_queued_jobs: int = 0,
    min_free_bytes: int = 0,
    retention_max_bytes: int = 0,
) -> dict:
    profile_snapshot = inspect_model_profiles()
    profile_statuses = {item.id: item for item in profile_snapshot.profiles}

    def grouped_availability(profile_ids: tuple[str, ...]) -> dict[str, object]:
        statuses = [profile_statuses[profile_id] for profile_id in profile_ids]
        if any(item.available for item in statuses):
            return {"available": True, "unavailable_reason": None}
        enabled = [item for item in statuses if item.enabled]
        if not enabled:
            return {
                "available": False,
                "unavailable_reason": "no profile in this model family is enabled",
            }
        return {
            "available": False,
            "unavailable_reason": "; ".join(
                f"{item.id}: {item.unavailable_reason or 'unavailable'}" for item in enabled
            ),
        }

    backend_availability = {
        MultiInstrumentModel.YOURMT3.value: grouped_availability(YOURMT3_PROFILE_IDS),
        MultiInstrumentModel.MIROS.value: grouped_availability(("miros",)),
        MultiInstrumentModel.MUSCRIPTOR.value: grouped_availability(MUSCRIPTOR_PROFILE_IDS),
    }
    mode_ids = [
        (ProcessingMode.SMART.value, "direct"),
        (ProcessingMode.VOCAL_SPLIT.value, "separation"),
        (ProcessingMode.SIX_STEM_SPLIT.value, "separation"),
        (ProcessingMode.PIANO_TRANSKUN.value, "direct"),
        (ProcessingMode.PIANO_TRANSKUN_V2_AUG.value, "direct"),
        (ProcessingMode.PIANO_ARIA_AMT.value, "direct"),
        (ProcessingMode.PIANO_BYTEDANCE_PEDAL.value, "direct"),
    ]
    # Labels come from the same shared i18n catalog the desktop, Space and
    # Colab read, so every platform presents identical wording.
    translator_zh = Translator("zh_CN")
    translator_en = Translator("en_US")
    modes = []
    for mode_id, kind in mode_ids:
        if mode_id == ProcessingMode.SMART.value:
            availability = grouped_availability(
                (*YOURMT3_PROFILE_IDS, "miros", *MUSCRIPTOR_PROFILE_IDS)
            )
        else:
            status_item = profile_statuses[mode_id]
            availability = {
                "available": status_item.available,
                "unavailable_reason": status_item.unavailable_reason,
            }
        modes.append(
            {
                "id": mode_id,
                "label_zh": translator_zh.t(f"main.mode.{mode_id}"),
                "label_en": translator_en.t(f"main.mode.{mode_id}"),
                "kind": kind,
                **availability,
            }
        )
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
            manual_labels[f"yourmt3:{model.value}"] = (
                f"YourMT3+ · {YOURMT3_MODELS[model.value]['ui_label']}"
            )
    runtime_error = None
    accelerator = "unavailable"
    accelerator_device = None
    accelerator_devices = []
    accelerator_ready = False
    try:
        import torch

        from src.utils.gpu_utils import (
            ensure_accelerator_runtime_compatibility,
            get_accelerator_type,
            get_device,
            get_gpu_info,
        )

        cuda_available = bool(torch.cuda.is_available())
        devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        xpu = getattr(torch, "xpu", None)
        xpu_available = bool(xpu is not None and xpu.is_available())
        torch_version = str(torch.__version__)
        detected = get_accelerator_type()
        requested = os.environ.get("MUSIC_TO_MIDI_ACCELERATOR", "").strip().lower()
        accelerator = requested or detected
        if accelerator not in {"cuda", "xpu"}:
            raise RuntimeError("完整工作流需要 CUDA 或 Intel XPU，" f"当前检测结果为 {detected!r}")
        if requested and requested != detected:
            raise RuntimeError(
                "MUSIC_TO_MIDI_ACCELERATOR 与当前 PyTorch 运行时不一致: "
                f"requested={requested}, detected={detected}"
            )
        accelerator_device = get_device(prefer_gpu=True, gpu_index=0)
        ensure_accelerator_runtime_compatibility(accelerator_device)
        accelerator_devices = get_gpu_info()
        accelerator_ready = True
    except Exception as exc:
        runtime_error = str(exc)
        if "torch_version" not in locals():
            cuda_available = False
            xpu_available = False
            devices = []
            torch_version = f"unavailable: {exc}"
    return {
        "api_version": API_VERSION,
        "deployment": _deployment_capabilities(),
        "modes": modes,
        "backends": [model.value for model in MultiInstrumentModel],
        "backend_availability": backend_availability,
        "yourmt3_models": [
            {
                "id": model.value,
                "label": YOURMT3_MODELS[model.value]["ui_label"],
                "available": profile_statuses[f"yourmt3:{model.value}"].available,
                "unavailable_reason": profile_statuses[f"yourmt3:{model.value}"].unavailable_reason,
            }
            for model in YourMT3Model
            if model is not YourMT3Model.LEGACY_MC13
        ],
        "muscriptor_models": [
            {
                "id": model.value,
                "label_zh": translator_zh.t(f"main.engine.muscriptor_models.{model.value}"),
                "label_en": translator_en.t(f"main.engine.muscriptor_models.{model.value}"),
                "available": profile_statuses[
                    "muscriptor" if model is MuscriptorModel.LARGE else f"muscriptor:{model.value}"
                ].available,
                "unavailable_reason": profile_statuses[
                    "muscriptor" if model is MuscriptorModel.LARGE else f"muscriptor:{model.value}"
                ].unavailable_reason,
            }
            for model in (MuscriptorModel.LARGE, MuscriptorModel.MEDIUM, MuscriptorModel.SMALL)
        ],
        "muscriptor_instruments": [
            {
                "id": instrument,
                "label_zh": muscriptor_instrument_label(instrument, "zh_CN"),
                "label_en": muscriptor_instrument_label(instrument, "en_US"),
            }
            for instrument in MUSCRIPTOR_INSTRUMENTS
        ],
        "muscriptor_processing_chains": [
            {
                "id": chain.value,
                "label_zh": translator_zh.t(
                    f"main.engine.muscriptor_processing_chains.{chain.value}"
                ),
                "label_en": translator_en.t(
                    f"main.engine.muscriptor_processing_chains.{chain.value}"
                ),
            }
            for chain in (
                MuscriptorProcessingChain.OFFICIAL,
                MuscriptorProcessingChain.TELKNET,
            )
        ],
        "midi_track_modes": [mode.value for mode in MidiTrackMode],
        "midi_quantization": {
            "grids": list(MIDI_QUANTIZE_GRIDS),
            "default_grid": DEFAULT_MIDI_QUANTIZE_GRID,
            "default_enabled": False,
            "scopes": [DEFAULT_MIDI_QUANTIZE_SCOPE],
            "default_scope": DEFAULT_MIDI_QUANTIZE_SCOPE,
        },
        "tempo_modes": [
            {
                "id": mode.value,
                "label_zh": translator_zh.t(f"main.tempo.{mode.value}"),
                "label_en": translator_en.t(f"main.tempo.{mode.value}"),
            }
            for mode in TempoMode
        ],
        "manual_midi_routes": [
            {
                "id": route,
                "label": manual_labels[route],
                "available": profile_statuses[route].available,
                "unavailable_reason": profile_statuses[route].unavailable_reason,
            }
            for route in MANUAL_MIDI_ROUTES
        ],
        "limits": {
            "custom_bpm_min": MIN_TEMPO_BPM,
            "custom_bpm_max": MAX_TEMPO_BPM,
            "max_upload_bytes": max_upload_bytes or _max_upload_bytes(),
            "max_queued_jobs": max_queued_jobs,
            "min_free_bytes": min_free_bytes,
            "retention_max_bytes": retention_max_bytes,
            "audio_extensions": sorted(SUPPORTED_AUDIO_SUFFIXES),
        },
        "runtime": {
            "torch": torch_version,
            "cuda_available": cuda_available,
            "cuda_devices": devices,
            "xpu_available": xpu_available,
            "accelerator": accelerator,
            "accelerator_device": accelerator_device,
            "accelerator_devices": accelerator_devices,
            "accelerator_ready": accelerator_ready,
            "accelerator_error": runtime_error,
            "gpu_queue_concurrency": 1,
        },
        "model_profiles": profile_snapshot.to_dict(),
    }


def create_app(
    *,
    manager: JobManager | None = None,
    data_root: str | Path | None = None,
    cors_origins: list[str] | None = None,
    frontend_root: str | Path | None = None,
    max_upload_bytes: int | None = None,
    max_queued_jobs: int = 0,
    min_free_bytes: int = 0,
    retention_days: int = 30,
    retention_max_jobs: int = 200,
    retention_max_bytes: int = 0,
) -> FastAPI:
    upload_limit = _max_upload_bytes() if max_upload_bytes is None else int(max_upload_bytes)
    if upload_limit <= 0:
        raise ValueError("max_upload_bytes must be positive")
    owned_manager = manager is None
    job_manager = manager or JobManager(
        data_root or _default_data_root(),
        retention_days=retention_days,
        retention_max_jobs=retention_max_jobs,
        retention_max_bytes=retention_max_bytes,
        max_queued_jobs=max_queued_jobs,
        min_free_bytes=min_free_bytes,
    )
    effective_max_queued_jobs = int(getattr(job_manager, "max_queued_jobs", max_queued_jobs))
    effective_min_free_bytes = int(getattr(job_manager, "min_free_bytes", min_free_bytes))
    effective_retention_max_bytes = int(
        getattr(job_manager, "retention_max_bytes", retention_max_bytes)
    )

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

    @app.exception_handler(QueueCapacityError)
    async def queue_capacity_error(_request: Request, exc: QueueCapacityError) -> JSONResponse:
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    @app.exception_handler(InsufficientStorageError)
    async def insufficient_storage_error(
        _request: Request, exc: InsufficientStorageError
    ) -> JSONResponse:
        return JSONResponse(status_code=507, content={"detail": str(exc)})

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

    def capability_snapshot() -> dict:
        return _capabilities(
            max_upload_bytes=upload_limit,
            max_queued_jobs=effective_max_queued_jobs,
            min_free_bytes=effective_min_free_bytes,
            retention_max_bytes=effective_retention_max_bytes,
        )

    def require_public_accelerator_options(parsed: InferenceOptions | ManualMidiOptions) -> None:
        if not _environment_flag("MUSIC_TO_MIDI_PUBLIC_DEPLOYMENT"):
            return
        if not parsed.use_gpu:
            raise HTTPException(
                status_code=422,
                detail="public deployment requires accelerator inference; use_gpu must be true",
            )
        if parsed.gpu_device != 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    "public container exposes one selected NVIDIA device as gpu_device=0; "
                    f"received gpu_device={parsed.gpu_device}"
                ),
            )

    def require_public_runtime_ready() -> None:
        if not _environment_flag("MUSIC_TO_MIDI_PUBLIC_DEPLOYMENT"):
            return
        state = job_manager.health()
        capabilities_payload = capability_snapshot()
        errors: list[str] = []
        if not state["worker_alive"]:
            errors.append("inference worker is not alive")
        if not capabilities_payload["runtime"]["accelerator_ready"]:
            errors.append(
                str(
                    capabilities_payload["runtime"]["accelerator_error"]
                    or "accelerator runtime is unavailable"
                )
            )
        if not capabilities_payload["model_profiles"]["ready"]:
            errors.append(
                str(
                    capabilities_payload["model_profiles"]["readiness_error"]
                    or "model profiles are unavailable"
                )
            )
        if not capabilities_payload["deployment"]["configuration_ready"]:
            errors.append(str(capabilities_payload["deployment"]["configuration_error"]))
        if errors:
            raise HTTPException(
                status_code=503,
                detail="public inference service is not ready: " + "; ".join(errors),
            )

    def require_primary_options(parsed: InferenceOptions) -> None:
        require_public_accelerator_options(parsed)
        try:
            profile_id = primary_profile_id(
                processing_mode=parsed.processing_mode,
                transcription_backend=parsed.transcription_backend,
                yourmt3_model=parsed.yourmt3_model,
                muscriptor_model=parsed.muscriptor_model,
            )
            require_profile_available(profile_id, inspect_model_profiles())
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        require_public_runtime_ready()

    def require_manual_options(parsed: ManualMidiOptions) -> None:
        require_public_accelerator_options(parsed)
        try:
            require_profile_available(parsed.route, inspect_model_profiles())
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        require_public_runtime_ready()

    @app.get("/api/v1/health")
    def health() -> dict:
        state = job_manager.health()
        if not state["worker_alive"]:
            raise HTTPException(status_code=503, detail="inference worker is not alive")
        return {"status": "ok", "api_version": API_VERSION, **state}

    @app.get("/api/v1/ready")
    def ready() -> dict:
        state = job_manager.health()
        capabilities_payload = capability_snapshot()
        errors: list[str] = []
        if not state["worker_alive"]:
            errors.append("inference worker is not alive")
        if not capabilities_payload["runtime"]["accelerator_ready"]:
            errors.append(
                str(
                    capabilities_payload["runtime"]["accelerator_error"]
                    or "accelerator runtime is unavailable"
                )
            )
        if not capabilities_payload["model_profiles"]["ready"]:
            errors.append(
                str(
                    capabilities_payload["model_profiles"]["readiness_error"]
                    or "model profiles are unavailable"
                )
            )
        if not capabilities_payload["deployment"]["configuration_ready"]:
            errors.append(str(capabilities_payload["deployment"]["configuration_error"]))
        storage = state["storage"]
        if storage["free_bytes"] < storage["min_free_bytes"]:
            errors.append(
                "job storage is below its free-space floor: "
                f"free={storage['free_bytes']}, required={storage['min_free_bytes']}"
            )
        if errors:
            raise HTTPException(
                status_code=503,
                detail={"status": "not_ready", "errors": errors, "api_version": API_VERSION},
            )
        return {
            "status": "ready",
            "api_version": API_VERSION,
            "enabled_profiles": capabilities_payload["model_profiles"]["enabled_profiles"],
            "accelerator": capabilities_payload["runtime"]["accelerator"],
        }

    @app.get("/api/v1/capabilities")
    def capabilities() -> dict:
        return capability_snapshot()

    @app.get("/api/v1/jobs", response_model=list[JobSnapshot])
    def list_jobs() -> list[dict]:
        return job_manager.list_snapshots()

    @app.post("/api/v1/jobs", response_model=JobSnapshot, status_code=202)
    async def create_primary_job(
        audio: Annotated[UploadFile, File(...)],
        options: Annotated[str, Form(...)],
    ) -> dict:
        parsed = _parse_json_model(options, InferenceOptions)
        require_primary_options(parsed)
        job_manager.require_submission_capacity()
        suffix = _safe_upload_suffix(audio.filename)
        job_id, job_dir = job_manager.reserve_job_dir()
        destination = job_dir / "input" / f"source{suffix}"
        try:
            await _write_upload(audio, destination, upload_limit, effective_min_free_bytes)
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
        require_manual_options(parsed)
        job_manager.require_submission_capacity()
        suffix = _safe_upload_suffix(audio.filename)
        job_id, job_dir = job_manager.reserve_job_dir()
        destination = job_dir / "input" / f"source{suffix}"
        try:
            await _write_upload(audio, destination, upload_limit, effective_min_free_bytes)
            return job_manager.submit_reserved(
                job_id=job_id,
                kind="manual_midi",
                source_path=destination,
                original_filename=Path(audio.filename or destination.name).name,
                options=parsed.model_dump(),
                tempo_source_path=destination,
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
        require_manual_options(options)
        job_manager.require_submission_capacity()
        try:
            return job_manager.submit_parent_track(job_id, track_id, options.model_dump())
        except (QueueCapacityError, InsufficientStorageError):
            raise
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
        existing = _job_or_http(job_manager, job_id)
        try:
            if existing["kind"] == "primary":
                require_primary_options(InferenceOptions.model_validate(existing["request"]))
            else:
                require_manual_options(ManualMidiOptions.model_validate(existing["request"]))
            return job_manager.retry(job_id)
        except (QueueCapacityError, InsufficientStorageError):
            raise
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.delete("/api/v1/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, cascade: bool = False) -> Response:
        try:
            job_manager.delete(job_id, cascade=cascade)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(status_code=204)

    @app.post("/api/v1/jobs/{job_id}/sheet-music", response_model=JobSnapshot)
    def generate_sheet_music(
        job_id: str,
        artifact_id: str | None = None,
        quantize_grid: str = DEFAULT_MIDI_QUANTIZE_GRID,
    ) -> dict:
        try:
            return job_manager.generate_sheet_music(
                job_id,
                artifact_id=artifact_id,
                quantize_grid=quantize_grid,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except MuseScoreRuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except SheetMusicExportError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

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
