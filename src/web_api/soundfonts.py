"""Job-bound SoundFont import, preset discovery, and verified WAV exports."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from src.core.muscriptor_result_assets import read_midi_roll_notes, render_midi_audio_export
from src.core.soundfont_library import (
    MAX_SOUNDFONT_BYTES,
    import_soundfont,
    read_soundfont,
    soundfont_selection,
)
from src.gui.web.soundfont_runtime import SOUNDFONT_JS
from src.models.gm_instruments import get_instrument_name
from src.models.muscriptor_instruments import muscriptor_instrument_label


def install_soundfont_routes(app, manager, *, prefix="/api/v1/jobs", include_runtime=True, on_import=None):
    def context(job_id, artifact_id):
        source, artifact = manager.resolve_artifact(job_id, artifact_id)
        if artifact.kind != "midi":
            raise ValueError("音色库只能用于 MIDI 结果")
        identity = hashlib.sha256(artifact_id.encode()).hexdigest()[:16]
        root = source.parent / "soundfonts" / identity
        return source, root

    def load_library(root, identity):
        if not isinstance(identity, str) or re.fullmatch(r"[a-f0-9]{64}", identity) is None:
            raise ValueError("音色库编号无效")
        metadata = json.loads((root / (identity + ".json")).read_text(encoding="utf-8"))
        name = metadata["name"]
        suffix = Path(name).suffix.lower()
        if suffix not in {".sf2", ".sf3"}:
            raise ValueError("音色库文件格式无效")
        library = read_soundfont(root / (identity + suffix), name=name)
        if library.sha256 != identity:
            raise ValueError("已导入的音色库校验失败")
        return library

    def save_import(job_id, artifact_id, path):
        # Publish under the job lock so retention/deletion cannot race the copy.
        with manager._lock:
            _source, root = context(job_id, artifact_id)
            library = import_soundfont(path, root)
            (root / (library.sha256 + ".json")).write_text(
                json.dumps({"name": library.name}, ensure_ascii=False), encoding="utf-8"
            )
            if on_import is not None:
                on_import(job_id, [library.path, root / (library.sha256 + ".json")])
            return library.to_dict()

    def render(job_id, artifact_id, payload):
        with manager._lock:
            source, root = context(job_id, artifact_id)
            if set(payload) - {"soundfont", "preset"}:
                raise ValueError("音频导出字段无效")
            options = {}
            selected = payload.get("soundfont")
            if selected is not None:
                if not isinstance(selected, dict) or set(selected) != {"id", "assignments"}:
                    raise ValueError("音色库选择字段无效")
                library = load_library(root, selected["id"])
                options["soundfont_selection"] = soundfont_selection(
                    library, selected["assignments"]
                )
            temporary = tempfile.TemporaryDirectory(prefix="music-to-midi-api-soundfont-")
            stage = Path(temporary.name)
            snapshot = stage / "source.mid"
            shutil.copyfile(source, snapshot)
            if options:
                # Keep the active render independent of subsequent job deletion.
                from dataclasses import replace

                selection = options["soundfont_selection"]
                bank_copy = stage / ("bank" + selection.library.path.suffix)
                shutil.copyfile(selection.library.path, bank_copy)
                options["soundfont_selection"] = replace(
                    selection, library=replace(selection.library, path=bank_copy)
                )
        try:
            result = render_midi_audio_export(
                snapshot,
                stage / "soundfont.wav",
                str(payload.get("preset", "pcm24_48000")),
                **options,
            )
            context(job_id, artifact_id)
            return temporary, result
        except BaseException:
            temporary.cleanup()
            raise

    def error(exc):
        code = (
            404
            if isinstance(exc, (KeyError, FileNotFoundError))
            else 422 if isinstance(exc, ValueError) else 409
        )
        return HTTPException(status_code=code, detail=str(exc))

    def soundfont_ui():
        return Response(SOUNDFONT_JS, media_type="text/javascript")

    if include_runtime:
        app.add_api_route("/api/v1/soundfont-ui.js", soundfont_ui, methods=["GET"], include_in_schema=False)

    @app.get(prefix + "/{job_id}/soundfonts/{artifact_id}")
    def soundfont_catalog(job_id: str, artifact_id: str):
        try:
            with manager._lock:
                source, root = context(job_id, artifact_id)
                notes = read_midi_roll_notes(source, muscriptor_groups=False)
                sources = {(note.program, note.is_drum): note.instrument for note in notes}
                return {
                    "sources": [
                        dict(
                            program=key[0],
                            is_drum=key[1],
                            name=name,
                            labels={
                                language: (
                                    muscriptor_instrument_label("drums", language)
                                    if key[1]
                                    else get_instrument_name(key[0], language)
                                )
                                for language in ("zh_CN", "en_US")
                            },
                        )
                        for key, name in sources.items()
                    ],
                    "libraries": [
                        load_library(root, path.stem).to_dict()
                        for path in sorted(root.glob("*.json"))
                    ],
                }
        except (KeyError, OSError, ValueError, RuntimeError) as exc:
            raise error(exc) from exc

    @app.post(prefix + "/{job_id}/soundfonts/{artifact_id}")
    async def upload_soundfont(job_id: str, artifact_id: str, file: UploadFile = File(...)):
        try:
            context(job_id, artifact_id)
            name = Path((file.filename or "").replace("\\", "/")).name
            if Path(name).suffix.lower() not in {".sf2", ".sf3"}:
                raise ValueError("音色库必须是 SF2 或 SF3 文件")
            with tempfile.TemporaryDirectory(prefix="music-to-midi-soundfont-upload-") as folder:
                temporary = Path(folder) / name
                size = 0
                with temporary.open("wb") as stream:
                    while block := await file.read(1024 * 1024):
                        size += len(block)
                        if size > MAX_SOUNDFONT_BYTES:
                            raise HTTPException(status_code=413, detail="音色库超过 1 GiB 大小限制")
                        stream.write(block)
                return await run_in_threadpool(save_import, job_id, artifact_id, temporary)
        except (KeyError, OSError, ValueError, RuntimeError) as exc:
            raise error(exc) from exc
        finally:
            await file.close()

    @app.post(prefix + "/{job_id}/soundfonts/{artifact_id}/render")
    def render_soundfont(job_id: str, artifact_id: str, payload: dict):
        try:
            temporary, result = render(job_id, artifact_id, payload)
            return FileResponse(
                result.path,
                media_type="audio/wav",
                filename="soundfont.wav",
                background=BackgroundTask(temporary.cleanup),
            )
        except (KeyError, OSError, ValueError, RuntimeError) as exc:
            raise error(exc) from exc
