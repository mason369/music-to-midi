"""Serve real shared result controls and the standalone UI against local MIDI fixtures.

Run with the project venv, then open /timing/zh_CN and /timing/en_US. No model
inference, external requests, production state or user audio is needed.
"""

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mido
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from src.core.midi_timing import midi_timeline_manifest
from src.core.muscriptor_result_assets import prepare_midi_playback_assets
from src.gui.web.edited_midi_preview import EditedMidiPreviewRegistry
from src.gui.web.muscriptor_result_runtime import (
    build_muscriptor_result_html,
    muscriptor_result_head,
)
from src.gui.web.sheet_music_export import SheetMusicExportRegistry
from src.i18n.translator import Translator
from src.web_api.app import create_app


def create_fixture_app():
    root = Path(tempfile.mkdtemp(prefix="midi-timing-ui-"))
    source = root / "variable.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500000),
                mido.MetaMessage("time_signature", numerator=4, denominator=4),
                mido.MetaMessage("set_tempo", tempo=750000, time=1920),
            ]
        )
    )
    notes = mido.MidiTrack([mido.Message("program_change", program=0)])
    previous = 0
    for index in range(16):
        start = 2400 + index * 240 + (31 if index % 2 else 17)
        notes.append(
            mido.Message("note_on", note=60 + index % 5, velocity=90, time=start - previous)
        )
        notes.append(mido.Message("note_off", note=60 + index % 5, velocity=23, time=180))
        previous = start + 180
    midi.tracks.append(notes)
    midi.save(source)
    original = root / "original.wav"
    time = np.arange(44100 * 12) / 44100
    sf.write(original, (0.08 * np.sin(2 * np.pi * 261.63 * time)).astype("float32"), 44100)
    assets = prepare_midi_playback_assets(
        source, original, root / "playback", muscriptor_groups=False
    )
    previews, sheets = EditedMidiPreviewRegistry(), SheetMusicExportRegistry()
    token = previews.register(
        request_dir=root,
        source_midi_path=source,
        original_audio_path=original,
        reference_bpm=120,
        muscriptor_groups=False,
        duration_seconds=assets.duration,
    )
    sheet_token = sheets.register(request_dir=root, source_midi_path=source)
    app = FastAPI()

    @app.get("/gradio_api/file={path:path}")
    def file(path: str):
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise HTTPException(404)
        return FileResponse(resolved)

    @app.post("/timing-preview")
    async def preview(request: Request):
        body = await request.json()
        try:
            return {"data": [previews.render(body["data"][0])]}
        except (ValueError, RuntimeError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/timing-sheet")
    async def sheet(request: Request):
        body = await request.json()
        return {"data": [sheets.render(body["data"][0])]}

    @app.get("/timing/{language}")
    def page(language: str):
        if language not in Translator.AVAILABLE_LANGUAGES:
            raise HTTPException(404)
        translator = Translator(language)
        state = dict(
            notes=[asdict(n) for n in assets.notes],
            duration=assets.duration,
            reference_bpm=120,
            target_bpm=120,
            time_signature=(4, 4),
            timeline=midi_timeline_manifest(source, 120),
            detected_instruments=list(assets.instrument_wavs),
            selected_instruments=[],
            instrument_wavs={key: str(path) for key, path in assets.instrument_wavs.items()},
            playback_audio_path=str(assets.original_wav),
            midi_path=str(source),
            transcription_wav=str(assets.live_transcription_wav),
            stereo_mix_wav=str(assets.stereo_mix_wav),
            preview_api="/timing-preview",
            preview_token=token,
            sheet_api="/timing-sheet",
            sheet_token=sheet_token,
            backend_label="MIDI timing fixture",
        )
        html = build_muscriptor_result_html(state, translator.t, language)
        return HTMLResponse(
            f'<!doctype html><html lang="{language}"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">{muscriptor_result_head()}'
            f'</head><body style="margin:16px;background:#0f1a2d">{html}</body></html>'
        )

    app.mount(
        "/",
        create_app(
            data_root=root / "backend",
            frontend_root=Path(__file__).resolve().parents[1] / "web",
        ),
    )
    print(
        json.dumps({"fixture_root": str(root), "source": str(source)}, ensure_ascii=False),
        flush=True,
    )
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    uvicorn.run(create_fixture_app(), host="127.0.0.1", port=args.port, log_level="warning")
