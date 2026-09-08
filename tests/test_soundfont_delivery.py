"""HTTP, cache-isolation and shared-platform regressions for SoundFont results."""

import html
import json
import re
import threading
from types import SimpleNamespace

import gradio as gr
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.gui.web.edited_midi_preview import EditedMidiPreviewRegistry
from src.gui.web.muscriptor_result_runtime import build_muscriptor_result_html
from src.web_api.soundfonts import install_soundfont_routes
from tests.test_soundfont_library import make_midi, soundfont_directory


def test_soundfont_http_import_is_bound_to_midi_and_persists_catalog(tmp_path):
    source = make_midi(tmp_path / "source.mid")
    bank = soundfont_directory(tmp_path / "extra.sf3", terminal_name=b"")

    class Manager:
        _lock = threading.RLock()

        def resolve_artifact(self, job_id, artifact_id):
            if (job_id, artifact_id) != ("owned-job", "midi"):
                raise KeyError("unknown job or artifact")
            return source, SimpleNamespace(kind="midi")

    app = FastAPI()
    install_soundfont_routes(app, Manager())
    endpoint = "/api/v1/jobs/owned-job/soundfonts/midi"
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/jobs/other-job/soundfonts/midi",
                files={"file": (bank.name, bank.read_bytes())},
            ).status_code
            == 404
        )
        assert client.post(endpoint, files={"file": ("bad.sf2", b"invalid")}).status_code == 422
        response = client.post(endpoint, files={"file": (bank.name, bank.read_bytes())})
        assert response.status_code == 200, response.text
        library = response.json()
        catalog = client.get(endpoint)
        assert catalog.status_code == 200
        assert catalog.json()["libraries"] == [library]
        assert catalog.json()["sources"][0]["program"] == 0
        assert "path" not in library
        assert (
            client.post(
                endpoint + "/render", json={"soundfont": {"id": "../outside", "assignments": []}}
            ).status_code
            == 422
        )
        assert client.get("/api/v1/soundfont-ui.js").status_code == 200


def test_gradio_presets_are_token_bound_and_cache_identity_includes_bank(tmp_path):
    source = make_midi(tmp_path / "source.mid")
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"RIFF")
    registry = EditedMidiPreviewRegistry()
    context = dict(
        request_dir=tmp_path,
        source_midi_path=source,
        original_audio_path=audio,
        reference_bpm=97,
        muscriptor_groups=False,
        duration_seconds=1,
    )
    token, other_token = registry.register(**context), registry.register(**context)
    library = registry.import_soundfont_file(token, str(soundfont_directory(tmp_path / "bank.sf2")))
    request = {
        "soundfont": {
            "id": library["id"],
            "assignments": [dict(source_program=0, is_drum=False, bank=16, program=25)],
        }
    }
    first = registry._soundfont_options(token, request)
    request["soundfont"]["assignments"][0].update(bank=0, program=0)
    second = registry._soundfont_options(token, request)
    assert registry._soundfont_identity(first) != registry._soundfont_identity(second)
    with pytest.raises(ValueError, match="未导入"):
        registry._soundfont_options(other_token, request)
    first["soundfont_selection"].library.path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        registry._soundfont_options(token, request)


def test_actual_gradio_route_accepts_every_result_export_manifest_endpoint():
    names = [
        "render_edited_midi_preview",
        "render_edited_midi_audio_export",
        "render_edited_midi_stem_export",
        "render_sheet_music_export",
    ]
    state = dict(
        playback_audio_path="source.wav",
        midi_path="source.mid",
        transcription_wav="result.wav",
        stereo_mix_wav="stereo.wav",
        instrument_wavs={},
        notes=[],
        duration=1,
    )
    for key, name in zip(
        ["preview_api", "audio_export_api", "audio_stem_export_api", "sheet_api"], names
    ):
        state[key] = "./api/" + name
    markup = build_muscriptor_result_html(state, lambda key: key, "zh_CN")
    manifest = json.loads(
        html.unescape(re.search(r'<pre class="msr-manifest" hidden>(.*?)</pre>', markup).group(1))
    )
    with gr.Blocks() as demo:
        value = gr.Textbox()
        for name in names:
            gr.Button().click(lambda payload: payload, value, value, api_name=name, queue=False)
    with TestClient(gr.routes.App.create_app(demo)) as client:
        for key in ["previewApi", "audioExportApi", "audioStemExportApi", "sheetApi"]:
            response = client.post(manifest[key].removeprefix("."), json={"data": ["verified"]})
            assert response.status_code == 200, (key, response.text)
            assert response.json()["data"] == ["verified"]
