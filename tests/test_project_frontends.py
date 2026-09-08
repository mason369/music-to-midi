from __future__ import annotations

import io
import time
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from PyQt6.QtCore import QMimeData, QPointF, Qt, QUrl
from PyQt6.QtGui import QDropEvent

from tests.test_persistent_projects import Engine, audio, project
from tests.test_web_api import available_model_profiles
from src.projects import ProjectRunner, ProjectStore
from src.web_api.app import create_app
from src.web_api.jobs import JobManager


def wait_project(client, project_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/projects/{project_id}")
        assert response.status_code == 200, response.text
        snapshot = response.json()
        if not snapshot["active"]:
            assert not snapshot.get("service_error"), snapshot
            return snapshot
        time.sleep(0.01)
    raise AssertionError("project did not finish")


def test_http_create_add_configure_run_reopen_continue_and_archive(tmp_path):
    engine = Engine()
    manager = JobManager(tmp_path / "api", engine=engine)
    try:
        with TestClient(create_app(manager=manager)) as client:
            created = client.post("/api/v1/projects", json={"name": "test"})
            assert created.status_code == 201, created.text
            pid = created.json()["storage_id"]
            wav = audio(tmp_path / "song.wav")
            with wav.open("rb") as stream:
                added = client.post(
                    f"/api/v1/projects/{pid}/songs", files={"file": (wav.name, stream, "audio/wav")}
                )
            assert added.status_code == 200, added.text
            workflow = {
                "primary": {"processing_mode": "six_stem_split"},
                "stems": {"piano": {"route": "piano_transkun"}},
            }
            configured = client.put(
                f"/api/v1/projects/{pid}/workflow",
                json={"workflow": workflow, "revision": added.json()["revision"]},
            )
            assert configured.status_code == 200, configured.text
            assert not engine.calls
            assert client.post(f"/api/v1/projects/{pid}/run", json={}).status_code == 202
            done = wait_project(client, pid)
            assert done["status"] == "succeeded", done
            assert len(engine.calls) == 2
            assert client.post(f"/api/v1/projects/{pid}/run", json={}).status_code == 202
            wait_project(client, pid)
            assert len(engine.calls) == 2
            exported = client.get(f"/api/v1/projects/{pid}/archive")
            assert exported.status_code == 200, exported.text
            imported = client.post(
                "/api/v1/projects/import",
                files={"file": ("test.mtmproject", exported.content, "application/zip")},
            )
            assert imported.status_code == 201, imported.text
            new_pid = imported.json()["storage_id"]
            client.post(f"/api/v1/projects/{new_pid}/run", json={})
            assert wait_project(client, new_pid)["status"] == "succeeded"
            assert len(engine.calls) == 2
    finally:
        manager.close()


def test_qt_multifile_drop_and_reopen_separation_with_completed_track(qtbot, tmp_path):
    from src.gui.main_window import MainWindow
    from src.models.data_models import Config

    store = project(tmp_path, {"piano": "piano_transkun"})
    ProjectRunner(store, Engine).run()
    document = store.load()
    for step in document.songs[0].checkpoints.values():
        if any(a.kind == "midi" for a in step.artifacts):
            step.result["beat"] = {"bpm": 120, "source_bpm": 120}
    store.save(document)
    with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
        window = MainWindow(Config())
    qtbot.addWidget(window)
    window.output_dir_edit.setText(str(tmp_path / "output"))
    panel = window.project_panel
    panel.store = store
    panel.refresh()
    panel.restore_result()
    assert window._last_separation_result is not None
    assert len(window.audio_mixer._backends) == 6
    assert window.audio_mixer._backends["piano"].row._midi_route == "piano_transkun"
    first = audio(tmp_path / "more1.wav", 2)
    second = audio(tmp_path / "more2.wav", 3)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(first)), QUrl.fromLocalFile(str(second))])
    event = QDropEvent(
        QPointF(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.dropzone.dropEvent(event)
    assert len(panel.store.load().songs) == 3
    assert window.worker is None
    window.close()


@pytest.mark.parametrize("with_extra_track,midi_route", [
    (False, "piano_transkun"), (True, "piano_transkun"),
    (True, "muscriptor"), (True, "muscriptor:medium"), (True, "muscriptor:small"),
])
def test_reopened_project_midi_opens_editor_without_repeating_inference(
    qtbot, tmp_path, with_extra_track, midi_route
):
    from PyQt6.QtWidgets import QPushButton
    from src.gui.main_window import MainWindow
    from src.models.data_models import Config
    from src.projects import Workflow
    from src.web_api.schemas import ManualMidiOptions
    from dataclasses import replace

    class EditorEngine(Engine):
        def run(self, **values):
            result = super().run(**values)
            # Match the real engine's persisted BPM contract; the fixture MIDI
            # uses the standard 500000 us/beat (120 BPM) tempo.
            if any(a.kind == "midi" for a in result.artifacts):
                result = replace(result, result={
                    **result.result,
                    "beat": {
                        "bpm": 120, "source_bpm": 120, "time_signature": [4, 4],
                        "beat_times": [0.0, 0.5],
                    },
                })
            return result

    engine = EditorEngine()
    if with_extra_track:
        store = project(tmp_path, {"piano": midi_route})
        song_id = store.load().songs[0].id
        extra_id = store.add_track(song_id, audio(tmp_path / "extra.wav", 12))
        workflow = store.load().songs[0].workflow.model_copy(deep=True)
        workflow.stems[extra_id] = ManualMidiOptions(route="piano_transkun")
        store.configure(workflow, [song_id])
    else:
        store = ProjectStore(tmp_path / "direct")
        store.create(workflow=Workflow())
        store.add([audio(tmp_path / "source.wav")])
    assert ProjectRunner(store, lambda: engine).run()["status"] == "succeeded"
    count = len(engine.calls)
    with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
        window = MainWindow(Config())
    qtbot.addWidget(window)
    panel = window.project_panel
    panel.store = ProjectStore(store.path)
    try:
        panel.refresh()
        panel.restore_result()
        song = store.load().songs[0]
        widget = window.muscriptor_result_widget
        assert widget is not None
        if with_extra_track:
            assert len(window.audio_mixer._backends) == 7
            assert window._last_separation_result is not None
            button = window.findChild(QPushButton, "projectMidiResult-piano")
            assert button is not None
            button.click()
            widget = window.muscriptor_result_widget
            checkpoint = song.checkpoints[song.track_keys["piano"]]
            assert widget.source_track_name == "piano"
            assert widget.muscriptor_groups == midi_route.startswith("muscriptor")
        else:
            checkpoint = song.checkpoints[song.primary_key]
            assert Path(widget.audio_path) == store.verify(song.source)
        midi = next(a for a in checkpoint.artifacts if a.kind == "midi")
        assert Path(widget.midi_path) == store.verify(midi)
        qtbot.waitUntil(lambda: widget._asset_worker is None, timeout=15000)
        assert widget._playback_engine.is_configured, widget.status_label.text()
        previous_render = widget._original_assets.transcription_wav
        previous_cache = widget._preview_root
        panel.restore_result()
        if with_extra_track:
            panel.open_midi("piano")
        widget = window.muscriptor_result_widget
        qtbot.waitUntil(lambda: widget._asset_worker is None, timeout=15000)
        assert widget._playback_engine.is_configured, widget.status_label.text()
        assert widget._original_assets.transcription_wav != previous_render
        assert not previous_cache.exists()
        assert len(engine.calls) == count
        assert store.verify(midi).is_file()
    finally:
        window.close()


def test_gradio_shared_project_ui_builds_without_loading_models(tmp_path):
    import gradio as gr
    from src.gui.web.project_workbench import build_project_workbench

    engine = Engine()
    with gr.Blocks() as demo:
        build_project_workbench(tmp_path, engine_factory=lambda: engine)
    config = demo.get_config_file()
    assert any(c["type"] == "browserstate" for c in config["components"])
    assert any(
        c["type"] == "file" and c["props"].get("file_count") == "multiple"
        for c in config["components"]
    )
    assert not engine.calls


def test_project_configuration_allows_frontend_cors_preflight(tmp_path):
    manager = JobManager(tmp_path, start_worker=False)
    try:
        with TestClient(create_app(manager=manager)) as client:
            response = client.options(
                "/api/v1/projects/anything/workflow",
                headers={
                    "Origin": "http://127.0.0.1:5173",
                    "Access-Control-Request-Method": "PUT",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            assert response.status_code == 200, response.text
            assert "PUT" in response.headers["access-control-allow-methods"]
    finally:
        manager.close()


def test_project_midi_soundfont_context_survives_export_import(tmp_path):
    from src.projects.results import ProjectResults
    from src.projects.service import ProjectService
    from src.web_api.soundfonts import install_soundfont_routes
    from tests.test_soundfont_library import soundfont_directory
    from fastapi import FastAPI

    service = ProjectService(tmp_path / "projects", Engine)
    created = service.create("test")
    store = service.store(created["storage_id"])
    store.add([audio(tmp_path / "song.wav")])
    ProjectRunner(store, Engine).run()
    song = store.load().songs[0]
    rid = f"project-{created['storage_id']}-{song.id}-{song.primary_key}"
    results = ProjectResults(service)
    app = FastAPI()
    install_soundfont_routes(
        app, results, prefix="/api/v1/project-results", on_import=results.register_soundfont
    )
    bank = soundfont_directory(tmp_path / "extra.sf3", terminal_name=b"")
    endpoint = f"/api/v1/project-results/{rid}/soundfonts/midi"
    with TestClient(app) as client:
        response = client.post(endpoint, files={"file": (bank.name, bank.read_bytes())})
        assert response.status_code == 200, response.text
        assert len(client.get(endpoint).json()["libraries"]) == 1
    archive = store.export_archive(tmp_path / "bank.mtmproject")
    moved = ProjectStore.import_archive(archive, tmp_path / "moved")
    assert (
        sum(
            a.kind == "soundfont"
            for a in moved.load().songs[0].checkpoints[song.primary_key].artifacts
        )
        == 2
    )
