"""Checkpoint identity, real process loss, and cross-platform delivery contracts."""

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.core.manual_midi import MANUAL_MIDI_ROUTES
from src.models.data_models import ProcessingMode, ProcessingProgress, ProcessingStage
from src.projects import ProjectStore, ProjectRunner, Workflow
from src.projects.service import ProjectService
from src.web_api.schemas import InferenceOptions, ManualMidiOptions
from tests.test_persistent_projects import Engine, audio, project


@pytest.mark.parametrize("mode", [m.value for m in ProcessingMode if m is not ProcessingMode.PIANO])
def test_every_primary_mode_roundtrips_without_reinference(tmp_path, mode):
    store = ProjectStore(tmp_path / "project")
    store.create(workflow=Workflow(primary=InferenceOptions(processing_mode=mode)))
    store.add([audio(tmp_path / "source.wav")])
    engine = Engine()
    assert ProjectRunner(store, lambda: engine).run()["status"] == "succeeded"
    assert ProjectRunner(ProjectStore(store.path), lambda: engine).run()["status"] == "succeeded"
    assert len(engine.calls) == 1


@pytest.mark.parametrize("route", MANUAL_MIDI_ROUTES)
def test_all_manual_routes_preserve_model_and_checkpoint(tmp_path, route):
    store = project(tmp_path, {"piano": route})
    engine = Engine()
    ProjectRunner(store, lambda: engine).run()
    ProjectRunner(ProjectStore(store.path), lambda: engine).run()
    assert len(engine.calls) == 2
    assert engine.calls[1][2]["route"] == route


def test_model_change_clears_only_current_result_and_partial_run_is_pending(tmp_path):
    store = project(tmp_path, {"piano": "piano_transkun"}, count=2)
    engine = Engine()
    sid = store.load().songs[0].id
    snapshot = ProjectRunner(store, lambda: engine).run([sid])
    assert snapshot["status"] == "pending" and snapshot["last_run_status"] == "succeeded"
    before = store.load().songs[0]
    changed = before.workflow.model_copy(deep=True)
    changed.stems["piano"] = ManualMidiOptions(route="piano_aria_amt")
    store.configure(changed, [sid])
    after = store.load().songs[0]
    assert after.primary_key == before.primary_key
    assert not after.track_keys
    assert after.status == "pending" and before.track_keys["piano"] in after.checkpoints


def test_process_kill_preserves_completed_song_and_unlocks_resume(tmp_path):
    store = project(tmp_path, count=3)
    marker = tmp_path / "running.txt"
    script = tmp_path / "worker.py"
    script.write_text(
        """import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from tests.test_persistent_projects import Engine
from src.projects import ProjectRunner,ProjectStore
engine=Engine()
def hook(value):
    if len(engine.calls)==2:
        Path(sys.argv[3]).write_text('ready')
        time.sleep(120)
engine.hook=hook
ProjectRunner(ProjectStore(sys.argv[2]),lambda:engine).run()
""",
        encoding="utf-8",
    )
    with (tmp_path / "worker.log").open("wb") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                str(script),
                str(Path(__file__).parents[1]),
                str(store.path),
                str(marker),
            ],
            stdout=output,
            stderr=output,
        )
        try:
            deadline = time.monotonic() + 15
            while not marker.exists() and time.monotonic() < deadline and process.poll() is None:
                time.sleep(0.05)
            assert marker.exists()
            assert store.is_busy()
            assert store.load().songs[0].status == "succeeded"
        finally:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=True,
                    capture_output=True,
                )
            else:
                process.kill()
            process.wait(timeout=10)
    assert not store.is_busy()
    engine = Engine()
    assert ProjectRunner(ProjectStore(store.path), lambda: engine).run()["status"] == "succeeded"
    assert len(engine.calls) == 2


def test_zero_gpu_transports_only_serializable_arguments_and_progress(tmp_path, monkeypatch):
    from src.gui.web import project_zero_gpu as module

    captured = []

    def decorator(**config):
        def wrap(fn):
            def invoke(request):
                pickle.dumps(request)
                captured.append(request)
                from src.core.transcription_stream import append_jsonl_event

                append_jsonl_event(
                    request["events_path"],
                    ProcessingProgress(
                        ProcessingStage.TRANSCRIPTION, 0.5, 0.5, "progress"
                    ).to_dict(),
                )
                return "complete"

            return invoke

        return wrap

    factory = module.make_zerogpu_engine(decorator, lambda *a: 60)
    updates, processors = [], []
    result = factory().run(
        kind="primary",
        source_path=tmp_path / "source.wav",
        output_dir=tmp_path,
        options=InferenceOptions().model_dump(),
        progress_callback=updates.append,
        processor_callback=processors.append,
        track_id=None,
        tempo_source_path=None,
    )
    assert result == "complete" and len(updates) == 1
    assert not {"progress_callback", "processor_callback"} & captured[0].keys()
    assert processors[-1] is None


def test_source_delivery_includes_project_dependency_tree():
    root = Path(__file__).parents[1]
    sync = (root / ".github/workflows/sync_to_hf.yml").read_text(encoding="utf-8")
    assert "core models utils i18n projects web_api" in sync
    assert "- 'src/model_profile_runtime_probe.py'" in sync
    assert "src/model_profiles.py src/model_profile_runtime_probe.py src/web_contract.py" in sync
    spec = (root / "MusicToMidi.spec").read_text(encoding="utf-8")
    assert "'src.projects.cli'" in spec
    assert "'src.gui.widgets.project_panel'" in spec
    for relative in ("docker/backend.Dockerfile", "docker/gateway.Dockerfile"):
        assert (
            "COPY src ./src" in (root / relative).read_text()
            if "backend" in relative
            else "COPY web /srv/web" in (root / relative).read_text()
        )


def test_primary_button_does_not_execute_saved_manual_plan(tmp_path):
    store = project(tmp_path, {"piano": "piano_transkun"})
    engine = Engine()
    result = ProjectRunner(store, lambda: engine).run(primary_only=True)
    assert result["status"] == "pending"
    assert [call[0] for call in engine.calls] == ["primary"]
    assert ProjectRunner(store, lambda: engine).run()["status"] == "succeeded"
    assert [call[0] for call in engine.calls] == ["primary", "manual_midi"]


def test_concurrent_manifest_reads_and_atomic_writes(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    store = project(tmp_path)

    def writer():
        for _ in range(60):
            document = store.load()
            store.save(document)

    def reader():
        for _ in range(100):
            assert ProjectStore(store.root).load().songs[0].source.sha256

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(writer), *(pool.submit(reader) for _ in range(4))]
        for future in futures:
            future.result(timeout=15)


def test_external_track_is_portable_and_keeps_completed_separation(tmp_path):
    store = project(tmp_path)
    engine = Engine()
    ProjectRunner(store, lambda: engine).run()
    sid = store.load().songs[0].id
    key = store.add_track(sid, audio(tmp_path / "extra.wav", 5))
    workflow = store.load().songs[0].workflow
    workflow.stems[key] = ManualMidiOptions(route="miros")
    store.configure(workflow, [sid])
    assert ProjectRunner(store, lambda: engine).run()["status"] == "succeeded"
    assert len(engine.calls) == 2 and engine.calls[-1][1] == key
    archive = store.export_archive(tmp_path / "extra.mtmproject")
    restored = ProjectStore.import_archive(archive, tmp_path / "restored")
    assert ProjectRunner(restored, lambda: engine).run()["status"] == "succeeded"
    assert len(engine.calls) == 2


def test_shared_workflow_keeps_external_tracks_with_their_own_song(tmp_path):
    store = project(tmp_path)
    first = store.load().songs[0].id
    second = store.add([audio(tmp_path / "second.wav", 2)])[0]
    key1 = store.add_track(first, audio(tmp_path / "extra1.wav", 3))
    key2 = store.add_track(second, audio(tmp_path / "extra2.wav", 4))
    store.update_song_state(first, {}, {key1: ManualMidiOptions(route="miros")})
    store.update_song_state(second, {}, {key2: ManualMidiOptions(route="piano_transkun")})
    workflow = store.load().songs[0].workflow
    workflow.stems["piano"] = ManualMidiOptions(route="piano_aria_amt")
    store.configure(workflow)
    songs = store.load().songs
    assert set(songs[0].workflow.stems) == {"piano", key1}
    assert set(songs[1].workflow.stems) == {"piano", key2}
    third = store.add([audio(tmp_path / "third.wav", 6)])[0]
    assert set(next(s for s in store.load().songs if s.id == third).workflow.stems) == {"piano"}
    path = tmp_path / "workflow.json"
    workflow.save(path)
    assert set(Workflow.load(path).stems) == {"piano"}


@pytest.mark.parametrize("direct_mode", [False, True])
def test_clear_custom_stems_keeps_source_and_history(tmp_path, direct_mode):
    store = project(tmp_path)
    sid = store.load().songs[0].id
    key = store.add_track(sid, audio(tmp_path / "extra.wav", 7))
    store.update_song_state(sid, {}, {key: ManualMidiOptions(route="miros")})
    engine = Engine()
    ProjectRunner(store, lambda: engine).run()
    old = store.load().songs[0]
    workflow = Workflow() if direct_mode else old.workflow.reusable()
    store.configure(workflow, clear_custom_stems=not direct_mode)
    song = store.load().songs[0]
    assert not song.workflow.stems
    assert key in song.extra_tracks
    assert song.checkpoints == old.checkpoints


def test_unknown_external_track_in_workflow_is_an_explicit_error(tmp_path):
    store = project(tmp_path)
    before = store.load()
    workflow = before.workflow.model_copy(deep=True)
    workflow.stems["custom-" + "a" * 32] = ManualMidiOptions(route="miros")
    with pytest.raises(ValueError, match="不存在的外部音轨"):
        store.configure(workflow)
    assert store.load() == before
