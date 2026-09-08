from __future__ import annotations

import json
import threading
import wave
import zipfile
from pathlib import Path
from types import SimpleNamespace

import mido
import pytest

from src.projects import ProjectRunner, ProjectStore, Workflow
from src.projects.store import PROJECT_FILE, SPLIT_STEMS, exclusive
from src.web_api.engine import ArtifactSpec, ExecutionResult
from src.web_api.schemas import InferenceOptions, ManualMidiOptions


def audio(path: Path, value: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(value.to_bytes(2, "little", signed=True) * 1600)
    return path


class Engine:
    def __init__(self):
        self.calls = []
        self.fail_track = None
        self.hook = None

    def run(self, **values):
        self.calls.append((values["kind"], values.get("track_id"), values["options"]))
        if self.hook:
            self.hook(values)
        if values.get("track_id") == self.fail_track and self.fail_track:
            raise RuntimeError("test failure")
        values["progress_callback"](
            SimpleNamespace(
                stage=SimpleNamespace(value="transcription"), overall_progress=0.5, message="test"
            )
        )
        directory = values["output_dir"]
        mode = values["options"].get("processing_mode")
        if mode in SPLIT_STEMS:
            artifacts = tuple(
                ArtifactSpec(
                    f"track-{stem}", "audio_track", audio(directory / f"{stem}.wav"), track_id=stem
                )
                for stem in SPLIT_STEMS[mode]
            )
            return ExecutionResult(
                {
                    "mode": mode,
                    "manual_midi_required": True,
                    "tracks": [
                        {"id": a.track_id, "name": a.track_id, "artifact_id": a.id}
                        for a in artifacts
                    ],
                },
                artifacts,
            )
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        track.extend(
            [mido.Message("note_on", note=60), mido.Message("note_off", note=60, time=480)]
        )
        midi.tracks.append(track)
        path = directory / "result.mid"
        midi.save(path)
        return ExecutionResult(
            {
                "mode": mode or "manual_midi",
                "route": values["options"].get("route"),
                "total_notes": 1,
            },
            (ArtifactSpec("midi", "midi", path, track_id=values.get("track_id")),),
        )


def project(tmp_path, stems=None, count=1):
    store = ProjectStore(tmp_path / "project")
    store.create(
        workflow=Workflow(
            primary=InferenceOptions(processing_mode="six_stem_split"),
            stems={s: ManualMidiOptions(route=r) for s, r in (stems or {}).items()},
        )
    )
    store.add([audio(tmp_path / f"song{i}.wav", i) for i in range(count)])
    return store


def test_restart_reuses_six_stems_and_completed_midi_only_runs_remaining(tmp_path):
    store = project(tmp_path, {"piano": "piano_transkun", "bass": "miros"})
    first = Engine()
    first.fail_track = "bass"
    assert ProjectRunner(store, lambda: first).run()["status"] == "failed"
    second = Engine()
    reopened = ProjectStore(store.path)
    assert ProjectRunner(reopened, lambda: second).run()["status"] == "succeeded"
    assert [(kind, stem) for kind, stem, _ in second.calls] == [("manual_midi", "bass")]
    assert len(reopened.load().songs) == 1


def test_profile_edit_only_invalidates_changed_track_and_never_executes(tmp_path):
    store = project(tmp_path, {"piano": "piano_transkun", "bass": "miros"})
    engine = Engine()
    ProjectRunner(store, lambda: engine).run()
    workflow = store.load().workflow
    workflow.stems["piano"] = ManualMidiOptions(route="piano_aria_amt")
    store.configure(workflow)
    assert len(engine.calls) == 3
    ProjectRunner(store, lambda: engine).run()
    assert len(engine.calls) == 4
    assert engine.calls[-1][1:] == ("piano", workflow.stems["piano"].model_dump(mode="json"))


def test_batch_cancel_resume_keeps_finished_songs(tmp_path):
    store = project(tmp_path, count=4)
    engine = Engine()
    runner = ProjectRunner(store, lambda: engine)
    engine.hook = lambda _values: runner.cancel() if len(engine.calls) == 3 else None
    assert runner.run()["status"] == "cancelled"
    assert sum(s.status == "succeeded" for s in store.load().songs) == 2
    engine2 = Engine()
    assert ProjectRunner(store, lambda: engine2).run()["status"] == "succeeded"
    assert len(engine2.calls) == 2


def test_missing_or_changed_asset_is_explicit_failure_without_retranscription(tmp_path):
    store = project(tmp_path)
    engine = Engine()
    ProjectRunner(store, lambda: engine).run()
    song = store.load().songs[0]
    store.resolve(song.checkpoints[song.primary_key].artifacts[0].path).write_bytes(b"corrupt")
    result = ProjectRunner(store, lambda: engine).run()
    assert result["status"] == "failed"
    assert "资源丢失或内容发生变化" in result["error"]
    assert len(engine.calls) == 1


def test_archive_roundtrip_relocation_preserves_every_checkpoint(tmp_path):
    store = project(tmp_path, {"piano": "piano_transkun"})
    ProjectRunner(store, Engine).run()
    archive = store.export_archive(tmp_path / "saved.mtmproject")
    moved = ProjectStore.import_archive(archive, tmp_path / "moved")
    engine = Engine()
    assert ProjectRunner(moved, lambda: engine).run()["status"] == "succeeded"
    assert engine.calls == []


@pytest.mark.parametrize("name", ["../evil", "/absolute", "C:/escape", "songs/../../escape"])
def test_archive_path_escape_is_rejected_without_writing_outside(tmp_path, name):
    store = project(tmp_path)
    document = store.load().model_dump(mode="json")
    document["songs"][0]["source"]["path"] = name
    archive = tmp_path / "bad.mtmproject"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(PROJECT_FILE, json.dumps(document))
        z.writestr(name, b"bad")
    with pytest.raises(ValueError):
        ProjectStore.import_archive(archive, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_concurrent_run_and_stale_configuration_cannot_overwrite_progress(tmp_path):
    store = project(tmp_path)
    stale = store.load()
    with exclusive(store.run_lock):
        with pytest.raises(RuntimeError, match="其他窗口"):
            ProjectRunner(store, Engine).run()
        with pytest.raises(RuntimeError, match="其他窗口"):
            store.configure(Workflow())
    store.configure(store.load().workflow)
    with pytest.raises(RuntimeError, match="项目已被更新"):
        store.save(stale)


def test_duplicate_input_uses_existing_song_without_new_outputs(tmp_path):
    store = project(tmp_path)
    song = store.load().songs[0]
    assert store.add([tmp_path / "song0.wav"]) == [song.id]
    assert len(store.load().songs) == 1


def test_project_cli_create_configure_run_status(tmp_path):
    from src.cli.app import main
    import io

    path = tmp_path / "p"
    source = audio(tmp_path / "a.wav")
    engine = Engine()

    def invoke(args):
        out, err = io.StringIO(), io.StringIO()
        code = main(
            ["project", *args, "--json"],
            engine_factory=lambda: engine,
            stdout=out,
            stderr=err,
            install_signal_handlers=False,
        )
        assert code == 0, err.getvalue()
        return json.loads(out.getvalue().splitlines()[-1])

    invoke(
        [
            "create",
            str(path),
            str(source),
            "--mode",
            "six-stem-split",
            "--stem",
            "piano=piano_transkun",
        ]
    )
    assert len(engine.calls) == 0
    assert invoke(["run", str(path)])["status"] == "succeeded"
    assert len(engine.calls) == 2
    assert invoke(["run", str(path)])["status"] == "succeeded"
    assert len(engine.calls) == 2
