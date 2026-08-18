from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from src.models.data_models import (
    Config,
    ProcessingProgress,
    ProcessingResult,
    ProcessingStage,
)
from src.web_api.app import create_app
from src.web_api.engine import ArtifactSpec, ExecutionResult, InferenceEngine
from src.web_api.jobs import JobManager


class FakeInferenceEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def run(
        self,
        *,
        kind,
        source_path,
        output_dir,
        options,
        progress_callback,
        processor_callback,
        track_id=None,
    ):
        self.calls.append((kind, options.get("processing_mode") or options.get("route")))
        processor_callback(self)
        progress_callback(
            ProcessingProgress(
                stage=ProcessingStage.TRANSCRIPTION,
                stage_progress=0.5,
                overall_progress=0.5,
                message="fake inference",
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        if kind == "primary" and options["processing_mode"] == "vocal_split":
            artifacts = []
            tracks = []
            for index, name in enumerate(("vocals", "accompaniment")):
                path = output_dir / f"{name}.wav"
                path.write_bytes(b"RIFF-fake-track-" + name.encode("ascii"))
                artifact_id = f"track-{index + 1}-{name}"
                artifacts.append(ArtifactSpec(artifact_id, "audio_track", path, track_id=name))
                tracks.append({"id": name, "name": name, "artifact_id": artifact_id})
            return ExecutionResult(
                {
                    "mode": "vocal_split",
                    "processing_time": 0.01,
                    "track_count": 2,
                    "tracks": tracks,
                    "manual_midi_required": True,
                },
                tuple(artifacts),
            )
        path = output_dir / "result.mid"
        path.write_bytes(b"MThd-fake-midi")
        return ExecutionResult(
            {
                "mode": options.get("processing_mode", "manual_midi"),
                "processing_time": 0.01,
                "total_notes": 3,
                "track_count": 1,
                "route": options.get("route"),
            },
            (ArtifactSpec("midi", "midi", path, track_id=track_id),),
        )

    def cancel(self) -> None:
        return None


class BlockingInferenceEngine(FakeInferenceEngine):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, **kwargs):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test engine release was not signalled")
        return super().run(**kwargs)


def inference_options(mode: str = "smart") -> dict:
    return {
        "processing_mode": mode,
        "transcription_backend": "yourmt3",
        "yourmt3_model": "yptf_moe_multi_nops",
        "muscriptor_model": "large",
        "muscriptor_instruments": [],
        "midi_track_mode": "multi_track",
        "custom_bpm": None,
        "use_gpu": True,
        "gpu_device": 0,
        "language": "zh_CN",
    }


def wait_for_terminal(manager: JobManager, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    snapshot = manager.snapshot(job_id)
    while snapshot["status"] not in {"succeeded", "failed", "cancelled"}:
        remaining = deadline - time.monotonic()
        assert remaining > 0, snapshot
        update = manager.wait_for_revision(job_id, snapshot["revision"], min(remaining, 0.5))
        if update is not None:
            snapshot = update
    return snapshot


def post_audio(client: TestClient, mode: str = "smart"):
    return client.post(
        "/api/v1/jobs",
        files={"audio": ("sample.wav", b"RIFF-fake-input", "audio/wav")},
        data={"options": json.dumps(inference_options(mode))},
    )


def test_capabilities_publish_all_current_modes_and_manual_routes(tmp_path):
    manager = JobManager(tmp_path, engine=FakeInferenceEngine())
    try:
        with TestClient(create_app(manager=manager, cors_origins=["http://testserver"])) as client:
            response = client.get("/api/v1/capabilities")
            assert response.status_code == 200
            payload = response.json()
            assert {item["id"] for item in payload["modes"]} == {
                "smart",
                "vocal_split",
                "six_stem_split",
                "piano_transkun",
                "piano_transkun_v2_aug",
                "piano_aria_amt",
                "piano_bytedance_pedal",
            }
            assert len(payload["manual_midi_routes"]) == 13
            assert payload["runtime"]["gpu_queue_concurrency"] == 1
    finally:
        manager.close()


def test_processing_result_reports_specialized_route_not_default_backend():
    result = ProcessingResult(midi_path="unused.mid")

    specialized = InferenceEngine._processing_payload(
        result,
        Config(
            processing_mode="piano_transkun",
            transcription_backend="yourmt3",
        ),
    )
    smart = InferenceEngine._processing_payload(
        result,
        Config(processing_mode="smart", transcription_backend="miros"),
    )

    assert specialized["transcription_backend"] == "piano_transkun"
    assert smart["transcription_backend"] == "miros"


def test_primary_job_runs_outside_request_thread_and_publishes_artifact(
    tmp_path,
):
    engine = FakeInferenceEngine()
    manager = JobManager(tmp_path, engine=engine)
    try:
        with TestClient(create_app(manager=manager)) as client:
            response = post_audio(client)
            assert response.status_code == 202, response.text
            submitted = response.json()
            assert submitted["status"] in {"queued", "running", "succeeded"}
            completed = wait_for_terminal(manager, submitted["id"])
            assert completed["status"] == "succeeded"
            assert completed["request"]["processing_mode"] == "smart"
            assert completed["result"]["total_notes"] == 3
            assert completed["progress"]["overall_progress"] == 1.0
            assert len(completed["artifacts"]) == 1

            artifact = completed["artifacts"][0]
            download = client.get(artifact["download_url"])
            assert download.status_code == 200
            assert download.content == b"MThd-fake-midi"
            assert download.headers["content-type"].startswith("audio/midi")
            assert engine.calls == [("primary", "smart")]
    finally:
        manager.close()


def test_sse_stream_delivers_terminal_snapshot(tmp_path):
    manager = JobManager(tmp_path, engine=FakeInferenceEngine())
    try:
        with TestClient(create_app(manager=manager)) as client:
            submitted = post_audio(client).json()
            completed = wait_for_terminal(manager, submitted["id"])
            response = client.get(f"/api/v1/jobs/{submitted['id']}/events")
            assert response.status_code == 200
            content_type = response.headers["content-type"]
            assert content_type.startswith("text/event-stream")
            assert "event: job" in response.text
            assert f'id: {completed["revision"]}' in response.text
            assert '"status": "succeeded"' in response.text
    finally:
        manager.close()


def test_separation_track_requires_explicit_manual_midi_child_job(tmp_path):
    engine = FakeInferenceEngine()
    manager = JobManager(tmp_path, engine=engine)
    try:
        with TestClient(create_app(manager=manager)) as client:
            parent = post_audio(client, "vocal_split").json()
            parent = wait_for_terminal(manager, parent["id"])
            assert parent["status"] == "succeeded"
            assert parent["result"]["manual_midi_required"] is True
            assert {item["track_id"] for item in parent["artifacts"]} == {
                "vocals",
                "accompaniment",
            }

            child_response = client.post(
                f"/api/v1/jobs/{parent['id']}/tracks/vocals/midi",
                json={
                    "route": "piano_aria_amt",
                    "muscriptor_instruments": [],
                    "custom_bpm": None,
                    "use_gpu": True,
                    "gpu_device": 0,
                    "language": "zh_CN",
                },
            )
            assert child_response.status_code == 202, child_response.text
            child = wait_for_terminal(manager, child_response.json()["id"])
            assert child["status"] == "succeeded"
            assert child["parent_job_id"] == parent["id"]
            assert child["track_id"] == "vocals"
            assert child["result"]["route"] == "piano_aria_amt"
            assert engine.calls == [
                ("primary", "vocal_split"),
                ("manual_midi", "piano_aria_amt"),
            ]
    finally:
        manager.close()


def test_queued_job_can_be_cancelled_without_running_the_engine(tmp_path):
    engine = BlockingInferenceEngine()
    manager = JobManager(tmp_path, engine=engine)
    first_id = None
    try:
        for index in range(2):
            job_id, job_dir = manager.reserve_job_dir()
            (job_dir / "input" / "source.wav").write_bytes(b"RIFF")
            manager.submit_reserved(
                job_id=job_id,
                kind="primary",
                source_path=job_dir / "input" / "source.wav",
                original_filename=f"{index}.wav",
                options=inference_options(),
            )
            if index == 0:
                first_id = job_id
            else:
                second_id = job_id
        assert engine.started.wait(timeout=2)
        cancelled = manager.cancel(second_id)
        assert cancelled["status"] == "cancelled"
        engine.release.set()
        assert wait_for_terminal(manager, first_id)["status"] == "succeeded"
        assert len(engine.calls) == 1
    finally:
        engine.release.set()
        manager.close()


def test_restart_marks_nonterminal_manifest_failed_not_resumed(tmp_path):
    first = JobManager(tmp_path, engine=FakeInferenceEngine(), start_worker=False)
    job_id, job_dir = first.reserve_job_dir()
    source = job_dir / "input" / "source.wav"
    source.write_bytes(b"RIFF")
    first.submit_reserved(
        job_id=job_id,
        kind="primary",
        source_path=source,
        original_filename="source.wav",
        options=inference_options(),
    )
    first.close()

    second = JobManager(tmp_path, engine=FakeInferenceEngine())
    try:
        recovered = second.snapshot(job_id)
        assert recovered["status"] == "failed"
        assert "不可安全续跑" in recovered["error"]
    finally:
        second.close()


def test_static_client_avoids_inference_imports_and_has_full_workbench():
    root = Path(__file__).resolve().parents[1]
    source = (root / "web" / "app.js").read_text(encoding="utf-8")
    markup = (root / "web" / "index.html").read_text(encoding="utf-8")
    assert "MusicToMidiPipeline" not in source
    assert "AudioSeparationService" not in source
    assert "/api/v1/jobs" in source
    assert "EventSource" in source
    for control_id in (
        "playAll",
        "pauseAll",
        "restartAll",
        "zoomSlider",
        "fitTracks",
        "alignTracks",
        "addTrack",
        "stopButton",
    ):
        assert f'id="{control_id}"' in markup


def test_static_client_locales_are_complete_and_ui_has_no_hardcoded_chinese():
    root = Path(__file__).resolve().parents[1]
    source = (root / "web" / "app.js").read_text(encoding="utf-8")
    markup = (root / "web" / "index.html").read_text(encoding="utf-8")
    catalogs = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (root / "web" / "locales").glob("*.json")
    }

    assert catalogs.keys() == {"zh_CN", "en_US"}
    assert catalogs["zh_CN"].keys() == catalogs["en_US"].keys()
    assert not re.search(r"[\u3400-\u9fff]", source + markup)

    declared_keys = set(
        re.findall(
            r'data-i18n(?:-title|-aria-label|-placeholder)?="([^"]+)"',
            markup,
        )
    )
    static_script_keys = set(re.findall(r'\bt\("([^"]+)"', source))
    assert declared_keys | static_script_keys <= catalogs["zh_CN"].keys()


def test_static_client_matches_the_existing_desktop_app_visual_language():
    styles_path = Path(__file__).resolve().parents[1] / "web" / "styles.css"
    styles = styles_path.read_text(encoding="utf-8")

    assert "--canvas: #1a1a2e" in styles
    assert "--panel-input: #16213e" in styles
    assert "--blue: #4a9eff" in styles
    assert "--control-corner: 5px" in styles
    assert "[hidden] { display: none !important; }" in styles
    assert "border-radius: 18px" not in styles
    assert "radial-gradient" not in styles
    assert ".route-rail" not in styles
    assert ".surface-grid" not in styles
    assert "min-height: 300px" in styles
    assert "alignment-console" not in styles
    assert "chord-lane" not in styles


def test_static_client_guidance_tracks_real_job_state():
    root = Path(__file__).resolve().parents[1]
    source = (root / "web" / "app.js").read_text(encoding="utf-8")
    markup = (root / "web" / "index.html").read_text(encoding="utf-8")
    styles = (root / "web" / "styles.css").read_text(encoding="utf-8")

    for control_id in (
        "workflowGuide",
        "guideTitle",
        "guideDescription",
        "guideAction",
        "configurationPanel",
        "actionDeck",
    ):
        assert f'id="{control_id}"' in markup

    assert "function updateWorkflowGuide()" in source
    assert "state.submissionPending" in source
    assert "isJobRunning(job)" in source
    assert "performGuideAction" in source
    assert '$("#startButton").click()' not in source
    assert 'id="modeSelect"' in markup
    assert '$("#modeSelect").addEventListener("change"' in source
    assert "select.value = state.selectedMode" in source
    assert ".guidance-target.is-guided" in styles
    assert ".guide-steps" not in styles
    assert 'class="guide-steps"' not in markup
def test_terminal_job_retry_copies_input_and_preserves_options(tmp_path):
    engine = FakeInferenceEngine()
    manager = JobManager(tmp_path, engine=engine)
    try:
        with TestClient(create_app(manager=manager)) as client:
            original = wait_for_terminal(manager, post_audio(client).json()["id"])
            retry_response = client.post(f"/api/v1/jobs/{original['id']}/retry")
            assert retry_response.status_code == 202, retry_response.text
            retried = wait_for_terminal(manager, retry_response.json()["id"])

            assert retried["status"] == "succeeded"
            assert retried["retry_of_job_id"] == original["id"]
            assert retried["request"] == original["request"]
            assert retried["original_filename"] == original["original_filename"]
            assert (
                tmp_path / retried["id"] / "input" / "source.wav"
            ).read_bytes() == b"RIFF-fake-input"
            assert engine.calls == [("primary", "smart"), ("primary", "smart")]
    finally:
        manager.close()


def test_artifact_sha256_is_persisted_exposed_and_rechecked(tmp_path):
    manager = JobManager(tmp_path, engine=FakeInferenceEngine())
    try:
        with TestClient(create_app(manager=manager)) as client:
            completed = wait_for_terminal(manager, post_audio(client).json()["id"])
            artifact = completed["artifacts"][0]
            assert re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])

            download = client.get(artifact["download_url"])
            assert download.status_code == 200
            assert download.headers["x-content-sha256"] == artifact["sha256"]
            assert download.headers["etag"] == f'"sha256-{artifact["sha256"]}"'

            artifact_path = tmp_path / completed["id"] / "output" / "result.mid"
            original = artifact_path.read_bytes()
            artifact_path.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
            changed = client.get(artifact["download_url"])
            assert changed.status_code == 409
            assert "checksum changed" in changed.json()["detail"]
    finally:
        manager.close()


def test_legacy_manifest_artifact_hash_is_migrated_on_restart(tmp_path):
    first = JobManager(tmp_path, engine=FakeInferenceEngine())
    try:
        job_id, job_dir = first.reserve_job_dir()
        source = job_dir / "input" / "source.wav"
        source.write_bytes(b"RIFF-fake-input")
        first.submit_reserved(
            job_id=job_id,
            kind="primary",
            source_path=source,
            original_filename="sample.wav",
            options=inference_options(),
        )
        completed = wait_for_terminal(first, job_id)
    finally:
        first.close()

    manifest_path = tmp_path / completed["id"] / JobManager.MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in payload["artifacts"]:
        artifact.pop("sha256", None)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    second = JobManager(tmp_path, engine=FakeInferenceEngine(), start_worker=False)
    try:
        migrated = second.snapshot(completed["id"])
        assert re.fullmatch(r"[0-9a-f]{64}", migrated["artifacts"][0]["sha256"])
        persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert persisted["artifacts"][0]["sha256"] == migrated["artifacts"][0]["sha256"]
    finally:
        second.close()


def test_static_client_api_discovery_and_launcher_use_the_real_host_and_ports():
    root = Path(__file__).resolve().parents[1]
    source = (root / "web" / "app.js").read_text(encoding="utf-8")
    launcher = (root / "run_web.ps1").read_text(encoding="utf-8")

    assert "window.location.hostname" in source
    assert 'new URLSearchParams(window.location.search).get("api")' in source
    assert "normalizeApiBase" in source
    assert "EscapeDataString($apiBase)" in launcher
    assert "MUSIC_TO_MIDI_ALLOWED_ORIGINS" in launcher
    assert "-PublicHost is required" in launcher
