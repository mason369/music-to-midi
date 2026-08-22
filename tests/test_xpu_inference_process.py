from __future__ import annotations

import json
import sys

import pytest

import src.main as main_module
import src.utils.gpu_utils as gpu_utils
import src.web_api.inference_process as inference_process
from src.models.data_models import ProcessingProgress, ProcessingStage
from src.web_api.engine import ArtifactSpec, ExecutionResult, InferenceEngine


def test_xpu_isolation_is_required_only_for_gpu_jobs_outside_the_worker(monkeypatch):
    monkeypatch.delenv("MUSIC_TO_MIDI_WEB_INFERENCE_WORKER", raising=False)
    monkeypatch.setattr(gpu_utils, "get_accelerator_type", lambda: "xpu")

    assert inference_process.should_isolate_xpu_inference({"use_gpu": True}) is True
    assert inference_process.should_isolate_xpu_inference({"use_gpu": False}) is False

    monkeypatch.setenv("MUSIC_TO_MIDI_WEB_INFERENCE_WORKER", "1")
    assert inference_process.should_isolate_xpu_inference({"use_gpu": True}) is False


def test_inference_engine_delegates_xpu_job_without_running_parent_cleanup(monkeypatch, tmp_path):
    artifact = tmp_path / "result.mid"
    artifact.write_bytes(b"midi")
    expected = ExecutionResult(
        result={"ok": True},
        artifacts=(ArtifactSpec("midi", "midi", artifact),),
    )
    calls = []

    class FakeIsolatedProcess:
        def run(self, **kwargs):
            calls.append(kwargs)
            return expected

    monkeypatch.setattr(inference_process, "should_isolate_xpu_inference", lambda _options: True)
    monkeypatch.setattr(inference_process, "XpuInferenceProcess", FakeIsolatedProcess)
    monkeypatch.setattr(
        InferenceEngine,
        "_run_direct",
        lambda *_args, **_kwargs: pytest.fail("parent process must not run XPU inference"),
    )
    processors = []

    actual = InferenceEngine().run(
        kind="primary",
        source_path=tmp_path / "source.wav",
        output_dir=tmp_path / "output",
        options={"use_gpu": True},
        progress_callback=lambda _progress: None,
        processor_callback=processors.append,
    )

    assert actual is expected
    assert len(calls) == 1
    assert isinstance(processors[0], FakeIsolatedProcess)
    assert processors[1] is None


def test_worker_serializes_progress_result_and_artifacts(monkeypatch, tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"wav")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    artifact = output_dir / "result.mid"
    artifact.write_bytes(b"midi")
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    events_path = tmp_path / "events.jsonl"
    request_path.write_text(
        json.dumps(
            {
                "kind": "primary",
                "source_path": str(source),
                "output_dir": str(output_dir),
                "options": {"use_gpu": True},
                "track_id": None,
                "tempo_source_path": None,
            }
        ),
        encoding="utf-8",
    )

    def fake_run_direct(_self, **kwargs):
        kwargs["progress_callback"](
            ProcessingProgress(
                stage=ProcessingStage.TRANSCRIPTION,
                stage_progress=0.5,
                overall_progress=0.4,
                message="real progress",
            )
        )
        return ExecutionResult(
            result={"total_notes": 4},
            artifacts=(ArtifactSpec("midi", "midi", artifact),),
        )

    monkeypatch.setattr(InferenceEngine, "_run_direct", fake_run_direct)

    exit_code = inference_process.run_inference_worker(
        [
            "--request-json",
            str(request_path),
            "--response-json",
            str(response_path),
            "--events-jsonl",
            str(events_path),
        ]
    )

    response = json.loads(response_path.read_text(encoding="utf-8"))
    event = json.loads(events_path.read_text(encoding="utf-8").strip())
    assert exit_code == 0
    assert response == {
        "ok": True,
        "result": {"total_notes": 4},
        "artifacts": [
            {
                "id": "midi",
                "kind": "midi",
                "path": str(artifact),
                "track_id": None,
            }
        ],
    }
    assert event["stage"] == "transcription"
    assert event["message"] == "real progress"


def test_worker_failure_is_explicit_and_contains_traceback(monkeypatch, tmp_path):
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    events_path = tmp_path / "events.jsonl"
    request_path.write_text(
        json.dumps(
            {
                "kind": "primary",
                "source_path": str(tmp_path / "source.wav"),
                "output_dir": str(tmp_path / "output"),
                "options": {"use_gpu": True},
                "track_id": None,
                "tempo_source_path": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        InferenceEngine,
        "_run_direct",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("worker exploded")),
    )

    exit_code = inference_process.run_inference_worker(
        [
            "--request-json",
            str(request_path),
            "--response-json",
            str(response_path),
            "--events-jsonl",
            str(events_path),
        ]
    )

    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert response["ok"] is False
    assert response["error"] == "worker exploded"
    assert "RuntimeError: worker exploded" in response["traceback"]


def test_parent_reports_nonzero_worker_exit_and_removes_control_directory(monkeypatch, tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"wav")
    output = tmp_path / "job" / "output"
    output.mkdir(parents=True)

    class FailedProcess:
        returncode = 23

        def poll(self):
            return 23

        def wait(self, timeout=None):
            return 23

    monkeypatch.setattr(
        inference_process.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FailedProcess(),
    )
    monkeypatch.setattr(
        inference_process,
        "_worker_command",
        lambda *_args: ["fake-worker"],
    )

    with pytest.raises(RuntimeError, match="exited with code 23"):
        inference_process.XpuInferenceProcess().run(
            kind="primary",
            source_path=source,
            output_dir=output,
            options={"use_gpu": True},
            progress_callback=lambda _progress: None,
            track_id=None,
            tempo_source_path=None,
        )

    assert list((tmp_path / "job").glob(".xpu-inference-*")) == []


def test_parent_propagates_failure_manifest_before_nonzero_worker_exit(monkeypatch, tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"wav")
    output = tmp_path / "job" / "output"
    output.mkdir(parents=True)

    class FailedProcess:
        returncode = 1

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

    def fake_worker_command(_request_path, response_path, _events_path):
        response_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "error": "ByteDance XPU initialization failed",
                    "traceback": "RuntimeError: exact worker traceback",
                }
            ),
            encoding="utf-8",
        )
        return ["fake-worker"]

    monkeypatch.setattr(
        inference_process.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FailedProcess(),
    )
    monkeypatch.setattr(inference_process, "_worker_command", fake_worker_command)

    with pytest.raises(RuntimeError) as excinfo:
        inference_process.XpuInferenceProcess().run(
            kind="primary",
            source_path=source,
            output_dir=output,
            options={"use_gpu": True},
            progress_callback=lambda _progress: None,
            track_id=None,
            tempo_source_path=None,
        )

    message = str(excinfo.value)
    assert "exited with code 1" in message
    assert "ByteDance XPU initialization failed" in message
    assert "RuntimeError: exact worker traceback" in message
    assert list((tmp_path / "job").glob(".xpu-inference-*")) == []


def test_parent_cancel_before_worker_start_is_terminal_and_cleans_control_directory(
    monkeypatch, tmp_path
):
    source = tmp_path / "source.wav"
    source.write_bytes(b"wav")
    output = tmp_path / "job" / "output"
    output.mkdir(parents=True)

    class RunningProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(
        inference_process.subprocess,
        "Popen",
        lambda *_args, **_kwargs: RunningProcess(),
    )
    monkeypatch.setattr(
        inference_process,
        "_worker_command",
        lambda *_args: ["fake-worker"],
    )
    isolated = inference_process.XpuInferenceProcess()
    isolated.cancel()

    with pytest.raises(InterruptedError, match="cancelled before startup"):
        isolated.run(
            kind="primary",
            source_path=source,
            output_dir=output,
            options={"use_gpu": True},
            progress_callback=lambda _progress: None,
            track_id=None,
            tempo_source_path=None,
        )

    assert list((tmp_path / "job").glob(".xpu-inference-*")) == []


def test_frozen_worker_command_reuses_backend_executable(monkeypatch, tmp_path):
    monkeypatch.setattr(inference_process, "is_frozen_app", lambda: True)
    monkeypatch.setattr(inference_process.sys, "executable", r"C:\bundle\MusicToMidiBackendXpu.exe")

    command = inference_process._worker_command(
        tmp_path / "request.json",
        tmp_path / "response.json",
        tmp_path / "events.jsonl",
    )

    assert command[:2] == [r"C:\bundle\MusicToMidiBackendXpu.exe", "--web-inference-worker"]


def test_main_dispatches_web_inference_worker_before_gui_startup(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["MusicToMidiBackendXpu.exe", "--web-inference-worker", "--request-json", "job.json"],
    )
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(
        inference_process,
        "run_inference_worker",
        lambda argv: calls.append(argv) or 9,
    )
    monkeypatch.setattr(
        main_module,
        "_prepare_torch_runtime_before_pyqt",
        lambda: pytest.fail("worker dispatch must happen before GUI/Torch startup"),
    )

    with pytest.raises(SystemExit) as excinfo:
        main_module.main()

    assert excinfo.value.code == 9
    assert calls == [["--request-json", "job.json"]]


def test_frozen_main_hard_exits_after_web_inference_worker(monkeypatch):
    calls = []
    hard_exits = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["MusicToMidiBackendXpu.exe", "--web-inference-worker", "--request-json", "job.json"],
    )
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        inference_process,
        "run_inference_worker",
        lambda argv: calls.append(argv) or 9,
    )
    monkeypatch.setattr(main_module.os, "_exit", hard_exits.append)

    main_module.main()

    assert calls == [["--request-json", "job.json"]]
    assert hard_exits == [9]
