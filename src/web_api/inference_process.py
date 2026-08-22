"""One-fresh-process-per-job isolation for Windows Intel XPU Web inference.

Intel client GPUs share the Windows system commit budget with host allocations.
Keeping every large model backend in the long-lived HTTP process can therefore
leave enough committed host memory behind to starve a later backend even after
the XPU caching allocator is emptied.  This module keeps the API process light:
each XPU job owns a fresh process and the operating system reclaims its complete
address space at the terminal state.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from src.core.transcription_stream import append_jsonl_event, read_new_jsonl_events
from src.models.data_models import ProcessingProgress, ProcessingStage
from src.utils.runtime_paths import get_executable_dir, get_project_root, is_frozen_app

logger = logging.getLogger(__name__)

_WORKER_SWITCH = "--web-inference-worker"
_WORKER_ENV = "MUSIC_TO_MIDI_WEB_INFERENCE_WORKER"


def should_isolate_xpu_inference(options: dict[str, Any]) -> bool:
    """Return whether this HTTP job must run outside the persistent API process."""

    if os.environ.get(_WORKER_ENV) == "1" or not bool(options.get("use_gpu", True)):
        return False
    from src.utils.gpu_utils import get_accelerator_type

    return get_accelerator_type() == "xpu"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _tail_text(path: Path, *, limit: int = 12000) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace").strip()


def _worker_command(request_path: Path, response_path: Path, events_path: Path) -> list[str]:
    arguments = [
        "--request-json",
        str(request_path),
        "--response-json",
        str(response_path),
        "--events-jsonl",
        str(events_path),
    ]
    if is_frozen_app():
        return [sys.executable, _WORKER_SWITCH, *arguments]
    return [sys.executable, "-m", "src.web_api.inference_process", *arguments]


class XpuInferenceProcess:
    """Own one isolated inference process and expose cooperative job cancellation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._cancelled = False

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    @staticmethod
    def _terminate_and_reap(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        else:
            process.wait(timeout=0)

    def run(
        self,
        *,
        kind: str,
        source_path: Path,
        output_dir: Path,
        options: dict[str, Any],
        progress_callback,
        track_id: str | None,
        tempo_source_path: Path | None,
    ):
        from src.web_api.engine import ArtifactSpec, ExecutionResult

        source = Path(source_path).resolve()
        output = Path(output_dir).resolve()
        tempo_source = Path(tempo_source_path).resolve() if tempo_source_path else None
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"isolated XPU job input is missing or empty: {source}")
        output.mkdir(parents=True, exist_ok=True)
        worker_dir = output.parent / f".xpu-inference-{uuid.uuid4().hex}"
        worker_dir.mkdir(parents=False, exist_ok=False)
        request_path = worker_dir / "request.json"
        response_path = worker_dir / "response.json"
        events_path = worker_dir / "progress.jsonl"
        stdout_path = worker_dir / "stdout.log"
        stderr_path = worker_dir / "stderr.log"
        request_payload = {
            "kind": str(kind),
            "source_path": str(source),
            "output_dir": str(output),
            "options": dict(options),
            "track_id": track_id,
            "tempo_source_path": str(tempo_source) if tempo_source is not None else None,
        }
        _atomic_write_json(request_path, request_payload)

        command = _worker_command(request_path, response_path, events_path)
        process_env = dict(os.environ)
        process_env[_WORKER_ENV] = "1"
        process_env["PYTHONIOENCODING"] = "utf-8"
        process_env["PYTHONUTF8"] = "1"
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        event_offset = 0
        process: subprocess.Popen[bytes] | None = None

        def forward_progress() -> None:
            nonlocal event_offset
            events, event_offset = read_new_jsonl_events(events_path, event_offset)
            for payload in events:
                progress_callback(
                    ProcessingProgress(
                        stage=ProcessingStage(str(payload["stage"])),
                        stage_progress=float(payload["stage_progress"]),
                        overall_progress=float(payload["overall_progress"]),
                        message=str(payload["message"]),
                        bpm_display=payload.get("bpm_display"),
                        source_bpm=payload.get("source_bpm"),
                        target_bpm=payload.get("target_bpm"),
                    )
                )

        try:
            logger.info("Starting isolated Intel XPU inference process: %s", command)
            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                process = subprocess.Popen(
                    command,
                    cwd=str(get_executable_dir() if is_frozen_app() else get_project_root()),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=process_env,
                    creationflags=creationflags,
                )
                with self._lock:
                    self._process = process
                    cancelled = self._cancelled
                if cancelled:
                    self._terminate_and_reap(process)
                    raise InterruptedError("Intel XPU inference job was cancelled before startup")

                while process.poll() is None:
                    forward_progress()
                    with self._lock:
                        cancelled = self._cancelled
                    if cancelled:
                        self._terminate_and_reap(process)
                        raise InterruptedError("Intel XPU inference job was cancelled")
                    time.sleep(0.1)
                process.wait(timeout=0)
                forward_progress()
                with self._lock:
                    cancelled = self._cancelled
                if cancelled:
                    raise InterruptedError("Intel XPU inference job was cancelled")

            response: dict[str, Any] | None = None
            response_read_error: str | None = None
            if response_path.is_file():
                try:
                    payload = json.loads(response_path.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise TypeError(
                            "isolated Intel XPU response manifest must contain a JSON object"
                        )
                    response = payload
                except Exception as exc:
                    response_read_error = (
                        "isolated Intel XPU response manifest is invalid: "
                        f"{type(exc).__name__}: {exc}"
                    )

            if process.returncode != 0:
                response_detail = ""
                if response is not None:
                    if bool(response.get("ok")):
                        response_detail = (
                            "isolated Intel XPU worker wrote a success response before "
                            "exiting with a nonzero code"
                        )
                    else:
                        response_detail = "\n".join(
                            part
                            for part in (
                                str(response.get("error") or "").strip(),
                                str(response.get("traceback") or "").strip(),
                            )
                            if part
                        )
                detail = "\n".join(
                    part
                    for part in (
                        f"isolated Intel XPU inference exited with code {process.returncode}",
                        response_detail,
                        response_read_error or "",
                        _tail_text(stdout_path),
                        _tail_text(stderr_path),
                    )
                    if part
                )
                raise RuntimeError(detail)
            if response_read_error:
                raise RuntimeError(response_read_error)
            if response is None:
                raise RuntimeError(
                    "isolated Intel XPU inference exited successfully without a response manifest"
                )
            if not bool(response.get("ok")):
                detail = "\n".join(
                    part
                    for part in (
                        str(response.get("error") or "isolated Intel XPU inference failed"),
                        str(response.get("traceback") or "").strip(),
                        _tail_text(stderr_path),
                    )
                    if part
                )
                raise RuntimeError(detail)

            artifacts = []
            for payload in response.get("artifacts") or []:
                artifact_path = Path(str(payload["path"])).resolve()
                if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
                    raise RuntimeError(
                        f"isolated Intel XPU worker reported a missing artifact: {artifact_path}"
                    )
                artifacts.append(
                    ArtifactSpec(
                        id=str(payload["id"]),
                        kind=str(payload["kind"]),
                        path=artifact_path,
                        track_id=payload.get("track_id"),
                    )
                )
            return ExecutionResult(
                result=dict(response["result"]),
                artifacts=tuple(artifacts),
            )
        finally:
            if process is not None and process.poll() is None:
                self._terminate_and_reap(process)
            with self._lock:
                self._process = None
            shutil.rmtree(worker_dir, ignore_errors=False)


def run_inference_worker(argv: list[str] | None = None) -> int:
    """Execute one serialized HTTP inference job inside the isolated process."""

    parser = argparse.ArgumentParser(description="Internal Web inference worker")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--response-json", required=True)
    parser.add_argument("--events-jsonl", required=True)
    args = parser.parse_args(argv)
    request_path = Path(args.request_json).resolve()
    response_path = Path(args.response_json).resolve()
    events_path = Path(args.events_jsonl).resolve()
    os.environ[_WORKER_ENV] = "1"

    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        from src.web_api.engine import InferenceEngine

        def progress_callback(progress: ProcessingProgress) -> None:
            append_jsonl_event(events_path, progress.to_dict())

        execution = InferenceEngine()._run_direct(
            kind=str(request["kind"]),
            source_path=Path(str(request["source_path"])),
            output_dir=Path(str(request["output_dir"])),
            options=dict(request["options"]),
            progress_callback=progress_callback,
            processor_callback=lambda _processor: None,
            track_id=request.get("track_id"),
            tempo_source_path=(
                Path(str(request["tempo_source_path"]))
                if request.get("tempo_source_path")
                else None
            ),
        )
        _atomic_write_json(
            response_path,
            {
                "ok": True,
                "result": execution.result,
                "artifacts": [
                    {
                        "id": artifact.id,
                        "kind": artifact.kind,
                        "path": str(artifact.path),
                        "track_id": artifact.track_id,
                    }
                    for artifact in execution.artifacts
                ],
            },
        )
        return 0
    except Exception as exc:
        try:
            _atomic_write_json(
                response_path,
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        except Exception:
            traceback.print_exc()
        logger.error("Isolated Web inference worker failed: %s", exc, exc_info=True)
        return 1


def main(argv: list[str] | None = None) -> int:
    return run_inference_worker(argv)


if __name__ == "__main__":
    raise SystemExit(main())
