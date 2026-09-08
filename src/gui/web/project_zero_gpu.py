"""Serializable ZeroGPU requests with real progress and cooperative cancellation."""

from contextvars import copy_context
from pathlib import Path
import threading

from src.core.transcription_stream import append_jsonl_event, read_new_jsonl_events
from src.models.data_models import ProcessingProgress, ProcessingStage
from src.web_api.engine import InferenceEngine


def run_gpu_project_step(request, *, engine_factory=InferenceEngine):
    """Runs inside the GPU process; no parent callbacks or locks cross IPC."""
    request = dict(request)
    event_path = Path(request.pop("events_path"))
    cancel_path = Path(request.pop("cancel_path"))
    finished = threading.Event()
    processor = [None]
    lock = threading.Lock()

    def check_cancel():
        if cancel_path.exists():
            raise InterruptedError("用户已停止项目处理")

    def bind(value):
        with lock:
            processor[0] = value
            if value is not None and cancel_path.exists():
                value.cancel()

    def watch_cancel():
        while not finished.wait(0.1):
            if cancel_path.exists():
                with lock:
                    if processor[0] is not None:
                        processor[0].cancel()
                return

    def progress(value):
        check_cancel()
        append_jsonl_event(event_path, value.to_dict())

    monitor = threading.Thread(target=watch_cancel, name="project-gpu-cancel")
    monitor.start()
    try:
        check_cancel()
        for key in ("source_path", "output_dir", "tempo_source_path"):
            if request.get(key) is not None:
                request[key] = Path(request[key])
        result = engine_factory().run(
            **request, progress_callback=progress, processor_callback=bind
        )
        check_cancel()
        return result
    finally:
        finished.set()
        monitor.join()


def make_zerogpu_engine(gpu_decorator, estimate_duration):
    def duration(request):
        options = request["options"]
        config = (
            InferenceEngine._primary_config(options)
            if request["kind"] == "primary"
            else InferenceEngine._manual_config(options)
        )
        return estimate_duration(
            request["source_path"],
            config.processing_mode,
            config.transcription_backend,
            config.yourmt3_model,
            config.muscriptor_model,
            config.muscriptor_instruments,
            config.custom_bpm,
            config.tempo_mode,
            config.muscriptor_processing_chain,
        )

    @gpu_decorator(duration=duration, size="large")
    def execute_step(request):
        return run_gpu_project_step(request)

    class SpaceProjectEngine:
        def run(self, **kwargs):
            progress_callback = kwargs.pop("progress_callback")
            processor_callback = kwargs.pop("processor_callback")
            directory = Path(kwargs["output_dir"])
            events_path = directory / ".gpu-progress.jsonl"
            cancel_path = directory / ".gpu-cancel"
            request = {k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()}
            request.update(events_path=str(events_path), cancel_path=str(cancel_path))
            completed = threading.Event()
            result, errors = [], []

            class Cancellation:
                def cancel(self):
                    cancel_path.touch(exist_ok=True)

            def execute():
                try:
                    result.append(execute_step(request))
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    completed.set()

            thread = threading.Thread(
                target=copy_context().run, args=(execute,), name="project-zerogpu"
            )
            processor_callback(Cancellation())
            thread.start()
            offset = 0
            try:
                while True:
                    done = completed.wait(0.1)
                    updates, offset = read_new_jsonl_events(events_path, offset)
                    for update in updates:
                        update["stage"] = ProcessingStage(update["stage"])
                        progress_callback(ProcessingProgress(**update))
                    if done:
                        break
                if errors:
                    raise errors[0]
                return result[0]
            finally:
                if thread.is_alive():
                    cancel_path.touch(exist_ok=True)
                thread.join()
                processor_callback(None)

    return SpaceProjectEngine
