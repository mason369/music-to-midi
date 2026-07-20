"""TransKun piano transcription wrapper."""

from __future__ import annotations

import importlib.util
import logging
import math
import multiprocessing
import queue
from collections import defaultdict
from contextlib import ExitStack
from importlib import metadata, resources
from pathlib import Path
from typing import Callable, Optional

from src.i18n.translator import Translator
from src.models.data_models import Config
from src.utils.artifact_identity import validate_file_identity
from src.utils.gpu_utils import (
    clear_gpu_memory,
    ensure_cuda_runtime_compatibility,
    get_device,
    rewrite_cuda_runtime_error,
)
from src.utils.midi_output import (
    publish_midi_output,
    remove_temporary_midi,
    unique_midi_temp_path,
)
from src.core.transcription_stream import piano_notes_payload, snapshot_event

logger = logging.getLogger(__name__)

TRANSKUN_PACKAGE_NAME = "transkun"
TRANSKUN_PACKAGE_VERSION = "2.0.1"
TRANSKUN_WEIGHT_NAME = "2.0.pt"
TRANSKUN_WEIGHT_SIZE = 56_408_978
TRANSKUN_WEIGHT_SHA256 = "50a80010effc2a59ffcd068a95cd2b29bd7f23a27a3515bc3ccd209c89a3d44c"
TRANSKUN_CONF_NAME = "2.0.conf"
TRANSKUN_CONF_SIZE = 782
TRANSKUN_CONF_SHA256 = "d3d989214eb148230ee5df476d994dcde6af595904d3f968f1221d2e3bea5ac6"


def _transkun_worker(
    audio_path: str,
    output_path: str,
    weight_path: str,
    conf_path: str,
    device: str,
    result_queue,
) -> None:
    try:
        import moduleconf
        import soxr
        import torch
        import torch.nn.functional as F
        from transkun.Data import resolveOverlapping, writeMidi
        from transkun.Util import makeFrame
        from transkun.transcribe import readAudio

        conf_manager = moduleconf.parseFromFile(str(conf_path))
        transkun_cls = conf_manager["Model"].module.TransKun
        conf = conf_manager["Model"].config

        checkpoint = torch.load(str(weight_path), map_location=device)
        model = transkun_cls(conf=conf).to(device)

        state_dict = checkpoint.get("best_state_dict") or checkpoint.get("state_dict")
        if state_dict is None:
            raise RuntimeError("TransKun checkpoint 缺少 state_dict")
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        fs, audio = readAudio(str(audio_path))
        if fs != model.fs:
            audio = soxr.resample(audio, fs, model.fs)

        source_duration = len(audio) / float(model.fs)
        x = torch.from_numpy(audio).to(device).transpose(-1, -2)
        step_seconds = model.segmentHopSizeInSecond
        segment_seconds = model.segmentSizeInSecond
        pad_time_begin = segment_seconds - step_seconds
        x = F.pad(
            x,
            (
                math.ceil(pad_time_begin * model.fs),
                math.ceil(model.fs * pad_time_begin),
            ),
        )
        n_sample = x.shape[-1]
        events_by_type = defaultdict(list)
        start_frame_index = math.floor(pad_time_begin * model.fs / model.hopSize)
        start_positions = [start_frame_index] * len(model.targetMIDIPitch)
        step_size = math.ceil(step_seconds * model.fs / model.hopSize) * model.hopSize
        segment_size = math.ceil(segment_seconds * model.fs)
        total_segments = len(range(0, n_sample, step_size))

        with torch.no_grad():
            for segment_index, i in enumerate(range(0, n_sample, step_size), start=1):
                j = min(i + segment_size, n_sample)
                begin_time = i / model.fs - pad_time_begin
                current_slice = x[:, i:j]
                if current_slice.shape[-1] < segment_size:
                    current_slice = F.pad(
                        current_slice,
                        (0, segment_size - current_slice.shape[-1]),
                    )
                current_frames = makeFrame(
                    current_slice,
                    model.hopSize,
                    model.windowSize,
                )
                last_frame_index = round(segment_size / model.hopSize)
                current_events, last_positions = model.transcribeFrames(
                    current_frames.unsqueeze(0),
                    forcedStartPos=start_positions,
                    velocityCriteron="hamming",
                    onsetBound=None,
                    lastFrameIdx=last_frame_index,
                )
                current_events = current_events[0]
                start_positions = [
                    max(position - int(step_size / model.hopSize), 0) for position in last_positions
                ]
                for event in current_events:
                    event.start = max(event.start + begin_time, 0)
                    event.end = max(event.end + begin_time, event.start)
                    if (
                        events_by_type[event.pitch]
                        and event.start < events_by_type[event.pitch][-1].end
                    ):
                        previous = events_by_type[event.pitch][-1]
                        if event.hasOnset:
                            events_by_type[event.pitch][-1] = event
                        else:
                            previous.hasOffset = event.hasOffset
                            previous.end = max(event.end, previous.end)
                        continue
                    if event.hasOnset:
                        events_by_type[event.pitch].append(event)

                is_final = segment_index == total_segments
                if is_final:
                    for pitch_events in events_by_type.values():
                        if pitch_events:
                            pitch_events[-1].hasOffset = True
                    frontier = source_duration
                else:
                    frontier = min(source_duration, max(0.0, i / model.fs))
                stable_events = [
                    event
                    for pitch_events in events_by_type.values()
                    for event in pitch_events
                    if event.hasOffset and event.end <= frontier + 1e-6
                ]
                if not is_final:
                    result_queue.put(
                        {
                            "event": snapshot_event(
                                backend="TransKun",
                                completed=segment_index,
                                total=total_segments,
                                frontier_seconds=frontier,
                                duration_seconds=source_duration,
                                notes=piano_notes_payload(
                                    stable_events,
                                    frontier_seconds=frontier,
                                ),
                            )
                        }
                    )

        notes_est = [event for events in events_by_type.values() for event in events]
        notes_est = [event for event in notes_est if event.hasOffset]
        notes_est = resolveOverlapping(notes_est)
        result_queue.put(
            {
                "event": snapshot_event(
                    backend="TransKun",
                    completed=total_segments,
                    total=total_segments,
                    frontier_seconds=source_duration,
                    duration_seconds=source_duration,
                    notes=piano_notes_payload(
                        notes_est,
                        frontier_seconds=source_duration,
                    ),
                )
            }
        )

        output_midi = writeMidi(notes_est)
        output_midi.write(str(output_path))
        result_queue.put({"ok": str(output_path)})
    except Exception as exc:  # pragma: no cover - exercised through parent wrapper
        result_queue.put({"error": str(exc)})


class TranskunTranscriber:
    """Run TransKun's packaged piano model and export a single-track MIDI."""

    WEIGHT_NAME = TRANSKUN_WEIGHT_NAME
    CONF_NAME = TRANSKUN_CONF_NAME

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._cancelled = False
        self._cancel_check: Optional[Callable[[], bool]] = None
        self._process = None
        self._translator = Translator(getattr(self.config, "language", Translator.DEFAULT_LANGUAGE))

    def _pt(self, key: str, **kwargs) -> str:
        return self._translator.t(key, **kwargs)

    @staticmethod
    def is_available() -> bool:
        return TranskunTranscriber.get_unavailable_reason() == ""

    @staticmethod
    def get_unavailable_reason() -> str:
        try:
            if importlib.util.find_spec("transkun.transcribe") is None:
                return (
                    "TransKun 未安装。请执行: "
                    f"python -m pip install transkun=={TRANSKUN_PACKAGE_VERSION}"
                )
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            return f"TransKun 模块不可用: {exc}"

        try:
            installed_version = metadata.version(TRANSKUN_PACKAGE_NAME)
        except metadata.PackageNotFoundError:
            return "TransKun 缺少 distribution metadata，无法验证包版本"
        if installed_version != TRANSKUN_PACKAGE_VERSION:
            return (
                "TransKun 包版本不匹配: "
                f"expected {TRANSKUN_PACKAGE_VERSION}, got {installed_version}。"
                "请执行: python -m pip install --force-reinstall "
                f"transkun=={TRANSKUN_PACKAGE_VERSION}"
            )
        return ""

    @classmethod
    def _get_packaged_resource(cls, name: str):
        if not cls.is_available():
            return None
        try:
            return resources.files("transkun").joinpath("pretrained", name)
        except (ModuleNotFoundError, FileNotFoundError):
            return None

    def is_model_available(self) -> bool:
        weight = self._get_packaged_resource(self.WEIGHT_NAME)
        conf = self._get_packaged_resource(self.CONF_NAME)
        if not weight or not conf or not weight.is_file() or not conf.is_file():
            return False
        try:
            with ExitStack() as stack:
                weight_path = stack.enter_context(resources.as_file(weight))
                conf_path = stack.enter_context(resources.as_file(conf))
                validate_file_identity(
                    weight_path,
                    expected_size=TRANSKUN_WEIGHT_SIZE,
                    expected_sha256=TRANSKUN_WEIGHT_SHA256,
                    label="TransKun 2.0.pt",
                )
                validate_file_identity(
                    conf_path,
                    expected_size=TRANSKUN_CONF_SIZE,
                    expected_sha256=TRANSKUN_CONF_SHA256,
                    label="TransKun 2.0.conf",
                )
            return True
        except (OSError, RuntimeError):
            return False

    def set_cancel_check(self, callback: Optional[Callable[[], bool]]) -> None:
        self._cancel_check = callback

    def cancel(self) -> None:
        self._cancelled = True
        process = self._process
        if process is not None and self._process_is_alive(process):
            logger.info("正在终止 TransKun 子进程...")
            process.terminate()

    @staticmethod
    def _process_is_alive(process) -> bool:
        is_alive = getattr(process, "is_alive", None)
        if callable(is_alive):
            return bool(is_alive())
        poll = getattr(process, "poll", None)
        if callable(poll):
            return poll() is None
        return False

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise InterruptedError("TransKun 转写处理已取消")
        if self._cancel_check and self._cancel_check():
            raise InterruptedError("TransKun 转写处理已取消")

    def _resolve_runtime_device(self) -> str:
        preferred = get_device(self.config.use_gpu, self.config.gpu_device)
        if preferred.startswith("cuda"):
            ensure_cuda_runtime_compatibility(preferred)
            return preferred
        if preferred != "cpu":
            raise RuntimeError(
                "TransKun 当前仅对 CPU/CUDA 路径做了集成验证，"
                f"检测到设备 {preferred}。请切换到 CPU 或 CUDA 后重试。"
            )
        return "cpu"

    def transcribe(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        input_path = Path(audio_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        unavailable_reason = self.get_unavailable_reason()
        if unavailable_reason:
            raise RuntimeError(unavailable_reason)
        if not self.is_model_available():
            raise RuntimeError(
                "TransKun 预训练资源缺失或身份校验失败。"
                f"2.0.pt 期望 {TRANSKUN_WEIGHT_SIZE} bytes / {TRANSKUN_WEIGHT_SHA256}；"
                f"2.0.conf 期望 {TRANSKUN_CONF_SIZE} bytes / {TRANSKUN_CONF_SHA256}。"
                "请执行: python -m pip install --force-reinstall "
                f"transkun=={TRANSKUN_PACKAGE_VERSION}"
            )

        self._check_cancelled()
        device = self._resolve_runtime_device()
        logger.info("Running TransKun transcription on %s: %s", device, input_path)

        temp_output_path = unique_midi_temp_path(out_path, "transkun")
        process = None
        result_queue = None
        try:
            with ExitStack() as stack:
                weight_path = stack.enter_context(
                    resources.as_file(self._get_packaged_resource(self.WEIGHT_NAME))
                )
                conf_path = stack.enter_context(
                    resources.as_file(self._get_packaged_resource(self.CONF_NAME))
                )

                if progress_callback:
                    progress_callback(0.05, self._pt("progress.preparing_transkun", device=device))

                result_queue = multiprocessing.Queue()
                process = multiprocessing.Process(
                    target=_transkun_worker,
                    args=(
                        str(input_path),
                        str(temp_output_path),
                        str(weight_path),
                        str(conf_path),
                        device,
                        result_queue,
                    ),
                )
                self._process = process
                process.start()

                if progress_callback:
                    progress_callback(0.30, self._pt("progress.running_transkun"))

                result = None
                event_callback = getattr(self, "_event_callback", None)

                def _consume(message) -> None:
                    nonlocal result
                    if "event" in message:
                        if event_callback is not None:
                            event_callback(message["event"])
                        return
                    result = message

                while process.is_alive():
                    self._check_cancelled()
                    while True:
                        try:
                            _consume(result_queue.get_nowait())
                        except queue.Empty:
                            break
                    process.join(timeout=0.2)

                self._process = None
                self._check_cancelled()

                while result is None:
                    try:
                        _consume(result_queue.get(timeout=2.0))
                    except queue.Empty as exc:
                        exit_code = getattr(process, "exitcode", None)
                        raise RuntimeError(
                            "TransKun 子进程未返回结果 " f"(exit code: {exit_code})"
                        ) from exc
                if "error" in result:
                    raise RuntimeError(result["error"])
                if "ok" not in result:
                    raise RuntimeError(f"TransKun 子进程返回未知结果: {result!r}")
                publish_midi_output(temp_output_path, out_path, "TransKun")

        except InterruptedError:
            raise
        except Exception as exc:
            friendly_message = rewrite_cuda_runtime_error(exc, device)
            raise RuntimeError(f"TransKun 转写失败: {friendly_message}") from exc
        finally:
            cleanup_process = self._process or process
            if cleanup_process is not None and self._process_is_alive(cleanup_process):
                cleanup_process.terminate()
                join = getattr(cleanup_process, "join", None)
                if callable(join):
                    join(timeout=5.0)
                if self._process_is_alive(cleanup_process):
                    kill = getattr(cleanup_process, "kill", None)
                    if callable(kill):
                        kill()
                        if callable(join):
                            join(timeout=5.0)
            self._process = None
            if result_queue is not None:
                close = getattr(result_queue, "close", None)
                if callable(close):
                    close()
                join_thread = getattr(result_queue, "join_thread", None)
                if callable(join_thread):
                    join_thread()
            try:
                remove_temporary_midi(temp_output_path)
            finally:
                try:
                    clear_gpu_memory()
                except Exception:
                    pass

        if progress_callback:
            progress_callback(1.0, self._pt("progress.transkun_complete"))

        logger.info("TransKun output: %s", out_path)
        return str(out_path)

    def set_event_callback(self, callback: Optional[Callable[[dict], None]]) -> None:
        self._event_callback = callback
