"""TransKun V2 Aug piano transcription wrapper.

This route uses the official ``checkpointTransformerAug.zip`` model-card
artifact instead of the smaller checkpoint bundled with the PyPI package.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing
import queue
from pathlib import Path
from typing import Any, Callable, Optional

from src.core.transkun_transcriber import TranskunTranscriber, _transkun_worker
from src.models.data_models import Config
from src.utils.gpu_utils import clear_gpu_memory, rewrite_cuda_runtime_error
from src.utils.midi_output import (
    publish_midi_output,
    remove_temporary_midi,
    unique_midi_temp_path,
)
from src.utils.runtime_paths import get_bundle_roots

logger = logging.getLogger(__name__)

TRANSKUN_V2_AUG_GOOGLE_DRIVE_FILE_ID = "1Hg5ua8vYdtg1Y-MnXD0mLyhRK9Srd7hm"
TRANSKUN_V2_AUG_MODEL_CARD_URL = (
    f"https://drive.google.com/file/d/{TRANSKUN_V2_AUG_GOOGLE_DRIVE_FILE_ID}/view?usp=drive_link"
)
TRANSKUN_V2_AUG_DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download"
    f"?id={TRANSKUN_V2_AUG_GOOGLE_DRIVE_FILE_ID}&export=download&confirm=t"
)
TRANSKUN_V2_AUG_ARCHIVE_NAME = "checkpointTransformerAug.zip"
TRANSKUN_V2_AUG_ARCHIVE_SIZE = 50_694_377
TRANSKUN_V2_AUG_ARCHIVE_SHA256 = "f61ebf6467d89081fde9728b659895a3e3d65b4c89516964178967167fae6590"
TRANSKUN_V2_AUG_MODEL_DIR_NAME = "checkpointMSimplerAug"
TRANSKUN_V2_AUG_CHECKPOINT_NAME = "checkpoint.pt"
TRANSKUN_V2_AUG_CHECKPOINT_SIZE = 56_423_254
TRANSKUN_V2_AUG_CHECKPOINT_SHA256 = (
    "8bd6b4b5ddf9ce8c5f296a57859eec9f166cd337c35245ec2a2576d90be68c4c"
)
TRANSKUN_V2_AUG_CONFIG_NAME = "model.conf"
TRANSKUN_V2_AUG_CONFIG_SIZE = 782
TRANSKUN_V2_AUG_CONFIG_SHA256 = "d3d989214eb148230ee5df476d994dcde6af595904d3f968f1221d2e3bea5ac6"
TRANSKUN_V2_AUG_CACHE_ROOT = Path.home() / ".cache" / "music_ai_models" / "transkun_v2_aug"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_transkun_v2_aug_model_files(model_dir: Path) -> str:
    """Return an explicit validation error, or an empty string when valid."""

    model_dir = Path(model_dir)
    expected = (
        (
            model_dir / TRANSKUN_V2_AUG_CHECKPOINT_NAME,
            TRANSKUN_V2_AUG_CHECKPOINT_SIZE,
            TRANSKUN_V2_AUG_CHECKPOINT_SHA256,
        ),
        (
            model_dir / TRANSKUN_V2_AUG_CONFIG_NAME,
            TRANSKUN_V2_AUG_CONFIG_SIZE,
            TRANSKUN_V2_AUG_CONFIG_SHA256,
        ),
    )

    for path, expected_size, expected_sha256 in expected:
        if not path.is_file():
            return f"缺少 TransKun V2 Aug 文件: {path.resolve()}"
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            return (
                f"TransKun V2 Aug 文件大小不匹配: {path.resolve()} "
                f"({actual_size} != {expected_size})"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            return (
                f"TransKun V2 Aug 文件 SHA256 不匹配: {path.resolve()} "
                f"({actual_sha256} != {expected_sha256})"
            )
    return ""


class TranskunV2AugTranscriber(TranskunTranscriber):
    """Run the official TransKun V2 data-augmentation checkpoint."""

    def __init__(
        self,
        config: Optional[Config] = None,
        model_dir: Optional[Path] = None,
    ):
        super().__init__(config)
        self.model_dir = Path(model_dir) if model_dir is not None else self.default_model_dir()
        self.checkpoint_path = self.model_dir / TRANSKUN_V2_AUG_CHECKPOINT_NAME
        self.conf_path = self.model_dir / TRANSKUN_V2_AUG_CONFIG_NAME

    @staticmethod
    def default_model_dir() -> Path:
        for root in get_bundle_roots():
            for relative_dir in (
                Path("models") / "transkun_v2_aug" / TRANSKUN_V2_AUG_MODEL_DIR_NAME,
                Path("assets") / "models" / "transkun_v2_aug" / TRANSKUN_V2_AUG_MODEL_DIR_NAME,
            ):
                candidate = root / relative_dir
                if candidate.is_dir():
                    return candidate
        return TRANSKUN_V2_AUG_CACHE_ROOT / TRANSKUN_V2_AUG_MODEL_DIR_NAME

    def get_model_validation_error(self) -> str:
        return validate_transkun_v2_aug_model_files(self.model_dir)

    def is_model_available(self) -> bool:
        return self.get_model_validation_error() == ""

    @staticmethod
    def _format_missing_output_error(
        input_path: Path,
        output_path: Path,
        device: str,
    ) -> str:
        return "\n".join(
            (
                "TransKun V2 Aug 未生成 MIDI 输出。",
                f"输入音频: {input_path.resolve()}",
                f"期望输出: {output_path.resolve()}",
                f"运行设备: {device}",
            )
        )

    def transcribe(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        input_path = Path(audio_path)
        out_path = Path(output_path)

        if not input_path.is_file():
            raise FileNotFoundError(f"TransKun V2 Aug 输入音频不存在: {input_path.resolve()}")
        unavailable_reason = self.get_unavailable_reason()
        if unavailable_reason:
            raise RuntimeError(unavailable_reason)

        validation_error = self.get_model_validation_error()
        if validation_error:
            raise RuntimeError(
                f"{validation_error}\n请执行: python download_transkun_v2_aug_model.py"
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._check_cancelled()
        device = self._resolve_runtime_device()
        logger.info(
            "Running TransKun V2 Aug transcription on %s: %s",
            device,
            input_path,
        )

        temp_output_path = unique_midi_temp_path(out_path, "transkun-v2-aug")
        result_queue: Any = None
        try:
            if progress_callback:
                progress_callback(
                    0.05,
                    self._pt("progress.preparing_transkun", device=device),
                )

            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=_transkun_worker,
                args=(
                    str(input_path),
                    str(temp_output_path),
                    str(self.checkpoint_path),
                    str(self.conf_path),
                    device,
                    result_queue,
                ),
            )
            setattr(self, "_process", process)
            process.start()

            if progress_callback:
                progress_callback(0.30, self._pt("progress.running_transkun"))

            while process.is_alive():
                self._check_cancelled()
                process.join(timeout=0.2)

            setattr(self, "_process", None)
            self._check_cancelled()

            try:
                result = result_queue.get(timeout=2.0)
            except queue.Empty as exc:
                exit_code = getattr(process, "exitcode", None)
                raise RuntimeError(
                    f"TransKun V2 Aug 子进程未返回结果（exit code: {exit_code}）"
                ) from exc

            if "error" in result:
                raise RuntimeError(str(result["error"]))
            if "ok" not in result:
                raise RuntimeError(f"TransKun V2 Aug 子进程返回未知结果: {result!r}")
            if not temp_output_path.is_file() or temp_output_path.stat().st_size == 0:
                raise RuntimeError(self._format_missing_output_error(input_path, out_path, device))
            publish_midi_output(temp_output_path, out_path, "TransKun V2 Aug")

        except InterruptedError:
            raise
        except Exception as exc:
            friendly_message = rewrite_cuda_runtime_error(exc, device)
            raise RuntimeError(f"TransKun V2 Aug 转写失败: {friendly_message}") from exc
        finally:
            cleanup_process = getattr(self, "_process", None)
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
            setattr(self, "_process", None)
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
                clear_gpu_memory()

        if progress_callback:
            progress_callback(1.0, self._pt("progress.transkun_complete"))

        logger.info("TransKun V2 Aug output: %s", out_path)
        return str(out_path)
