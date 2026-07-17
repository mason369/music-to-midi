"""ByteDance pedal-aware piano transcription wrapper."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from importlib import metadata
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
    clip_midi_to_duration,
    publish_midi_output,
    remove_temporary_midi,
    unique_midi_temp_path,
)
from src.utils.runtime_paths import get_bytedance_piano_dir

logger = logging.getLogger(__name__)

BYTEDANCE_PIANO_CHECKPOINT_NAME = "note_F1=0.9677_pedal_F1=0.9186.pth"
BYTEDANCE_PIANO_CHECKPOINT_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
BYTEDANCE_PIANO_PACKAGE_NAME = "piano-transcription-inference"
BYTEDANCE_PIANO_PACKAGE_VERSION = "0.0.6"
BYTEDANCE_PIANO_CHECKPOINT_SIZE = 171_966_578
BYTEDANCE_PIANO_CHECKPOINT_SHA256 = (
    "c3fa9730725bf4a762f1c14bc80cd5986eacda01b026f5a4a2525cd607876141"
)
# Compatibility alias for callers that previously used a minimum-size threshold.
BYTEDANCE_PIANO_MIN_CHECKPOINT_BYTES = BYTEDANCE_PIANO_CHECKPOINT_SIZE


def validate_bytedance_piano_checkpoint(path: Path) -> Path:
    """Require the exact official ByteDance piano checkpoint artifact."""

    return validate_file_identity(
        path,
        expected_size=BYTEDANCE_PIANO_CHECKPOINT_SIZE,
        expected_sha256=BYTEDANCE_PIANO_CHECKPOINT_SHA256,
        label="ByteDance Piano checkpoint",
    )


class ByteDancePianoTranscriber:
    """Run ByteDance's piano transcription model and keep pedal CC events."""

    def __init__(
        self,
        config: Optional[Config] = None,
        checkpoint_path: Optional[Path] = None,
    ):
        self.config = config or Config()
        if checkpoint_path is None:
            checkpoint_path = self.default_checkpoint_path()
        self.checkpoint_path = Path(checkpoint_path)
        self._cancelled = False
        self._translator = Translator(getattr(self.config, "language", Translator.DEFAULT_LANGUAGE))

    def _pt(self, key: str, **kwargs) -> str:
        return self._translator.t(key, **kwargs)

    @staticmethod
    def default_checkpoint_path() -> Path:
        return get_bytedance_piano_dir() / BYTEDANCE_PIANO_CHECKPOINT_NAME

    @staticmethod
    def is_available() -> bool:
        return ByteDancePianoTranscriber.get_unavailable_reason() == ""

    @staticmethod
    def get_unavailable_reason() -> str:
        try:
            if importlib.util.find_spec("piano_transcription_inference") is None:
                return (
                    "ByteDance Piano 未安装。请执行: "
                    "python -m pip install piano-transcription-inference==0.0.6 "
                    "torchlibrosa matplotlib"
                )

            try:
                installed_version = metadata.version(BYTEDANCE_PIANO_PACKAGE_NAME)
            except metadata.PackageNotFoundError:
                return "ByteDance Piano 缺少 distribution metadata，无法验证包版本"
            if installed_version != BYTEDANCE_PIANO_PACKAGE_VERSION:
                return (
                    "ByteDance Piano 包版本不匹配: "
                    f"expected {BYTEDANCE_PIANO_PACKAGE_VERSION}, got {installed_version}。"
                    "请执行: python -m pip install --force-reinstall "
                    "piano-transcription-inference==0.0.6"
                )

            # The upstream package imports matplotlib.pyplot at module import time.
            # Use a non-interactive backend so PyInstaller builds do not need a GUI backend.
            os.environ.setdefault("MPLBACKEND", "Agg")
            importlib.import_module("piano_transcription_inference")
            return ""
        except (ImportError, ModuleNotFoundError) as exc:
            return (
                f"ByteDance Piano 运行依赖缺失: {exc}。请执行: "
                "python -m pip install piano-transcription-inference==0.0.6 "
                "torchlibrosa matplotlib"
            )
        except Exception as exc:
            return f"ByteDance Piano 后端导入失败: {exc}"

    def is_model_available(self) -> bool:
        try:
            validate_bytedance_piano_checkpoint(self.checkpoint_path)
            return True
        except (OSError, RuntimeError):
            return False

    def cancel(self) -> None:
        self._cancelled = True

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise InterruptedError("ByteDance Piano 转写处理已取消")

    def _resolve_runtime_device(self) -> str:
        preferred = get_device(self.config.use_gpu, self.config.gpu_device)
        if preferred.startswith("cuda"):
            ensure_cuda_runtime_compatibility(preferred)
            return "cuda"
        if preferred == "cpu":
            return "cpu"
        raise RuntimeError(
            "ByteDance Piano 当前仅对 CPU/CUDA 路径做了集成验证，"
            f"检测到设备 {preferred}。请切换到 CPU 或 CUDA 后重试。"
        )

    @staticmethod
    def _format_missing_output_error(input_path: Path, out_path: Path, device: str) -> str:
        lines = [
            "ByteDance Piano 未生成 MIDI 输出",
            f"输入音频: {input_path.resolve()}",
            f"期望输出: {out_path.resolve()}",
            f"输出目录: {out_path.parent.resolve()}",
            f"运行设备: {device}",
        ]
        if out_path.parent.exists():
            midi_candidates = sorted(
                candidate.resolve()
                for candidate in out_path.parent.glob("*.mid")
                if candidate.is_file()
            )
            if midi_candidates:
                lines.append("输出目录中的 MIDI 文件:")
                lines.extend(f"  {candidate}" for candidate in midi_candidates[:12])
                if len(midi_candidates) > 12:
                    lines.append(f"  ... 另外 {len(midi_candidates) - 12} 个")
            else:
                lines.append("输出目录中未发现 .mid 文件")
        else:
            lines.append("输出目录不存在")
        return "\n".join(lines)

    def _format_missing_checkpoint_error(self) -> str:
        return "\n".join(
            [
                "ByteDance Piano checkpoint 缺失或不完整。",
                f"期望文件名: {BYTEDANCE_PIANO_CHECKPOINT_NAME}",
                f"期望大小: {BYTEDANCE_PIANO_CHECKPOINT_SIZE} bytes",
                f"期望 SHA-256: {BYTEDANCE_PIANO_CHECKPOINT_SHA256}",
                f"当前检查路径: {self.checkpoint_path.resolve()}",
                "如果曾将 checkpoint 改名为 matplotlib.pth，请改回上面的原始文件名。",
                "matplotlib 是 Python 依赖，需要通过 pip 安装，不能通过重命名模型文件提供。",
                "请执行: python download_bytedance_piano_model.py",
            ]
        )

    def transcribe(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        input_path = Path(audio_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.is_available():
            raise RuntimeError(self.get_unavailable_reason())
        if not self.is_model_available():
            raise RuntimeError(self._format_missing_checkpoint_error())

        self._check_cancelled()
        device = self._resolve_runtime_device()
        logger.info("Running ByteDance Piano transcription on %s: %s", device, input_path)

        if progress_callback:
            progress_callback(0.05, self._pt("progress.loading_bytedance_piano", device=device))

        temp_output_path = unique_midi_temp_path(out_path, "bytedance-piano")
        try:
            os.environ.setdefault("MPLBACKEND", "Agg")
            module = importlib.import_module("piano_transcription_inference")
            import librosa

            audio, _sample_rate = librosa.load(
                str(input_path),
                sr=module.sample_rate,
                mono=True,
            )
            audio_duration_seconds = len(audio) / float(module.sample_rate)
            self._check_cancelled()

            if progress_callback:
                progress_callback(0.20, self._pt("progress.running_bytedance_piano"))

            transcriptor = module.PianoTranscription(
                device=device,
                checkpoint_path=str(self.checkpoint_path),
            )
            self._check_cancelled()
            transcriptor.transcribe(audio, str(temp_output_path))
            self._check_cancelled()
            if not temp_output_path.is_file() or temp_output_path.stat().st_size == 0:
                raise RuntimeError(self._format_missing_output_error(input_path, out_path, device))
            clip_midi_to_duration(
                temp_output_path,
                audio_duration_seconds,
                "ByteDance Piano",
            )
            publish_midi_output(temp_output_path, out_path, "ByteDance Piano")
        except InterruptedError:
            raise
        except Exception as exc:
            friendly_message = rewrite_cuda_runtime_error(
                exc,
                "cuda:0" if device == "cuda" else device,
            )
            raise RuntimeError(f"ByteDance Piano 转写失败: {friendly_message}") from exc
        finally:
            try:
                remove_temporary_midi(temp_output_path)
            finally:
                clear_gpu_memory()

        if progress_callback:
            progress_callback(1.0, self._pt("progress.bytedance_piano_complete"))

        logger.info("ByteDance Piano output: %s", out_path)
        return str(out_path)
