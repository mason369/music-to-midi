"""ByteDance pedal-aware piano transcription wrapper."""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Callable, Optional

from src.i18n.translator import Translator
from src.models.data_models import Config
from src.utils.gpu_utils import clear_gpu_memory, get_device
from src.utils.runtime_paths import get_bytedance_piano_dir

logger = logging.getLogger(__name__)

BYTEDANCE_PIANO_CHECKPOINT_NAME = "note_F1=0.9677_pedal_F1=0.9186.pth"
BYTEDANCE_PIANO_CHECKPOINT_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
BYTEDANCE_PIANO_MIN_CHECKPOINT_BYTES = 160_000_000


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
        try:
            return importlib.util.find_spec("piano_transcription_inference") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def is_model_available(self) -> bool:
        return (
            self.checkpoint_path.exists()
            and self.checkpoint_path.stat().st_size >= BYTEDANCE_PIANO_MIN_CHECKPOINT_BYTES
        )

    def cancel(self) -> None:
        self._cancelled = True

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise InterruptedError("ByteDance Piano 转写处理已取消")

    def _resolve_runtime_device(self) -> str:
        preferred = get_device(self.config.use_gpu, self.config.gpu_device)
        if preferred.startswith("cuda"):
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
            raise RuntimeError(
                "ByteDance Piano 未安装。请执行: "
                "python -m pip install piano-transcription-inference torchlibrosa"
            )
        if not self.is_model_available():
            raise RuntimeError(
                "ByteDance Piano checkpoint 缺失或不完整。请执行: "
                "python download_bytedance_piano_model.py"
            )

        self._cancelled = False
        self._check_cancelled()
        device = self._resolve_runtime_device()
        logger.info("Running ByteDance Piano transcription on %s: %s", device, input_path)

        if progress_callback:
            progress_callback(0.05, self._pt("progress.loading_bytedance_piano", device=device))

        try:
            module = importlib.import_module("piano_transcription_inference")
            import librosa

            audio, _sample_rate = librosa.load(
                str(input_path),
                sr=module.sample_rate,
                mono=True,
            )
            self._check_cancelled()

            if progress_callback:
                progress_callback(0.20, self._pt("progress.running_bytedance_piano"))

            transcriptor = module.PianoTranscription(
                device=device,
                checkpoint_path=str(self.checkpoint_path),
            )
            self._check_cancelled()
            transcriptor.transcribe(audio, str(out_path))
            self._check_cancelled()
        except InterruptedError:
            raise
        except Exception as exc:
            raise RuntimeError(f"ByteDance Piano 转写失败: {exc}") from exc
        finally:
            clear_gpu_memory()

        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(self._format_missing_output_error(input_path, out_path, device))

        if progress_callback:
            progress_callback(1.0, self._pt("progress.bytedance_piano_complete"))

        logger.info("ByteDance Piano output: %s", out_path)
        return str(out_path)
