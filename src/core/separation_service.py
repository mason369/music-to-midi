"""GUI-independent WAV-only separation service used by every interface."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.core.pipeline import MusicToMidiPipeline
from src.i18n.translator import Translator
from src.models.data_models import (
    Config,
    ProcessingMode,
    ProcessingProgress,
    ProcessingStage,
)

ProgressCallback = Callable[[ProcessingProgress], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class SeparationResult:
    """Verified separated WAV outputs from one separation-only job."""

    mode: str
    source_path: str
    output_dir: str
    separated_audio: dict[str, str]
    processing_time: float


class AudioSeparationService:
    """Run a supported split mode and stop before any MIDI transcription."""

    def __init__(
        self,
        config: Config,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> None:
        self.config = config
        self._translator = Translator(config.language)
        self._progress_callback = progress_callback
        self._external_cancel_check = cancel_check
        self._cancelled = False
        self._separator = None

    def _pt(self, key: str, **kwargs) -> str:
        return self._translator.t(key, **kwargs)

    def _is_cancelled(self) -> bool:
        external_cancelled = (
            bool(self._external_cancel_check())
            if self._external_cancel_check is not None
            else False
        )
        return self._cancelled or external_cancelled

    def _check_cancelled(self) -> None:
        if self._is_cancelled():
            raise InterruptedError("用户取消了处理")

    def _emit_progress(
        self,
        stage: ProcessingStage,
        stage_progress: float,
        overall_progress: float,
        message: str,
    ) -> None:
        if self._progress_callback is None:
            return
        self._progress_callback(
            ProcessingProgress(
                stage=stage,
                stage_progress=max(0.0, min(1.0, float(stage_progress))),
                overall_progress=max(0.0, min(1.0, float(overall_progress))),
                message=str(message),
            )
        )

    def _on_separator_progress(self, progress: float, message: str) -> None:
        normalized = max(0.0, min(1.0, float(progress)))
        self._emit_progress(
            ProcessingStage.SEPARATION,
            normalized,
            0.05 + normalized * 0.90,
            message,
        )

    @staticmethod
    def _validate_outputs(
        separated_audio: dict[str, str],
        expected_names: tuple[str, ...],
    ) -> dict[str, str]:
        actual_names = set(separated_audio)
        expected = set(expected_names)
        if actual_names != expected:
            missing = sorted(expected - actual_names)
            unexpected = sorted(actual_names - expected)
            raise RuntimeError(
                "分离输出不符合契约："
                f"缺少={missing or '无'}，意外输出={unexpected or '无'}"
            )

        validated: dict[str, str] = {}
        for name in expected_names:
            path = Path(separated_audio[name]).resolve()
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(
                    f"分离音轨不存在或为空：track={name}, path={path}"
                )
            validated[name] = str(path)
        return validated

    def _separate_vocals(self, wav_path: str, output_dir: str) -> dict[str, str]:
        from src.core.vocal_separator import VocalSeparator

        if not VocalSeparator.is_available():
            raise RuntimeError(
                "人声分离不可用。请安装: "
                "pip install audio-separator==0.44.1 --no-deps"
            )
        if not VocalSeparator.is_model_available():
            raise RuntimeError(
                "人声分离模型未下载。请运行 python download_sota_models.py 后重试。"
            )

        device = "cuda:0" if self.config.use_gpu else "cpu"
        separator = VocalSeparator(
            language=self.config.language,
            primary_device=device,
            accompaniment_device=device,
        )
        separator.set_cancel_check(self._is_cancelled)
        self._separator = separator
        separated = separator.separate(
            audio_path=wav_path,
            output_dir=output_dir,
            progress_callback=self._on_separator_progress,
        )
        return self._validate_outputs(separated, ("vocals", "accompaniment"))

    def _separate_six_stems(self, wav_path: str, output_dir: str) -> dict[str, str]:
        from src.core.multi_stem_separator import STEM_KEYS, SixStemSeparator

        if not SixStemSeparator.is_available():
            raise RuntimeError(
                "六声部分离不可用，请安装: "
                "pip install audio-separator==0.44.1 --no-deps"
            )

        device = "cuda:0" if self.config.use_gpu else "cpu"
        separator = SixStemSeparator(
            language=self.config.language,
            target_device=device,
        )
        if not separator.is_model_available():
            raise RuntimeError(
                "六声部分离模型未下载。请运行 python download_sota_models.py 后重试。"
            )
        set_cancel_check = getattr(separator, "set_cancel_check", None)
        if callable(set_cancel_check):
            set_cancel_check(self._is_cancelled)
        self._separator = separator
        separated = separator.separate(
            audio_path=wav_path,
            output_dir=output_dir,
            progress_callback=self._on_separator_progress,
        )
        return self._validate_outputs(separated, tuple(STEM_KEYS))

    def process(self, audio_path: str | Path, output_dir: str | Path) -> SeparationResult:
        start_time = time.time()
        self.config.validate()
        mode = self.config.processing_mode
        if mode not in {
            ProcessingMode.VOCAL_SPLIT.value,
            ProcessingMode.SIX_STEM_SPLIT.value,
        }:
            raise ValueError(
                f"AudioSeparationService does not support processing mode {mode!r}"
            )

        source_path = Path(audio_path).resolve()
        if not source_path.is_file() or source_path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Input audio does not exist or is empty: {source_path}"
            )
        resolved_output_dir = Path(output_dir).resolve()
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        self._emit_progress(
            ProcessingStage.PREPROCESSING,
            0.0,
            0.0,
            self._pt("progress.analyzing_audio"),
        )
        self._check_cancelled()
        wav_path = MusicToMidiPipeline._ensure_wav(
            str(source_path),
            str(resolved_output_dir),
        )
        self._check_cancelled()
        self._emit_progress(
            ProcessingStage.PREPROCESSING,
            1.0,
            0.05,
            self._pt("progress.separation_only_ready"),
        )

        if mode == ProcessingMode.VOCAL_SPLIT.value:
            separated_audio = self._separate_vocals(
                wav_path, str(resolved_output_dir)
            )
        else:
            separated_audio = self._separate_six_stems(
                wav_path, str(resolved_output_dir)
            )
        self._check_cancelled()

        processing_time = time.time() - start_time
        self._emit_progress(
            ProcessingStage.COMPLETE,
            1.0,
            1.0,
            self._pt(
                "progress.separation_only_complete",
                seconds=f"{processing_time:.1f}",
            ),
        )
        return SeparationResult(
            mode=mode,
            source_path=str(source_path),
            output_dir=str(resolved_output_dir),
            separated_audio=separated_audio,
            processing_time=processing_time,
        )

    def cancel(self) -> None:
        self._cancelled = True
        cancel = getattr(self._separator, "cancel", None)
        if callable(cancel):
            cancel()
