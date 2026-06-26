"""人声分离模块 - 使用 audio-separator RoFormer ensembles 分离人声与伴奏。"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from src.i18n.translator import Translator
from src.utils.audio_separator_compat import (
    execute_audio_separator_job,
    get_separator_cls,
)
from src.utils.gpu_utils import (
    clear_gpu_memory,
    is_unsupported_cuda_architecture_error,
    rewrite_cuda_runtime_error,
)
from src.utils.runtime_paths import get_audio_separator_model_dir

logger = logging.getLogger(__name__)

ROFORMER_MODEL = "ensemble:vocal_rvc"
ROFORMER_REQUIRED_MODELS = (
    "melband_roformer_big_beta6x.ckpt",
    "mel_band_roformer_vocals_fv4_gabox.ckpt",
)

KARAOKE_MODEL = "ensemble:karaoke"
KARAOKE_REQUIRED_MODELS = (
    "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "mel_band_roformer_karaoke_becruily.ckpt",
)

MDXC_QUALITY_PARAMS = {
    "segment_size": 256,
    "override_model_segment_size": False,
    "batch_size": 1,
    "overlap": 50,
    "pitch_shift": 0,
}


def _find_existing_model(cache_dir: Path, model_name: str) -> Optional[Path]:
    direct = cache_dir / model_name
    if direct.is_file() and direct.stat().st_size > 0:
        return direct
    for path in cache_dir.rglob(model_name):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _resolve_output_files(output_files: Iterable[object], output_dir: Path) -> list[Path]:
    resolved: list[Path] = []
    seen: set[str] = set()
    for item in output_files:
        path = Path(str(item))
        if not path.is_absolute():
            path = output_dir / path
        if path.exists():
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                resolved.append(path)

    if resolved:
        return resolved

    for path in sorted(output_dir.glob("*.wav"), key=lambda candidate: candidate.stat().st_mtime):
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            resolved.append(path)
    return resolved


def _classify_vocal_rvc_output(file_name: str) -> Optional[str]:
    lower_name = file_name.lower()
    if any(marker in lower_name for marker in ("(instrumental)", "(other)", "(no_vocal", "(no vocal")):
        return "accompaniment"
    if any(marker in lower_name for marker in ("(vocals)", "(vocal)", "(primary)", "(lead)")):
        return "vocals"
    if "instrumental" in lower_name or "no_vocal" in lower_name:
        return "accompaniment"
    if "vocal" in lower_name or "primary" in lower_name:
        return "vocals"
    return None


def _classify_karaoke_output(file_name: str) -> Optional[str]:
    lower_name = file_name.lower()
    if any(marker in lower_name for marker in ("(instrumental)", "(other)", "(backing)", "(no_vocal", "(no vocal")):
        return "backing"
    if any(marker in lower_name for marker in ("(vocals)", "(vocal)", "(lead)", "(main_vocal)", "(main vocals)")):
        return "lead"
    if "instrumental" in lower_name or "other" in lower_name or "backing" in lower_name:
        return "backing"
    if "vocal" in lower_name or "lead" in lower_name:
        return "lead"
    return None


def _move_role_output(
    output_files: Iterable[object],
    output_dir: Path,
    classifier: Callable[[str], Optional[str]],
    expected_roles: dict[str, Path],
) -> dict[str, Path]:
    resolved_files = _resolve_output_files(output_files, output_dir)
    role_paths: dict[str, Path] = {}

    for path in resolved_files:
        role = classifier(path.name)
        if role in expected_roles and role not in role_paths:
            role_paths[role] = path

    missing = [role for role in expected_roles if role not in role_paths]
    if missing:
        raise RuntimeError(
            "audio-separator 输出角色识别失败；"
            f"缺少={missing}, files={[path.name for path in resolved_files]}"
        )

    final_paths: dict[str, Path] = {}
    for role, source_path in role_paths.items():
        final_path = expected_roles[role]
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != final_path.resolve():
            if final_path.exists():
                final_path.unlink()
            shutil.move(str(source_path), str(final_path))
        final_paths[role] = final_path

    return final_paths


def _match_channels(audio, channels: int):
    import numpy as np

    current = audio.shape[1]
    if current == channels:
        return audio
    if current == 1:
        return np.repeat(audio, channels, axis=1)
    if channels == 1:
        return np.mean(audio, axis=1, keepdims=True)
    if current > channels:
        return audio[:, :channels]
    repeats = channels - current
    return np.concatenate([audio, np.repeat(audio[:, -1:], repeats, axis=1)], axis=1)


def _resample_audio(audio, orig_sr: int, target_sr: int):
    if orig_sr == target_sr:
        return audio
    import librosa
    import numpy as np

    channels = [
        librosa.resample(audio[:, channel], orig_sr=orig_sr, target_sr=target_sr)
        for channel in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1)


def _mix_backing_into_accompaniment(
    *,
    backing_vocals_path: Path,
    accompaniment_path: Path,
    output_path: Path,
) -> Path:
    import numpy as np
    import soundfile as sf

    accompaniment, accompaniment_sr = sf.read(
        str(accompaniment_path),
        dtype="float32",
        always_2d=True,
    )
    backing, backing_sr = sf.read(
        str(backing_vocals_path),
        dtype="float32",
        always_2d=True,
    )

    backing = _resample_audio(backing, backing_sr, accompaniment_sr)
    backing = _match_channels(backing, accompaniment.shape[1])

    max_len = max(accompaniment.shape[0], backing.shape[0])
    if accompaniment.shape[0] < max_len:
        accompaniment = np.pad(
            accompaniment,
            ((0, max_len - accompaniment.shape[0]), (0, 0)),
            mode="constant",
        )
    if backing.shape[0] < max_len:
        backing = np.pad(backing, ((0, max_len - backing.shape[0]), (0, 0)), mode="constant")

    mixed = accompaniment + backing
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), mixed, accompaniment_sr)
    return output_path


class VocalSeparator:
    """使用 AI-RVC 风格 RoFormer ensembles 做高质量人声/伴奏分离。"""

    def __init__(
        self,
        language: str = Translator.DEFAULT_LANGUAGE,
        *,
        primary_device: Optional[str] = None,
        karaoke_device: Optional[str] = None,
    ):
        self._cancelled = False
        self._cancel_check: Optional[Callable[[], bool]] = None
        self._translator = Translator(language)
        self.primary_device = primary_device
        self.karaoke_device = karaoke_device

    def _pt(self, key: str, **kwargs) -> str:
        return self._translator.t(key, **kwargs)

    def set_cancel_check(self, fn: Callable[[], bool]) -> None:
        self._cancel_check = fn

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise InterruptedError("用户取消了处理")
        if self._cancel_check and self._cancel_check():
            raise InterruptedError("用户取消了处理")

    def cancel(self) -> None:
        self._cancelled = True

    @staticmethod
    def is_available() -> bool:
        """检查 audio-separator 是否可用。"""
        try:
            get_separator_cls()
            return True
        except ImportError:
            return False

    @staticmethod
    def is_model_available() -> bool:
        cache_dir = get_audio_separator_model_dir()
        required = ROFORMER_REQUIRED_MODELS + KARAOKE_REQUIRED_MODELS
        return all(_find_existing_model(cache_dir, model_name) is not None for model_name in required)

    @staticmethod
    def _get_model_cache_dir() -> str:
        return str(get_audio_separator_model_dir())

    def _run_separator(
        self,
        *,
        model_name: str,
        audio_path: str,
        output_dir: Path,
        progress_callback: Optional[Callable[[float, str], None]],
        progress_start: float,
        progress_span: float,
        target_device: Optional[str] = None,
    ) -> list[object]:
        Separator = get_separator_cls()
        separator = None

        def _progress(local: float, message: str) -> None:
            if progress_callback is not None:
                progress_callback(progress_start + local * progress_span, message)

        def _after_load(active_separator) -> None:
            device = getattr(active_separator, "torch_device", "unknown")
            logger.info("Loaded audio-separator model %s on %s", model_name, device)
            _progress(0.15, f"loaded {model_name}")

        def _action(active_separator):
            self._check_cancelled()
            _progress(0.25, f"separating with {model_name}")
            return active_separator.separate(audio_path)

        try:
            separator, output_files, _used_cpu, _reason = execute_audio_separator_job(
                Separator,
                separator_kwargs={
                    "output_dir": str(output_dir),
                    "model_file_dir": self._get_model_cache_dir(),
                    "output_format": "WAV",
                    "use_soundfile": True,
                    "mdxc_params": dict(MDXC_QUALITY_PARAMS),
                },
                model_name=model_name,
                action=_action,
                logger=logger,
                progress_callback=progress_callback,
                fallback_progress=(
                    progress_start,
                    self._pt("progress.cpu_retry"),
                ),
                target_device=target_device,
                after_load=_after_load,
            )
            _progress(1.0, f"finished {model_name}")
            return list(output_files or [])
        finally:
            if separator is not None:
                try:
                    del separator
                except Exception:
                    logger.debug("Failed to release audio-separator instance", exc_info=True)
            clear_gpu_memory()

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, str]:
        """分离音频为人声、伴奏，并额外输出主唱/和声感知 stem。"""
        self._cancelled = False
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stem = Path(Path(audio_path).name).stem

        if progress_callback:
            progress_callback(0.0, self._pt("progress.loading_vocal_separator"))

        primary_dir = output_path / "_vocal_rvc"
        karaoke_dir = output_path / "_karaoke"
        primary_dir.mkdir(parents=True, exist_ok=True)
        karaoke_dir.mkdir(parents=True, exist_ok=True)

        vocals_with_harmony_path = output_path / f"{stem}_vocals_with_harmony.wav"
        accompaniment_path = output_path / f"{stem}_accompaniment.wav"
        original_vocals_path = output_path / f"{stem}_original_vocals.wav"
        backing_vocals_path = output_path / f"{stem}_backing_vocals.wav"
        accompaniment_with_harmony_path = output_path / f"{stem}_accompaniment_with_harmony.wav"

        try:
            logger.info(
                "Starting quality-first vocal split: primary=%s primary_device=%s karaoke=%s karaoke_device=%s mdxc=%s",
                ROFORMER_MODEL,
                self.primary_device or "auto",
                KARAOKE_MODEL,
                self.karaoke_device or "auto",
                MDXC_QUALITY_PARAMS,
            )

            primary_outputs = self._run_separator(
                model_name=ROFORMER_MODEL,
                audio_path=audio_path,
                output_dir=primary_dir,
                progress_callback=progress_callback,
                progress_start=0.02,
                progress_span=0.50,
                target_device=self.primary_device,
            )
            primary_paths = _move_role_output(
                primary_outputs,
                primary_dir,
                _classify_vocal_rvc_output,
                {
                    "vocals": vocals_with_harmony_path,
                    "accompaniment": accompaniment_path,
                },
            )
            self._check_cancelled()

            karaoke_outputs = self._run_separator(
                model_name=KARAOKE_MODEL,
                audio_path=str(primary_paths["vocals"]),
                output_dir=karaoke_dir,
                progress_callback=progress_callback,
                progress_start=0.52,
                progress_span=0.34,
                target_device=self.karaoke_device,
            )
            _move_role_output(
                karaoke_outputs,
                karaoke_dir,
                _classify_karaoke_output,
                {
                    "lead": original_vocals_path,
                    "backing": backing_vocals_path,
                },
            )
            self._check_cancelled()

            if progress_callback:
                progress_callback(0.90, self._pt("progress.saving_separation_results"))
            _mix_backing_into_accompaniment(
                backing_vocals_path=backing_vocals_path,
                accompaniment_path=accompaniment_path,
                output_path=accompaniment_with_harmony_path,
            )

            for path in (
                vocals_with_harmony_path,
                accompaniment_path,
                original_vocals_path,
                accompaniment_with_harmony_path,
            ):
                if not path.exists() or path.stat().st_size <= 0:
                    raise RuntimeError(f"separator did not create a valid output file: {path}")

            logger.info("Vocal split complete: vocals_with_harmony=%s", vocals_with_harmony_path)
            logger.info("Vocal split complete: accompaniment=%s", accompaniment_path)
            logger.info("Vocal split complete: original_vocals=%s", original_vocals_path)
            logger.info(
                "Vocal split complete: accompaniment_with_harmony=%s",
                accompaniment_with_harmony_path,
            )
        except InterruptedError:
            raise
        except RuntimeError as exc:
            if is_unsupported_cuda_architecture_error(exc):
                friendly = rewrite_cuda_runtime_error(exc)
                logger.error("人声分离 CUDA 架构不兼容:\n%s", friendly)
                raise RuntimeError(f"人声分离失败（GPU 不兼容）:\n{friendly}") from exc
            logger.error("音频分离失败: %s", exc)
            raise RuntimeError(f"音频分离失败: {exc}") from exc
        except Exception as exc:
            logger.error("音频分离失败: %s", exc)
            raise RuntimeError(f"音频分离失败: {exc}") from exc
        finally:
            clear_gpu_memory()

        if progress_callback:
            progress_callback(1.0, self._pt("progress.separation_complete"))

        return {
            "vocals": str(vocals_with_harmony_path),
            "no_vocals": str(accompaniment_path),
            "original_vocals": str(original_vocals_path),
            "vocals_with_harmony": str(vocals_with_harmony_path),
            "accompaniment_with_harmony": str(accompaniment_with_harmony_path),
        }
