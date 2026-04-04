"""Six-stem source separation using audio-separator."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from download_multistem_model import (
    ROFORMER_SW_MODEL,
    download_multistem_model,
)
from src.utils.audio_separator_compat import (
    execute_audio_separator_job,
    get_separator_cls,
    patch_separator_package_metadata,
)
from src.utils.runtime_paths import get_audio_separator_model_dir

logger = logging.getLogger(__name__)

STEM_KEYS = ("bass", "drums", "guitar", "piano", "vocals", "other")
BS_ROFORMER_SW_MODEL = ROFORMER_SW_MODEL


class SixStemSeparator:
    def __init__(
        self,
        *,
        separator_cls=None,
        ensure_assets_fn: Optional[Callable[..., object]] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.separator_cls = separator_cls
        self.ensure_assets_fn = ensure_assets_fn or download_multistem_model
        self.cache_dir = Path(cache_dir) if cache_dir is not None else get_audio_separator_model_dir()

    @staticmethod
    def is_available() -> bool:
        try:
            get_separator_cls()
            return True
        except Exception:
            return False

    def _get_separator_cls(self):
        if self.separator_cls is not None:
            return patch_separator_package_metadata(self.separator_cls)
        return get_separator_cls()

    def _attach_sw_model_registry(self, separator) -> None:
        if not hasattr(separator, "list_supported_model_files"):
            return
        original = separator.list_supported_model_files

        def _patched_list_supported_model_files():
            models = original()
            mdxc_models = dict(models.get("MDXC", {}))
            if "Roformer Model: BS-Roformer-SW" not in mdxc_models:
                mdxc_models["Roformer Model: BS-Roformer-SW"] = {
                    "filename": BS_ROFORMER_SW_MODEL,
                    "scores": {},
                    "stems": list(STEM_KEYS),
                    "target_stem": None,
                    "download_files": [BS_ROFORMER_SW_MODEL, "config_bs_roformer_sw.yaml"],
                }
            models["MDXC"] = mdxc_models
            return models

        separator.list_supported_model_files = _patched_list_supported_model_files

    def _prepare_separator(self, separator) -> None:
        self.ensure_assets_fn(cache_dir=self.cache_dir, printer=logger.info)
        self._attach_sw_model_registry(separator)

    @staticmethod
    def _to_existing_paths(output_dir: Path, output_files: Iterable[str]) -> list[Path]:
        paths: list[Path] = []
        for entry in output_files:
            path = Path(entry)
            if not path.is_absolute():
                path = output_dir / path
            if path.exists() and path.is_file():
                paths.append(path)

        if paths:
            return paths

        return [p for p in output_dir.glob("*.wav") if p.is_file()]

    @staticmethod
    def _detect_stem_key(name: str) -> Optional[str]:
        lowered = name.lower()
        if "vocals" in lowered and "instrumental" not in lowered:
            return "vocals"
        if "drum" in lowered:
            return "drums"
        if "bass" in lowered:
            return "bass"
        if "guitar" in lowered:
            return "guitar"
        if "piano" in lowered:
            return "piano"
        if "other" in lowered or "instrumental" in lowered:
            return "other"
        return None

    def _normalize_outputs(self, audio_path: str, output_dir: Path, output_files: Iterable[str]) -> Dict[str, str]:
        source_paths = self._to_existing_paths(output_dir, output_files)
        source_by_stem: Dict[str, Path] = {}

        for path in source_paths:
            stem_key = self._detect_stem_key(path.name)
            if stem_key and stem_key not in source_by_stem:
                source_by_stem[stem_key] = path

        input_stem = Path(audio_path).stem
        result: Dict[str, str] = {}
        for stem_key in STEM_KEYS:
            if stem_key not in source_by_stem:
                continue
            normalized_path = output_dir / f"{input_stem}_{stem_key}.wav"
            src = source_by_stem[stem_key]
            if src.resolve() != normalized_path.resolve():
                os.replace(src, normalized_path)
            result[stem_key] = str(normalized_path)

        missing = [stem for stem in STEM_KEYS if stem not in result]
        if missing:
            raise RuntimeError(f"Six-stem separation output is incomplete, missing stems: {missing}")

        return result

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, str]:
        if progress_callback:
            progress_callback(0.0, "Preparing six-stem separator...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        separator_cls = self._get_separator_cls()

        def _after_load(_active_separator):
            if progress_callback:
                progress_callback(0.25, f"Loaded model: {BS_ROFORMER_SW_MODEL}")

        _separator, output_files, _used_cpu_fallback, _fallback_reason = execute_audio_separator_job(
            separator_cls,
            separator_kwargs={
                "output_dir": str(output_path),
                "model_file_dir": str(self.cache_dir),
                "output_format": "WAV",
            },
            model_name=BS_ROFORMER_SW_MODEL,
            action=lambda active_separator: active_separator.separate(audio_path),
            logger=logger,
            progress_callback=progress_callback,
            fallback_progress=(
                0.1,
                "Detected an unsupported NVIDIA GPU architecture, retrying six-stem separation on CPU...",
            ),
            prepare_separator=self._prepare_separator,
            after_load=_after_load,
        )
        if progress_callback:
            progress_callback(0.85, "Normalizing stem files...")

        normalized = self._normalize_outputs(audio_path, output_path, output_files)

        if progress_callback:
            progress_callback(1.0, "Six-stem separation complete")
        return normalized
