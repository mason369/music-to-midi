"""Six-stem source separation using audio-separator."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from download_multistem_model import (
    DEFAULT_CACHE_DIR,
    ROFORMER_SW_MODEL,
    download_multistem_model,
)

logger = logging.getLogger(__name__)

STEM_KEYS = ("bass", "drums", "guitar", "piano", "vocals", "other")
BS_ROFORMER_SW_MODEL = ROFORMER_SW_MODEL
HTDEMUCS_6S_MODEL = "htdemucs_6s.yaml"


class SixStemSeparator:
    def __init__(
        self,
        backend: str = "bs_roformer_sw",
        *,
        allow_fallback: bool = True,
        separator_cls=None,
        ensure_assets_fn: Optional[Callable[..., object]] = None,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ):
        self.backend = backend
        self.allow_fallback = allow_fallback
        self.separator_cls = separator_cls
        self.ensure_assets_fn = ensure_assets_fn or download_multistem_model
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def is_available() -> bool:
        try:
            from audio_separator.separator import Separator  # noqa: F401
            return True
        except Exception:
            return False

    def _get_separator_cls(self):
        if self.separator_cls is not None:
            return self.separator_cls
        from audio_separator.separator import Separator

        return Separator

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

    def _load_model(self, separator) -> str:
        selected_backend = self.backend

        if selected_backend == "bs_roformer_sw":
            try:
                self.ensure_assets_fn(cache_dir=self.cache_dir, printer=logger.info)
                self._attach_sw_model_registry(separator)
                separator.load_model(BS_ROFORMER_SW_MODEL)
                return BS_ROFORMER_SW_MODEL
            except Exception:
                if not self.allow_fallback:
                    raise
                logger.warning("Failed to load BS Roformer SW, fallback to htdemucs_6s", exc_info=True)
                separator.load_model(HTDEMUCS_6S_MODEL)
                return HTDEMUCS_6S_MODEL

        if selected_backend == "htdemucs_6s":
            separator.load_model(HTDEMUCS_6S_MODEL)
            return HTDEMUCS_6S_MODEL

        raise ValueError(f"Unsupported six-stem backend: {selected_backend}")

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
        separator = separator_cls(
            output_dir=str(output_path),
            model_file_dir=str(self.cache_dir),
            output_format="WAV",
        )

        model_name = self._load_model(separator)
        if progress_callback:
            progress_callback(0.25, f"Loaded model: {model_name}")

        output_files = separator.separate(audio_path)
        if progress_callback:
            progress_callback(0.85, "Normalizing stem files...")

        normalized = self._normalize_outputs(audio_path, output_path, output_files)

        if progress_callback:
            progress_callback(1.0, "Six-stem separation complete")
        return normalized
