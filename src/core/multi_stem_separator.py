"""Six-stem source separation using audio-separator."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from download_multistem_model import (
    ROFORMER_SW_CONFIG,
    ROFORMER_SW_DISPLAY_NAME,
    ROFORMER_SW_MODEL,
    ROFORMER_SW_REGISTRY_NAME,
    download_multistem_model,
    is_multistem_model_available,
)
from src.i18n.translator import Translator
from src.utils.audio_separator_compat import (
    execute_audio_separator_job,
    get_separator_cls,
    patch_separator_package_metadata,
)
from src.utils.runtime_paths import get_audio_separator_model_dir

logger = logging.getLogger(__name__)

STEM_KEYS = ("bass", "drums", "guitar", "piano", "vocals", "other")
BS_ROFORMER_SW_MODEL = ROFORMER_SW_MODEL
BS_ROFORMER_SW_CONFIG = ROFORMER_SW_CONFIG
BS_ROFORMER_SW_DISPLAY_NAME = ROFORMER_SW_DISPLAY_NAME
BS_ROFORMER_SW_REGISTRY_NAME = ROFORMER_SW_REGISTRY_NAME


class SixStemSeparator:
    def __init__(
        self,
        *,
        separator_cls=None,
        ensure_assets_fn: Optional[Callable[..., object]] = None,
        cache_dir: Optional[Path] = None,
        language: str = Translator.DEFAULT_LANGUAGE,
        target_device: Optional[str] = None,
    ):
        self.separator_cls = separator_cls
        self.ensure_assets_fn = ensure_assets_fn or download_multistem_model
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else get_audio_separator_model_dir()
        )
        self._translator = Translator(language)
        self.target_device = target_device

    def _pt(self, key: str, **kwargs) -> str:
        return self._translator.t(key, **kwargs)

    @staticmethod
    def is_available() -> bool:
        try:
            get_separator_cls()
            return True
        except Exception:
            return False

    def is_model_available(self) -> bool:
        return is_multistem_model_available(cache_dir=self.cache_dir)

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
            if BS_ROFORMER_SW_REGISTRY_NAME not in mdxc_models:
                mdxc_models[BS_ROFORMER_SW_REGISTRY_NAME] = {
                    "filename": BS_ROFORMER_SW_MODEL,
                    "scores": {},
                    "stems": list(STEM_KEYS),
                    "target_stem": None,
                    "download_files": [BS_ROFORMER_SW_MODEL, BS_ROFORMER_SW_CONFIG],
                }
            models["MDXC"] = mdxc_models
            return models

        separator.list_supported_model_files = _patched_list_supported_model_files

    def _prepare_separator(self, separator) -> None:
        if not self.is_model_available():
            self.ensure_assets_fn(cache_dir=self.cache_dir, printer=logger.info)
        self._attach_sw_model_registry(separator)

    @staticmethod
    def _to_existing_paths(output_dir: Path, output_files: Iterable[str]) -> list[Path]:
        entries = list(output_files)
        if not entries:
            raise RuntimeError("Six-stem separator returned no output files")

        output_root = output_dir.resolve()
        paths: list[Path] = []
        resolved_paths: set[Path] = set()
        for entry in entries:
            path = Path(entry)
            if not path.is_absolute():
                path = output_dir / path
            resolved_path = path.resolve()
            try:
                resolved_path.relative_to(output_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Six-stem separator returned a file outside the current output directory: {path}"
                ) from exc
            if not resolved_path.exists() or not resolved_path.is_file():
                raise RuntimeError(f"Six-stem separator output file does not exist: {path}")
            if resolved_path in resolved_paths:
                raise RuntimeError(f"Six-stem separator returned a duplicate output file: {path}")
            resolved_paths.add(resolved_path)
            paths.append(resolved_path)
        return paths

    @staticmethod
    def _detect_stem_key(name: str, input_stem: str) -> Optional[str]:
        # audio-separator 0.44.1 writes
        #   <sanitized input>_(<stem>)_<model>.wav
        # Match only that stem slot. Searching the complete filename would let
        # words such as "piano" in the original song title override "(other)".
        sanitized_input = re.sub(r'[<>:"/\\|?*]', "_", input_stem)
        sanitized_input = re.sub(r"_+", "_", sanitized_input).strip("_. ")
        if not sanitized_input:
            return None

        filename_stem = Path(name).stem
        suffix = (
            filename_stem[len(sanitized_input) :]
            if filename_stem.casefold().startswith(sanitized_input.casefold())
            else ""
        )
        match = re.fullmatch(
            r"_\((bass|drums|guitar|piano|vocals|other)\)(?:_.+)?",
            suffix,
            flags=re.IGNORECASE,
        )
        return match.group(1).lower() if match else None

    def _normalize_outputs(
        self, audio_path: str, output_dir: Path, output_files: Iterable[str]
    ) -> Dict[str, str]:
        source_paths = self._to_existing_paths(output_dir, output_files)
        source_by_stem: Dict[str, Path] = {}
        input_stem = Path(audio_path).stem

        for path in source_paths:
            stem_key = self._detect_stem_key(path.name, input_stem)
            if stem_key is None:
                raise RuntimeError(f"Unrecognized six-stem separator output filename: {path.name}")
            if stem_key in source_by_stem:
                raise RuntimeError(
                    f"Six-stem separator returned multiple files for stem '{stem_key}': "
                    f"{source_by_stem[stem_key].name}, {path.name}"
                )
            source_by_stem[stem_key] = path

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
            raise RuntimeError(
                f"Six-stem separation output is incomplete, missing stems: {missing}"
            )

        return result

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, str]:
        if progress_callback:
            progress_callback(0.0, self._pt("progress.preparing_six_stem_separator"))

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        separator_cls = self._get_separator_cls()

        def _after_load(_active_separator):
            device = getattr(_active_separator, "torch_device", "unknown")
            logger.info(
                "Loaded six-stem separator model %s on %s",
                BS_ROFORMER_SW_DISPLAY_NAME,
                device,
            )
            if progress_callback:
                progress_callback(
                    0.25, self._pt("progress.loaded_model", model=BS_ROFORMER_SW_DISPLAY_NAME)
                )

        _separator, output_files, _used_cpu_fallback, _fallback_reason = (
            execute_audio_separator_job(
                separator_cls,
                separator_kwargs={
                    "output_dir": str(output_path),
                    "model_file_dir": str(self.cache_dir),
                    "output_format": "WAV",
                    "mdxc_params": {
                        "segment_size": 128,
                        "override_model_segment_size": True,
                        "batch_size": 1,
                        "overlap": 8,
                        "pitch_shift": 0,
                    },
                },
                model_name=BS_ROFORMER_SW_MODEL,
                action=lambda active_separator: active_separator.separate(audio_path),
                logger=logger,
                progress_callback=progress_callback,
                fallback_progress=(0.1, self._pt("progress.cpu_retry")),
                target_device=self.target_device,
                prepare_separator=self._prepare_separator,
                after_load=_after_load,
            )
        )
        if progress_callback:
            progress_callback(0.85, self._pt("progress.normalizing_stem_files"))

        normalized = self._normalize_outputs(audio_path, output_path, output_files)

        if progress_callback:
            progress_callback(1.0, self._pt("progress.six_stem_separation_complete"))
        return normalized
