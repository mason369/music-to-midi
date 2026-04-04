from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Callable, Optional, TypeVar

from src.utils.gpu_utils import (
    ensure_cuda_runtime_compatibility,
    get_device,
    is_unsupported_cuda_architecture_error,
    rewrite_cuda_runtime_error,
)
from src.utils.runtime_paths import activate_audio_separator_runtime


logger = logging.getLogger(__name__)

_PATCH_ATTR = "_music_to_midi_audio_separator_metadata_patch"
_FALLBACK_DISTRIBUTION = SimpleNamespace(version="unknown")
_CPU_FALLBACK_PROVIDER = ["CPUExecutionProvider"]

T = TypeVar("T")


def patch_separator_package_metadata(separator_cls: Any) -> Any:
    """Patch audio-separator metadata lookup for frozen builds missing dist-info."""
    if separator_cls is None or getattr(separator_cls, _PATCH_ATTR, False):
        return separator_cls

    original = getattr(separator_cls, "get_package_distribution", None)
    if not callable(original):
        return separator_cls

    def _patched_get_package_distribution(self, package_name):
        distribution = original(self, package_name)
        if distribution is None and package_name == "audio-separator":
            logger.debug(
                "audio-separator package metadata is unavailable; using placeholder version"
            )
            return _FALLBACK_DISTRIBUTION
        return distribution

    separator_cls.get_package_distribution = _patched_get_package_distribution
    setattr(separator_cls, _PATCH_ATTR, True)
    return separator_cls


def get_separator_cls():
    activate_audio_separator_runtime()
    from audio_separator.separator import Separator

    return patch_separator_package_metadata(Separator)


def _force_separator_cpu(separator: Any) -> Any:
    cpu_device = getattr(separator, "torch_device_cpu", None)
    if cpu_device is None:
        try:
            import torch

            cpu_device = torch.device("cpu")
        except Exception:
            cpu_device = "cpu"

    separator.torch_device = cpu_device
    separator.onnx_execution_provider = list(_CPU_FALLBACK_PROVIDER)
    return separator


def _resolve_cpu_fallback_reason(prefer_gpu: bool = True) -> Optional[str]:
    device = str(get_device(prefer_gpu=prefer_gpu))
    if not device.startswith("cuda"):
        return None

    try:
        ensure_cuda_runtime_compatibility(device)
    except RuntimeError as error:
        if is_unsupported_cuda_architecture_error(error):
            return rewrite_cuda_runtime_error(error, device)
        raise

    return None


def execute_audio_separator_job(
    separator_cls: Any,
    *,
    separator_kwargs: dict[str, Any],
    model_name: str,
    action: Callable[[Any], T],
    logger: logging.Logger,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    fallback_progress: Optional[tuple[float, str]] = None,
    prepare_separator: Optional[Callable[[Any], None]] = None,
    after_load: Optional[Callable[[Any], None]] = None,
) -> tuple[Any, T, bool, Optional[str]]:
    fallback_reason = _resolve_cpu_fallback_reason()
    force_cpu = fallback_reason is not None

    if force_cpu:
        logger.warning(
            "Detected unsupported CUDA architecture for audio-separator; falling back to CPU.\n%s",
            fallback_reason,
        )
        if progress_callback is not None and fallback_progress is not None:
            progress_callback(*fallback_progress)

    def _build_separator(use_cpu: bool) -> Any:
        separator = separator_cls(**separator_kwargs)
        if use_cpu:
            _force_separator_cpu(separator)
        if prepare_separator is not None:
            prepare_separator(separator)
        return separator

    separator = _build_separator(force_cpu)

    try:
        separator.load_model(model_name)
        if after_load is not None:
            after_load(separator)
        result = action(separator)
        return separator, result, force_cpu, fallback_reason
    except RuntimeError as error:
        if force_cpu or not is_unsupported_cuda_architecture_error(error):
            raise

        fallback_reason = rewrite_cuda_runtime_error(
            error,
            str(getattr(separator, "torch_device", "cuda:0")),
        )
        logger.warning(
            "audio-separator hit an unsupported CUDA architecture at runtime; retrying on CPU.\n%s",
            fallback_reason,
        )
        if progress_callback is not None and fallback_progress is not None:
            progress_callback(*fallback_progress)

        separator = _build_separator(True)
        separator.load_model(model_name)
        if after_load is not None:
            after_load(separator)
        result = action(separator)
        return separator, result, True, fallback_reason
