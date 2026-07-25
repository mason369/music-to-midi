"""Model-specific inference precision selection for supported NVIDIA GPUs.

The project keeps published model parameters in FP32.  Reduced precision is
used only through autocast for model families whose verified upstream runtime
does so.  Capability decisions are based on the active CUDA runtime and a real
matrix-multiply probe rather than GPU-name allowlists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

YOURMT3 = "yourmt3"
MUSCRIPTOR = "muscriptor"
ARIA_AMT = "aria_amt"
MIROS = "miros"
LEAP_XE = "leap_xe"
POLARFORMER = "polarformer"

_KNOWN_MODELS = {
    YOURMT3,
    MUSCRIPTOR,
    ARIA_AMT,
    MIROS,
    LEAP_XE,
    POLARFORMER,
}


@dataclass(frozen=True)
class PrecisionCapabilities:
    device: str
    device_name: str
    compute_capability: str
    fp32: bool
    fp16: bool
    bf16: bool
    tf32: bool
    fp16_probe_error: str = ""
    bf16_probe_error: str = ""


@dataclass(frozen=True)
class InferencePrecisionPlan:
    model: str
    device: str
    parameter_dtype: str
    compute_dtype: str
    autocast: bool
    allow_tf32: bool
    capabilities: PrecisionCapabilities
    reason: str

    def torch_dtype(self, torch_module=None):
        if torch_module is None:
            import torch as torch_module

        names = {
            "float32": "float32",
            "float16": "float16",
            "bfloat16": "bfloat16",
        }
        return getattr(torch_module, names[self.compute_dtype])


def _probe_cuda_dtype(torch_module, device, dtype) -> tuple[bool, str]:
    try:
        with torch_module.inference_mode():
            left = torch_module.tensor(
                [[1.0, 2.0], [3.0, 4.0]],
                device=device,
                dtype=dtype,
            )
            result = torch_module.mm(left, left)
            if result.dtype != dtype:
                raise RuntimeError(f"unexpected output dtype {result.dtype}")
            if not bool(torch_module.isfinite(result).all().item()):
                raise RuntimeError("matrix multiplication produced a non-finite value")
        torch_module.cuda.synchronize(device)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _native_bf16_supported(torch_module, device_index: int, major: int) -> bool:
    checker = getattr(torch_module.cuda, "is_bf16_supported", None)
    if not callable(checker):
        return False
    try:
        return bool(checker(including_emulation=False))
    except TypeError:
        # Older PyTorch releases do not expose including_emulation.  Native
        # NVIDIA BF16 Tensor Cores begin at compute capability 8.x.
        return major >= 8 and bool(checker())


def _probe_precision_capabilities_impl(torch_module, device_name: str) -> PrecisionCapabilities:
    device = torch_module.device(device_name)
    if device.type != "cuda":
        return PrecisionCapabilities(
            device=str(device),
            device_name=device.type.upper(),
            compute_capability="n/a",
            fp32=True,
            fp16=False,
            bf16=False,
            tf32=False,
        )

    if not torch_module.cuda.is_available():
        raise RuntimeError(f"CUDA precision probe requested but CUDA is unavailable: {device}")

    device_index = device.index
    if device_index is None:
        device_index = int(torch_module.cuda.current_device())
    properties = torch_module.cuda.get_device_properties(device_index)
    major, minor = torch_module.cuda.get_device_capability(device_index)
    resolved_device = torch_module.device("cuda", device_index)

    fp32, fp32_error = _probe_cuda_dtype(torch_module, resolved_device, torch_module.float32)
    if not fp32:
        raise RuntimeError(
            "CUDA device failed the required FP32 inference probe: "
            f"device={resolved_device}, gpu={properties.name}, error={fp32_error}"
        )

    fp16, fp16_error = _probe_cuda_dtype(torch_module, resolved_device, torch_module.float16)
    native_bf16 = _native_bf16_supported(torch_module, device_index, int(major))
    if native_bf16:
        bf16, bf16_error = _probe_cuda_dtype(
            torch_module,
            resolved_device,
            torch_module.bfloat16,
        )
    else:
        bf16 = False
        bf16_error = "native BF16 Tensor Core support is unavailable"

    return PrecisionCapabilities(
        device=str(resolved_device),
        device_name=str(properties.name),
        compute_capability=f"{major}.{minor}",
        fp32=True,
        fp16=fp16,
        bf16=bf16,
        tf32=int(major) >= 8,
        fp16_probe_error=fp16_error,
        bf16_probe_error=bf16_error,
    )


@lru_cache(maxsize=32)
def _probe_precision_capabilities_cached(device_name: str) -> PrecisionCapabilities:
    import torch

    return _probe_precision_capabilities_impl(torch, device_name)


def probe_precision_capabilities(
    device: str,
    *,
    torch_module=None,
) -> PrecisionCapabilities:
    """Probe FP32/FP16/BF16 execution on the active CUDA device."""

    if torch_module is None:
        return _probe_precision_capabilities_cached(str(device))
    return _probe_precision_capabilities_impl(torch_module, str(device))


def select_inference_precision(
    model: str,
    device: str,
    *,
    torch_module=None,
) -> InferencePrecisionPlan:
    """Return the verified precision policy for one model family."""

    normalized_model = str(model).strip().lower()
    if normalized_model not in _KNOWN_MODELS:
        raise ValueError(f"Unknown inference precision policy: {model!r}")

    capabilities = probe_precision_capabilities(device, torch_module=torch_module)
    if not capabilities.device.startswith("cuda"):
        return InferencePrecisionPlan(
            model=normalized_model,
            device=capabilities.device,
            parameter_dtype="float32",
            compute_dtype="float32",
            autocast=False,
            allow_tf32=False,
            capabilities=capabilities,
            reason="non-CUDA execution uses explicit FP32",
        )

    if normalized_model in {YOURMT3, POLARFORMER}:
        compute_dtype = "float32"
        reason = "quality-sensitive verified path uses FP32"
    elif normalized_model in {MUSCRIPTOR, LEAP_XE}:
        if not capabilities.fp16:
            raise RuntimeError(
                f"{normalized_model} requires CUDA FP16 autocast, but the active GPU "
                f"failed the FP16 probe: {capabilities.fp16_probe_error}"
            )
        compute_dtype = "float16"
        reason = "verified upstream path uses FP16 autocast with FP32 parameters"
    elif normalized_model == ARIA_AMT:
        if capabilities.bf16:
            compute_dtype = "bfloat16"
            reason = "native BF16 is available and matches the verified Aria-AMT path"
        else:
            compute_dtype = "float32"
            reason = (
                "native BF16 is unavailable; verified Aria-AMT path keeps FP32 "
                "instead of substituting FP16"
            )
    elif normalized_model == MIROS:
        if capabilities.bf16:
            compute_dtype = "bfloat16"
            reason = "native BF16 is available and matches the verified MIROS path"
        elif capabilities.fp16:
            compute_dtype = "float16"
            reason = (
                "native BF16 is unavailable; use supported FP16 Tensor Core autocast "
                "with FP32 parameters"
            )
        else:
            raise RuntimeError(
                "MIROS requires CUDA BF16 or FP16 autocast, but neither precision "
                "passed the active-device probe"
            )
    else:  # pragma: no cover - guarded by _KNOWN_MODELS
        raise AssertionError(normalized_model)

    return InferencePrecisionPlan(
        model=normalized_model,
        device=capabilities.device,
        parameter_dtype="float32",
        compute_dtype=compute_dtype,
        autocast=compute_dtype != "float32",
        allow_tf32=False,
        capabilities=capabilities,
        reason=reason,
    )


def configure_torch_precision(plan: InferencePrecisionPlan, *, torch_module=None) -> None:
    """Apply the plan's global CUDA math safeguards in the current process."""

    if torch_module is None:
        import torch as torch_module

    if not plan.device.startswith("cuda"):
        return

    torch_module.set_float32_matmul_precision("highest")
    torch_module.backends.cuda.matmul.allow_tf32 = plan.allow_tf32
    torch_module.backends.cudnn.allow_tf32 = plan.allow_tf32

    matmul_backend = torch_module.backends.cuda.matmul
    if hasattr(matmul_backend, "allow_fp16_reduced_precision_reduction"):
        matmul_backend.allow_fp16_reduced_precision_reduction = False
    if hasattr(matmul_backend, "allow_bf16_reduced_precision_reduction"):
        matmul_backend.allow_bf16_reduced_precision_reduction = False


def verify_float32_model_parameters(
    model: Any,
    *,
    model_name: str,
    torch_module=None,
) -> None:
    """Reject accidental whole-model FP16/BF16 conversion."""

    if torch_module is None:
        import torch as torch_module

    parameter_dtypes = {
        parameter.dtype for parameter in model.parameters() if bool(parameter.is_floating_point())
    }
    if not parameter_dtypes:
        raise RuntimeError(f"{model_name} exposes no floating-point model parameters")
    if parameter_dtypes != {torch_module.float32}:
        formatted = ", ".join(sorted(str(dtype) for dtype in parameter_dtypes))
        raise RuntimeError(
            f"{model_name} parameter dtype contract violated: expected only torch.float32, "
            f"got {formatted}"
        )


def log_precision_plan(
    target_logger: logging.Logger,
    plan: InferencePrecisionPlan,
) -> None:
    capabilities = plan.capabilities
    target_logger.info(
        "Inference precision plan | model=%s device=%s gpu=%s cc=%s "
        "parameters=%s compute=%s autocast=%s fp32=%s fp16=%s bf16=%s "
        "tf32_supported=%s tf32_enabled=%s reduced_precision_reduction=false reason=%s",
        plan.model,
        plan.device,
        capabilities.device_name,
        capabilities.compute_capability,
        plan.parameter_dtype,
        plan.compute_dtype,
        plan.autocast,
        capabilities.fp32,
        capabilities.fp16,
        capabilities.bf16,
        capabilities.tf32,
        plan.allow_tf32,
        plan.reason,
    )
