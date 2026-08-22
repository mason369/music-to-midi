"""Strict runtime gate for the isolated CUDA and Intel XPU environments."""

from __future__ import annotations

import argparse
from importlib import metadata
import os
from pathlib import Path
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CUDA_PYTORCH_VERSIONS = {
    "torch": "2.7.0",
    "torchaudio": "2.7.0",
    "torchvision": "0.22.0",
}
XPU_PYTORCH_VERSIONS = {
    "torch": "2.11.0",
    "torchaudio": "2.11.0",
    "torchvision": "0.26.0",
}
XPU_RUNTIME_VERSIONS = {
    "onnxruntime-openvino": "1.24.1",
    "openvino": "2025.4.1",
}
XPU_HARDWARE_BASELINE = (
    "Intel Arc B-Series (Battlemage) and Intel Core Ultra Series 3 (Panther Lake)"
)
CUDA_RUNTIME_VERSIONS = {"onnxruntime-gpu": "1.23.2"}


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Required runtime distribution is missing: {name}") from exc


def _require_distribution_versions(expected: dict[str, str]) -> dict[str, str]:
    actual = {name: _distribution_version(name) for name in expected}
    mismatches = {
        name: version
        for name, version in actual.items()
        if version.split("+", 1)[0] != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"Runtime distribution mismatch: expected={expected}, actual={actual}")
    return actual


def _reject_distributions(names: tuple[str, ...]) -> None:
    installed = {}
    for name in names:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    if installed:
        raise RuntimeError(f"Conflicting runtime distributions are installed: {installed}")


def _validate_windows_xpu_openmp_runtime(torch) -> None:
    if sys.platform != "win32":
        return

    torch_package = Path(torch.__file__).resolve().parent
    torch_lib = torch_package / "lib"
    required = torch_lib / "libiomp5md.dll"
    conflicting = torch_lib / "libomp140.x86_64.dll"
    if not required.is_file():
        raise RuntimeError(
            "The official PyTorch XPU Intel OpenMP runtime is missing: " f"{required}"
        )
    if conflicting.exists():
        raise RuntimeError(
            "The isolated XPU environment contains a conflicting LLVM OpenMP runtime: "
            f"{conflicting}. Recreate venv-xpu with install_xpu.ps1; do not copy "
            "the CUDA libomp140 repair into an XPU environment."
        )


def _validate_torch_xpu(torch) -> tuple[str, str]:
    versions = _require_distribution_versions(XPU_PYTORCH_VERSIONS)
    wrong_flavors = {name: version for name, version in versions.items() if "+xpu" not in version}
    if wrong_flavors:
        raise RuntimeError(f"Expected native PyTorch XPU wheels, got: {wrong_flavors}")
    if getattr(torch.version, "cuda", None) is not None:
        raise RuntimeError(f"Intel XPU environment unexpectedly exposes CUDA={torch.version.cuda}")
    if getattr(torch.version, "hip", None) is not None:
        raise RuntimeError(f"Intel XPU environment unexpectedly exposes ROCm={torch.version.hip}")
    _validate_windows_xpu_openmp_runtime(torch)
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available():
        raise RuntimeError("torch.xpu.is_available() is False")
    if int(xpu.device_count()) <= 0:
        raise RuntimeError("torch.xpu did not report an Intel GPU")
    device = "xpu:0"
    probe = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    result = torch.mm(probe, probe)
    if result.device.type != "xpu" or result.device.index != 0:
        raise RuntimeError(f"XPU probe ran on the wrong device: {result.device}")
    if result.cpu().tolist() != [[7.0, 10.0], [15.0, 22.0]]:
        raise RuntimeError("XPU matrix multiplication produced an invalid result")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bf16_input = torch.randn(1, 1, 32, 32, device=device, dtype=torch.bfloat16)
        bf16_conv = torch.nn.Conv2d(1, 8, kernel_size=3, bias=True).eval().to(device)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="xpu",
                dtype=torch.bfloat16,
                enabled=True,
            ),
        ):
            bf16_output = bf16_conv(bf16_input)
        signal = torch.linspace(-1.0, 1.0, steps=4096, device=device)
        spectrum = torch.fft.rfft(signal)
        restored = torch.fft.irfft(spectrum, n=signal.shape[-1])
        window = torch.hann_window(512, device=device)
        stft = torch.stft(
            signal,
            n_fft=512,
            hop_length=128,
            window=window,
            return_complex=True,
        )
        reconstructed = torch.istft(
            stft,
            n_fft=512,
            hop_length=128,
            window=window,
            length=signal.shape[-1],
        )
        xpu.synchronize(0)

    fallback_messages = []
    for warning in caught:
        message = str(warning.message)
        lowered = message.lower()
        if "xpu" in lowered and "cpu" in lowered and "fallback" in lowered:
            fallback_messages.append(message)
        elif "xpu" in lowered and "cpu" in lowered and "falling back" in lowered:
            fallback_messages.append(message)
    if fallback_messages:
        raise RuntimeError(
            "PyTorch XPU required operators fell back to CPU: " + " | ".join(fallback_messages)
        )

    if bf16_output.device.type != "xpu" or bf16_output.dtype != torch.bfloat16:
        raise RuntimeError(
            "XPU BF16 autocast convolution returned the wrong device or dtype: "
            f"device={bf16_output.device}, dtype={bf16_output.dtype}"
        )
    if not bool(torch.isfinite(bf16_output).all().item()):
        raise RuntimeError("XPU BF16 autocast convolution produced non-finite values")

    rfft_error = float((signal - restored).abs().max().cpu())
    stft_error = float((signal - reconstructed).abs().max().cpu())
    if rfft_error > 1e-4 or stft_error > 1e-4:
        raise RuntimeError(
            "XPU FFT/STFT numerical probe failed: "
            f"rfft_error={rfft_error}, stft_error={stft_error}"
        )
    xpu.synchronize(0)
    return device, xpu.get_device_name(0)


def _validate_torch_cuda(torch) -> tuple[str, str]:
    versions = _require_distribution_versions(CUDA_PYTORCH_VERSIONS)
    wrong_flavors = {name: version for name, version in versions.items() if "+cu128" not in version}
    if wrong_flavors:
        raise RuntimeError(f"Expected PyTorch CUDA 12.8 wheels, got: {wrong_flavors}")
    if getattr(torch.version, "hip", None) is not None:
        raise RuntimeError(f"ROCm is unsupported: HIP={torch.version.hip}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected CUDA 12.8, got {torch.version.cuda!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    device = "cuda:0"
    probe = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    result = torch.mm(probe, probe)
    if result.device.type != "cuda" or result.device.index != 0:
        raise RuntimeError(f"CUDA probe ran on the wrong device: {result.device}")
    if result.cpu().tolist() != [[7.0, 10.0], [15.0, 22.0]]:
        raise RuntimeError("CUDA matrix multiplication produced an invalid result")
    torch.cuda.synchronize(0)
    return device, torch.cuda.get_device_name(0)


def _validate_openvino_gpu_session() -> list[str]:
    import numpy as np
    from src.utils.openvino_runtime import initialize_openvino_gpu_runtime

    initialize_openvino_gpu_runtime()
    import onnxruntime as ort
    from onnx import TensorProto, helper

    _require_distribution_versions(XPU_RUNTIME_VERSIONS)
    providers = ort.get_available_providers()
    if "OpenVINOExecutionProvider" not in providers:
        raise RuntimeError(f"OpenVINOExecutionProvider is unavailable: providers={providers}")
    weights = helper.make_tensor(
        "weights",
        TensorProto.FLOAT,
        [2, 2],
        [2.0, 0.0, 0.0, 3.0],
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weights"], ["output"])],
        "music_to_midi_openvino_xpu_probe",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [weights],
    )
    model = helper.make_model(
        graph,
        producer_name="music-to-midi",
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
    )
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=[
            (
                "OpenVINOExecutionProvider",
                {"device_type": "GPU.0"},
            )
        ],
    )
    active = session.get_providers()
    if not active or active[0] != "OpenVINOExecutionProvider":
        raise RuntimeError(
            "OpenVINO session did not keep the GPU provider primary: " f"active={active}"
        )
    session.disable_fallback()
    value = np.asarray([[1.25, -2.5]], dtype=np.float32)
    output = session.run(["output"], {"input": value})[0]
    expected = np.asarray([[2.5, -7.5]], dtype=np.float32)
    if not np.array_equal(output, expected):
        raise RuntimeError("OpenVINO GPU MatMul probe produced an invalid result")
    return active


def validate_runtime(accelerator: str) -> dict[str, object]:
    if accelerator == "xpu":
        os.environ["PYTORCH_DEBUG_XPU_FALLBACK"] = "1"

    import torch
    import torchaudio  # noqa: F401
    import torchvision  # noqa: F401

    if accelerator == "xpu":
        _reject_distributions(
            (
                "intel-extension-for-pytorch",
                "onnxruntime",
                "onnxruntime-gpu",
            )
        )
        device, name = _validate_torch_xpu(torch)
        providers = _validate_openvino_gpu_session()
    elif accelerator == "cuda":
        import onnxruntime as ort

        _reject_distributions(("intel-extension-for-pytorch", "onnxruntime-openvino"))
        _require_distribution_versions(CUDA_RUNTIME_VERSIONS)
        device, name = _validate_torch_cuda(torch)
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(f"CUDAExecutionProvider is unavailable: {providers}")
    else:
        raise ValueError(f"Unsupported accelerator: {accelerator!r}")

    return {
        "accelerator": accelerator,
        "device": device,
        "name": name,
        "torch": str(torch.__version__),
        "providers": providers,
        **({"hardware_baseline": XPU_HARDWARE_BASELINE} if accelerator == "xpu" else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", required=True, choices=("cuda", "xpu"))
    args = parser.parse_args()
    result = validate_runtime(args.accelerator)
    print("Accelerator runtime verified:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
