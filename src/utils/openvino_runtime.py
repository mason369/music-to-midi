"""Initialize the pip OpenVINO runtime before ONNX Runtime loads its provider."""

from __future__ import annotations


def initialize_openvino_gpu_runtime() -> tuple[str, ...]:
    """Load OpenVINO's Windows DLL search paths and prove a GPU device exists."""

    try:
        import openvino as openvino
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "OpenVINO could not be imported; install the pinned openvino runtime "
            "before creating an OpenVINOExecutionProvider session"
        ) from exc

    try:
        devices = tuple(str(device) for device in openvino.Core().available_devices)
    except Exception as exc:
        raise RuntimeError(f"OpenVINO runtime initialization failed: {exc}") from exc

    if not any(device == "GPU" or device.startswith("GPU.") for device in devices):
        raise RuntimeError(f"OpenVINO did not report an Intel GPU: devices={list(devices)}")
    return devices
