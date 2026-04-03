import types

import pytest

from src.models.data_models import Config
import src.core.yourmt3_transcriber as yourmt3_transcriber
import src.utils.gpu_utils as gpu_utils


class _FakeCudaProps:
    name = "NVIDIA GeForce GTX 780"
    major = 3
    minor = 5


class _FakeCudaModule:
    @staticmethod
    def get_device_properties(_index):
        return _FakeCudaProps()

    @staticmethod
    def get_arch_list():
        return ["sm_50", "sm_60", "sm_75", "sm_86", "sm_90"]

    @staticmethod
    def synchronize(_index=0):
        return None


class _FakeTorchNoKernelImage:
    __version__ = "2.4.0+cu121"
    version = types.SimpleNamespace(cuda="12.1")
    cuda = _FakeCudaModule()

    @staticmethod
    def zeros(*_args, **_kwargs):
        raise RuntimeError(
            "CUDA error: no kernel image is available for execution on the device"
        )


def test_cuda_preflight_reports_unsupported_gpu_architecture(monkeypatch):
    monkeypatch.setattr(gpu_utils, "_get_torch", lambda: _FakeTorchNoKernelImage())

    with pytest.raises(RuntimeError) as excinfo:
        gpu_utils.ensure_cuda_runtime_compatibility("cuda:0")

    message = str(excinfo.value)
    assert "显卡架构" in message
    assert "NVIDIA GeForce GTX 780" in message
    assert "3.5" in message
    assert "sm_50, sm_60, sm_75, sm_86, sm_90" in message
    assert "CPU 版发布包" in message


def test_transcribe_precise_rewrites_no_kernel_image_error(monkeypatch):
    monkeypatch.setattr(yourmt3_transcriber, "get_device", lambda *_args: "cuda:0")
    transcriber = yourmt3_transcriber.YourMT3Transcriber(Config())
    monkeypatch.setattr(transcriber, "is_available", lambda: True)

    def _raise_no_kernel_image(*_args, **_kwargs):
        raise RuntimeError(
            "CUDA error: no kernel image is available for execution on the device"
        )

    monkeypatch.setattr(transcriber, "_prepare_and_infer", _raise_no_kernel_image)

    with pytest.raises(RuntimeError) as excinfo:
        transcriber.transcribe_precise("demo.wav")

    message = str(excinfo.value)
    assert "显卡架构" in message
    assert "CPU 版发布包" in message
    assert "no kernel image is available for execution on the device" not in message
