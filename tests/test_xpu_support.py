from __future__ import annotations

import builtins
import inspect
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import src.utils.gpu_utils as gpu_utils
import src.utils.openvino_runtime as openvino_runtime
import src.core.yourmt3_transcriber as yourmt3_transcriber
import tools.validate_accelerator_runtime as runtime_validator
from src.core.aria_amt_transcriber import (
    ARIA_AMT_ONSET_VELOCITY_TOKEN_END,
    ARIA_AMT_ONSET_VELOCITY_TOKEN_START,
    ARIA_AMT_TOKEN_VOCAB_SIZE,
    AriaAmtTranscriber,
    _process_aria_segments_on_device,
    _recalculate_aria_token_ids_on_device,
)
from src.core.beat_this_tracker import BeatThisTracker
from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber
from src.core.multi_stem_separator import STEM_KEYS, SixStemSeparator
from src.core.transkun_transcriber import TranskunTranscriber
from src.core.vocal_separator import (
    _create_strict_onnx_session,
    _describe_onnx_session,
    _resolve_onnx_providers,
)
from src.core.yourmt3_transcriber import YourMT3Transcriber
from src.models.data_models import Config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_xpu_overlay_pins_safetensors_pread_release():
    requirements = (REPO_ROOT / "requirements-xpu.txt").read_text(encoding="utf-8")

    assert "safetensors==0.8.0" in requirements.splitlines()


class _FakeXpuTensor:
    def __init__(self, value: float = 1.0, index: int = 0):
        self._value = value
        self.device = SimpleNamespace(type="xpu", index=index)

    def add(self, value: float):
        return _FakeXpuTensor(self._value + value, self.device.index)

    def item(self):
        return self._value


class _FakeXpuRuntime:
    def __init__(self, *, available: bool = True, count: int = 1):
        self.available = available
        self.count = count
        self.synchronized = []

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def synchronize(self, index):
        self.synchronized.append(index)

    @staticmethod
    def get_device_name(index):
        return f"Intel Test XPU {index}"


class _FakeXpuTorch:
    def __init__(self, *, available: bool = True, count: int = 1):
        self.xpu = _FakeXpuRuntime(available=available, count=count)

    @staticmethod
    def ones(_size, *, device):
        return _FakeXpuTensor(index=int(str(device).split(":", 1)[1]))


def test_native_xpu_probe_executes_reads_back_and_synchronizes(monkeypatch):
    fake_torch = _FakeXpuTorch()
    monkeypatch.setattr(gpu_utils, "_get_torch", lambda: fake_torch)

    gpu_utils.ensure_xpu_runtime_compatibility("xpu:0")

    assert fake_torch.xpu.synchronized == [0]


def test_native_xpu_probe_rejects_missing_runtime_and_bad_index(monkeypatch):
    unavailable = _FakeXpuTorch(available=False)
    monkeypatch.setattr(gpu_utils, "_get_torch", lambda: unavailable)
    with pytest.raises(RuntimeError, match=r"torch\.xpu\.is_available\(\) 为 False"):
        gpu_utils.ensure_xpu_runtime_compatibility("xpu:0")

    one_device = _FakeXpuTorch(count=1)
    monkeypatch.setattr(gpu_utils, "_get_torch", lambda: one_device)
    with pytest.raises(RuntimeError, match="requested=1, count=1"):
        gpu_utils.ensure_xpu_runtime_compatibility("xpu:1")


def test_get_device_returns_only_a_probed_native_xpu(monkeypatch):
    fake_torch = _FakeXpuTorch()
    monkeypatch.setattr(gpu_utils, "_get_torch", lambda: fake_torch)
    monkeypatch.setattr(gpu_utils, "get_accelerator_type", lambda: "xpu")

    assert gpu_utils.get_device(prefer_gpu=True, gpu_index=0) == "xpu:0"
    assert fake_torch.xpu.synchronized == [0]


def test_xpu_cleanup_collects_then_synchronizes_before_emptying_cache(monkeypatch):
    events = []

    class CleanupXpu:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def synchronize():
            events.append("synchronize")

        @staticmethod
        def empty_cache():
            events.append("empty_cache")

    fake_torch = SimpleNamespace(xpu=CleanupXpu())
    monkeypatch.setattr(gpu_utils, "_get_torch", lambda: fake_torch)
    monkeypatch.setattr(gpu_utils, "get_accelerator_type", lambda: "xpu")
    monkeypatch.setattr(gpu_utils.gc, "collect", lambda: events.append("gc_collect"))

    gpu_utils.clear_gpu_memory()

    assert events == ["gc_collect", "synchronize", "empty_cache"]


class _FakeSessionOptions:
    def __init__(self):
        self.entries = {}

    def add_session_config_entry(self, key, value):
        self.entries[key] = value


class _FakeOrtSession:
    last = None

    def __init__(self, path, *, sess_options, providers):
        self.path = path
        self.sess_options = sess_options
        self.providers = providers
        self.fallback_disabled = False
        type(self).last = self

    @staticmethod
    def get_providers():
        return ["OpenVINOExecutionProvider"]

    def disable_fallback(self):
        self.fallback_disabled = True


class _FakeOpenVinoOrt:
    SessionOptions = _FakeSessionOptions
    InferenceSession = _FakeOrtSession

    @staticmethod
    def get_available_providers():
        return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]


def test_polarformer_xpu_uses_openvino_gpu_without_cpu_provider(tmp_path):
    with patch(
        "src.core.vocal_separator.initialize_openvino_gpu_runtime",
        return_value=("CPU", "GPU"),
    ) as initialize_openvino:
        providers = _resolve_onnx_providers("xpu:0", _FakeOpenVinoOrt)
    assert providers == [("OpenVINOExecutionProvider", {"device_type": "GPU.0"})]
    initialize_openvino.assert_called_once_with()

    session = _create_strict_onnx_session(
        _FakeOpenVinoOrt,
        tmp_path / "polarformer.onnx",
        providers,
        "xpu:0",
    )

    assert session.get_providers() == ["OpenVINOExecutionProvider"]
    assert session.sess_options.entries == {"session.disable_cpu_ep_fallback": "1"}
    assert session.providers == providers
    assert session.fallback_disabled
    assert _describe_onnx_session(session, "xpu:0") == (
        "ONNX Runtime · OpenVINOExecutionProvider " "(GPU.0; CPU fallback disabled)"
    )


def test_polarformer_cuda_keeps_required_auxiliary_cpu_nodes(tmp_path):
    class CudaSession(_FakeOrtSession):
        @staticmethod
        def get_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    fake_ort = SimpleNamespace(
        SessionOptions=_FakeSessionOptions,
        InferenceSession=CudaSession,
    )
    providers = [("CUDAExecutionProvider", {"device_id": 0})]

    session = _create_strict_onnx_session(
        fake_ort,
        tmp_path / "polarformer.onnx",
        providers,
        "cuda:0",
    )

    assert session.get_providers() == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert session.sess_options.entries == {}
    assert session.providers == providers
    assert not session.fallback_disabled


def test_polarformer_xpu_rejects_missing_openvino_and_cpu_append(tmp_path):
    missing = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="OpenVINOExecutionProvider"):
        _resolve_onnx_providers("xpu:0", missing)

    class CpuAppendingSession(_FakeOrtSession):
        @staticmethod
        def get_providers():
            return ["CPUExecutionProvider", "OpenVINOExecutionProvider"]

    fake_ort = SimpleNamespace(
        SessionOptions=_FakeSessionOptions,
        InferenceSession=CpuAppendingSession,
    )
    with pytest.raises(RuntimeError, match="unexpected provider|provider contract"):
        _create_strict_onnx_session(
            fake_ort,
            tmp_path / "polarformer.onnx",
            [("OpenVINOExecutionProvider", {"device_type": "GPU.0"})],
            "xpu:0",
        )


def test_polarformer_xpu_accepts_registered_cpu_provider_when_fallback_is_disabled(
    tmp_path,
):
    class OrtRegisteredCpuSession(_FakeOrtSession):
        @staticmethod
        def get_providers():
            return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

    fake_ort = SimpleNamespace(
        SessionOptions=_FakeSessionOptions,
        InferenceSession=OrtRegisteredCpuSession,
    )
    providers = [("OpenVINOExecutionProvider", {"device_type": "GPU.0"})]

    session = _create_strict_onnx_session(
        fake_ort,
        tmp_path / "polarformer.onnx",
        providers,
        "xpu:0",
    )

    assert session.get_providers() == [
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert session.sess_options.entries == {"session.disable_cpu_ep_fallback": "1"}
    assert session.fallback_disabled
    assert _describe_onnx_session(session, "xpu:0") == (
        "ONNX Runtime · OpenVINOExecutionProvider " "(GPU.0; CPU fallback disabled)"
    )


def test_polarformer_xpu_requires_runtime_fallback_disable_api(tmp_path):
    class NoRuntimeFallbackSession:
        def __init__(self, _path, *, sess_options, providers):
            self.sess_options = sess_options
            self.providers = providers

        @staticmethod
        def get_providers():
            return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

    fake_ort = SimpleNamespace(
        SessionOptions=_FakeSessionOptions,
        InferenceSession=NoRuntimeFallbackSession,
    )
    with pytest.raises(RuntimeError, match="disable_fallback"):
        _create_strict_onnx_session(
            fake_ort,
            tmp_path / "polarformer.onnx",
            [("OpenVINOExecutionProvider", {"device_type": "GPU.0"})],
            "xpu:0",
        )


def test_openvino_runtime_initialization_requires_a_real_gpu(monkeypatch):
    class FakeCore:
        available_devices = ["CPU", "GPU"]

    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=FakeCore))
    assert openvino_runtime.initialize_openvino_gpu_runtime() == ("CPU", "GPU")

    class CpuOnlyCore:
        available_devices = ["CPU"]

    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=CpuOnlyCore))
    with pytest.raises(RuntimeError, match="did not report an Intel GPU"):
        openvino_runtime.initialize_openvino_gpu_runtime()


@pytest.mark.parametrize(
    ("factory", "method_name", "patch_prefix"),
    (
        (
            lambda: TranskunTranscriber(Config(use_gpu=True)),
            "_resolve_runtime_device",
            "src.core.transkun_transcriber",
        ),
        (
            lambda: ByteDancePianoTranscriber(Config(use_gpu=True)),
            "_resolve_runtime_device",
            "src.core.bytedance_piano_transcriber",
        ),
        (
            lambda: AriaAmtTranscriber(config=Config(use_gpu=True)),
            "_resolve_runtime_device",
            "src.core.aria_amt_transcriber",
        ),
        (
            lambda: BeatThisTracker(Config(use_gpu=True)),
            "_target_device",
            "src.core.beat_this_tracker",
        ),
    ),
)
def test_backend_device_resolvers_preserve_explicit_xpu(
    factory,
    method_name,
    patch_prefix,
):
    with (
        patch(f"{patch_prefix}.get_device", return_value="xpu:0"),
        patch(f"{patch_prefix}.ensure_accelerator_runtime_compatibility") as ensure_runtime,
    ):
        instance = factory()
        assert getattr(instance, method_name)() == "xpu:0"

    ensure_runtime.assert_called_once_with("xpu:0")


def test_aria_xpu_route_replaces_both_upstream_cuda_hardcodings():
    source = inspect.getsource(
        __import__(
            "src.core.aria_amt_transcriber",
            fromlist=["_setup_aria_decoder_cache_on_xpu"],
        )
    )
    assert "def _setup_aria_decoder_cache_on_xpu" in source
    assert "block.attn.kv_cache = KVCache" in source
    assert "block.cross_attn.kv_cache = KVCache" in source
    assert "_process_aria_segments_on_device" in source
    assert 'if device.startswith("xpu")' in source


def test_aria_xpu_token_refinement_matches_pinned_math_without_cuda_literal():
    torch.manual_seed(7)
    logits = torch.randn(3, ARIA_AMT_TOKEN_VOCAB_SIZE)
    token_ids = torch.tensor([391, 1700, 3418], dtype=torch.long)
    column_indices = torch.arange(ARIA_AMT_TOKEN_VOCAB_SIZE).unsqueeze(0)
    interval_mask = (column_indices >= ARIA_AMT_ONSET_VELOCITY_TOKEN_START) & (
        column_indices <= ARIA_AMT_ONSET_VELOCITY_TOKEN_END
    )

    actual = _recalculate_aria_token_ids_on_device(
        logits,
        token_ids,
        column_indices=column_indices,
        interval_mask=interval_mask,
    )

    probabilities = torch.softmax(logits, dim=-1)
    expanded = token_ids.unsqueeze(1)
    beam_mask = (column_indices <= expanded + 2) & (column_indices >= expanded - 2)
    original_mask = torch.zeros_like(probabilities, dtype=torch.bool)
    original_mask.scatter_(1, expanded, 1)
    probabilities[~((interval_mask & beam_mask) | original_mask)] = 0
    expected = (
        ((probabilities * column_indices.float()).sum(dim=1) / (probabilities.sum(dim=1) + 1e-9))
        .round()
        .to(torch.long)
    )

    assert torch.equal(actual, expected)
    route_source = inspect.getsource(_process_aria_segments_on_device)
    assert "_recalculate_aria_token_ids_on_device" in route_source
    assert "transcribe_module.recalculate_tok_ids" not in route_source


def test_xpu_install_launch_build_and_packaging_are_isolated():
    requirements = (REPO_ROOT / "requirements-xpu.txt").read_text(encoding="utf-8")
    installer = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")
    launcher = (REPO_ROOT / "run.ps1").read_text(encoding="utf-8")
    spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")
    builder = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")
    validator = (REPO_ROOT / "tools" / "validate_accelerator_runtime.py").read_text(
        encoding="utf-8"
    )

    for expected in (
        "numpy==1.26.4",
        "torch==2.11.0+xpu",
        "torchaudio==2.11.0+xpu",
        "torchvision==0.26.0+xpu",
        "onnxruntime-openvino==1.24.1",
        "openvino==2025.4.1",
    ):
        assert expected in requirements
    assert "intel-extension-for-pytorch" not in requirements.lower()
    assert "onnxruntime-gpu" not in requirements.lower()
    assert "https://download.pytorch.org/whl/xpu" in installer
    assert (
        "expected = {'torch': '2.11.0', 'torchaudio': '2.11.0', " "'torchvision': '0.26.0'}"
    ) in installer
    assert (
        '& "$PIP" install "torch==2.11.0" "torchaudio==2.11.0" ' '"torchvision==0.26.0"'
    ) in installer
    assert runtime_validator.CUDA_PYTORCH_VERSIONS == {
        "torch": "2.7.0",
        "torchaudio": "2.7.0",
        "torchvision": "0.22.0",
    }
    assert runtime_validator.XPU_PYTORCH_VERSIONS == {
        "torch": "2.11.0",
        "torchaudio": "2.11.0",
        "torchvision": "0.26.0",
    }
    assert runtime_validator.XPU_RUNTIME_VERSIONS == {
        "onnxruntime-openvino": "1.24.1",
        "openvino": "2025.4.1",
    }
    assert "Panther Lake" in runtime_validator.XPU_HARDWARE_BASELINE
    assert "Arc B-Series" in runtime_validator.XPU_HARDWARE_BASELINE
    assert '"venv-xpu"' in installer
    assert "validate_accelerator_runtime.py" in launcher
    assert "collect_all('openvino')" in spec
    assert "MusicToMidi-XPU" in spec
    assert "_collect_xpu_library_bin_binaries" in spec
    assert '"ur_adapter_level_zero.dll"' in spec
    assert '"ur_adapter_opencl.dll"' in spec
    assert '"mkl_sycl_blas.5.dll"' in spec
    assert "_require_xpu_library_bin_binaries(a.binaries" in spec
    assert "pyinstaller-$Accelerator" in builder
    assert "Assert-XpuPortableRuntimeDlls" in builder
    assert "Get-FileHash -LiteralPath $packagedDll -Algorithm SHA256" in builder
    assert '"session.disable_cpu_ep_fallback", "1"' in validator
    assert "session.disable_fallback()" in validator
    assert '"device_type": "GPU.0"' in validator
    assert 'os.environ["PYTORCH_DEBUG_XPU_FALLBACK"] = "1"' in validator
    assert "sys.path.insert(0, str(REPO_ROOT))" in validator
    assert "torch.stft(" in validator


def test_xpu_runtime_gate_requires_native_bf16_autocast_convolution():
    validator = (REPO_ROOT / "tools" / "validate_accelerator_runtime.py").read_text(
        encoding="utf-8"
    )

    assert 'device_type="xpu"' in validator
    assert "dtype=torch.bfloat16" in validator
    assert "bf16_conv(bf16_input)" in validator
    assert "XPU BF16 autocast convolution" in validator


def test_xpu_wrappers_select_only_the_isolated_xpu_contract():
    for filename, target in (
        ("install_xpu.ps1", "install.ps1"),
        ("run_xpu.ps1", "run.ps1"),
        ("build_portable_xpu.ps1", "build_web_executables.ps1"),
    ):
        source = (REPO_ROOT / filename).read_text(encoding="utf-8")
        assert target in source
        assert "-Accelerator xpu" in source
        assert "fallback" not in source.lower()


def test_xpu_portable_build_does_not_mix_cuda_openmp_repair():
    builder = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")
    spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

    assert 'if ($Accelerator -eq "cuda" -and (Test-Path $TorchRuntimeRepair))' in builder
    assert "if torch_lib_dir and not IS_XPU:" in spec


def test_windows_xpu_openmp_contract_requires_intel_and_rejects_llvm(
    monkeypatch,
    tmp_path,
):
    torch_package = tmp_path / "torch"
    torch_lib = torch_package / "lib"
    torch_lib.mkdir(parents=True)
    fake_torch = SimpleNamespace(__file__=str(torch_package / "__init__.py"))
    monkeypatch.setattr(runtime_validator.sys, "platform", "win32")

    with pytest.raises(RuntimeError, match="Intel OpenMP runtime is missing"):
        runtime_validator._validate_windows_xpu_openmp_runtime(fake_torch)

    (torch_lib / "libiomp5md.dll").write_bytes(b"intel-openmp")
    runtime_validator._validate_windows_xpu_openmp_runtime(fake_torch)

    (torch_lib / "libomp140.x86_64.dll").write_bytes(b"llvm-openmp")
    with pytest.raises(RuntimeError, match="conflicting LLVM OpenMP runtime"):
        runtime_validator._validate_windows_xpu_openmp_runtime(fake_torch)


def test_torch_dll_guidance_matches_the_selected_accelerator(monkeypatch):
    monkeypatch.setenv("MUSIC_TO_MIDI_ACCELERATOR", "xpu")
    xpu_guidance = gpu_utils.get_torch_dll_troubleshooting(frozen=False)
    assert "Intel GPU" in xpu_guidance
    assert "libiomp5md.dll" in xpu_guidance
    assert "venv-xpu" in xpu_guidance

    monkeypatch.setenv("MUSIC_TO_MIDI_ACCELERATOR", "cuda")
    cuda_guidance = gpu_utils.get_torch_dll_troubleshooting(frozen=False)
    assert "NVIDIA" in cuda_guidance
    assert "CUDA 12.8" in cuda_guidance
    assert "libomp140.x86_64.dll" in cuda_guidance

    assert "get_torch_dll_troubleshooting" in inspect.getsource(gpu_utils._get_torch)
    assert "get_torch_dll_troubleshooting" in inspect.getsource(yourmt3_transcriber._import_torch)


def test_xpu_torch_import_failures_surface_the_intel_runtime_contract(
    monkeypatch,
    caplog,
):
    real_import = builtins.__import__

    def fail_torch_import(name, *args, **kwargs):
        if name == "torch":
            raise OSError('Error loading "C:\\torch\\c10.dll"')
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("MUSIC_TO_MIDI_ACCELERATOR", "xpu")
    monkeypatch.setattr(builtins, "__import__", fail_torch_import)
    monkeypatch.setattr(gpu_utils, "_fix_torch_dll_path", lambda: None)
    monkeypatch.setattr(gpu_utils, "_torch_module", None)
    monkeypatch.setattr(gpu_utils, "_torch_checked", False)
    caplog.set_level(logging.WARNING, logger="src.utils.gpu_utils")

    with pytest.raises(RuntimeError, match="libiomp5md.dll") as exc_info:
        yourmt3_transcriber._import_torch()
    assert "venv-xpu" in str(exc_info.value)

    assert gpu_utils._get_torch() is None
    assert "Intel GPU" in caplog.text
    assert "libiomp5md.dll" in caplog.text


def test_six_stem_xpu_job_releases_model_before_emptying_xpu_cache(tmp_path):
    class FakeSeparator:
        def __init__(self, **kwargs):
            self.output_dir = Path(kwargs["output_dir"])
            self.model_instance = None

        def load_model(self, _model_name):
            self.model_instance = SimpleNamespace(model_run=object())

        def separate(self, audio_path):
            source_stem = Path(audio_path).stem
            outputs = []
            for stem in STEM_KEYS:
                output = self.output_dir / f"{source_stem}_({stem})_test.wav"
                output.write_bytes(b"valid-test-wave")
                outputs.append(str(output))
            return outputs

    source = tmp_path / "song.wav"
    source.write_bytes(b"input")
    separator = SixStemSeparator(
        separator_cls=FakeSeparator,
        cache_dir=tmp_path / "models",
        target_device="xpu:0",
    )
    separator.is_model_available = lambda: True

    with (
        patch("src.utils.audio_separator_compat.ensure_accelerator_runtime_compatibility"),
        patch("src.utils.audio_separator_compat.ensure_module_on_device"),
        patch("src.core.multi_stem_separator.clear_gpu_memory") as clear_xpu,
    ):
        result = separator.separate(str(source), str(tmp_path / "outputs"))

    assert tuple(result) == STEM_KEYS
    assert all(Path(path).is_file() for path in result.values())
    clear_xpu.assert_called_once_with()


def test_yourmt3_xpu_moves_only_the_current_batch_to_the_device(monkeypatch):
    moved_devices = []
    clock = {"value": 0.0}
    monkeypatch.setattr(
        "time.time",
        lambda: (clock.__setitem__("value", clock["value"] + 0.1) or clock["value"]),
    )

    class FakeBatch:
        def __init__(self, size):
            self.shape = (size, 1, 32)

        def to(self, device):
            moved_devices.append(device)
            return self

    class FakeSegments:
        shape = (3, 1, 32)

        def __getitem__(self, item):
            start = int(item.start or 0)
            stop = min(int(item.stop), self.shape[0])
            return FakeBatch(stop - start)

    class FakeModel:
        @staticmethod
        def inference(batch, _task_tokens):
            return torch.zeros((batch.shape[0], 1, 4), dtype=torch.float32)

    with patch("src.core.yourmt3_transcriber.get_device", return_value="xpu:0"):
        transcriber = YourMT3Transcriber(Config())

    predictions = transcriber._inference_with_oom_retry(
        FakeModel(),
        bsz=1,
        audio_segments=FakeSegments(),
    )

    assert len(predictions) == 3
    assert moved_devices == ["xpu:0", "xpu:0", "xpu:0"]
