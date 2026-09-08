"""Accompaniment selection, cancellation, precision, and short-clip contracts."""

from pathlib import Path
import ast
import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from src.core import vocal_separator
from src.i18n.translator import Translator


@pytest.fixture
def leap_runtime(monkeypatch):
    config = {
        "audio": {"sample_rate": 44100},
        "model": {"stft_hop_length": 512},
        "inference": {"dim_t": 1101},
        "training": {"target_instrument": "other"},
    }
    audio = np.linspace(-0.4, 0.4, 100, dtype=np.float32).reshape(2, 50)
    state = SimpleNamespace(
        events=[], config=config, audio=audio, runtime=None, error=None, malformed=None
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, batch):
            state.events.append("forward")
            if state.error:
                raise state.error
            return batch * 0.25

    class Runtime:
        def __init__(self, *, common_config, arch_config):
            state.events.append("load")
            state.runtime = self
            self.model_run = Model()
            self.common_config = common_config
            self.arch_config = arch_config

        def demix(self, mix):
            state.input_shape = mix.shape
            output = self.model_run(torch.from_numpy(mix)).numpy()
            if state.malformed == "shape":
                output = output[:, :-1]
            if state.malformed == "nan":
                output[:, 0] = np.nan
            return {"other": output, "vocals": np.full_like(output, 0.75)}

    # Unit tests exercise the adapter with a deterministic upstream interface.
    # Native DLL imports/inference are validated in separate real-runtime runs,
    # as in model_profile_runtime_probe, outside the Qt pytest process.
    mdxc_module = ModuleType("audio_separator.separator.architectures.mdxc_separator")
    mdxc_module.MDXCSeparator = Runtime
    uvr_module = ModuleType("audio_separator.separator.uvr_lib_v5")
    uvr_module.spec_utils = SimpleNamespace(normalize=lambda *, wave, **_kwargs: wave)
    monkeypatch.setitem(sys.modules, mdxc_module.__name__, mdxc_module)
    monkeypatch.setitem(sys.modules, uvr_module.__name__, uvr_module)
    monkeypatch.setattr(vocal_separator, "activate_audio_separator_runtime", lambda: None)
    monkeypatch.setattr(vocal_separator, "_load_yaml", lambda _: config)
    monkeypatch.setattr(vocal_separator, "_load_stereo_audio", lambda *_: audio.copy())
    monkeypatch.setattr(vocal_separator, "_resolve_torch_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(vocal_separator, "clear_gpu_memory", lambda: None)
    return state


def run_leg(state, *, callback=None, cancel_check=lambda: None):
    return vocal_separator._run_leap_accompaniment_leg(
        audio_path="original.wav",
        checkpoint_path=Path("bs_roformer_leap_inst.ckpt"),
        config_path=Path("bs_leap_inst_conf.yaml"),
        requested_device="cpu",
        progress_callback=callback,
        translate=Translator("en_US").t,
        cancel_check=cancel_check,
    )


def test_returns_predicted_instrumental_and_preserves_short_clip_length(leap_runtime):
    output, sr = run_leg(leap_runtime)
    np.testing.assert_allclose(output, leap_runtime.audio * 0.25, rtol=0, atol=1e-7)
    assert sr == 44100 and output.shape == (2, 50)
    assert leap_runtime.input_shape == (2, 563200)
    assert leap_runtime.runtime.arch_config == {
        "overlap": 8,
        "batch_size": 1,
        "pitch_shift": 0,
        "override_model_segment_size": False,
    }
    assert leap_runtime.runtime.common_config["model_data"]["model_type"] == "bs_roformer"
    assert "model_type" not in leap_runtime.config
    assert not leap_runtime.runtime.model_run._forward_pre_hooks
    assert not leap_runtime.runtime.model_run._forward_hooks


def test_reports_loading_and_chunk_before_work(leap_runtime):
    def report(_value, message):
        if "Loading Leap Instrumental" in message:
            leap_runtime.events.append("loading")
        elif "is processing accompaniment" in message:
            leap_runtime.events.append("running")

    run_leg(leap_runtime, callback=report)
    assert leap_runtime.events == ["loading", "load", "running", "forward"]


def test_cancel_after_inference_discards_result_and_removes_hooks(leap_runtime):
    def cancel_check():
        if "forward" in leap_runtime.events:
            raise InterruptedError("cancelled by user")

    with pytest.raises(InterruptedError, match="cancelled by user"):
        run_leg(leap_runtime, cancel_check=cancel_check)
    assert not leap_runtime.runtime.model_run._forward_pre_hooks
    assert not leap_runtime.runtime.model_run._forward_hooks


def test_cancel_before_loading_does_not_construct_model(leap_runtime):
    def cancel_check():
        raise InterruptedError("cancelled by user")

    with pytest.raises(InterruptedError):
        run_leg(leap_runtime, cancel_check=cancel_check)
    assert leap_runtime.runtime is None


def test_inference_failure_is_not_converted_to_success(leap_runtime):
    leap_runtime.error = RuntimeError("real model failure")
    with pytest.raises(RuntimeError, match="real model failure"):
        run_leg(leap_runtime)
    assert not leap_runtime.runtime.model_run._forward_hooks


@pytest.mark.parametrize("malformed", ["shape", "nan"])
def test_invalid_instrumental_output_is_rejected(leap_runtime, malformed):
    leap_runtime.malformed = malformed
    with pytest.raises(RuntimeError, match="输出"):
        run_leg(leap_runtime)


def test_vocal_target_cannot_be_mistaken_for_instrumental(leap_runtime):
    leap_runtime.config["training"]["target_instrument"] = "vocals"
    with pytest.raises(RuntimeError, match="other"):
        run_leg(leap_runtime)
    assert leap_runtime.runtime is None


def test_portable_manifest_excludes_retired_accompaniment_cache(tmp_path):
    source = Path("MusicToMidi.spec").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = []
    for node in tree.body:
        segment = ast.get_source_segment(source, node) or ""
        if (isinstance(node, ast.FunctionDef) and node.name == "_collect_tree") or (
            isinstance(node, (ast.Assign, ast.AugAssign))
            and "retired_accompaniment_assets" in segment
        ):
            nodes.append(node)
    for name in (
        "bs_roformer_leap_inst.ckpt",
        "bs_leap_inst_conf.yaml",
        "bs_polarformer_fp16.onnx",
        "model_bs_polarformer_float16.yaml",
        "bs_polarformer.onnx",
        "model_bs_polarformer.yaml",
    ):
        (tmp_path / name).write_bytes(b"manifest fixture")
    namespace = {"os": os, "audio_separator_models_dir": str(tmp_path), "datas": []}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "MusicToMidi.spec", "exec"), namespace)
    assert {Path(source).name for source, _ in namespace["datas"]} == {
        "bs_roformer_leap_inst.ckpt",
        "bs_leap_inst_conf.yaml",
    }
