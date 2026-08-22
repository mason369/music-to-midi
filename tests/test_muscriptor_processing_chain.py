"""Contracts for the explicit official/TelkNet MuScriptor processing switch."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.muscriptor_transcriber import MuscriptorTranscriber
from src.models.data_models import Config, MuscriptorProcessingChain
from src.web_api.app import _capabilities
from src.web_api.engine import InferenceEngine
from src.web_api.schemas import InferenceOptions, ManualMidiOptions


def test_config_defaults_to_official_and_round_trips_fix_chain():
    default = Config()
    default.validate()
    assert default.muscriptor_processing_chain == MuscriptorProcessingChain.OFFICIAL.value
    assert default.to_dict()["muscriptor_processing_chain"] == "official"

    fix_chain = Config.from_dict(
        {
            **default.to_dict(),
            "muscriptor_processing_chain": MuscriptorProcessingChain.TELKNET.value,
        }
    )
    fix_chain.validate()
    assert fix_chain.muscriptor_processing_chain == MuscriptorProcessingChain.TELKNET.value

    with pytest.raises(ValueError, match="invalid muscriptor_processing_chain"):
        Config(muscriptor_processing_chain="silent-fallback")


def test_transcriber_reloads_the_selected_processing_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    loaded_chains: list[str] = []

    def fake_loader(_weights, _device, *, processing_chain):
        loaded_chains.append(processing_chain)
        return object()

    monkeypatch.setattr(
        MuscriptorTranscriber,
        "_runtime_unavailable_reason",
        staticmethod(lambda: ""),
    )
    monkeypatch.setattr(
        "src.core.muscriptor_transcriber.get_cached_muscriptor_paths",
        lambda *_args, **_kwargs: (weights, config_path),
    )
    monkeypatch.setattr(
        "src.core.muscriptor_transcriber.get_device",
        lambda *_args, **_kwargs: "cpu",
    )
    monkeypatch.setattr(
        "src.core.muscriptor_transcriber.ensure_accelerator_runtime_compatibility",
        lambda _device: None,
    )
    monkeypatch.setattr(
        "src.core.muscriptor_transcriber.load_muscriptor_model_memory_bounded",
        fake_loader,
    )
    monkeypatch.setattr("src.core.muscriptor_transcriber.clear_gpu_memory", lambda: None)

    config = Config(
        use_gpu=False,
        muscriptor_processing_chain=MuscriptorProcessingChain.OFFICIAL.value,
    )
    transcriber = MuscriptorTranscriber(config)
    official_model = transcriber.load_model()
    assert transcriber._runtime_details["processing_chain"] == "official"
    assert transcriber._runtime_details["quality_mode"] == "official_v0.3.0"
    assert transcriber._runtime_details["boundary_recovery"] == "official_v0.3.0"

    config.muscriptor_processing_chain = MuscriptorProcessingChain.TELKNET.value
    telknet_model = transcriber.load_model()
    assert telknet_model is not official_model
    assert loaded_chains == ["official", "telknet"]
    assert transcriber._runtime_details["processing_chain"] == "telknet"
    assert transcriber._runtime_details["boundary_recovery"] == (
        "telknet_issue74_single_program_v1"
    )


def test_web_api_accepts_both_chains_and_defaults_to_telknet():
    assert InferenceOptions().muscriptor_processing_chain == "official"
    assert ManualMidiOptions(route="muscriptor").muscriptor_processing_chain == "official"

    primary = InferenceOptions(muscriptor_processing_chain="telknet", use_gpu=False)
    primary_config = InferenceEngine._primary_config(primary.model_dump())
    assert primary_config.muscriptor_processing_chain == "telknet"

    manual = ManualMidiOptions(
        route="muscriptor",
        muscriptor_processing_chain="telknet",
        use_gpu=False,
    )
    manual_config = InferenceEngine._manual_config(manual.model_dump())
    assert manual_config.muscriptor_processing_chain == "telknet"

    with pytest.raises(ValidationError, match="unsupported MuScriptor processing chain"):
        InferenceOptions(muscriptor_processing_chain="unknown")

    capabilities = _capabilities()
    assert [item["id"] for item in capabilities["muscriptor_processing_chains"]] == [
        "official",
        "telknet",
    ]
    assert capabilities["muscriptor_processing_chains"][0]["label_zh"] == ("官方处理链路（默认）")
    assert capabilities["muscriptor_processing_chains"][0]["label_en"] == (
        "Official processing path (default)"
    )
    assert capabilities["muscriptor_processing_chains"][1]["label_zh"] == ("分段边界连续性修复链路")
    assert capabilities["muscriptor_processing_chains"][1]["label_en"] == (
        "Segment-boundary continuity fix path"
    )


def test_all_frontends_submit_the_same_processing_chain_field():
    space_source = Path("space/app.py").read_text(encoding="utf-8")
    colab_source = Path("colab_notebook.ipynb").read_text(encoding="utf-8")
    web_source = Path("web/app.js").read_text(encoding="utf-8")
    web_capabilities_source = Path("src/web_api/app.py").read_text(encoding="utf-8")

    for source in (space_source, colab_source, web_source):
        assert "muscriptor_processing_chain" in source
    for source in (space_source, colab_source):
        assert "telknet" in source
    for source in (space_source, colab_source, web_capabilities_source):
        assert "official" in source.lower()
    assert "muscriptorProcessingChainSelect" in web_source
    assert "muscriptor_processing_chain=MuscriptorProcessingChain.OFFICIAL.value" in space_source
    assert "muscriptor_processing_chain=MuscriptorProcessingChain.OFFICIAL.value" in colab_source
    assert 'muscriptorProcessingChain.value || "official"' in web_source
