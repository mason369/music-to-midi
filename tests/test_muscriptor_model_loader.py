from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.core.muscriptor_model_loader import _load_safetensors_state_in_place


class _SingleCodebookModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Linear(3, 2, bias=False)
        self.linear = torch.nn.Linear(2, 4, bias=False)


def test_safetensors_are_strictly_remapped_and_loaded_in_place(tmp_path: Path):
    model = _SingleCodebookModel()
    original_parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    checkpoint = tmp_path / "model.safetensors"
    expected_embedding = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expected_linear = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    save_file(
        {
            "emb.0.weight": expected_embedding,
            "linears.0.weight": expected_linear,
        },
        checkpoint,
    )

    loaded = _load_safetensors_state_in_place(model, checkpoint, "cpu")

    assert loaded == 2
    assert torch.equal(model.emb.weight, expected_embedding)
    assert torch.equal(model.linear.weight, expected_linear)
    assert {
        name: id(parameter) for name, parameter in model.named_parameters()
    } == original_parameter_ids


def test_safetensors_loader_preserves_upstream_model_dtype(tmp_path: Path):
    model = _SingleCodebookModel().to(dtype=torch.float16)
    checkpoint = tmp_path / "model-fp16.safetensors"
    expected_embedding = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    expected_linear = torch.arange(8, dtype=torch.float16).reshape(4, 2)
    save_file(
        {
            "emb.0.weight": expected_embedding,
            "linears.0.weight": expected_linear,
        },
        checkpoint,
    )

    loaded = _load_safetensors_state_in_place(model, checkpoint, "cpu")

    assert loaded == 2
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float16}
    assert torch.equal(model.emb.weight, expected_embedding)
    assert torch.equal(model.linear.weight, expected_linear)


def test_safetensors_dtype_mismatch_fails_before_mutating_model(tmp_path: Path):
    model = _SingleCodebookModel()
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    checkpoint = tmp_path / "wrong-dtype.safetensors"
    save_file(
        {
            "emb.0.weight": torch.zeros_like(model.emb.weight, dtype=torch.float16),
            "linears.0.weight": torch.zeros_like(model.linear.weight, dtype=torch.float16),
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="checkpoint/model dtype mismatch"):
        _load_safetensors_state_in_place(model, checkpoint, "cpu")

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, original[name])


def test_safetensors_key_mismatch_fails_before_mutating_model(tmp_path: Path):
    model = _SingleCodebookModel()
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    checkpoint = tmp_path / "incomplete.safetensors"
    save_file(
        {"emb.0.weight": torch.zeros_like(model.emb.weight)},
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="checkpoint/model key mismatch"):
        _load_safetensors_state_in_place(model, checkpoint, "cpu")

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, original[name])


def test_safetensors_reject_multi_codebook_checkpoint(tmp_path: Path):
    model = _SingleCodebookModel()
    checkpoint = tmp_path / "multi-codebook.safetensors"
    save_file(
        {
            "emb.0.weight": torch.zeros_like(model.emb.weight),
            "emb.1.weight": torch.zeros_like(model.emb.weight),
            "linears.0.weight": torch.zeros_like(model.linear.weight),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="more than one codebook"):
        _load_safetensors_state_in_place(model, checkpoint, "cpu")
