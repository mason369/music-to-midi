"""Memory-bounded loading for the pinned MuScriptor safetensors checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _remap_checkpoint_key(key: str) -> str:
    """Apply the pinned upstream single-codebook compatibility mapping."""

    if key.startswith(("emb.1.", "linears.1.")):
        raise ValueError(
            "Checkpoint has more than one codebook (n_q > 1); "
            "only single-stream models are supported."
        )
    if key.startswith("emb.0."):
        return "emb." + key[len("emb.0.") :]
    if key.startswith("linears.0."):
        return "linear." + key[len("linears.0.") :]
    return key


def _load_safetensors_state_in_place(model, weights_path: str | Path, device) -> int:
    """Copy one verified tensor at a time into an upstream-created model.

    MuScriptor builds the destination model directly on its target device. Its
    upstream ``load_file(..., device='cuda')`` then materializes a second full
    device-side state dict before ``load_state_dict`` copies it into that model.
    The large checkpoint therefore needs roughly two complete GPU allocations
    during loading. This loader does not choose a precision: it preserves the
    dtype created by MuScriptor's official model builder and strictly requires
    every checkpoint tensor to match it while copying one CPU-mapped tensor at
    a time into the already allocated destination storage.
    """

    import torch
    from safetensors import safe_open

    checkpoint = Path(weights_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MuScriptor checkpoint does not exist: {checkpoint}")
    target_device = torch.device(device)
    destination = model.state_dict()

    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        source_keys = list(source.keys())
        key_map = {key: _remap_checkpoint_key(key) for key in source_keys}
        if len(set(key_map.values())) != len(key_map):
            raise RuntimeError("MuScriptor checkpoint key remapping produced duplicates")

        missing = sorted(set(destination) - set(key_map.values()))
        unexpected = sorted(set(key_map.values()) - set(destination))
        if missing or unexpected:
            raise RuntimeError(
                "MuScriptor checkpoint/model key mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

        # Validate the complete checkpoint before mutating any parameter.
        for source_key, target_key in key_map.items():
            target = destination[target_key]
            tensor_slice = source.get_slice(source_key)
            source_shape = tuple(int(value) for value in tensor_slice.get_shape())
            if source_shape != tuple(target.shape):
                raise RuntimeError(
                    "MuScriptor checkpoint tensor shape mismatch: "
                    f"key={source_key}, checkpoint={source_shape}, model={tuple(target.shape)}"
                )
            source_dtype = tensor_slice.get_dtype()
            expected_source_dtype = {
                torch.bool: "BOOL",
                torch.uint8: "U8",
                torch.int8: "I8",
                torch.int16: "I16",
                torch.int32: "I32",
                torch.int64: "I64",
                torch.float16: "F16",
                torch.bfloat16: "BF16",
                torch.float32: "F32",
                torch.float64: "F64",
            }.get(target.dtype)
            if expected_source_dtype is None:
                raise RuntimeError(
                    "MuScriptor model uses an unsupported destination dtype: "
                    f"key={target_key}, model={target.dtype}"
                )
            if source_dtype != expected_source_dtype:
                raise RuntimeError(
                    "MuScriptor checkpoint/model dtype mismatch: "
                    f"key={source_key}, checkpoint={source_dtype}, model={target.dtype}"
                )
            # Upstream _build_model(device) constructs the transformer on the
            # requested accelerator but leaves a small conditioner projection
            # on CPU until its final model.to(device) call after state loading.
            if target.device.type != "cpu" and target.device != target_device:
                raise RuntimeError(
                    "MuScriptor model parameter is on the wrong device before loading: "
                    f"key={target_key}, expected={target_device} or cpu, "
                    f"actual={target.device}"
                )

        with torch.no_grad():
            for source_key, target_key in key_map.items():
                # Windows can invalidate a large safetensors memory-mapped view
                # while torch is still consuming it for a CUDA transfer. Keep
                # only one small owned CPU tensor alive and synchronize before
                # releasing its storage; the largest pinned tensor is ~38 MiB.
                mapped_tensor = source.get_tensor(source_key)
                tensor = mapped_tensor.clone()
                del mapped_tensor
                target = destination[target_key]
                target.copy_(tensor, non_blocking=False)
                if target.is_cuda:
                    torch.cuda.synchronize(target.device)
                del tensor

    return len(source_keys)


def load_muscriptor_model_memory_bounded(weights_path: str | Path, device: str):
    """Build the pinned official model and load its exact weights in place."""

    import torch
    import muscriptor.transcription_model as upstream

    checkpoint = Path(weights_path).resolve()
    target_device = torch.device(device)
    config = upstream._resolve_config(checkpoint, checkpoint)
    model = upstream._build_model(target_device, config)
    model.eval()
    tensor_count = _load_safetensors_state_in_place(model, checkpoint, target_device)
    model.to(target_device)
    wrong_devices = sorted(
        name for name, tensor in model.state_dict().items() if tensor.device != target_device
    )
    if wrong_devices:
        raise RuntimeError(
            "MuScriptor model did not finish moving to the requested device: "
            f"device={target_device}, keys={wrong_devices}"
        )

    tokenizer = upstream.MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS",
        max_shift_steps=1001,
    )
    logger.info(
        "Loaded %s MuScriptor tensors in place on %s using the upstream dtype policy",
        tensor_count,
        target_device,
    )
    return upstream.TranscriptionModel(
        model=model,
        tokenizer=tokenizer,
        device=target_device,
    )


__all__ = ["load_muscriptor_model_memory_bounded"]
