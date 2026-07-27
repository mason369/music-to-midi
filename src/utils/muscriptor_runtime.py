"""Verified runtime optimizations for the pinned MuScriptor implementation."""

from __future__ import annotations

from types import MethodType
from typing import Any


def configure_muscriptor_kv_cache(
    model: Any,
    *,
    compute_dtype: str,
    torch_module=None,
) -> int:
    """Store and reuse MuScriptor KV caches in the verified compute dtype.

    MuScriptor's published parameters remain FP32. Its Q/K/V projections run
    under FP16 autocast, but the upstream cache allocator derives its dtype from
    the FP32 projection weights. That stores already-FP16 K/V values as FP32 and
    recasts them for every attention step. Keeping those exact FP16 values in a
    persistent cache removes the redundant conversion, halves cache storage,
    and avoids allocating every layer's full cache for every five-second chunk.
    """

    if torch_module is None:
        import torch

        torch_module = torch

    if str(compute_dtype) != "float16":
        raise ValueError(
            "MuScriptor KV-cache optimization requires verified FP16 compute, "
            f"got {compute_dtype!r}"
        )

    from muscriptor.modules.transformer import StreamingMultiheadAttention

    cache_dtype = torch_module.float16
    configured = 0

    for module in model.modules():
        if not isinstance(module, StreamingMultiheadAttention):
            continue

        def init_optimized_state(self, batch_size: int, sequence_length: int):
            weight = self.in_proj_weight
            shape = (
                2,
                int(batch_size),
                int(sequence_length),
                int(self.num_heads),
                int(self.dim_per_head),
            )
            cache = getattr(self, "_music_to_midi_kv_cache", None)
            if (
                cache is None
                or tuple(cache.shape) != shape
                or cache.device != weight.device
                or cache.dtype != cache_dtype
            ):
                cache = torch_module.empty(
                    shape,
                    device=weight.device,
                    dtype=cache_dtype,
                )
                self._music_to_midi_kv_cache = cache
            return {
                "cache": cache,
                "offset": torch_module.zeros(
                    1,
                    dtype=torch_module.long,
                    device=weight.device,
                ),
            }

        module.init_state = MethodType(init_optimized_state, module)
        configured += 1

    if configured <= 0:
        raise RuntimeError(
            "Pinned MuScriptor runtime exposes no StreamingMultiheadAttention "
            "layers; KV-cache optimization cannot be verified"
        )
    return configured
