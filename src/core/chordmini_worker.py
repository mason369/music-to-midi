"""Subprocess-only official BTC inference, using TelkNet's fixed default recipe."""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
import sys
import types
from pathlib import Path


def run_chord_worker(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from src.utils.chordmini_runtime import (
        CHECKPOINT_SHA256,
        CHORDMINI_COMMIT,
        RECIPE_ID,
        get_chordmini_runtime,
    )

    source = get_chordmini_runtime()
    import numpy as np
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("ChordMini requires exactly one assigned CUDA device")
    # Both applications are named src. Only this disposable worker may replace
    # that namespace; explicit __path__ also bypasses the frozen app's src loader.
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    package = types.ModuleType("src")
    package.__path__ = [str(source / "src")]
    package.__spec__ = importlib.machinery.ModuleSpec("src", loader=None, is_package=True)
    sys.modules["src"] = package
    sys.path[:0] = [str(source), str(source / "src")]
    from mir_eval.chord import encode

    from src.evaluation.test import _extract_song_features_root_compatible
    from src.evaluation.utils.inference import predict_sliding_windows
    from src.models import load_model
    from src.utils import HParams, extract_model_state_dict, idx2voca_chord, set_random_seed

    set_random_seed(42, include_python_random=True)
    config = HParams.load(str(source / "config/ChordMini.yaml"))
    feature, step, duration = _extract_song_features_root_compatible(str(args.audio), config)
    vocabulary = idx2voca_chord()
    checkpoint = source / "checkpoints/btc_model_best.pth"
    model, mean, std = load_model(str(checkpoint), "BTC", config, torch.device("cuda:0"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not {"mean", "std"} <= state.get("normalization", state).keys():
        raise RuntimeError("Checkpoint normalization missing")
    declared = state.get("idx_to_chord")
    if declared is None:
        raise RuntimeError("Checkpoint vocabulary missing")
    for index, label in vocabulary.items():
        saved = (
            declared.get(index, declared.get(str(index)))
            if isinstance(declared, dict)
            else declared[index]
        )
        left, right = encode(label), encode(saved)
        if left[0] != right[0] or not np.array_equal(left[1], right[1]) or left[2] != right[2]:
            raise RuntimeError(f"Checkpoint vocabulary mismatch at {index}")
    model.load_state_dict(extract_model_state_dict(state), strict=True)
    if mean is None or std is None or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("Invalid normalization")
    model.eval()
    model.idx_to_chord = vocabulary
    seq_len = int(state.get("timestep") or getattr(model, "timestep", 108))
    with torch.inference_mode():
        values = predict_sliding_windows(
            model,
            feature,
            mean,
            std,
            seq_len,
            16,
            "BTC",
            170,
            vote_aggregation="logit",
            use_overlap=True,
            overlap_ratio=0.5,
            smooth_logits=True,
            smooth_predictions=True,
            kernel_size=9,
            use_gaussian=True,
        )
    count = int(np.floor(duration / step))
    if count > 0:
        values = values[:count]
    values = np.asarray(values).reshape(-1)
    boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    segments = [
        {
            "start": float(left * step),
            "end": float(right * step),
            "label": vocabulary[int(values[left])],
        }
        for left, right in zip(boundaries[:-1], boundaries[1:])
        if right > left
    ]
    torch.cuda.synchronize()
    record = {
        "id": "chordmini_btc",
        "name": "ChordMini BTC",
        "recipe": RECIPE_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_revision": CHORDMINI_COMMIT,
        "device": torch.cuda.get_device_name(),
        "visible_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "torch_version": torch.__version__,
        "duration": float(duration),
        "frame_seconds": float(step),
        "segments": segments,
    }
    args.output.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(run_chord_worker())
