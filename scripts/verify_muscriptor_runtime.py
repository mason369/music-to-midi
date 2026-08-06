"""Run strict real-model MuScriptor precision and deterministic-MIDI checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import mido
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.beat_detector import BeatDetector
from src.core.muscriptor_transcriber import MuscriptorTranscriber
from src.models.data_models import Config
from src.utils.muscriptor_source_patch import (
    MUSCRIPTOR_QUALITY_PATCH_COMMIT,
    MUSCRIPTOR_SOURCE_COMMIT,
    validate_muscriptor_runtime_identity,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_note_ons(path: Path) -> int:
    return sum(
        1
        for track in mido.MidiFile(str(path)).tracks
        for message in track
        if (not message.is_meta and message.type == "note_on" and int(message.velocity) > 0)
    )


def _verified_note_track_tempo_maps(path: Path) -> dict[int, list[tuple[int, int]]]:
    midi = mido.MidiFile(str(path))
    canonical: list[tuple[int, int]] | None = None
    result: dict[int, list[tuple[int, int]]] = {}
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        tempos: list[tuple[int, int]] = []
        has_notes = False
        for message in track:
            tick += int(message.time)
            if message.is_meta and message.type == "set_tempo":
                tempos.append((tick, int(message.tempo)))
            elif not message.is_meta and message.type in {"note_on", "note_off"}:
                has_notes = True
        if track_index == 0:
            canonical = tempos
        if has_notes:
            result[track_index] = tempos
    if not canonical:
        raise RuntimeError(f"MuScriptor MIDI has no conductor tempo map: {path}")
    if not result:
        raise RuntimeError(f"MuScriptor MIDI has no note-bearing tracks: {path}")
    mismatches = {index: tempos for index, tempos in result.items() if tempos != canonical}
    if mismatches:
        raise RuntimeError(
            "MuScriptor note tracks do not repeat the conductor tempo map: "
            f"canonical={canonical!r}, mismatches={mismatches!r}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", choices=("small", "medium", "large"), default="large")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    if args.repeat < 1:
        raise RuntimeError("--repeat must be at least 1")
    source = args.audio.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    identity_error = validate_muscriptor_runtime_identity()
    if identity_error:
        raise RuntimeError(identity_error)
    if not torch.cuda.is_available():
        raise RuntimeError("Real MuScriptor verification requires CUDA")

    config = Config(
        use_gpu=True,
        gpu_device=0,
        multi_instrument_model="muscriptor",
        transcription_backend="muscriptor",
        muscriptor_model=args.model,
        muscriptor_instruments=[],
    )
    transcriber = MuscriptorTranscriber(config)
    transcriber.set_beat_info(BeatDetector(config).detect(str(source)))
    model = transcriber.load_model()
    inner_model = getattr(model, "_model", None)
    if inner_model is None:
        raise RuntimeError("MuScriptor runtime no longer exposes its loaded torch model")

    parameter_dtypes = Counter(str(parameter.dtype) for parameter in inner_model.parameters())
    if set(parameter_dtypes) != {"torch.float32"}:
        raise RuntimeError(
            "MuScriptor parameters do not match the official CUDA default "
            f"(FP32 parameters with FP16 autocast compute): {dict(parameter_dtypes)!r}"
        )
    autocast = getattr(inner_model, "autocast", None)
    if not getattr(autocast, "enabled", False) or getattr(autocast, "dtype", None) != torch.float16:
        raise RuntimeError(
            "MuScriptor CUDA autocast is not the expected enabled FP16 policy: "
            f"enabled={getattr(autocast, 'enabled', None)!r}, "
            f"dtype={getattr(autocast, 'dtype', None)!r}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    for index in range(args.repeat):
        output = (args.output_dir / f"{source.stem}-{args.model}-{index + 1}.mid").resolve()
        started = time.perf_counter()
        transcriber.transcribe_to_midi(str(source), str(output))
        note_track_tempos = _verified_note_track_tempo_maps(output)
        runs.append(
            {
                "output": str(output),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "sha256": _sha256(output),
                "note_ons": _count_note_ons(output),
                "bytes": output.stat().st_size,
                "onset_delay_seconds": transcriber.last_onset_delay_seconds,
                "bar_offset_seconds": transcriber.last_bar_offset_seconds,
                "note_track_tempo_maps": note_track_tempos,
            }
        )

    hashes = {str(run["sha256"]) for run in runs}
    if len(hashes) != 1:
        raise RuntimeError(f"Repeated MuScriptor MIDI hashes differ: {runs!r}")
    note_counts = {run["note_ons"] for run in runs}
    if len(note_counts) != 1:
        raise RuntimeError(f"Repeated MuScriptor note counts differ: {runs!r}")

    print(
        json.dumps(
            {
                "source_commit": MUSCRIPTOR_SOURCE_COMMIT,
                "quality_patch_commit": MUSCRIPTOR_QUALITY_PATCH_COMMIT,
                "model": args.model,
                "gpu": torch.cuda.get_device_name(0),
                "parameter_dtypes": dict(parameter_dtypes),
                "precision_policy": "official_cuda_default",
                "autocast_enabled": True,
                "autocast_dtype": str(torch.float16),
                "quality_mode": "overlap_restart",
                "overlap_seconds": 2.5,
                "allow_reset": True,
                "onset_phase_correction": "official_v0.3.0",
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    transcriber.unload_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
