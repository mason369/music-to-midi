"""Run one real, file-backed end-to-end alignment check."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from mido import MidiFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pipeline import MusicToMidiPipeline
from src.models.data_models import Config


def _midi_stats(path: str) -> dict:
    midi_path = Path(path)
    midi = MidiFile(str(midi_path))
    note_on = 0
    cc64 = 0
    for track in midi.tracks:
        for message in track:
            if message.is_meta:
                continue
            if message.type == "note_on" and message.velocity > 0:
                note_on += 1
            elif message.type == "control_change" and message.control == 64:
                cc64 += 1
    return {
        "path": str(midi_path.resolve()),
        "bytes": midi_path.stat().st_size,
        "tracks": len(midi.tracks),
        "note_on": note_on,
        "cc64": cc64,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--backend", default="yourmt3")
    parser.add_argument("--yourmt3-model", default="yptf_moe_multi_nops")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--discard-separated", action="store_true")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    output_dir = Path(args.output_dir)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Input audio does not exist: {audio_path.resolve()}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory must be empty: {output_dir.resolve()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        processing_mode=args.mode,
        transcription_backend=args.backend,
        multi_instrument_model=args.backend,
        yourmt3_model=args.yourmt3_model,
        vocal_split_merge_midi=args.merge,
        save_separated_tracks=not args.discard_separated,
        output_dir=str(output_dir),
    )

    started = time.time()

    def on_progress(progress) -> None:
        print(
            f"[{progress.overall_progress * 100:6.2f}%] "
            f"{progress.stage.value}: {progress.message}",
            flush=True,
        )

    try:
        result = MusicToMidiPipeline(config).process(
            str(audio_path),
            str(output_dir),
            progress_callback=on_progress,
        )
        midi_paths = []
        for value in (
            result.midi_path,
            result.merged_midi_path,
            result.vocal_midi_path,
            result.accompaniment_midi_path,
        ):
            if value and value not in midi_paths:
                midi_paths.append(value)
        for value in (result.stem_midi_paths or {}).values():
            if value not in midi_paths:
                midi_paths.append(value)

        payload = {
            "status": "PASS",
            "mode": args.mode,
            "backend": args.backend,
            "yourmt3_model": args.yourmt3_model,
            "wall_seconds": round(time.time() - started, 3),
            "pipeline_seconds": round(result.processing_time, 3),
            "total_notes": result.total_notes,
            "midi": [_midi_stats(path) for path in midi_paths],
            "separated_audio": {
                name: {
                    "path": str(Path(path).resolve()),
                    "bytes": Path(path).stat().st_size,
                }
                for name, path in (result.separated_audio or {}).items()
            },
        }
        print("ALIGNMENT_RESULT=" + json.dumps(payload, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        payload = {
            "status": "FAIL",
            "mode": args.mode,
            "backend": args.backend,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        print("ALIGNMENT_RESULT=" + json.dumps(payload, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
