"""Run one real full-chain conversion matrix case.

This helper is intentionally file-backed because Windows multiprocessing cannot
re-import a stdin-based ``__main__`` module.
"""
import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

from mido import MidiFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pipeline import MusicToMidiPipeline
from src.models.data_models import Config


def midi_stats(path: str) -> dict:
    midi_path = Path(path)
    midi_file = MidiFile(str(midi_path))
    note_on = 0
    tracks_with_notes = 0

    for track in midi_file.tracks:
        track_note_count = sum(
            1
            for message in track
            if not message.is_meta
            and message.type == "note_on"
            and getattr(message, "velocity", 0) > 0
        )
        if track_note_count:
            tracks_with_notes += 1
            note_on += track_note_count

    return {
        "path": str(midi_path),
        "exists": midi_path.exists(),
        "bytes": midi_path.stat().st_size if midi_path.exists() else 0,
        "tracks": len(midi_file.tracks),
        "tracks_with_notes": tracks_with_notes,
        "note_on": note_on,
    }


def run_case(case: dict, mix_audio: Path, piano_audio: Path, root: Path) -> dict:
    audio_path = piano_audio if case.get("audio") == "piano" else mix_audio
    output_dir = root / case["name"]
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        processing_mode=case["mode"],
        transcription_backend=case["backend"],
        multi_instrument_model=case.get("multi_model", "yourmt3"),
        midi_track_mode=case.get("track_mode", "multi_track"),
        vocal_split_merge_midi=case.get("merge", False),
        transcription_quality="fast",
        output_dir=str(output_dir),
        save_separated_tracks=True,
    )

    progress_tail = []

    def progress(progress_item) -> None:
        progress_tail.append(getattr(progress_item, "message", str(progress_item)))
        del progress_tail[:-8]

    result = MusicToMidiPipeline(config).process(str(audio_path), str(output_dir), progress_callback=progress)

    midi_paths = []
    for attr in ("midi_path", "merged_midi_path", "vocal_midi_path", "accompaniment_midi_path"):
        value = getattr(result, attr, None)
        if value:
            midi_paths.append(value)
    midi_paths.extend((getattr(result, "stem_midi_paths", None) or {}).values())

    unique_paths = []
    for path in midi_paths:
        if path not in unique_paths:
            unique_paths.append(path)

    stats = [midi_stats(path) for path in unique_paths]
    if not stats:
        raise RuntimeError("no MIDI paths returned")
    if any(item["bytes"] <= 0 for item in stats):
        raise RuntimeError("empty MIDI output")

    return {
        "case": case["name"],
        "status": "PASS",
        "audio": str(audio_path),
        "mode": case["mode"],
        "backend": config.transcription_backend,
        "multi_model": config.multi_instrument_model,
        "track_mode": config.midi_track_mode,
        "merge": config.vocal_split_merge_midi,
        "total_notes": getattr(result, "total_notes", None),
        "midi": stats,
        "progress_tail": progress_tail,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--mix-audio", required=True)
    parser.add_argument("--piano-audio", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    case = json.loads(args.case_json)
    try:
        result = run_case(case, Path(args.mix_audio), Path(args.piano_audio), Path(args.root))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "case": case.get("name"),
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
