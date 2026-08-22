"""Replay the published TelkNet v10 real-song tempo acceptance table.

The validator runs the pinned local Beat This final0 checkpoint four times per
source by default, proves byte-identical raw timestamp arrays, applies this
project's three-state tempo implementation, and compares the published BPM,
tempo-event count, isolated-point repair count, and SMF-quantized phase error.
Missing source audio is an explicit non-zero outcome, never a skipped pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.beat_this_tracker import BeatThisTracker, analyze_beat_this_grid
from src.core.telknet_beat_grid_v10 import normalize_beat_grid
from src.core.telknet_tempo_map import tempo_map_max_phase_error
from src.models.data_models import Config, TempoMode
from src.utils.runtime_paths import get_ffmpeg_executable


@dataclass(frozen=True)
class BenchmarkSong:
    id: int
    source: str
    relative_path: str | None
    expected_bpm: float | None
    expected_bpm_range: tuple[float, float] | None
    expected_tempo_events: int
    expected_isolated_repairs: int
    expected_phase_error: float
    source_kind: str = "music"


SONGS = (
    BenchmarkSong(
        12,
        "Fairy Tail - Masayume Chasing",
        "Fairy Tail - Masayume Chasing.mp3",
        129.002138,
        None,
        1,
        4,
        0.15967224887070586,
    ),
    BenchmarkSong(
        13,
        "【Refrain】色づ世界の明日から",
        "【Refrain】色づ世界の明日から.mp3",
        78.000974,
        None,
        1,
        0,
        0.051549554679354515,
    ),
    BenchmarkSong(
        14,
        "Far far away - 小林愛香",
        "apps/web/public/tool-assets/music-demos/far-far-away/source.mp3",
        121.967626,
        None,
        1,
        0,
        0.09070322441627354,
        "telknet",
    ),
    BenchmarkSong(
        15,
        "Far far away_缩混",
        "Far far away_缩混.mp3",
        121.977009,
        None,
        1,
        1,
        0.19942426854453288,
    ),
    BenchmarkSong(
        16,
        "Moon Halo AmaLee Ver",
        "Honkai Impact 3rd - Moon Halo  AmaLee Ver.mp3",
        135.766160,
        None,
        1,
        2,
        0.3430360957242655,
    ),
    BenchmarkSong(
        17, "July - 바람에 쓰는 편지", None, 87.010441, None, 1, 0, 0.04311079916237759, "july"
    ),
    BenchmarkSong(
        18,
        "Project_1",
        "Project_1/Project_1.flac",
        None,
        (157.697, 161.290),
        5,
        0,
        0.05265789473685345,
    ),
    BenchmarkSong(
        19, "Refrain - Aimer", "Refrain - Aimer.mp3", 78.102252, None, 1, 1, 0.23205731661599316
    ),
    BenchmarkSong(20, "群青", "群青.mp3", 135.021212, None, 1, 10, 0.10829817158935384),
    BenchmarkSong(
        21,
        "だから僕は音楽を辞めた",
        "apps/web/public/tool-assets/music-demos/dakara-boku-wa-ongaku-o-yameta/source.mp3",
        124.994987,
        None,
        1,
        0,
        0.05674567048384915,
        "telknet",
    ),
    BenchmarkSong(22, "群青钢琴", "群青钢琴.mp3", 140.003118, None, 1, 4, 0.08640523424855928),
    BenchmarkSong(
        23,
        "Moon Halo",
        "Honkai Impact 3rd-Moon Halo.mp3",
        135.763230,
        None,
        1,
        7,
        0.29206735664538996,
    ),
    BenchmarkSong(24, "群青缩混", "群青缩混.mp3", 135.024465, None, 1, 26, 0.334581559262233),
)


def _source_path(
    song: BenchmarkSong,
    *,
    music_dir: Path,
    telknet_root: Path | None,
    july_source: Path | None,
) -> Path | None:
    if song.source_kind == "july":
        return july_source.resolve() if july_source is not None else None
    if song.source_kind == "telknet":
        if telknet_root is None:
            return None
        return (telknet_root / str(song.relative_path)).resolve()
    return (music_dir / str(song.relative_path)).resolve()


def _array_digest(beats, downbeats) -> str:
    digest = hashlib.sha256()
    for values in (beats, downbeats):
        array = np.asarray(values, dtype=np.float64)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _smf_bpm(bpm: float) -> float:
    tempo_microseconds = round(60_000_000.0 / float(bpm))
    return 60_000_000.0 / tempo_microseconds


def _published_tempo_map(info) -> tuple[tuple[float, float], ...]:
    source = info.tempo_map or [(0.0, info.bpm)]
    return tuple((float(start), _smf_bpm(float(bpm))) for start, bpm in source)


def _prepare_muscriptor_audio_input(source: Path, temp_dir: Path) -> Path:
    """Mirror the production MuScriptor/Beat This shared PCM input contract."""
    with source.open("rb") as stream:
        if stream.read(4) == b"RIFF":
            return source

    normalized = temp_dir / ".muscriptor-input.wav"
    process = subprocess.run(
        [
            get_ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "pcm_s16le",
            str(normalized),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0 or not normalized.is_file() or normalized.stat().st_size <= 44:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(
            "MuScriptor input audio normalization failed: "
            f"{detail or f'ffmpeg exited with code {process.returncode}'}"
        )
    with normalized.open("rb") as stream:
        if stream.read(4) != b"RIFF":
            raise RuntimeError("MuScriptor input normalization did not produce RIFF WAV")
    return normalized


def _validate_song(
    tracker: BeatThisTracker,
    song: BenchmarkSong,
    source: Path,
    *,
    runs: int,
) -> dict:
    raw_runs = []
    run_seconds = []
    with tempfile.TemporaryDirectory(prefix="telknet-tempo-audio-") as temp_dir:
        detector_input = _prepare_muscriptor_audio_input(source, Path(temp_dir))
        for run_index in range(runs):
            started = time.perf_counter()
            beats, downbeats = tracker.track_raw(str(detector_input))
            run_seconds.append(time.perf_counter() - started)
            raw_runs.append((np.asarray(beats), np.asarray(downbeats)))
            print(
                f"[{song.id}] {song.source}: raw run {run_index + 1}/{runs} "
                f"completed in {run_seconds[-1]:.3f}s",
                file=sys.stderr,
                flush=True,
            )

    digests = [_array_digest(beats, downbeats) for beats, downbeats in raw_runs]
    deterministic = len(set(digests)) == 1
    beats, downbeats = raw_runs[0]
    normalized = normalize_beat_grid(beats.tolist())
    mode = TempoMode.ADAPTIVE.value if song.id == 18 else TempoMode.FIXED_AUTO.value
    info = analyze_beat_this_grid(
        beats.tolist(),
        downbeats.tolist(),
        tempo_mode=mode,
    )
    tempo_map = _published_tempo_map(info)
    phase_error = tempo_map_max_phase_error(info.beat_times, tempo_map)
    observed_bpms = [bpm for _start, bpm in tempo_map]

    checks = {
        "raw_runs_byte_identical": deterministic,
        "tempo_event_count": len(tempo_map) == song.expected_tempo_events,
        "isolated_repairs": (
            normalized.isolated_phase_outliers_repaired == song.expected_isolated_repairs
        ),
        "phase_error": math.isclose(
            phase_error,
            song.expected_phase_error,
            rel_tol=0.0,
            abs_tol=5e-6,
        ),
    }
    if song.expected_bpm is not None:
        checks["bpm"] = math.isclose(
            float(info.bpm),
            song.expected_bpm,
            rel_tol=0.0,
            # The public table stores display-rounded six-decimal BPM values.
            abs_tol=5e-4,
        )
    else:
        assert song.expected_bpm_range is not None
        checks["bpm_range"] = math.isclose(
            min(observed_bpms),
            song.expected_bpm_range[0],
            rel_tol=0.0,
            abs_tol=1e-3,
        ) and math.isclose(
            max(observed_bpms),
            song.expected_bpm_range[1],
            rel_tol=0.0,
            abs_tol=1e-3,
        )

    return {
        "id": song.id,
        "source": song.source,
        "path": str(source),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "runs": runs,
        "run_seconds": run_seconds,
        "raw_digest": digests[0],
        "beat_count": len(info.beat_times),
        "downbeat_count": len(info.downbeats or []),
        "bpm": float(info.bpm),
        "tempo_bpms": observed_bpms,
        "tempo_events": len(tempo_map),
        "isolated_repairs": normalized.isolated_phase_outliers_repaired,
        "phase_error_beats": phase_error,
        "expected_phase_error_beats": song.expected_phase_error,
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--music-dir", type=Path, default=Path.home() / "Music")
    parser.add_argument("--telknet-root", type=Path)
    parser.add_argument("--july-source", type=Path)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--ids", type=int, nargs="*")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    selected = [song for song in SONGS if not args.ids or song.id in set(args.ids)]
    missing_ids = sorted(set(args.ids or ()) - {song.id for song in selected})
    if missing_ids:
        parser.error(f"unknown benchmark ids: {missing_ids}")

    config = Config(
        use_gpu=True,
        gpu_device=0,
        tempo_mode=TempoMode.FIXED_AUTO.value,
    )
    wrapper = BeatThisTracker(config)
    wrapper._load_tracker()
    results = []
    for song in selected:
        source = _source_path(
            song,
            music_dir=args.music_dir.resolve(),
            telknet_root=args.telknet_root.resolve() if args.telknet_root else None,
            july_source=args.july_source,
        )
        if source is None or not source.is_file() or source.stat().st_size <= 0:
            results.append(
                {
                    "id": song.id,
                    "source": song.source,
                    "path": str(source) if source is not None else None,
                    "status": "missing_source",
                }
            )
            print(
                f"[{song.id}] {song.source}: source audio is missing",
                file=sys.stderr,
                flush=True,
            )
            continue
        results.append(_validate_song(wrapper, song, source, runs=args.runs))

    summary = {
        "songs_requested": len(selected),
        "songs_passed": sum(item["status"] == "passed" for item in results),
        "songs_failed": sum(item["status"] == "failed" for item in results),
        "songs_missing": sum(item["status"] == "missing_source" for item in results),
        "raw_detection_runs": sum(item.get("runs", 0) for item in results),
    }
    print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))
    if summary["songs_failed"]:
        return 1
    if summary["songs_missing"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
