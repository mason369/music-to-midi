"""Model identity, complete chord vocabulary, native timing and worker contracts."""

import copy
import json
import os
import sys
import time
from pathlib import Path

import pytest

from src.core.chord_analysis import chord_worker_command, run_chord_process
from src.core.midi_chords import QUALITIES, chord_pitches, read_chord_record, split_chords_at_bars
from src.utils.chordmini_runtime import CHECKPOINT_SHA256, CHORDMINI_COMMIT, RECIPE_ID

ROOT = Path(__file__).resolve().parents[1]


def chord_record():
    return dict(
        id="chordmini_btc",
        source_revision=CHORDMINI_COMMIT,
        checkpoint_sha256=CHECKPOINT_SHA256,
        recipe=RECIPE_ID,
        duration=4.02,
        frame_seconds=0.05,
        segments=[
            dict(start=0.0, end=1.95, label="C:maj7"),
            dict(start=1.95, end=3.1, label="D:min7"),
            dict(start=3.1, end=4.0, label="N"),
        ],
    )


@pytest.mark.parametrize("root", ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"])
@pytest.mark.parametrize("quality,intervals", list(QUALITIES.items()))
def test_entire_170_class_vocabulary_keeps_its_intervals(root, quality, intervals):
    notes = chord_pitches(f"{root}:{quality}")
    assert tuple(note - notes[0] for note in notes) == intervals
    assert chord_pitches("N") == chord_pitches("X") == ()


def test_slash_bass_and_unrecognized_labels_are_never_simplified():
    assert chord_pitches("C:maj7/3") == (52, 60, 64, 67, 71)
    assert chord_pitches("Bb:min/5") == (53, 70, 73, 77)
    for invalid in ("H:maj", "C:invented", "Cmaj7", "", "C/9"):
        with pytest.raises(ValueError):
            chord_pitches(invalid)


def test_source_boundaries_are_lossless_and_fractional_tail_is_not_fabricated():
    original = chord_record()
    snapshot = copy.deepcopy(original)
    chords = read_chord_record(original)
    bars = split_chords_at_bars(chords, 120)
    assert original == snapshot
    assert [(c.start, c.end) for c in bars] == [(0.0, 1.95), (1.95, 2.0), (2.0, 3.1), (3.1, 4.0)]
    assert bars[-1].pitches == ()
    assert bars[-1].end < original["duration"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "omnizart"),
        ("checkpoint_sha256", "0" * 64),
        ("source_revision", "master"),
        ("recipe", "legacy"),
        ("duration", float("nan")),
        ("frame_seconds", 0),
        ("segments", []),
    ],
)
def test_wrong_or_missing_model_output_is_rejected(field, value):
    record = chord_record()
    record[field] = value
    with pytest.raises(ValueError):
        read_chord_record(record)


@pytest.mark.parametrize(
    "bad_start,bad_end", [(2.0, 3.1), (1.9, 3.1), (1.95, 1.95), (1.95, float("inf"))]
)
def test_gaps_overlaps_empty_and_nonfinite_intervals_fail(bad_start, bad_end):
    record = chord_record()
    record["segments"][1].update(start=bad_start, end=bad_end)
    with pytest.raises(ValueError):
        read_chord_record(record)


def test_worker_cancellation_reaps_the_real_child(tmp_path):
    pidfile = tmp_path / "pid.txt"
    command = [
        sys.executable,
        "-c",
        "import os,time,pathlib; pathlib.Path("
        + repr(str(pidfile))
        + ").write_text(str(os.getpid())); time.sleep(120)",
    ]
    start = time.monotonic()
    with pytest.raises(InterruptedError):
        run_chord_process(
            command, env=os.environ.copy(), log_path=tmp_path / "log", cancel_check=pidfile.exists
        )
    assert time.monotonic() - start < 5
    import psutil

    assert not psutil.pid_exists(int(pidfile.read_text()))


def test_failed_worker_reports_original_error(tmp_path):
    with pytest.raises(RuntimeError, match="actual model failure"):
        run_chord_process(
            [sys.executable, "-c", "raise RuntimeError('actual model failure')"],
            env=os.environ.copy(),
            log_path=tmp_path / "log",
            cancel_check=lambda: False,
        )


def test_frozen_worker_dispatch_and_all_resource_delivery_contracts(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert chord_worker_command("in.wav", "out.json")[1] == "--chordmini-worker"
    main = (ROOT / "src/main.py").read_text(encoding="utf-8")
    assert main.index('"--chordmini-worker" in sys.argv') < main.index(
        '"--self-test-gui-conversion" in sys.argv'
    )
    spec = (ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")
    portable = (ROOT / "build_portable.ps1").read_text(encoding="utf-8")
    assert "resources/chordmini-source.json" in spec
    assert '_collect_tree(chordmini_source_dir, "models/chordmini")' in spec
    assert "validate_chordmini(chordmini_source_dir)" in spec
    assert "validate_chordmini(models_root / 'chordmini')" in portable
    assert "get_chordmini_runtime" in portable and "Staged ChordMini BTC identity" in portable
    manifest = json.loads((ROOT / "resources/chordmini-source.json").read_text())
    assert manifest["checkpoints/btc_model_best.pth"] == CHECKPOINT_SHA256
    assert {
        "LICENSE",
        "README.md",
        "config/ChordMini.yaml",
        "src/evaluation/utils/inference.py",
    } <= manifest.keys()
    assert not any("2e1d_model_best" in name or "large_voca.pt" in name for name in manifest)


def test_corrupt_source_is_rejected_even_after_a_successful_validation(tmp_path, monkeypatch):
    import hashlib

    from src.utils import chordmini_runtime as runtime

    code = tmp_path / "official.py"
    code.write_bytes(b"correct")
    monkeypatch.setattr(
        runtime, "source_manifest", lambda: {"official.py": hashlib.sha256(b"correct").hexdigest()}
    )
    runtime.validate_chordmini(tmp_path)
    code.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        runtime.validate_chordmini(tmp_path)
