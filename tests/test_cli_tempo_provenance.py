"""A CLI-created stem inherits the verified original mixture as its beat source."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cli import app


class TrackEngine:
    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        path = kwargs["output_dir"] / "piano.wav"
        path.write_bytes(b"separated audio")
        return SimpleNamespace(
            result={"manual_midi_required": True, "mode": "six_stem_split"},
            artifacts=(SimpleNamespace(id="piano", kind="audio_track", path=path),),
        )


def separate(tmp_path, name="original"):
    source = tmp_path / f"{name}.wav"
    source.write_bytes(name.encode())
    engine = TrackEngine()
    assert app.main(
        [str(source), "--mode", "six_stem_split", "-o", str(tmp_path / "split")],
        engine_factory=lambda: engine, stdout=io.StringIO(), stderr=io.StringIO(),
        install_signal_handlers=False,
    ) == 0
    return source, tmp_path / "split" / name / "piano.wav"


def manual(tmp_path, tracks, *extra):
    engine, text, error = TrackEngine(), io.StringIO(), io.StringIO()
    code = app.main(
        ["track-to-midi", *map(str, tracks), "--route", "piano_transkun",
         "-o", str(tmp_path / "midi"), *extra],
        engine_factory=lambda: engine, stdout=text, stderr=error,
        install_signal_handlers=False,
    )
    return code, engine, text.getvalue(), error.getvalue()


def test_stem_uses_verified_original_without_requiring_a_repeated_option(tmp_path):
    original, stem = separate(tmp_path)
    code, engine, _, error = manual(tmp_path, [stem])
    assert code == 0, error
    assert engine.calls[0]["tempo_source_path"] == original
    manifest = json.loads((tmp_path / "midi/piano" / app.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["tempo_source"]["path"] == str(original)


def test_batch_stems_resolve_each_original_independently(tmp_path):
    first_source, first_stem = separate(tmp_path, "first")
    second_source, second_stem = separate(tmp_path, "second")
    code, engine, _, error = manual(tmp_path, [first_stem, second_stem])
    assert code == 0, error
    assert [call["tempo_source_path"] for call in engine.calls] == [first_source, second_source]


@pytest.mark.parametrize("damage", ["missing_original", "changed_original", "changed_stem", "broken_manifest"])
def test_unverifiable_stem_origin_stops_before_loading_a_model(tmp_path, damage):
    original, stem = separate(tmp_path)
    if damage == "missing_original":
        original.unlink()
    elif damage == "changed_original":
        original.write_bytes(b"changed")
    elif damage == "changed_stem":
        stem.write_bytes(b"edited stem")
    else:
        (stem.parent / app.MANIFEST_NAME).write_text("null", encoding="utf-8")
    code, engine, _, error = manual(tmp_path, [stem], "--dry-run")
    assert code == 2
    assert error.strip()
    assert not engine.calls
    assert not (tmp_path / "midi").exists()


def test_explicit_tempo_source_takes_precedence_for_an_edited_stem(tmp_path):
    original, stem = separate(tmp_path)
    stem.write_bytes(b"user edited this stem")
    replacement = tmp_path / "new_reference.wav"
    replacement.write_bytes(b"new reference")
    code, engine, _, error = manual(tmp_path, [stem], "--tempo-source", str(replacement))
    assert code == 0, error
    assert engine.calls[0]["tempo_source_path"] == replacement


def test_dry_run_displays_the_original_that_will_actually_be_used(tmp_path):
    original, stem = separate(tmp_path)
    code, engine, text, error = manual(tmp_path, [stem], "--dry-run", "--json")
    assert code == 0, error
    planned = json.loads(text.splitlines()[0])
    assert planned["tempo_source"] == str(original)
    assert not engine.calls


def test_standalone_audio_has_no_invented_original(tmp_path):
    source = tmp_path / "standalone.wav"
    source.write_bytes(b"standalone")
    code, engine, _, error = manual(tmp_path, [source])
    assert code == 0, error
    assert engine.calls[0]["tempo_source_path"] is None
