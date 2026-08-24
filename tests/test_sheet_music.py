from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import mido
import pytest
from PyQt6.QtWidgets import QApplication

import src.core.sheet_music as sheet_music
import src.gui.web.sheet_music_export as browser_sheet_music
import src.gui.workers.sheet_music_export_worker as sheet_music_worker
import src.utils.musescore_runtime as musescore_runtime
from src.core.sheet_music import SheetMusicExportResult
from src.gui.widgets.muscriptor_result import MuscriptorResultWidget


def _write_test_midi(path: Path) -> Path:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(97), time=0))
    conductor.append(
        mido.MetaMessage(
            "time_signature",
            numerator=6,
            denominator=8,
            time=0,
        )
    )
    midi.tracks.append(conductor)
    notes = mido.MidiTrack()
    notes.append(mido.Message("program_change", program=24, channel=0, time=0))
    notes.append(mido.Message("note_on", note=64, velocity=91, channel=0, time=17))
    notes.append(mido.Message("note_off", note=64, velocity=0, channel=0, time=91))
    midi.tracks.append(notes)
    midi.save(path)
    return path


def _fake_project_xml() -> str:
    strings = "".join("<string/>" for _ in range(6))
    return (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<museScore><Score>"
        f"<Part><Instrument><StringData>{strings}</StringData></Instrument>"
        "<Staff><StaffType group='pitched'><name>stdNormal</name></StaffType></Staff></Part>"
        "<Part><Instrument/><Staff><StaffType group='pitched'>"
        "<name>stdNormal</name></StaffType></Staff></Part>"
        "</Score></museScore>"
    )


def _fake_musescore_run(_binary, arguments, *, cancel_check):
    assert not cancel_check()
    if "--score-parts-pdf" in arguments:
        encoded = base64.b64encode(b"%PDF-1.4\npart\n").decode("ascii")
        payload = {
            "parts": ["Acoustic Guitar, Guitar", "Acoustic Grand Piano"],
            "partsBin": [encoded, encoded],
        }
        return subprocess.CompletedProcess(arguments, 0, json.dumps(payload), "")
    output = Path(arguments[arguments.index("-o") + 1])
    if output.suffix == ".mscx":
        output.write_text(_fake_project_xml(), encoding="utf-8")
    elif output.suffix == ".musicxml":
        output.write_text(
            "<?xml version='1.0'?><score-partwise version='4.0'/>",
            encoding="utf-8",
        )
    elif output.suffix == ".pdf":
        output.write_bytes(b"%PDF-1.4\nfull score\n")
    else:
        raise AssertionError(arguments)
    return subprocess.CompletedProcess(arguments, 0, "", "")


def _patch_fake_musescore(monkeypatch, tmp_path: Path) -> Path:
    executable = tmp_path / "MuseScore4.exe"
    executable.write_bytes(b"fake executable")
    monkeypatch.setattr(sheet_music, "get_musescore_executable", lambda: executable)
    monkeypatch.setattr(
        sheet_music,
        "validate_musescore_runtime",
        lambda path: Path(path).resolve(),
    )
    monkeypatch.setattr(sheet_music, "musescore_version", lambda _path: (4, 7, 4))
    monkeypatch.setattr(sheet_music, "_run_musescore", _fake_musescore_run)
    return executable


def test_sheet_export_quantizes_only_a_copy_and_publishes_complete_zip(
    tmp_path: Path,
    monkeypatch,
):
    _patch_fake_musescore(monkeypatch, tmp_path)
    source = _write_test_midi(tmp_path / "source.mid")
    source_bytes = source.read_bytes()
    destination = tmp_path / "source_sheet_music.zip"

    result = sheet_music.export_sheet_music_zip(source, destination, quantize_grid="1/32")

    assert source.read_bytes() == source_bytes
    assert result.path == destination.resolve()
    assert result.quantize_grid == "1/32"
    assert result.grid_ticks == 60
    assert result.paired_note_count == 1
    assert result.musescore_version == "4.7.4"
    assert result.members == (
        "score.mid",
        "score.musicxml",
        "full_score.pdf",
        "01_acoustic_guitar.pdf",
        "02_acoustic_grand_piano.pdf",
        "01_acoustic_guitar_tab.pdf",
        "sheet_music_manifest.json",
    )

    with zipfile.ZipFile(destination) as archive:
        assert tuple(archive.namelist()) == result.members
        manifest = json.loads(archive.read("sheet_music_manifest.json"))
        assert manifest["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
        assert manifest["quantize_grid"] == "1/32"
        assert manifest["files"] == list(result.members[:-1])
        quantized = mido.MidiFile(file=io.BytesIO(archive.read("score.mid")))
        absolute = 0
        note_ticks = []
        for message in quantized.tracks[1]:
            absolute += message.time
            if message.type in {"note_on", "note_off"}:
                note_ticks.append(absolute)
        assert note_ticks == [0, 120]
        assert archive.read("full_score.pdf").startswith(b"%PDF-")
        assert archive.read("score.musicxml").startswith(b"<?xml")


def test_sheet_export_failure_does_not_publish_or_change_source(tmp_path: Path, monkeypatch):
    _patch_fake_musescore(monkeypatch, tmp_path)
    source = _write_test_midi(tmp_path / "source.mid")
    source_bytes = source.read_bytes()
    destination = tmp_path / "failed.zip"

    monkeypatch.setattr(
        sheet_music,
        "_run_musescore",
        lambda _binary, arguments, *, cancel_check: subprocess.CompletedProcess(
            arguments, 3, "", "explicit fake failure"
        ),
    )

    with pytest.raises(sheet_music.SheetMusicExportError, match="failed to import"):
        sheet_music.export_sheet_music_zip(source, destination)

    assert source.read_bytes() == source_bytes
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.part.zip"))


def test_sheet_export_rejects_nonzero_process_even_when_it_wrote_output(
    tmp_path: Path,
    monkeypatch,
):
    _patch_fake_musescore(monkeypatch, tmp_path)
    source = _write_test_midi(tmp_path / "source.mid")
    destination = tmp_path / "nonzero-with-output.zip"

    def nonzero_import(_binary, arguments, *, cancel_check):
        completed = _fake_musescore_run(_binary, arguments, cancel_check=cancel_check)
        return subprocess.CompletedProcess(arguments, 9, completed.stdout, "explicit nonzero")

    monkeypatch.setattr(sheet_music, "_run_musescore", nonzero_import)

    with pytest.raises(sheet_music.SheetMusicExportError, match="returncode=9"):
        sheet_music.export_sheet_music_zip(source, destination)

    assert not destination.exists()


def test_sheet_export_rejects_nonzero_part_pdf_process_with_valid_json(
    tmp_path: Path,
    monkeypatch,
):
    _patch_fake_musescore(monkeypatch, tmp_path)
    source = _write_test_midi(tmp_path / "source.mid")
    destination = tmp_path / "nonzero-parts.zip"

    def nonzero_parts(_binary, arguments, *, cancel_check):
        completed = _fake_musescore_run(_binary, arguments, cancel_check=cancel_check)
        if "--score-parts-pdf" in arguments:
            return subprocess.CompletedProcess(
                arguments,
                11,
                completed.stdout,
                "explicit part failure",
            )
        return completed

    monkeypatch.setattr(sheet_music, "_run_musescore", nonzero_parts)

    with pytest.raises(sheet_music.SheetMusicExportError, match="returncode=11"):
        sheet_music.export_sheet_music_zip(source, destination)

    assert not destination.exists()


def test_pinned_musescore_distribution_requires_exact_upstream_license(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "MuseScore 4"
    executable = root / "bin" / "MuseScore4.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"runtime")
    license_bytes = b"GPL-3.0 test license"
    (root / "LICENSE.txt").write_bytes(license_bytes)
    monkeypatch.setattr(musescore_runtime, "musescore_version", lambda _path: (4, 7, 4))
    monkeypatch.setattr(musescore_runtime, "MUSESCORE_LICENSE_BYTES", len(license_bytes))
    monkeypatch.setattr(
        musescore_runtime,
        "MUSESCORE_LICENSE_SHA256",
        hashlib.sha256(license_bytes).hexdigest(),
    )

    assert (
        musescore_runtime.validate_pinned_musescore_distribution(executable) == executable.resolve()
    )
    (root / "LICENSE.txt").write_bytes(b"changed")
    with pytest.raises(musescore_runtime.MuseScoreRuntimeError, match="license size mismatch"):
        musescore_runtime.validate_pinned_musescore_distribution(executable)


def test_browser_sheet_registry_is_request_scoped_and_caches_identical_edit(
    tmp_path: Path,
    monkeypatch,
):
    request_dir = tmp_path / "request-000001"
    request_dir.mkdir()
    source = _write_test_midi(request_dir / "song.mid")
    calls = []

    def fake_export(midi_path, destination, *, quantize_grid):
        calls.append((Path(midi_path), Path(destination), quantize_grid))
        Path(destination).write_bytes(b"PK\x03\x04fake score archive")
        return SheetMusicExportResult(
            path=Path(destination),
            members=("score.mid", "full_score.pdf", "sheet_music_manifest.json"),
            quantize_grid=quantize_grid,
            grid_ticks=60,
            paired_note_count=1,
            musescore_version="4.7.4",
        )

    monkeypatch.setattr(browser_sheet_music, "export_sheet_music_zip", fake_export)
    registry = browser_sheet_music.SheetMusicExportRegistry()
    token = registry.register(request_dir=request_dir, source_midi_path=source)
    assert (
        registry.require_matching(
            token,
            request_dir=request_dir,
            source_midi_path=source,
        )
        == token
    )
    payload = json.dumps(
        {
            "token": token,
            "midi_base64": base64.b64encode(source.read_bytes()).decode("ascii"),
            "quantize_grid": "1/32",
        }
    )

    first = registry.render(payload)
    second = registry.render(payload)

    assert first == second
    assert first["filename"] == "song_sheet_music.zip"
    assert first["memberCount"] == 3
    assert first["quantizeGrid"] == "1/32"
    assert first["musescoreVersion"] == "4.7.4"
    assert first["url"].startswith("/gradio_api/file=")
    assert len(calls) == 1

    archive_path = next(request_dir.rglob("sheet_music.zip"))
    archive_path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed after export"):
        registry.render(payload)
    assert len(calls) == 1

    outside = _write_test_midi(tmp_path / "outside.mid")
    with pytest.raises(RuntimeError, match="does not match this MIDI result"):
        registry.require_matching(
            token,
            request_dir=request_dir,
            source_midi_path=outside,
        )
    with pytest.raises(RuntimeError, match="outside its request"):
        registry.register(request_dir=request_dir, source_midi_path=outside)


def test_desktop_sheet_action_exports_current_project_tempo_snapshot(
    tmp_path: Path,
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"desktop sheet export test audio placeholder")
    source_midi = _write_test_midi(tmp_path / "source.mid")
    source_bytes = source_midi.read_bytes()
    destination = tmp_path / "desktop_sheet_music.zip"
    calls = []

    def fake_export(midi_path, output_path, *, quantize_grid, cancel_check):
        snapshot = Path(midi_path)
        assert snapshot.name == "current-project-tempo.mid"
        assert snapshot.is_file()
        assert not cancel_check()
        Path(output_path).write_bytes(b"PK\x03\x04verified desktop score")
        calls.append((quantize_grid, snapshot.read_bytes()))
        return SheetMusicExportResult(
            path=Path(output_path).resolve(),
            members=("score.mid", "score.musicxml", "full_score.pdf"),
            quantize_grid=quantize_grid,
            grid_ticks=60,
            paired_note_count=1,
            musescore_version="4.7.4",
        )

    monkeypatch.setattr(sheet_music_worker, "export_sheet_music_zip", fake_export)
    monkeypatch.setattr(
        "src.gui.widgets.muscriptor_result.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), "ZIP (*.zip)"),
    )
    widget = MuscriptorResultWidget(str(source_audio), ["acoustic_guitar"])
    try:
        widget._midi_path = str(source_midi)
        widget.set_detected_bpm(97.0)
        widget._start_sheet_music_export()
        worker = widget._sheet_export_worker
        assert worker is not None
        assert not widget.download_sheet_music_action.isEnabled()
        assert worker.wait(5_000)
        for _ in range(10):
            app.processEvents()

        assert destination.is_file()
        assert calls and calls[0][0] == "1/32"
        assert calls[0][1] != b""
        assert source_midi.read_bytes() == source_bytes
        assert widget._sheet_export_worker is None
        assert widget.download_sheet_music_action.isEnabled()
        assert not any(widget._sheet_export_root.iterdir())
    finally:
        widget.shutdown()
        widget.close()
        app.processEvents()
