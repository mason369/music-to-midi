"""Non-destructive MIDI to MusicXML/PDF sheet-music export.

The engraving flow is adapted from MuScriptor's MIT-licensed ``utils/sheets.py``
at commit e34b397bf0584e67bfd81dc591c390e6dcb03350.  The project-specific layer
adds atomic ZIP publication, strict output verification, cancellation, and a
sheet-only copy passed through the existing MIDI quantization contract.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import mido

from src.core.midi_quantization import (
    DEFAULT_MIDI_QUANTIZE_GRID,
    MidiQuantizationReport,
    quantize_midi_notes,
)
from src.core.sheet_note_durations import restore_note_durations
from src.core.sheet_tablature import assign_score_tablature
from src.core.sheet_triplets import prepare_triplet_import, restore_triplet_musicxml
from src.utils.midi_output import validate_midi_output
from src.utils.musescore_runtime import (
    get_musescore_executable,
    get_musescore_subprocess_env,
    musescore_version,
    validate_musescore_runtime,
)

SHEET_MUSIC_ZIP_NAME = "sheet_music.zip"
SHEET_MUSIC_MANIFEST_NAME = "sheet_music_manifest.json"
SHEET_MUSIC_SCHEMA_VERSION = 2
MUSESCORE_RUN_TIMEOUT_SECONDS = 120.0


def quantization_export_options(strength=1.0, triplet=False, automatic=False):
    """Validate notation settings and retain only nondefaults for stable caches."""
    strength = float(strength)
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Quantization strength must be between zero and one")
    if not isinstance(triplet, bool) or not isinstance(automatic, bool):
        raise ValueError("Triplet and automatic quantization settings must be boolean")
    values = dict(quantize_strength=strength, quantize_triplet=triplet, quantize_auto=automatic)
    defaults = dict(quantize_strength=1.0, quantize_triplet=False, quantize_auto=False)
    return {key: value for key, value in values.items() if value != defaults[key]}


_IMPORT_OPTIONS = """<?xml version="1.0" encoding="UTF-8"?>
<MidiOptions>
  <QuantValue>3</QuantValue>
  <HumanPerformance>false</HumanPerformance>
  <RecognizePickupBar>false</RecognizePickupBar>
  <Duplets>false</Duplets>
  <Triplets>false</Triplets>
  <Quadruplets>false</Quadruplets>
  <Quintuplets>false</Quintuplets>
  <Septuplets>false</Septuplets>
  <Nonuplets>false</Nonuplets>
  <SimplifyDurations>false</SimplifyDurations>
  <DottedNotes>true</DottedNotes>
</MidiOptions>
"""
_TAB_PRESETS = {count: f"tab{count}StrCommon" for count in range(4, 10)}
_TAB_LINE_DISTANCE = "1.5"


def _midi_import_options(grid: str, strength: float, triplet: bool) -> str:
    """Engrave the existing music grid without a second beat reconstruction.

    MuseScore's QuantValue is a zero-based power-of-two subdivision (0=1/4).
    Partial quantization retains performance offsets, so use its finest import
    resolution instead of snapping those offsets to the full-strength grid.
    """
    options = ET.fromstring(_IMPORT_OPTIONS)
    resolution = {"1/4": 0, "1/8": 1, "1/16": 2, "1/32": 3, "1/64": 4}[grid]
    # A barline need not lie on the selected coarse grid (e.g. 9/8 + quarters).
    # The copy is already quantized; finer import resolution preserves that grid.
    options.find("QuantValue").text = str(max(resolution, 4) if strength == 1.0 else 8)
    options.find("Triplets").text = "true" if triplet else "false"
    return ET.tostring(options, encoding="unicode")


def _restore_score_tempos(midi_path: Path, musicxml: Path) -> None:
    """MIDI import drops off-note tempo events; write the source map explicitly."""
    midi = mido.MidiFile(midi_path)
    tick = 0
    tempos = {Fraction(0): 500000}
    for message in mido.merge_tracks(midi.tracks):
        tick += message.time
        if message.type == "set_tempo":
            tempos[Fraction(tick, midi.ticks_per_beat)] = message.tempo
    tree = ET.parse(musicxml)
    parts = tree.getroot().findall("part")
    if not parts:
        raise SheetMusicExportError("MuseScore MusicXML contains no score part")
    for part in parts:
        for measure in part.findall("measure"):
            for direction in measure.findall("direction"):
                if direction.find("./sound[@tempo]") is not None:
                    measure.remove(direction)
    start, divisions = Fraction(0), 1
    pending = dict(sorted(tempos.items()))
    measures = parts[0].findall("measure")
    for index, measure in enumerate(measures):
        divisions = int(measure.findtext("./attributes/divisions", str(divisions)))
        cursor, length = Fraction(0), Fraction(0)
        for child in measure:
            if child.tag in {"backup", "forward"}:
                cursor += (-1 if child.tag == "backup" else 1) * Fraction(
                    int(child.findtext("duration")), divisions
                )
            elif child.tag == "note" and child.find("chord") is None:
                cursor += Fraction(int(child.findtext("duration", "0")), divisions)
            length = max(length, cursor)
        changes = [
            (beat, tempo)
            for beat, tempo in pending.items()
            if start <= beat < start + length
            or (index == len(measures) - 1 and beat == start + length)
        ]
        if changes:
            resolution = math.lcm(
                divisions, *(beat.denominator for beat, _ in changes), start.denominator
            )
            factor = resolution // divisions
            for tag in ("duration", "offset"):
                for child in measure.findall(f".//{tag}"):
                    child.text = str(int(child.text) * factor)
            attributes = measure.find("attributes")
            if attributes is None:
                attributes = ET.Element("attributes")
                measure.insert(0, attributes)
            division_node = attributes.find("divisions")
            if division_node is None:
                division_node = ET.Element("divisions")
                attributes.insert(0, division_node)
            division_node.text = str(resolution)
            insertion = list(measure).index(attributes) + 1
            for beat, tempo in changes:
                direction = ET.Element("direction", placement="above")
                bpm = 60000000 / tempo
                if beat == 0:
                    kind = ET.SubElement(direction, "direction-type")
                    metronome = ET.SubElement(kind, "metronome")
                    ET.SubElement(metronome, "beat-unit").text = "quarter"
                    ET.SubElement(metronome, "per-minute").text = f"{bpm:.2f}".rstrip("0").rstrip(
                        "."
                    )
                else:
                    # Preserve adaptive playback without printing hundreds of tempo labels.
                    kind = ET.SubElement(direction, "direction-type")
                    ET.SubElement(kind, "words", {"print-object": "no"}).text = ""
                amount = int((beat - start) * resolution)
                # MuseScore's MusicXML importer deduplicates tempos at the current
                # cursor before applying <offset>. Give each event its real cursor
                # position, then restore it so the existing notes do not move.
                if amount:
                    forward = ET.Element("forward")
                    ET.SubElement(forward, "duration").text = str(amount)
                    measure.insert(insertion, forward)
                    insertion += 1
                ET.SubElement(direction, "offset").text = "0"
                ET.SubElement(direction, "sound", tempo=f"{bpm:.12g}")
                measure.insert(insertion, direction)
                insertion += 1
                if amount:
                    backup = ET.Element("backup")
                    ET.SubElement(backup, "duration").text = str(amount)
                    measure.insert(insertion, backup)
                    insertion += 1
                del pending[beat]
            # The next measure inherits the *original* divisions in this traversal;
            # explicitly write it there before changing the current XML declaration.
            if index + 1 < len(measures):
                following = measures[index + 1]
                if following.find("./attributes/divisions") is None:
                    attributes = following.find("attributes")
                    if attributes is None:
                        attributes = ET.Element("attributes")
                        following.insert(0, attributes)
                    node = ET.Element("divisions")
                    node.text = str(divisions)
                    attributes.insert(0, node)
        start += length
    if pending:
        raise SheetMusicExportError("Source tempo events lie outside the engraved score")
    tree.write(musicxml, encoding="UTF-8", xml_declaration=True)


class SheetMusicExportError(RuntimeError):
    """MuseScore ran but the requested score artifact was not verified."""


@dataclass(frozen=True)
class SheetMusicExportResult:
    """Verified result of one score ZIP publication."""

    path: Path
    members: tuple[str, ...]
    quantize_grid: str
    grid_ticks: int
    paired_note_count: int
    musescore_version: str


def _creation_flags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_musescore(
    binary: Path,
    arguments: list[str],
    *,
    cancel_check: Callable[[], bool],
) -> subprocess.CompletedProcess[str]:
    """Run one bounded MuseScore command while keeping cancellation responsive."""

    process = subprocess.Popen(
        [str(binary), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=get_musescore_subprocess_env(),
        creationflags=_creation_flags(),
    )
    deadline = time.monotonic() + MUSESCORE_RUN_TIMEOUT_SECONDS
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.2)
            return subprocess.CompletedProcess(
                process.args,
                int(process.returncode or 0),
                stdout,
                stderr,
            )
        except subprocess.TimeoutExpired:
            if cancel_check():
                _stop_process(process)
                raise InterruptedError("Sheet-music export cancelled")
            if time.monotonic() >= deadline:
                _stop_process(process)
                raise SheetMusicExportError(
                    f"MuseScore command timed out after {MUSESCORE_RUN_TIMEOUT_SECONDS:.0f}s: "
                    f"{arguments!r}"
                )


def _failure_detail(process: subprocess.CompletedProcess[str]) -> str:
    combined = "\n".join(
        value.strip() for value in (process.stderr or "", process.stdout or "") if value.strip()
    )
    lines = combined.splitlines()[-8:]
    return ("\n  " + "\n  ".join(lines)) if lines else ""


def _require_success(process: subprocess.CompletedProcess[str], what: str) -> None:
    if process.returncode != 0:
        raise SheetMusicExportError(
            f"MuseScore failed to {what} (returncode={process.returncode})."
            + _failure_detail(process)
        )


def _require_file(
    path: Path,
    what: str,
    process: subprocess.CompletedProcess[str],
    *,
    magic: bytes | None = None,
) -> Path:
    _require_success(process, what)
    if not path.is_file() or path.stat().st_size <= 0:
        raise SheetMusicExportError(
            f"MuseScore failed to {what} (returncode={process.returncode})."
            + _failure_detail(process)
        )
    if magic is not None:
        with path.open("rb") as handle:
            prefix = handle.read(len(magic))
        if not prefix.startswith(magic):
            raise SheetMusicExportError(f"MuseScore wrote an invalid {what}: {path}")
    return path


def _normalize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _part_slug(name: str) -> str:
    segments = [
        segment
        for segment in (_normalize_filename(value) for value in str(name).split(","))
        if segment
    ]
    words = [set(segment.split("_")) for segment in segments]
    keep = [
        segment
        for index, segment in enumerate(segments)
        if not any(
            words[index] < other or (words[index] == other and other_index < index)
            for other_index, other in enumerate(words)
            if other_index != index
        )
    ]
    return "_".join(keep) or "part"


def _score_root(project_path: Path) -> ET.Element:
    try:
        document = ET.parse(project_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise SheetMusicExportError(f"MuseScore project XML is invalid: {project_path}") from exc
    score = document.find("Score")
    if score is None:
        raise SheetMusicExportError(f"MuseScore project contains no <Score>: {project_path}")
    return score


def _string_counts(project_path: Path) -> list[int]:
    counts: list[int] = []
    for part in _score_root(project_path).findall("Part"):
        instrument = part.find("Instrument")
        counts.append(len(instrument.findall("StringData/string")) if instrument is not None else 0)
    return counts


def fretted_part_indices(project_path: str | Path) -> list[int]:
    """Return MuseScore part indices with a supported 4-9 string tablature preset."""

    return [
        index
        for index, string_count in enumerate(_string_counts(Path(project_path)))
        if string_count in _TAB_PRESETS
    ]


def convert_project_to_tablature(project_path: str | Path) -> list[int]:
    """Retype supported fretted staves in a temporary MuseScore project."""

    path = Path(project_path)
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError) as exc:
        raise SheetMusicExportError(f"MuseScore project XML is invalid: {path}") from exc
    score = tree.getroot().find("Score")
    if score is None:
        raise SheetMusicExportError(f"MuseScore project contains no <Score>: {path}")
    converted: list[int] = []
    for index, part in enumerate(score.findall("Part")):
        instrument = part.find("Instrument")
        string_count = len(instrument.findall("StringData/string")) if instrument is not None else 0
        preset = _TAB_PRESETS.get(string_count)
        staff = part.find("Staff")
        if preset is None or staff is None:
            continue
        staff_type = staff.find("StaffType")
        if staff_type is None:
            staff_type = ET.SubElement(staff, "StaffType")
        staff_type.set("group", "tablature")
        name = staff_type.find("name")
        if name is None:
            name = ET.SubElement(staff_type, "name")
        name.text = preset
        for tag in ("lines", "lineDistance"):
            for existing in staff_type.findall(tag):
                staff_type.remove(existing)
        ET.SubElement(staff_type, "lines").text = str(string_count)
        ET.SubElement(staff_type, "lineDistance").text = _TAB_LINE_DISTANCE
        converted.append(index)
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    return converted


def _write_part_pdfs(
    binary: Path,
    project_path: Path,
    output_dir: Path,
    *,
    suffix: str = "",
    only: set[int] | None = None,
    cancel_check: Callable[[], bool],
) -> list[Path]:
    process = _run_musescore(
        binary,
        ["--score-parts-pdf", str(project_path)],
        cancel_check=cancel_check,
    )
    _require_success(process, "generate per-instrument PDFs")
    try:
        payload = json.loads(process.stdout)
        names = payload["parts"]
        blobs = payload["partsBin"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SheetMusicExportError(
            "MuseScore failed to generate per-instrument PDFs "
            f"(returncode={process.returncode})." + _failure_detail(process)
        ) from exc
    if (
        not isinstance(names, list)
        or not isinstance(blobs, list)
        or len(names) != len(blobs)
        or not names
    ):
        raise SheetMusicExportError("MuseScore returned a malformed per-instrument PDF payload")
    written: list[Path] = []
    for index, (name, encoded) in enumerate(zip(names, blobs, strict=True)):
        if only is not None and index not in only:
            continue
        if cancel_check():
            raise InterruptedError("Sheet-music export cancelled")
        try:
            content = base64.b64decode(str(encoded), validate=True)
        except (ValueError, TypeError) as exc:
            raise SheetMusicExportError(
                f"MuseScore returned invalid PDF data for part {index + 1}"
            ) from exc
        if not content.startswith(b"%PDF-"):
            raise SheetMusicExportError(
                f"MuseScore returned a non-PDF payload for part {index + 1}"
            )
        path = output_dir / f"{index + 1:02d}_{_part_slug(str(name))}{suffix}.pdf"
        path.write_bytes(content)
        written.append(path)
    if not written:
        raise SheetMusicExportError("MuseScore generated no requested per-instrument PDFs")
    return written


def engrave_sheet_directory(
    quantized_midi_path: str | Path,
    output_dir: str | Path,
    *,
    quantize_grid: str = DEFAULT_MIDI_QUANTIZE_GRID,
    quantize_strength: float = 1.0,
    quantize_triplet: bool = False,
    musescore: str | Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[Path]:
    """Engrave a quantized MIDI copy into verified score files."""

    check_cancelled = cancel_check or (lambda: False)
    source_midi = validate_midi_output(quantized_midi_path, "sheet-music quantized MIDI").resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Sheet-music output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    binary = (
        validate_musescore_runtime(musescore)
        if musescore is not None
        else get_musescore_executable()
    )
    if check_cancelled():
        raise InterruptedError("Sheet-music export cancelled")

    score_midi = destination / "score.mid"
    shutil.copy2(source_midi, score_midi)
    validate_midi_output(score_midi, "sheet-music score MIDI")
    written = [score_midi]
    with tempfile.TemporaryDirectory(prefix="music-to-midi-engrave-") as temporary:
        work = Path(temporary)
        import_options = work / "import.xml"
        import_options.write_text(
            _midi_import_options(quantize_grid, quantize_strength, False),
            encoding="utf-8",
        )
        project = work / "score.mscx"
        musicxml = destination / "score.musicxml"
        if quantize_triplet:
            import_midi = work / "triplet-coordinates.mid"
            signatures = prepare_triplet_import(score_midi, import_midi)
            process = _run_musescore(
                binary,
                ["-M", str(import_options), "-o", str(musicxml), str(import_midi)],
                cancel_check=check_cancelled,
            )
            _require_file(musicxml, "write MusicXML", process)
            restore_note_durations(
                import_midi,
                musicxml,
                import_grid=Fraction(1, 16) if quantize_strength == 1.0 else Fraction(1, 256),
            )
            restore_triplet_musicxml(musicxml, signatures)
        else:
            process = _run_musescore(
                binary,
                ["-M", str(import_options), "-o", str(musicxml), str(score_midi)],
                cancel_check=check_cancelled,
            )
            _require_file(musicxml, "import the MIDI file", process)
            restore_note_durations(
                score_midi,
                musicxml,
                import_grid=Fraction(1, 16) if quantize_strength == 1.0 else Fraction(1, 256),
            )
        _restore_score_tempos(score_midi, musicxml)
        process = _run_musescore(
            binary, ["-o", str(project), str(musicxml)], cancel_check=check_cancelled
        )
        _require_file(project, "import the notation", process)
        _score_root(project)
        try:
            ET.parse(musicxml)
        except (ET.ParseError, OSError) as exc:
            raise SheetMusicExportError(f"MuseScore wrote invalid MusicXML: {musicxml}") from exc
        written.append(musicxml)

        full_score = destination / "full_score.pdf"
        process = _run_musescore(
            binary,
            ["-o", str(full_score), str(project)],
            cancel_check=check_cancelled,
        )
        written.append(
            _require_file(full_score, "render the full score PDF", process, magic=b"%PDF-")
        )
        written.extend(
            _write_part_pdfs(
                binary,
                project,
                destination,
                cancel_check=check_cancelled,
            )
        )

        fretted = fretted_part_indices(project)
        if fretted:
            tab_project = work / "score_tab.mscx"
            shutil.copy2(project, tab_project)
            converted = convert_project_to_tablature(tab_project)
            if converted != fretted:
                raise SheetMusicExportError(
                    f"Tablature conversion mismatch: expected={fretted}, converted={converted}"
                )
            tab_tree = ET.parse(tab_project)
            try:
                tab_report = assign_score_tablature(tab_tree.getroot().find("Score"), converted)
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise SheetMusicExportError(f"Could not validate tablature: {exc}") from exc
            tab_tree.write(tab_project, encoding="UTF-8", xml_declaration=True)
            valid_parts = {part["part"] - 1 for part in tab_report if part["status"] == "assigned"}
            if valid_parts:
                written.extend(
                    _write_part_pdfs(
                        binary,
                        tab_project,
                        destination,
                        suffix="_tab",
                        only=valid_parts,
                        cancel_check=check_cancelled,
                    )
                )
            if valid_parts != set(fretted):
                report_path = destination / "tablature_report.json"
                report_path.write_text(
                    json.dumps(tab_report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                explanation = destination / "TAB_README.txt"
                explanation.write_text(
                    "部分声部无法在该乐器的弦数、音域和持续音约束下生成完整 Tab。\n"
                    "这些声部的全部音符保留在 score.mid、MusicXML 和五线谱 PDF 中；没有为生成 Tab 删音、移调或截断持续音。\n"
                    "未输出存在弦位冲突的 Tab。具体声部、小节（从 1 起）、小节内四分音符拍偏移（从 0 起）和 MIDI 音高见 tablature_report.json。\n"
                    "需要人工编配这些位置；成功分配弦位也不等于已验证手型与演奏难度。\n\n"
                    "Some parts cannot be represented completely within the instrument's strings, range and sustained-note constraints.\n"
                    "All notes remain in score.mid, MusicXML and standard-notation PDFs; none were deleted, transposed or cut short to fit Tab.\n"
                    "Conflicting Tab PDFs were not emitted. tablature_report.json identifies each part, 1-based measure, 0-based quarter-note offset and MIDI pitch.\n"
                    "These passages require arrangement; assigned strings do not certify fingering comfort or playability.\n",
                    encoding="utf-8",
                )
                written.extend([report_path, explanation])
    return written


def _verify_zip(path: Path, expected_members: tuple[str, ...]) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SheetMusicExportError(f"Sheet-music ZIP was not written: {path}")
    with zipfile.ZipFile(path) as archive:
        names = tuple(archive.namelist())
        if names != expected_members or len(names) != len(set(names)):
            raise SheetMusicExportError(
                f"Sheet-music ZIP member mismatch: expected={expected_members!r}, actual={names!r}"
            )
        bad_member = archive.testzip()
        if bad_member is not None:
            raise SheetMusicExportError(f"Sheet-music ZIP CRC failed: {bad_member}")
        for info in archive.infolist():
            if info.is_dir() or info.file_size <= 0 or Path(info.filename).name != info.filename:
                raise SheetMusicExportError(
                    f"Sheet-music ZIP contains an invalid member: {info.filename!r}"
                )


def export_sheet_music_zip(
    midi_path: str | Path,
    destination: str | Path,
    *,
    quantize_grid: str = DEFAULT_MIDI_QUANTIZE_GRID,
    quantize_strength: float = 1.0,
    quantize_triplet: bool = False,
    quantize_auto: bool = False,
    musescore: str | Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> SheetMusicExportResult:
    """Quantize a private MIDI copy, engrave it, and atomically publish one ZIP."""

    check_cancelled = cancel_check or (lambda: False)
    source = validate_midi_output(midi_path, "sheet-music source MIDI").resolve()
    output = Path(destination).expanduser().resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError(f"Sheet-music destination must end in .zip: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    binary = (
        validate_musescore_runtime(musescore)
        if musescore is not None
        else get_musescore_executable()
    )
    version_tuple = musescore_version(binary)
    if version_tuple is None:
        raise SheetMusicExportError(f"MuseScore version became unavailable: {binary}")
    version_text = ".".join(str(value) for value in version_tuple)
    staged_zip = output.with_name(f".{output.name}.{secrets.token_hex(12)}.part.zip")
    try:
        with tempfile.TemporaryDirectory(prefix="music-to-midi-sheet-export-") as temporary:
            root = Path(temporary)
            quantized_midi = root / "quantized-source.mid"
            shutil.copy2(source, quantized_midi)
            quantization: MidiQuantizationReport = quantize_midi_notes(
                quantized_midi,
                str(quantize_grid),
                label="sheet-music copy quantization",
                strength=quantize_strength,
                triplet=quantize_triplet,
                automatic=quantize_auto,
            )
            if check_cancelled():
                raise InterruptedError("Sheet-music export cancelled")
            score_dir = root / "score"
            written = engrave_sheet_directory(
                quantization.path,
                score_dir,
                quantize_grid=quantization.grid,
                quantize_strength=quantization.strength,
                quantize_triplet=quantization.triplet,
                musescore=binary,
                cancel_check=check_cancelled,
            )
            members = tuple(path.name for path in written)
            if len(members) != len(set(members)):
                raise SheetMusicExportError(f"Duplicate sheet-music filenames: {members!r}")
            midi = mido.MidiFile(str(quantization.path))
            duration = float(midi.length)
            if not math.isfinite(duration) or duration <= 0:
                raise SheetMusicExportError(
                    f"Quantized sheet MIDI has invalid duration: {duration!r}"
                )
            manifest = {
                "schema_version": SHEET_MUSIC_SCHEMA_VERSION,
                "source_midi": source.name,
                "source_sha256": source_hash,
                "quantize_grid": quantization.grid,
                "grid_ticks": quantization.grid_ticks,
                "quantize_strength": quantization.strength,
                "quantize_triplet": quantization.triplet,
                "quantize_auto": quantization.automatic,
                "paired_note_count": quantization.paired_note_count,
                "duration_seconds": duration,
                "musescore_version": version_text,
                "files": list(members),
            }
            manifest_path = score_dir / SHEET_MUSIC_MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            archive_members = (*members, SHEET_MUSIC_MANIFEST_NAME)
            with zipfile.ZipFile(staged_zip, "w", compression=zipfile.ZIP_STORED) as archive:
                for member in archive_members:
                    if check_cancelled():
                        raise InterruptedError("Sheet-music export cancelled")
                    archive.write(score_dir / member, arcname=member)
            _verify_zip(staged_zip, archive_members)
            if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
                raise SheetMusicExportError(f"Sheet export changed its source MIDI: {source}")
            if check_cancelled():
                raise InterruptedError("Sheet-music export cancelled")
            staged_zip.replace(output)
            _verify_zip(output, archive_members)
            return SheetMusicExportResult(
                path=output,
                members=archive_members,
                quantize_grid=quantization.grid,
                grid_ticks=quantization.grid_ticks,
                paired_note_count=quantization.paired_note_count,
                musescore_version=version_text,
            )
    finally:
        staged_zip.unlink(missing_ok=True)


__all__ = [
    "MUSESCORE_RUN_TIMEOUT_SECONDS",
    "SHEET_MUSIC_MANIFEST_NAME",
    "SHEET_MUSIC_SCHEMA_VERSION",
    "SHEET_MUSIC_ZIP_NAME",
    "SheetMusicExportError",
    "SheetMusicExportResult",
    "convert_project_to_tablature",
    "engrave_sheet_directory",
    "export_sheet_music_zip",
    "fretted_part_indices",
]
