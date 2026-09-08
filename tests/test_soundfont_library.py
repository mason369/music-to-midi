"""SoundFont directory validation and immutable render-selection contracts."""

import hashlib
import struct
from pathlib import Path
from types import SimpleNamespace

import mido
import pytest

from src.core.soundfont_library import (
    import_soundfont,
    read_soundfont,
    soundfont_selection,
    write_soundfont_render_midi,
)


def soundfont_directory(path, *, terminal_name=b"EOP", presets=((0, 0), (16, 25), (128, 0))):
    """Minimal RIFF metadata fixture; actual SF2/SF3 synthesis is validated separately."""

    def chunk(tag, data):
        return tag + struct.pack("<I", len(data)) + data + b"\0" * (len(data) % 2)

    headers = b"".join(
        struct.pack("<20sHHHIII", f"Preset {i}".encode(), program, bank, i, 0, 0, 0)
        for i, (bank, program) in enumerate(presets)
    )
    headers += struct.pack("<20sHHHIII", terminal_name, 0, 0, len(presets), 0, 0, 0)
    pdta = chunk(b"phdr", headers) + chunk(b"pbag", b"\0" * 4 * (len(presets) + 1))
    for tag, size in [
        (b"pmod", 10),
        (b"pgen", 4),
        (b"inst", 22),
        (b"ibag", 4),
        (b"imod", 10),
        (b"igen", 4),
        (b"shdr", 46),
    ]:
        pdta += chunk(tag, b"\0" * size)
    payload = b"sfbk" + chunk(b"LIST", b"INFO" + chunk(b"ifil", struct.pack("<HH", 2, 1)))
    payload += chunk(b"LIST", b"sdta" + chunk(b"smpl", b"\0" * 100)) + chunk(
        b"LIST", b"pdta" + pdta
    )
    path.write_bytes(chunk(b"RIFF", payload))
    return path


@pytest.mark.parametrize("extension,terminal", [(".sf2", b"EOP"), (".sf3", b"")])
def test_real_writer_terminal_name_variants_and_extra_banks(tmp_path, extension, terminal):
    path = soundfont_directory(tmp_path / ("bank" + extension), terminal_name=terminal)
    library = read_soundfont(path)
    assert [(p.bank, p.program) for p in library.presets] == [(0, 0), (16, 25), (128, 0)]
    assert library.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("damage", ["truncated", "header", "duplicate", "extension"])
def test_invalid_soundfonts_fail_explicitly(tmp_path, damage):
    presets = ((0, 0), (0, 0)) if damage == "duplicate" else ((0, 0),)
    path = soundfont_directory(
        tmp_path / ("bank.wav" if damage == "extension" else "bank.sf2"), presets=presets
    )
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:-5])
    if damage == "header":
        path.write_bytes(b"BAD!" + path.read_bytes()[4:])
    with pytest.raises(ValueError):
        read_soundfont(path)


def test_import_is_content_addressed_and_retains_original(tmp_path):
    source = soundfont_directory(tmp_path / "user.sf3", terminal_name=b"")
    original = source.read_bytes()
    first = import_soundfont(source, tmp_path / "library")
    second = import_soundfont(source, tmp_path / "library")
    assert first == second
    assert source.read_bytes() == first.path.read_bytes() == original
    first.path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="校验失败"):
        import_soundfont(source, tmp_path / "library")


def make_midi(path):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack(
        [
            mido.MetaMessage("set_tempo", tempo=618557),
            mido.Message("program_change", program=0, time=17),
            mido.Message("control_change", control=64, value=127, time=4),
            mido.Message("pitchwheel", pitch=100, time=3),
            mido.Message("note_on", note=60, velocity=88, time=12),
            mido.Message("note_off", note=60, time=480),
            mido.Message("control_change", control=64, value=0, time=19),
        ]
    )
    midi.tracks.append(track)
    midi.save(path)
    return path


def events(path):
    tick = 0
    result = []
    for event in mido.merge_tracks(mido.MidiFile(path).tracks):
        tick += event.time
        if event.type == "program_change" or (
            event.type == "control_change" and event.control in {0, 32}
        ):
            continue
        result.append((tick, event.copy(time=0)))
    return result


def test_extra_bank_render_preserves_notes_pedal_tempo_and_source(tmp_path):
    library = read_soundfont(soundfont_directory(tmp_path / "bank.sf2"))
    selection = soundfont_selection(
        library, [dict(source_program=0, is_drum=False, bank=16, program=25)]
    )
    source = make_midi(tmp_path / "source.mid")
    original = source.read_bytes()
    rendered = write_soundfont_render_midi(source, tmp_path / "render.mid", selection)
    assert events(source) == events(rendered)
    assert source.read_bytes() == original
    messages = mido.MidiFile(rendered).tracks[0]
    assert any(m.type == "control_change" and m.control == 32 and m.value == 16 for m in messages)
    assert any(m.type == "program_change" and m.program == 25 for m in messages)
    library.path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="发生变化"):
        write_soundfont_render_midi(source, tmp_path / "changed.mid", selection)


@pytest.mark.parametrize(
    "assignment",
    [
        dict(source_program=0, is_drum=False, bank=1, program=25),
        dict(source_program=0, is_drum=True, bank=16, program=25),
        dict(source_program=True, is_drum=False, bank=16, program=25),
    ],
)
def test_invalid_or_incompatible_assignments_are_rejected(tmp_path, assignment):
    library = read_soundfont(soundfont_directory(tmp_path / "bank.sf2"))
    with pytest.raises(ValueError):
        soundfont_selection(library, [assignment])


def test_missing_original_preset_does_not_silently_use_piano(tmp_path):
    library = read_soundfont(soundfont_directory(tmp_path / "bank.sf2", presets=((16, 25),)))
    with pytest.raises(ValueError, match="缺少"):
        write_soundfont_render_midi(
            make_midi(tmp_path / "source.mid"),
            tmp_path / "render.mid",
            soundfont_selection(library, []),
        )


def test_desktop_selection_is_explicit_and_tracks_bank_and_program(qtbot, tmp_path):
    from src.gui.widgets.soundfont_panel import SoundFontPanel

    panel = SoundFontPanel(tmp_path / "imports")
    qtbot.addWidget(panel)
    panel.set_notes([SimpleNamespace(program=0, is_drum=False, instrument="Piano")])
    selections = []
    panel.apply_requested.connect(selections.append)
    panel._imported(read_soundfont(soundfont_directory(tmp_path / "bank.sf2")))
    panel.preset.setCurrentIndex(panel.preset.findText("Preset 1 · 16:25"))
    panel._remember_preset()
    assert selections == []
    panel.apply_button.click()
    assert selections[0].assignments == ((0, False, 16, 25),)
    panel.library.setCurrentIndex(0)
    assert len(selections) == 1
    panel.apply_button.click()
    assert selections[-1] is None
