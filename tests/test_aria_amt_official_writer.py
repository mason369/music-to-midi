import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mido
import pytest

pytest.importorskip("amt")
from amt.tokenizer import AmtTokenizer

from src.core.aria_amt_transcriber import (
    ARIA_AMT_SOURCE_ARCHIVE_URL,
    ARIA_AMT_SOURCE_REQUIREMENT,
    ARIA_UTILS_SOURCE_ARCHIVE_URL,
    ARIA_UTILS_SOURCE_REVISION,
    ARIA_UTILS_SOURCE_REQUIREMENT,
    AriaAmtTranscriber,
    get_aria_amt_runtime_unavailable_reason,
)


def test_official_writer_handles_zero_duration_without_an_unclosed_note(tmp_path):
    tokenizer = AmtTokenizer()
    sequence = [
        tokenizer.bos_tok,
        ("on", 60),
        ("onset", 0),
        ("vel", 80),
        ("off", 60),
        ("onset", 100),
        ("on", 55),
        ("onset", 100),
        ("vel", 65),
        ("off", 55),
        ("onset", 100),
        ("on", 64),
        ("onset", 150),
        ("vel", 75),
        ("off", 64),
        ("onset", 300),
        tokenizer.eos_tok,
    ]
    path = tmp_path / "official.mid"
    AriaAmtTranscriber._save_token_sequence_as_midi(tokenizer, sequence, path)
    midi = mido.MidiFile(path)
    active = {}
    notes = []
    tick = 0
    for event in midi.tracks[0]:
        tick += event.time
        if event.type == "note_on" and event.velocity:
            assert event.note not in active
            active[event.note] = (tick, event.velocity)
        elif event.type == "note_off" or (event.type == "note_on" and not event.velocity):
            assert event.note in active
            start, velocity = active.pop(event.note)
            assert tick > start
            notes.append((event.note, start, tick, velocity))
    assert not active
    assert notes == [(60, 0, 100, 80), (64, 150, 300, 75)]


def test_official_writer_preserves_pedal_events(tmp_path):
    tokenizer = AmtTokenizer()
    sequence = [
        tokenizer.bos_tok,
        ("pedal", 1),
        ("onset", 0),
        ("on", 60),
        ("onset", 0),
        ("vel", 80),
        ("off", 60),
        ("onset", 100),
        ("pedal", 0),
        ("onset", 200),
        tokenizer.eos_tok,
    ]
    path = tmp_path / "pedal.mid"
    AriaAmtTranscriber._save_token_sequence_as_midi(tokenizer, sequence, path)
    controls = []
    tick = 0
    for event in mido.MidiFile(path).tracks[0]:
        tick += event.time
        if event.type == "control_change" and event.control == 64:
            controls.append((tick, event.value))
    assert controls == [(0, 127), (200, 0)]


def test_all_install_surfaces_pin_the_same_official_midi_writer():
    root = Path(__file__).resolve().parents[1]
    requirement = (
        "ariautils @ https://github.com/EleutherAI/aria-utils/archive/"
        "7c815755c6889a784b6a1b3343e9e8daa3e7d128.zip"
    )
    for path in [root / "requirements.txt", root / "space/requirements.txt"]:
        assert requirement in path.read_text(encoding="utf-8").splitlines()
    notebook = json.loads((root / "colab_notebook.ipynb").read_text(encoding="utf-8"))
    assert requirement in "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "copy_metadata('ariautils')" in (root / "MusicToMidi.spec").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "source,available",
    [
        ({"vcs_info": {"commit_id": ARIA_UTILS_SOURCE_REVISION}}, True),
        ({"url": ARIA_UTILS_SOURCE_ARCHIVE_URL}, True),
        ({"vcs_info": {"commit_id": "93da092204e5b1189ed8e0259f6156266fd086a7"}}, False),
        ({}, False),
        ([], False),
    ],
)
def test_same_package_version_cannot_hide_outdated_official_source(source, available):
    def distribution(package):
        payload = {"url": ARIA_AMT_SOURCE_ARCHIVE_URL} if package == "aria-amt" else source
        return SimpleNamespace(version="0.0.1", read_text=lambda name: json.dumps(payload))

    with (
        patch("src.core.aria_amt_transcriber.importlib.util.find_spec", return_value=object()),
        patch("src.core.aria_amt_transcriber.metadata.distribution", side_effect=distribution),
    ):
        reason = get_aria_amt_runtime_unavailable_reason()
    assert (reason == "") is available
    if not available:
        assert ARIA_UTILS_SOURCE_REVISION in reason


def test_space_repairs_both_pinned_packages_without_changing_torch():
    root = Path(__file__).resolve().parents[1]
    source = ast.parse((root / "space/app.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "ensure_aria_amt_runtime"
    )
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=""))
    namespace = {
        "_aria_amt_runtime_available": lambda: False,
        "subprocess": SimpleNamespace(run=run, PIPE=-1, STDOUT=-2),
        "sys": SimpleNamespace(executable="python"),
        "logger": Mock(),
        "importlib": SimpleNamespace(invalidate_caches=Mock()),
        "ARIA_AMT_RUNTIME_REQUIREMENT": ARIA_AMT_SOURCE_REQUIREMENT,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<space installer>", "exec"),
        namespace,
    )
    with patch(
        "src.core.aria_amt_transcriber.get_aria_amt_runtime_unavailable_reason",
        side_effect=["old MIDI writer", ""],
    ):
        namespace["ensure_aria_amt_runtime"]()
    command = run.call_args.args[0]
    assert command[-2:] == [ARIA_AMT_SOURCE_REQUIREMENT, ARIA_UTILS_SOURCE_REQUIREMENT]
    assert "--no-deps" in command
    assert "--force-reinstall" in command
