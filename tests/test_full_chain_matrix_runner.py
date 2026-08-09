from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import full_chain_matrix_runner as runner


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 160)


class _FakePipeline:
    result: object

    def __init__(self, _config) -> None:
        pass

    def process(self, *_args, **_kwargs):
        return self.result


def test_matrix_runner_forwards_explicit_model_variants(tmp_path: Path) -> None:
    config = runner.build_case_config(
        {
            "name": "variant",
            "mode": "smart",
            "backend": "muscriptor",
            "multi_model": "muscriptor",
            "yourmt3_model": "yptf_single_nops",
            "muscriptor_model": "medium",
        },
        tmp_path,
    )

    assert config.yourmt3_model == "yptf_single_nops"
    assert config.muscriptor_model == "medium"


@pytest.mark.parametrize(
    ("mode", "keys"),
    (
        ("vocal_split", {"vocals", "accompaniment"}),
        (
            "six_stem_split",
            {"bass", "drums", "guitar", "piano", "vocals", "other"},
        ),
    ),
)
def test_matrix_runner_validates_wav_only_separation_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    keys: set[str],
) -> None:
    separated = {}
    for key in keys:
        path = tmp_path / "outputs" / f"{key}.wav"
        _write_wav(path)
        separated[key] = str(path)
    _FakePipeline.result = SimpleNamespace(
        midi_path="",
        separated_audio=separated,
        total_notes=0,
    )
    monkeypatch.setattr(runner, "MusicToMidiPipeline", _FakePipeline)

    result = runner.run_case(
        {"name": mode, "mode": mode, "backend": "yourmt3"},
        tmp_path / "mix.wav",
        tmp_path / "piano.wav",
        tmp_path / "matrix",
    )

    assert result["status"] == "PASS"
    assert result["midi"] == []
    assert set(result["separated_audio"]) == keys
    assert all(item["frames"] == 160 for item in result["separated_audio"].values())


def test_matrix_runner_rejects_hidden_midi_from_separation_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    separated = {}
    for key in ("vocals", "accompaniment"):
        path = tmp_path / "outputs" / f"{key}.wav"
        _write_wav(path)
        separated[key] = str(path)
    _FakePipeline.result = SimpleNamespace(
        midi_path=str(tmp_path / "unexpected.mid"),
        separated_audio=separated,
        total_notes=0,
    )
    monkeypatch.setattr(runner, "MusicToMidiPipeline", _FakePipeline)

    with pytest.raises(RuntimeError, match="unexpectedly returned MIDI"):
        runner.run_case(
            {"name": "vocal", "mode": "vocal_split", "backend": "yourmt3"},
            tmp_path / "mix.wav",
            tmp_path / "piano.wav",
            tmp_path / "matrix",
        )


def test_matrix_runner_rejects_missing_separation_stem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vocals = tmp_path / "outputs" / "vocals.wav"
    _write_wav(vocals)
    _FakePipeline.result = SimpleNamespace(
        midi_path="",
        separated_audio={"vocals": str(vocals)},
        total_notes=0,
    )
    monkeypatch.setattr(runner, "MusicToMidiPipeline", _FakePipeline)

    with pytest.raises(RuntimeError, match="separated WAV keys mismatch"):
        runner.run_case(
            {"name": "vocal", "mode": "vocal_split", "backend": "yourmt3"},
            tmp_path / "mix.wav",
            tmp_path / "piano.wav",
            tmp_path / "matrix",
        )
