from pathlib import Path

import pytest

from src.core.multi_stem_separator import STEM_KEYS
from src.core.pipeline import MusicToMidiPipeline
from src.models.data_models import Config


@pytest.mark.parametrize("save_separated_tracks", [True, False])
def test_six_stem_pipeline_stops_after_verified_wavs(
    monkeypatch,
    tmp_path,
    save_separated_tracks,
):
    audio_path = tmp_path / "original_mix.wav"
    audio_path.write_bytes(b"real input")
    output_dir = tmp_path / "output"
    separator_calls = []

    class FakeSixStemSeparator:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_model_available():
            return True

        def __init__(self, **_kwargs):
            self.cancel_check = None

        def set_cancel_check(self, cancel_check):
            self.cancel_check = cancel_check

        def separate(self, audio_path, output_dir, progress_callback=None):
            assert self.cancel_check is not None
            assert not self.cancel_check()
            separator_calls.append(Path(audio_path))
            root = Path(output_dir)
            root.mkdir(parents=True, exist_ok=True)
            outputs = {}
            for stem_name in STEM_KEYS:
                stem_path = root / f"original_mix_{stem_name}.wav"
                stem_path.write_bytes(stem_name.encode("utf-8"))
                outputs[stem_name] = str(stem_path)
            if progress_callback is not None:
                progress_callback(1.0, "separated")
            return outputs

    def unexpected_midi_work(*_args, **_kwargs):
        pytest.fail("split modes must stop before beat detection or MIDI transcription")

    config = Config(
        processing_mode="six_stem_split",
        save_separated_tracks=save_separated_tracks,
    )
    pipeline = MusicToMidiPipeline(config)
    for transcriber in (
        pipeline.yourmt3_transcriber,
        pipeline.miros_transcriber,
        pipeline.muscriptor_transcriber,
    ):
        monkeypatch.setattr(
            transcriber,
            "transcribe_to_midi",
            unexpected_midi_work,
            raising=False,
        )
    monkeypatch.setattr(pipeline, "_detect_beat_or_raise", unexpected_midi_work)
    monkeypatch.setattr(pipeline, "_merge_stem_midis", unexpected_midi_work)
    monkeypatch.setattr(
        "src.core.multi_stem_separator.SixStemSeparator",
        FakeSixStemSeparator,
    )

    result = pipeline.process(str(audio_path), str(output_dir))

    assert separator_calls == [audio_path]
    assert result.midi_path == ""
    assert result.total_notes == 0
    assert result.beat_info is None
    assert result.stem_midi_paths is None
    assert result.merged_midi_path is None
    assert set(result.separated_audio or {}) == set(STEM_KEYS)
    assert all(Path(path).is_file() for path in (result.separated_audio or {}).values())
    assert list(output_dir.rglob("*.mid")) == []
