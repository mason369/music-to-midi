import sys
import types
from pathlib import Path

import mido
import numpy as np
import pytest
import torch

from src.core import yourmt3_transcriber as yourmt3_module
from src.core.yourmt3_transcriber import YourMT3Transcriber


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        (
            "ymt3_plus",
            (None, None, "mt3_full_plus", None, None, None),
        ),
        (
            "yptf_single_nops",
            ("perceiver-tf", None, "mt3_full_plus", "spec", 300, None),
        ),
        (
            "yptf_multi_ps",
            ("perceiver-tf", "multi-t5", "mc13_full_plus_256", "spec", 300, None),
        ),
        (
            "yptf_moe_multi_nops",
            ("perceiver-tf", "multi-t5", "mc13_full_plus_256", "spec", 300, "moe"),
        ),
        (
            "yptf_moe_multi_ps",
            ("perceiver-tf", "multi-t5", "mc13_full_plus_256", "spec", 300, "moe"),
        ),
    ],
)
def test_official_checkpoint_arguments_match_published_space_modes(model_name, expected):
    args = YourMT3Transcriber._build_official_model_args(model_name, "experiment@model.ckpt")

    actual = (
        args.encoder_type,
        args.decoder_type,
        args.task,
        args.audio_codec,
        args.hop_length,
        args.ff_layer_type,
    )
    assert actual == expected
    assert args.project == "2024"
    assert args.write_model_output is True


def test_official_transcription_uses_torchaudio_nonoverlap_fixed_batch_and_writer(
    monkeypatch,
    tmp_path,
):
    calls = {}
    inverse_vocab = {0: (0, "Piano")}

    class FakeTaskManager:
        num_decoding_channels = 1

        def detokenize_list_batches(self, batches, start_secs, return_events):
            calls["detokenize"] = (batches, start_secs, return_events)
            return [], [], {}

    class FakeModel:
        task_manager = FakeTaskManager()
        midi_output_inverse_vocab = inverse_vocab

        def inference_file(self, bsz, audio_segments):
            calls["inference"] = (bsz, audio_segments.detach().cpu().clone())
            return [np.zeros((1, 1, 1), dtype=np.int64)], None

    fake_model = FakeModel()
    waveform = torch.tensor(
        [[1.0, 0.5, -0.5, -1.0], [-1.0, -0.5, 0.5, 1.0]],
        dtype=torch.float32,
    )

    torchaudio = types.ModuleType("torchaudio")

    def fake_load(*, uri):
        calls["load_uri"] = uri
        return waveform, 16_000

    def fake_resample(audio, source_rate, target_rate):
        calls["resample"] = (audio.detach().cpu().clone(), source_rate, target_rate)
        return audio

    torchaudio.load = fake_load
    torchaudio.functional = types.SimpleNamespace(resample=fake_resample)

    utils_package = types.ModuleType("utils")
    utils_package.__path__ = []
    audio_module = types.ModuleType("utils.audio")
    event2note_module = types.ModuleType("utils.event2note")
    note2event_module = types.ModuleType("utils.note2event")
    utils_module = types.ModuleType("utils.utils")

    def fake_slice(audio, slice_length, slice_hop):
        calls["slice"] = (audio.copy(), slice_length, slice_hop)
        return audio.copy()

    def fake_merge(events):
        calls["merge_events"] = events
        return [], {}

    def fake_mix(notes_by_channel):
        calls["mix"] = notes_by_channel
        return []

    def fake_write(notes, output_dir, track_name, received_inverse_vocab):
        calls["write"] = (notes, output_dir, track_name, received_inverse_vocab)
        midi_path = Path(output_dir) / "model_output" / f"{track_name}.mid"
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        midi = mido.MidiFile(type=1, ticks_per_beat=480)
        track = mido.MidiTrack()
        track.extend(
            [
                mido.MetaMessage("track_name", name="official", time=0),
                mido.Message("program_change", program=0, channel=0, time=0),
                mido.Message("note_on", note=60, velocity=100, channel=0, time=0),
                mido.Message("note_off", note=60, velocity=0, channel=0, time=1),
            ]
        )
        midi.tracks.append(track)
        midi.save(midi_path)

    audio_module.slice_padded_array = fake_slice
    event2note_module.merge_zipped_note_events_and_ties_to_notes = fake_merge
    note2event_module.mix_notes = fake_mix
    utils_module.write_model_output_as_midi = fake_write

    for name, module in {
        "torchaudio": torchaudio,
        "utils": utils_package,
        "utils.audio": audio_module,
        "utils.event2note": event2note_module,
        "utils.note2event": note2event_module,
        "utils.utils": utils_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    output_path = tmp_path / "song.mid"
    attempt_path = tmp_path / ".song.yourmt3.fixed.tmp.mid"
    monkeypatch.setattr(
        yourmt3_module,
        "unique_midi_temp_path",
        lambda _output, _purpose: attempt_path,
    )
    monkeypatch.setenv("MUSIC_TO_MIDI_YOURMT3_OFFICIAL_BATCH_SIZE", "1")
    monkeypatch.setattr(YourMT3Transcriber, "_model", fake_model)
    monkeypatch.setattr(
        YourMT3Transcriber,
        "_audio_cfg",
        {"sample_rate": 16_000, "input_frames": 4},
    )

    transcriber = object.__new__(YourMT3Transcriber)
    transcriber.device = "cpu"
    transcriber._cancelled = False
    transcriber._cancel_check_callback = None
    monkeypatch.setattr(transcriber, "is_selected_model_available", lambda: True)
    monkeypatch.setattr(transcriber, "_get_selected_model_name", lambda: "ymt3_plus")
    monkeypatch.setattr(transcriber, "_load_model", lambda **_kwargs: None)

    result = transcriber.transcribe_to_midi("source.wav", str(output_path))

    assert result == str(output_path.resolve())
    assert calls["load_uri"] == "source.wav"
    assert calls["resample"][1:] == (16_000, 16_000)
    assert torch.equal(calls["resample"][0], torch.zeros((1, 4)))
    assert calls["slice"][1:] == (4, 4)
    assert calls["inference"][0] == 8
    assert calls["inference"][1].shape == (1, 1, 4)
    assert calls["detokenize"][1:] == ([0.0], True)
    assert calls["write"][3] is inverse_vocab
    assert mido.MidiFile(output_path).tracks[0][3].time == 1


def test_official_transcription_emits_real_stable_prefix_snapshots_without_changing_writer(
    monkeypatch,
    tmp_path,
):
    class FakeTaskManager:
        num_decoding_channels = 1

        def detokenize_list_batches(self, _batches, start_secs, return_events):
            assert return_events is True
            return list(start_secs), [], {}

    class FakeModel:
        task_manager = FakeTaskManager()
        midi_output_inverse_vocab = {3: (24, "Guitar")}
        test_pitch_shift_layer = None

        def inference(self, x, _task_tokens):
            return torch.zeros((len(x), 1, 1), dtype=torch.long)

    fake_model = FakeModel()
    waveform = torch.zeros((1, 36), dtype=torch.float32)
    torchaudio = types.ModuleType("torchaudio")
    torchaudio.load = lambda *, uri: (waveform, 4)
    torchaudio.functional = types.SimpleNamespace(resample=lambda audio, _sr, _target: audio)

    utils_package = types.ModuleType("utils")
    utils_package.__path__ = []
    audio_module = types.ModuleType("utils.audio")
    event2note_module = types.ModuleType("utils.event2note")
    note2event_module = types.ModuleType("utils.note2event")
    utils_module = types.ModuleType("utils.utils")
    audio_module.slice_padded_array = lambda *_args: np.zeros((9, 4), dtype=np.float32)
    event2note_module.merge_zipped_note_events_and_ties_to_notes = lambda starts: (
        [
            types.SimpleNamespace(
                is_drum=False,
                program=3,
                onset=float(start),
                offset=float(start) + 0.5,
                pitch=60,
                velocity=1,
            )
            for start in starts
        ],
        {},
    )
    note2event_module.mix_notes = lambda notes_by_channel: list(notes_by_channel[0])

    def fake_write(_notes, output_dir, track_name, _inverse_vocab):
        path = Path(output_dir) / "model_output" / f"{track_name}.mid"
        path.parent.mkdir(parents=True, exist_ok=True)
        midi = mido.MidiFile(type=1, ticks_per_beat=480)
        track = mido.MidiTrack()
        track.extend(
            [
                mido.Message("program_change", program=24, channel=0, time=0),
                mido.Message("note_on", note=60, velocity=100, channel=0, time=0),
                mido.Message("note_off", note=60, velocity=0, channel=0, time=240),
            ]
        )
        midi.tracks.append(track)
        midi.save(path)

    utils_module.write_model_output_as_midi = fake_write
    for name, module in {
        "torchaudio": torchaudio,
        "utils": utils_package,
        "utils.audio": audio_module,
        "utils.event2note": event2note_module,
        "utils.note2event": note2event_module,
        "utils.utils": utils_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    output_path = tmp_path / "streamed.mid"
    attempt_path = tmp_path / ".streamed.yourmt3.tmp.mid"
    monkeypatch.setattr(yourmt3_module, "unique_midi_temp_path", lambda *_args: attempt_path)
    monkeypatch.setattr(YourMT3Transcriber, "_model", fake_model)
    monkeypatch.setattr(
        YourMT3Transcriber,
        "_audio_cfg",
        {"sample_rate": 4, "input_frames": 4},
    )
    transcriber = object.__new__(YourMT3Transcriber)
    transcriber.device = "cpu"
    transcriber._cancelled = False
    transcriber._cancel_check_callback = None
    monkeypatch.setattr(transcriber, "is_selected_model_available", lambda: True)
    monkeypatch.setattr(transcriber, "_get_selected_model_name", lambda: "ymt3_plus")
    monkeypatch.setattr(transcriber, "_load_model", lambda **_kwargs: None)
    events = []
    transcriber.set_event_callback(events.append)

    result = transcriber.transcribe_to_midi("source.wav", str(output_path))

    assert result == str(output_path.resolve())
    assert [(event["completed"], event["total"]) for event in events] == [(8, 9), (9, 9)]
    assert events[0]["frontier_seconds"] == pytest.approx(7.0)
    assert len(events[0]["notes"]) == 7
    assert events[0]["notes"][0]["instrument"] == "gm:024"
    assert events[-1]["frontier_seconds"] == pytest.approx(9.0)
    assert len(events[-1]["notes"]) == 9
    assert mido.MidiFile(output_path).tracks[0][2].time == 240
