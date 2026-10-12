"""Observe the real upstream writer without changing its MIDI or analysis count."""

from __future__ import annotations

import io
import logging

import mido
import numpy as np
import pytest

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.utils import beats as upstream_beats
from muscriptor.utils.beats import BeatGrid

from src.core.midi_tempo import read_muscriptor_bar_offset_seconds
from src.core.muscriptor_transcriber import MuscriptorTranscriber
from src.models.data_models import BeatInfo, Config


class EventReplay(TranscriptionModel):
    def __init__(self, events):
        self.events = events
        self._inst_to_program = {"acoustic_piano": 0}
        self.calls = 0

    def transcribe(self, _source, **_kwargs):
        self.calls += 1
        yield from self.events


def _events(delay):
    events = [ProgressEvent(0, 1)]
    for index in range(64):
        start = NoteStartEvent(
            index=index,
            instrument="acoustic_piano",
            pitch=60 + index % 12,
            start_time=0.5 * (index + 1) + delay,
        )
        events.extend((start, NoteEndEvent(start_event=start, end_time=start.start_time + 0.15)))
    # The writer, not the observer, owns duplicate/overlap cleanup and open notes.
    duplicate = NoteStartEvent(
        index=64, instrument="acoustic_piano", pitch=60, start_time=0.5 + delay
    )
    events.extend((duplicate, NoteEndEvent(start_event=duplicate, end_time=0.6 + delay)))
    events.append(NoteStartEvent(index=65, instrument="acoustic_piano", pitch=72, start_time=0.731))
    events.append(ProgressEvent(1, 1))
    return events


def _transcriber(events, meter=4, chain="official"):
    transcriber = MuscriptorTranscriber(Config(use_gpu=False, muscriptor_processing_chain=chain))
    transcriber._model = EventReplay(events)
    transcriber._loaded_processing_chain = chain
    transcriber._runtime_details = {"type": "runtime"}
    transcriber.set_beat_info(
        BeatInfo(
            bpm=120.0,
            beat_times=np.arange(0.0, 35.0, 0.5).tolist(),
            time_signature=(meter, 4) if meter else None,
        )
    )
    return transcriber


@pytest.mark.parametrize("chain", ["official", "telknet"])
@pytest.mark.parametrize("meter", [4, None])
@pytest.mark.parametrize("delay", [0.020, -0.020, 0.0])
def test_runtime_matches_real_writer_and_midi_is_byte_identical(
    tmp_path, monkeypatch, caplog, chain, meter, delay
):
    from unittest.mock import Mock

    measure = Mock(wraps=upstream_beats.estimate_onset_delay)
    monkeypatch.setattr(upstream_beats, "estimate_onset_delay", measure)
    events = _events(delay)
    transcriber = _transcriber(events, meter, chain)
    grid = BeatGrid(
        bpm=120.0,
        beats_per_bar=meter,
        first_downbeat=0.0,
        beats=np.asarray(transcriber._beat_info.beat_times),
    )
    reference = EventReplay(events).events_to_midi_bytes(iter(events), beat_grid=grid)
    assert measure.call_count == 1
    writer_onsets = measure.call_args.args[0]
    assert 0.731 not in writer_onsets
    output = tmp_path / "observed.mid"
    source = tmp_path / "source.wav"
    source.touch()
    runtime = []

    def observe(event):
        if event["type"] == "runtime":
            if event["onset_phase_correction"] != "pending":
                assert output.read_bytes() == reference
            runtime.append(event)

    transcriber.set_event_callback(observe)
    with caplog.at_level(logging.INFO, logger="src.core.muscriptor_transcriber"):
        transcriber.transcribe_to_midi(str(source), str(output))

    assert output.read_bytes() == reference
    assert transcriber._model.calls == 1
    assert measure.call_count == 2
    assert measure.call_args.args[0] == writer_onsets
    assert runtime[0]["bar_offset_seconds"] is None
    assert runtime[0]["onset_correction_applied"] is None
    details = runtime[-1]
    assert details["onset_phase_correction"] == "official_writer"
    assert details["onset_delay_seconds"] == pytest.approx(delay, abs=1e-12)
    expected_offset = (2.0 if meter else 0.5) if delay > 0 else 0.0
    assert details["bar_offset_seconds"] == pytest.approx(expected_offset)
    assert details["midi_time_shift_seconds"] == pytest.approx(expected_offset - delay)
    midi = mido.MidiFile(file=io.BytesIO(reference))
    assert read_muscriptor_bar_offset_seconds(midi) == pytest.approx(expected_offset)
    assert f"net_shift={expected_offset - delay:+.6f} s" in caplog.text
    assert "disabled_native_seconds" not in caplog.text


def test_reused_transcriber_resets_timing_until_the_next_writer_finishes(tmp_path):
    transcriber = _transcriber(_events(0.020))
    source = tmp_path / "source.wav"
    source.touch()
    runtime = []
    transcriber.set_event_callback(
        lambda event: runtime.append(event) if event["type"] == "runtime" else None
    )
    transcriber.transcribe_to_midi(str(source), str(tmp_path / "first.mid"))
    assert transcriber.last_bar_offset_seconds == pytest.approx(2.0)
    transcriber._model.events = _events(-0.020)
    transcriber.transcribe_to_midi(str(source), str(tmp_path / "second.mid"))
    assert runtime[2]["onset_delay_seconds"] is None
    assert runtime[2]["bar_offset_seconds"] is None
    assert transcriber.last_bar_offset_seconds == pytest.approx(0.0)
    assert transcriber.last_onset_delay_seconds == pytest.approx(-0.020)


def test_failed_writer_does_not_emit_completed_timing_or_publish(tmp_path, monkeypatch):
    transcriber = _transcriber(_events(0.020))
    source = tmp_path / "source.wav"
    source.touch()
    output = tmp_path / "output.mid"
    runtime = []
    transcriber.set_event_callback(
        lambda event: runtime.append(event) if event["type"] == "runtime" else None
    )

    def fail_writer(*_args, **_kwargs):
        raise RuntimeError("writer failed")

    monkeypatch.setattr(transcriber._model, "events_to_midi_bytes", fail_writer)
    with pytest.raises(RuntimeError, match="writer failed"):
        transcriber.transcribe_to_midi(str(source), str(output))
    assert len(runtime) == 1
    assert runtime[0]["onset_phase_correction"] == "pending"
    assert transcriber.last_bar_offset_seconds is None
    assert not output.exists()
