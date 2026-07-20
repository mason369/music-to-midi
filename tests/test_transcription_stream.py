from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.core.transcription_stream import (
    append_jsonl_event,
    model_notes_payload,
    read_new_jsonl_events,
    snapshot_event,
)
from src.gui.workers.processing_worker import ProcessingWorker
from src.models.data_models import Config


def test_model_snapshot_contains_only_stable_real_notes_and_writer_programs():
    notes = [
        SimpleNamespace(
            is_drum=False,
            program=7,
            onset=0.1,
            offset=0.8,
            pitch=60,
            velocity=1,
        ),
        SimpleNamespace(
            is_drum=False,
            program=7,
            onset=0.9,
            offset=1.4,
            pitch=64,
            velocity=1,
        ),
    ]

    payload = model_notes_payload(
        notes,
        frontier_seconds=1.0,
        inverse_vocab={7: (24, "Guitar")},
    )

    assert payload == [
        {
            "instrument": "gm:024",
            "program": 24,
            "is_drum": False,
            "pitch": 60,
            "velocity": 100,
            "start": 0.1,
            "end": 0.8,
        }
    ]


def test_jsonl_stream_reader_never_exposes_a_partial_event(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    first = snapshot_event(
        backend="MIROS",
        completed=1,
        total=3,
        frontier_seconds=5.0,
        duration_seconds=15.0,
        notes=[],
    )
    append_jsonl_event(path, first)
    with path.open("ab") as handle:
        handle.write(b'{"type":"snapshot"')

    events, offset = read_new_jsonl_events(path, 0)
    later, later_offset = read_new_jsonl_events(path, offset)

    assert events == [first]
    assert later == []
    assert later_offset == offset


def test_processing_worker_wires_every_backend_to_one_real_event_signal():
    QApplication.instance() or QApplication([])

    class FakeTranscriber:
        def __init__(self):
            self.callback = None

        def set_event_callback(self, callback):
            self.callback = callback

    class FakePipeline:
        def __init__(self, _config):
            for name in (
                "yourmt3_transcriber",
                "miros_transcriber",
                "muscriptor_transcriber",
                "transkun_transcriber",
                "transkun_v2_aug_transcriber",
                "aria_amt_transcriber",
                "bytedance_piano_transcriber",
            ):
                setattr(self, name, FakeTranscriber())

    with mock.patch(
        "src.gui.workers.processing_worker.MusicToMidiPipeline",
        FakePipeline,
    ):
        worker = ProcessingWorker("audio.wav", "output", Config())

    received = []
    worker.transcription_event.connect(received.append)
    payload = {"type": "snapshot", "completed": 1, "total": 1, "notes": []}
    for name in (
        "yourmt3_transcriber",
        "miros_transcriber",
        "muscriptor_transcriber",
        "transkun_transcriber",
        "transkun_v2_aug_transcriber",
        "aria_amt_transcriber",
        "bytedance_piano_transcriber",
    ):
        transcriber = getattr(worker.pipeline, name)
        assert transcriber.callback is not None
        transcriber.callback(payload)

    assert received == [payload] * 7
    worker.deleteLater()
