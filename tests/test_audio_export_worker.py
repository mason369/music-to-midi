from pathlib import Path

from src.gui.workers.audio_export_worker import AudioExportItem, AudioExportWorker


def _run_worker(worker: AudioExportWorker):
    succeeded = []
    failed = []
    cancelled = []
    worker.export_succeeded.connect(succeeded.append)
    worker.export_failed.connect(failed.append)
    worker.export_cancelled.connect(lambda: cancelled.append(True))
    worker.run()
    return succeeded, failed, cancelled


def test_audio_export_worker_publishes_complete_files_only(tmp_path: Path):
    source_a = tmp_path / "bass.wav"
    source_b = tmp_path / "drums.wav"
    source_a.write_bytes(b"bass-audio" * 128)
    source_b.write_bytes(b"drums-audio" * 128)
    destination = tmp_path / "saved"
    destination.mkdir()

    worker = AudioExportWorker(
        (
            AudioExportItem("bass", source_a, destination / source_a.name),
            AudioExportItem("drums", source_b, destination / source_b.name),
        )
    )
    succeeded, failed, cancelled = _run_worker(worker)

    assert failed == []
    assert cancelled == []
    assert succeeded == [(str(destination / "bass.wav"), str(destination / "drums.wav"))]
    assert (destination / "bass.wav").read_bytes() == source_a.read_bytes()
    assert (destination / "drums.wav").read_bytes() == source_b.read_bytes()
    assert list(destination.glob("*.music-to-midi.part")) == []


def test_audio_export_worker_reports_missing_source_without_fake_output(tmp_path: Path):
    destination = tmp_path / "saved.wav"
    worker = AudioExportWorker((AudioExportItem("missing", tmp_path / "missing.wav", destination),))

    succeeded, failed, cancelled = _run_worker(worker)

    assert succeeded == []
    assert cancelled == []
    assert len(failed) == 1
    assert "FileNotFoundError" in failed[0]
    assert not destination.exists()


def test_audio_export_worker_cancellation_leaves_no_partial_file(tmp_path: Path):
    source = tmp_path / "vocals.wav"
    source.write_bytes(b"vocal-audio")
    destination = tmp_path / "saved.wav"
    worker = AudioExportWorker((AudioExportItem("vocals", source, destination),))
    worker.cancel()

    succeeded, failed, cancelled = _run_worker(worker)

    assert succeeded == []
    assert failed == []
    assert cancelled == [True]
    assert not destination.exists()
