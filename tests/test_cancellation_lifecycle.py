# ruff: noqa: E402

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QDialog

from src.core.pipeline import MusicToMidiPipeline
from src.gui.main_window import MainWindow
from src.gui.workers.processing_worker import ProcessingWorker
from src.i18n.translator import t
from src.models.data_models import Config, ProcessingResult


class _Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class _FakeWorker:
    def __init__(self, *_args, **_kwargs):
        self.progress_updated = _Signal()
        self.processing_finished = _Signal()
        self.error_occurred = _Signal()
        self.finished = _Signal()
        self.running = False
        self.cancelled = False
        self.deleted = False

    def start(self):
        self.running = True

    def isRunning(self):
        return self.running

    def cancel(self):
        self.cancelled = True

    def deleteLater(self):
        self.deleted = True


class _CloseEvent:
    def __init__(self):
        self.ignored = False

    def ignore(self):
        self.ignored = True


class CancellationLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def _window(self):
        with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
            return MainWindow(Config())

    def test_pipeline_honors_cancel_delivered_before_process_entry(self):
        pipeline = object.__new__(MusicToMidiPipeline)
        pipeline._cancelled = True
        pipeline._progress_callback = None
        pipeline.config = mock.Mock()
        pipeline._ensure_wav = mock.Mock(return_value="song.wav")

        with self.assertRaisesRegex(InterruptedError, "用户取消"):
            pipeline.process("song.mp3", "output")

        pipeline.config.validate.assert_not_called()
        pipeline._ensure_wav.assert_not_called()

    def test_route_components_do_not_erase_pre_entry_cancel(self):
        from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber
        from src.core.miros_transcriber import MirosTranscriber
        from src.core.transkun_transcriber import TranskunTranscriber
        from src.core.transkun_v2_aug_transcriber import TranskunV2AugTranscriber
        from src.core.vocal_separator import VocalSeparator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"audio")
            output_path = root / "song.mid"

            bytedance = ByteDancePianoTranscriber(checkpoint_path=root / "bytedance.pth")
            bytedance.cancel()
            with (
                mock.patch.object(ByteDancePianoTranscriber, "is_available", return_value=True),
                mock.patch.object(bytedance, "is_model_available", return_value=True),
            ):
                with self.assertRaises(InterruptedError):
                    bytedance.transcribe(str(audio_path), str(output_path))

            transkun = TranskunTranscriber()
            transkun.cancel()
            with (
                mock.patch.object(transkun, "get_unavailable_reason", return_value=""),
                mock.patch.object(transkun, "is_model_available", return_value=True),
            ):
                with self.assertRaises(InterruptedError):
                    transkun.transcribe(str(audio_path), str(output_path))

            transkun_v2_aug = TranskunV2AugTranscriber(model_dir=root)
            transkun_v2_aug.cancel()
            with (
                mock.patch.object(
                    transkun_v2_aug,
                    "get_unavailable_reason",
                    return_value="",
                ),
                mock.patch.object(
                    transkun_v2_aug,
                    "get_model_validation_error",
                    return_value="",
                ),
            ):
                with self.assertRaises(InterruptedError):
                    transkun_v2_aug.transcribe(str(audio_path), str(output_path))

            miros = MirosTranscriber()
            miros.cancel()
            with (
                mock.patch.object(miros, "_repo_dir", return_value=root),
                mock.patch.object(miros, "_entrypoint_path", return_value=root / "run.py"),
                mock.patch.object(miros, "get_unavailable_reason", return_value=""),
            ):
                with self.assertRaises(InterruptedError):
                    miros.transcribe_to_midi(str(audio_path), str(output_path))

            vocal = VocalSeparator()
            vocal.cancel()
            with self.assertRaises(InterruptedError):
                vocal.separate(str(audio_path), str(root / "stems"))

    def test_pipeline_cancel_terminates_active_separator(self):
        pipeline = object.__new__(MusicToMidiPipeline)
        pipeline._cancelled = False
        pipeline._active_separator = mock.Mock()
        for name in (
            "yourmt3_transcriber",
            "miros_transcriber",
            "muscriptor_transcriber",
            "aria_amt_transcriber",
            "bytedance_piano_transcriber",
            "transkun_transcriber",
            "transkun_v2_aug_transcriber",
        ):
            setattr(pipeline, name, mock.Mock())

        pipeline.cancel()

        self.assertTrue(pipeline._cancelled)
        pipeline._active_separator.cancel.assert_called_once_with()

    def test_each_processing_worker_owns_a_fresh_single_job_pipeline(self):
        pipelines = [
            mock.Mock(name="first_pipeline"),
            mock.Mock(name="second_pipeline"),
        ]
        with mock.patch(
            "src.gui.workers.processing_worker.MusicToMidiPipeline",
            side_effect=pipelines,
        ) as pipeline_factory:
            first = ProcessingWorker("first.wav", "output-1", Config())
            second = ProcessingWorker("second.wav", "output-2", Config())

        self.assertIs(first.pipeline, pipelines[0])
        self.assertIs(second.pipeline, pipelines[1])
        self.assertIsNot(first.pipeline, second.pipeline)
        self.assertEqual(pipeline_factory.call_count, 2)
        first.deleteLater()
        second.deleteLater()

    def test_worker_finished_signal_is_connected_to_lifecycle_handler(self):
        window = self._window()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                window.current_file = str(Path(tmp) / "song.wav")
                window.output_dir_edit.setText(tmp)
                with mock.patch("src.gui.main_window.ProcessingWorker", _FakeWorker):
                    window._start_processing()

            worker = window.worker
            self.assertEqual(len(worker.finished.callbacks), 1)
            self.assertTrue(worker.running)
        finally:
            if window.worker is not None:
                window.worker.running = False
                window.worker.finished.callbacks[0]()
            window.close()

    def test_stop_remains_non_ready_until_worker_really_finishes(self):
        window = self._window()
        worker = _FakeWorker()
        worker.running = True
        window.worker = worker
        try:
            window._stop_processing()

            self.assertTrue(worker.cancelled)
            self.assertTrue(window._stopping)
            self.assertFalse(window.start_btn.isEnabled())
            self.assertFalse(window.stop_btn.isEnabled())
            self.assertFalse(window.track_panel._controls_enabled)
            self.assertEqual(window.status_label.text(), t("status.cancelling"))

            worker.running = False
            window._on_worker_thread_finished(worker)

            self.assertIsNone(window.worker)
            self.assertTrue(worker.deleted)
            self.assertFalse(window._stopping)
            self.assertTrue(window.start_btn.isEnabled())
            self.assertTrue(window.track_panel._controls_enabled)
            self.assertEqual(window.status_label.text(), t("status.cancelled"))
        finally:
            window.worker = None
            window.close()

    def test_close_is_deferred_until_running_worker_finishes(self):
        window = self._window()
        worker = _FakeWorker()
        worker.running = True
        window.worker = worker
        event = _CloseEvent()
        try:
            window.closeEvent(event)

            self.assertTrue(event.ignored)
            self.assertTrue(worker.cancelled)
            self.assertTrue(window._close_pending)
            self.assertIs(window.worker, worker)
            self.assertEqual(window.status_label.text(), t("status.cancelling"))

            worker.running = False
            with mock.patch.object(window, "close") as close:
                window._on_worker_thread_finished(worker)

            close.assert_called_once_with()
            self.assertIsNone(window.worker)
            self.assertTrue(worker.deleted)
            self.assertFalse(window._close_pending)
        finally:
            window.worker = None
            window.close()

    def test_success_keeps_start_disabled_until_exact_worker_finishes(self):
        window = self._window()
        worker = _FakeWorker()
        worker.running = True
        window.worker = worker
        window.start_btn.setEnabled(False)
        window.track_panel.set_processing_controls_enabled(False)
        result = ProcessingResult(midi_path="C:/output/song.mid", processing_time=1.0)
        try:
            with mock.patch.object(QDialog, "exec", return_value=0):
                window._on_finished(result)

            self.assertFalse(window.start_btn.isEnabled())
            self.assertFalse(window.track_panel._controls_enabled)
            self.assertIs(window.worker, worker)

            worker.running = False
            window._on_worker_thread_finished(worker)

            self.assertTrue(window.start_btn.isEnabled())
            self.assertTrue(window.track_panel._controls_enabled)
            self.assertIsNone(window.worker)
        finally:
            window.worker = None
            window.close()

    def test_error_keeps_start_disabled_until_exact_worker_finishes(self):
        window = self._window()
        worker = _FakeWorker()
        worker.running = True
        window.worker = worker
        window.start_btn.setEnabled(False)
        window.track_panel.set_processing_controls_enabled(False)
        try:
            with mock.patch.object(QDialog, "exec", return_value=0):
                window._on_error("failure")

            self.assertFalse(window.start_btn.isEnabled())
            self.assertFalse(window.track_panel._controls_enabled)
            self.assertIs(window.worker, worker)

            worker.running = False
            window._on_worker_thread_finished(worker)

            self.assertTrue(window.start_btn.isEnabled())
            self.assertTrue(window.track_panel._controls_enabled)
            self.assertIsNone(window.worker)
        finally:
            window.worker = None
            window.close()

    def test_stale_finished_signal_never_clears_or_deletes_current_worker(self):
        window = self._window()
        old_worker = _FakeWorker()
        current_worker = _FakeWorker()
        current_worker.running = True
        window.worker = current_worker
        try:
            window._on_worker_thread_finished(old_worker)

            self.assertIs(window.worker, current_worker)
            self.assertTrue(old_worker.deleted)
            self.assertFalse(current_worker.deleted)
        finally:
            window.worker = None
            window.close()


if __name__ == "__main__":
    unittest.main()
