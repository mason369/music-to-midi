"""Real Qt thread/event-loop regression for device detection during close."""

import threading
import time
from unittest.mock import patch

import pytest
from PyQt6.QtCore import QThread

from src.gui.main_window import MainWindow
from src.i18n.translator import t
from src.models.data_models import Config


@pytest.fixture
def pending_detector(qtbot):
    entered = threading.Event()
    release = threading.Event()

    def query():
        entered.set()
        if not release.wait(10):
            raise TimeoutError("Device query test was not released")
        return False

    with (
        patch("src.utils.gpu_utils.is_gpu_available", side_effect=query),
        patch("src.utils.gpu_utils.get_memory_info", return_value=None),
    ):
        window = MainWindow(Config())
        qtbot.addWidget(window)
        window.show()
        assert entered.wait(3)
        detector = window._gpu_detector
        try:
            yield window, detector, release
        finally:
            release.set()
            from PyQt6 import sip

            if not sip.isdeleted(detector):
                assert detector.wait(3000)
            qtbot.waitUntil(lambda: window._gpu_detector is None, timeout=3000)
            window.close()


def test_close_keeps_window_alive_until_device_thread_finishes(qtbot, pending_detector):
    window, detector, release = pending_detector
    started = time.monotonic()
    assert not window.close()
    assert time.monotonic() - started < 0.5
    assert window.isVisible()
    assert detector.isRunning()
    assert not window.start_btn.isEnabled()
    assert window.status_label.text() == t("status.closing")
    release.set()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=3000)
    assert window._gpu_detector is None
    assert not window._close_pending


def test_repeated_close_does_not_block_or_destroy_active_detector(qtbot, pending_detector):
    window, detector, release = pending_detector
    started = time.monotonic()
    for _ in range(30):
        assert not window.close()
    assert time.monotonic() - started < 0.5
    assert detector.isRunning()
    release.set()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=3000)


class _PendingProcessingThread(QThread):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled = False

    def run(self):
        self.entered.set()
        self.release.wait(10)

    def cancel(self):
        self.cancelled = True


@pytest.mark.parametrize("detector_first", [True, False])
def test_close_waits_for_both_native_threads_in_either_order(
    qtbot, pending_detector, detector_first
):
    window, detector, release_detector = pending_detector
    worker = _PendingProcessingThread()
    window.worker = worker
    worker.finished.connect(lambda: window._on_worker_thread_finished(worker))
    worker.start()
    assert worker.entered.wait(3)
    try:
        assert not window.close()
        assert worker.cancelled
        if detector_first:
            release_detector.set()
            qtbot.waitUntil(lambda: window._gpu_detector is None, timeout=3000)
            assert window.isVisible()
            worker.release.set()
        else:
            worker.release.set()
            qtbot.waitUntil(lambda: window.worker is None, timeout=3000)
            assert window.isVisible()
            release_detector.set()
        qtbot.waitUntil(lambda: not window.isVisible(), timeout=3000)
        assert window.worker is None
        assert window._gpu_detector is None
    finally:
        worker.release.set()
        release_detector.set()
        # Qt may already have processed deleteLater after the finished callback.
        from PyQt6 import sip

        if not sip.isdeleted(worker):
            assert worker.wait(3000)


def test_device_detection_exception_is_visible_and_logged(qtbot, caplog):
    with (
        patch("src.utils.gpu_utils.is_gpu_available", side_effect=RuntimeError("GPU probe failed")),
        patch("src.utils.gpu_utils.get_memory_info", return_value=None),
    ):
        window = MainWindow(Config())
        qtbot.addWidget(window)
        window.show()
        qtbot.waitUntil(lambda: window._gpu_detector is None, timeout=3000)
        assert window.device_label.text() == t("status.device_detection_failed")
        assert window.device_label.toolTip() == "GPU probe failed"
        records = [record for record in caplog.records if record.exc_info]
        assert any(str(record.exc_info[1]) == "GPU probe failed" for record in records)
        assert window.close()


def test_successful_device_detection_still_shows_cpu(qtbot):
    with (
        patch("src.utils.gpu_utils.is_gpu_available", return_value=False),
        patch("src.utils.gpu_utils.get_memory_info", return_value=None),
    ):
        window = MainWindow(Config())
        qtbot.addWidget(window)
        qtbot.waitUntil(lambda: window._gpu_detector is None, timeout=3000)
        assert window.device_label.text() == t("status.cpu")
        assert window.close()
