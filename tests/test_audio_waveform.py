import os
import tempfile
import time
import unittest
import wave
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QColor, QWheelEvent
from PyQt6.QtTest import QSignalSpy, QTest
from PyQt6.QtWidgets import QApplication

from src.gui.widgets.audio_waveform import (
    FfmpegWaveformLoader,
    TimelineRuler,
    WaveformEnvelope,
    WaveformLane,
)


class AudioWaveformTests(unittest.TestCase):
    SAMPLE_RATE = 8_000
    SAMPLES_PER_BUCKET = 256

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self._widgets = []
        self._loaders = []

    def tearDown(self):
        for loader in self._loaders:
            loader.cancel()
            loader.wait(5_000)
            loader.deleteLater()
        for widget in self._widgets:
            widget.close()
            widget.deleteLater()
        self._app.processEvents()
        self._temporary_directory.cleanup()

    def _known_wav(self, filename="known.wav") -> Path:
        bucket = self.SAMPLES_PER_BUCKET
        left = np.concatenate(
            [
                np.zeros(bucket, dtype=np.float32),
                np.full(bucket, 0.25, dtype=np.float32),
                np.full(bucket, 0.75, dtype=np.float32),
                np.linspace(-0.4, 0.8, bucket, dtype=np.float32),
            ]
        )
        right = left.copy()
        right[2 * bucket : 3 * bucket] = -0.75
        samples = np.column_stack((left, right))
        pcm = np.rint(samples * 32767.0).astype("<i2")
        path = Path(self._temporary_directory.name) / filename
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(2)
            stream.setsampwidth(2)
            stream.setframerate(self.SAMPLE_RATE)
            stream.writeframes(pcm.tobytes())
        return path

    def _load(self, path: Path) -> tuple[QSignalSpy, QSignalSpy]:
        loader = FfmpegWaveformLoader(samples_per_bucket=self.SAMPLES_PER_BUCKET)
        self._loaders.append(loader)
        loaded = QSignalSpy(loader.loaded)
        failed = QSignalSpy(loader.failed)
        loader.start(path)

        deadline = time.monotonic() + 10.0
        while loader.isRunning() and time.monotonic() < deadline:
            loader.wait(20)
            self._app.processEvents()
        self.assertTrue(loader.wait(1_000), "waveform loader did not finish")
        self._app.processEvents()
        return loaded, failed

    def test_ffmpeg_loader_measures_real_pcm_minimums_and_maximums(self):
        loaded, failed = self._load(self._known_wav())

        self.assertEqual(len(failed), 0)
        self.assertEqual(len(loaded), 1)
        envelope = loaded[0][0]
        self.assertIsInstance(envelope, WaveformEnvelope)
        self.assertEqual(envelope.sample_rate, self.SAMPLE_RATE)
        self.assertEqual(envelope.samples_per_bucket, self.SAMPLES_PER_BUCKET)
        self.assertEqual(envelope.frame_count, self.SAMPLES_PER_BUCKET * 4)
        self.assertEqual(envelope.bucket_count, 4)
        self.assertEqual(envelope.duration_ms, 128)

        np.testing.assert_allclose(envelope.minimums[0], 0.0, atol=1e-5)
        np.testing.assert_allclose(envelope.maximums[0], 0.0, atol=1e-5)
        np.testing.assert_allclose(envelope.minimums[1], 0.25, atol=2e-4)
        np.testing.assert_allclose(envelope.maximums[1], 0.25, atol=2e-4)
        # Opposite-polarity channels must remain visible instead of cancelling
        # through a mono downmix.
        np.testing.assert_allclose(envelope.minimums[2], -0.75, atol=2e-4)
        np.testing.assert_allclose(envelope.maximums[2], 0.75, atol=2e-4)
        self.assertLess(envelope.minimums[3], -0.39)
        self.assertGreater(envelope.maximums[3], 0.79)
        self.assertFalse(envelope.minimums.flags.writeable)
        self.assertFalse(envelope.maximums.flags.writeable)

    def test_loader_reports_missing_audio_as_an_explicit_failure(self):
        missing = Path(self._temporary_directory.name) / "missing.wav"
        loaded, failed = self._load(missing)

        self.assertEqual(len(loaded), 0)
        self.assertEqual(len(failed), 1)
        self.assertIn("does not exist", failed[0][0])
        self.assertIn("missing.wav", failed[0][0])

    def test_ffmpeg_loader_accepts_unicode_windows_filenames(self):
        loaded, failed = self._load(self._known_wav("小林愛香 - 参考音轨.wav"))

        self.assertEqual(len(failed), 0)
        self.assertEqual(len(loaded), 1)
        self.assertGreater(loaded[0][0].bucket_count, 0)

    def test_cancel_before_worker_entry_emits_no_false_result(self):
        loader = FfmpegWaveformLoader()
        self._loaders.append(loader)
        loaded = QSignalSpy(loader.loaded)
        failed = QSignalSpy(loader.failed)

        loader.start(self._known_wav("cancel.wav"))
        loader.cancel()
        self.assertTrue(loader.wait(5_000))
        self._app.processEvents()

        self.assertEqual(len(loaded), 0)
        self.assertEqual(len(failed), 0)

    def test_waveform_lane_renders_real_envelope_and_shared_playhead(self):
        envelope = WaveformEnvelope(
            minimums=np.asarray([-0.8, -0.25, -0.65, -0.1], dtype=np.float32),
            maximums=np.asarray([0.7, 0.5, 0.9, 0.2], dtype=np.float32),
            sample_rate=self.SAMPLE_RATE,
            frame_count=self.SAMPLES_PER_BUCKET * 4,
            samples_per_bucket=self.SAMPLES_PER_BUCKET,
        )
        lane = WaveformLane(self._known_wav(), QColor("#4a9eff"))
        self._widgets.append(lane)
        lane.resize(401, 80)
        lane.set_envelope(envelope)
        lane.set_timeline(128, 0, 128, 64, 0)
        lane.show()
        self._app.processEvents()

        image = lane.grab().toImage()
        waveform_color = QColor("#4a9eff").rgb()
        waveform_pixels = sum(
            image.pixel(x, y) == waveform_color
            for x in range(image.width())
            for y in range(image.height())
        )
        self.assertGreater(waveform_pixels, 150)

        playhead = image.pixelColor(200, 4)
        self.assertGreater(playhead.red(), 220)
        self.assertLess(playhead.green(), 150)
        self.assertLess(playhead.blue(), 150)

    def test_waveform_lane_click_drag_and_ctrl_wheel_map_to_timeline(self):
        lane = WaveformLane(self._known_wav(), "#4a9eff")
        self._widgets.append(lane)
        lane.resize(401, 80)
        lane.set_timeline(1_000, 100, 900, 400, 0)
        lane.show()
        self._app.processEvents()
        seeks = QSignalSpy(lane.seek_requested)
        zooms = QSignalSpy(lane.zoom_requested)

        QTest.mouseClick(lane, Qt.MouseButton.LeftButton, pos=QPoint(100, 40))
        self.assertEqual(seeks[-1][0], 300)

        QTest.mousePress(lane, Qt.MouseButton.LeftButton, pos=QPoint(50, 40))
        QTest.mouseMove(lane, QPoint(300, 40), delay=1)
        QTest.mouseRelease(lane, Qt.MouseButton.LeftButton, pos=QPoint(300, 40))
        self.assertEqual(seeks[-1][0], 700)

        wheel = QWheelEvent(
            QPointF(200, 40),
            QPointF(200, 40),
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.ControlModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )
        QApplication.sendEvent(lane, wheel)
        self.assertEqual(len(zooms), 0)

    def test_playhead_updates_reuse_the_rendered_waveform_cache(self):
        envelope = WaveformEnvelope(
            minimums=np.asarray([-0.8, -0.3, -0.6, -0.2], dtype=np.float32),
            maximums=np.asarray([0.7, 0.4, 0.9, 0.3], dtype=np.float32),
            sample_rate=self.SAMPLE_RATE,
            frame_count=self.SAMPLES_PER_BUCKET * 4,
            samples_per_bucket=self.SAMPLES_PER_BUCKET,
        )
        lane = WaveformLane(self._known_wav("cache.wav"), "#4a9eff")
        self._widgets.append(lane)
        lane.resize(401, 80)
        lane.set_envelope(envelope)
        lane.set_timeline(128, 0, 128, 32, 0)
        lane.show()
        self._app.processEvents()

        lane.grab()
        first_cache_key = lane._waveform_cache.cacheKey()
        lane.set_timeline(128, 0, 128, 96, 0)
        lane.grab()

        self.assertEqual(lane._waveform_cache.cacheKey(), first_cache_key)

        lane.set_timeline(128, 16, 112, 96, 0)
        lane.grab()
        self.assertNotEqual(lane._waveform_cache.cacheKey(), first_cache_key)

    def test_timeline_ruler_renders_playhead_and_supports_seek(self):
        ruler = TimelineRuler()
        self._widgets.append(ruler)
        ruler.resize(401, 32)
        ruler.set_timeline(1_000, 0, 1_000, 500)
        ruler.show()
        self._app.processEvents()
        seeks = QSignalSpy(ruler.seek_requested)

        image = ruler.grab().toImage()
        playhead = image.pixelColor(200, 16)
        self.assertGreater(playhead.red(), 220)
        self.assertLess(playhead.green(), 150)
        self.assertLess(playhead.blue(), 150)

        QTest.mouseClick(ruler, Qt.MouseButton.LeftButton, pos=QPoint(300, 16))
        self.assertEqual(seeks[-1][0], 750)


if __name__ == "__main__":
    unittest.main()
