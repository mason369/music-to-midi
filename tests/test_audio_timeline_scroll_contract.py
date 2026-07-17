import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QObject, QPoint, QPointF, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtTest import QSignalSpy
from PyQt6.QtWidgets import QApplication, QScrollArea, QSlider

from src.gui.widgets.audio_track_mixer import AudioTrackMixerWidget
from src.gui.widgets.audio_waveform import WaveformEnvelope
from src.gui.widgets.track_panel import TrackPanel
from src.gui.widgets.wheel_safe_controls import NoWheelComboBox, NoWheelSlider
from src.i18n.translator import set_language


class _FakePlayer(QObject):
    durationChanged = pyqtSignal(int)
    mediaStatusChanged = pyqtSignal(object)
    errorOccurred = pyqtSignal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio_output = None
        self._source = QUrl()
        self._position_ms = 0
        self._state = QMediaPlayer.PlaybackState.StoppedState

    def setAudioOutput(self, output):
        self._audio_output = output

    def setSource(self, source):
        self._source = source
        if source.isEmpty():
            self._position_ms = 0
            self._state = QMediaPlayer.PlaybackState.StoppedState
            return
        self.durationChanged.emit(30_000)
        self.mediaStatusChanged.emit(QMediaPlayer.MediaStatus.LoadedMedia)

    def errorString(self):
        return ""

    def position(self):
        return self._position_ms

    def setPosition(self, position_ms):
        self._position_ms = int(position_ms)

    def playbackState(self):
        return self._state

    def play(self):
        self._state = QMediaPlayer.PlaybackState.PlayingState

    def pause(self):
        self._state = QMediaPlayer.PlaybackState.PausedState

    def stop(self):
        self._state = QMediaPlayer.PlaybackState.StoppedState


class _FakeAudioOutput(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.volume = 1.0
        self.muted = False

    def setVolume(self, volume):
        self.volume = float(volume)

    def setMuted(self, muted):
        self.muted = bool(muted)


class _FakeWaveformLoader(QObject):
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cancelled = False

    def start(self, _path):
        self.loaded.emit(
            WaveformEnvelope(
                minimums=np.asarray([-0.7, -0.4, -0.8, -0.2], dtype=np.float32),
                maximums=np.asarray([0.6, 0.5, 0.9, 0.3], dtype=np.float32),
                sample_rate=1_000,
                frame_count=30_000,
                samples_per_bucket=7_500,
            )
        )

    def cancel(self):
        self.cancelled = True


class AudioTimelineScrollContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        set_language("en_US")
        self._temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self._temporary_directory.name)
        self.track_paths = {
            name: root / f"{name}.wav"
            for name in ("bass", "drums", "guitar", "piano", "vocals", "other")
        }
        for path in self.track_paths.values():
            path.write_bytes(b"test-audio-placeholder")
        self._widgets = []

    def tearDown(self):
        for widget in reversed(self._widgets):
            shutdown = getattr(widget, "shutdown", None)
            if shutdown is not None:
                shutdown()
            widget.close()
            widget.deleteLater()
        self._app.processEvents()
        self._temporary_directory.cleanup()
        set_language("zh_CN")

    def _mixer(self):
        mixer = AudioTrackMixerWidget(
            self.track_paths,
            player_factory=_FakePlayer,
            audio_output_factory=_FakeAudioOutput,
            waveform_loader_factory=_FakeWaveformLoader,
        )
        self._widgets.append(mixer)
        return mixer

    @staticmethod
    def _send_wheel(widget, modifiers, delta=120):
        local_point = widget.rect().center()
        event = QWheelEvent(
            QPointF(local_point),
            QPointF(widget.mapToGlobal(local_point)),
            QPoint(),
            QPoint(0, delta),
            Qt.MouseButton.NoButton,
            modifiers,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )
        QApplication.sendEvent(widget, event)

    def test_six_tracks_expand_without_nested_scroll_area(self):
        mixer = self._mixer()
        rows = [mixer._backends[name].row for name in mixer.track_names]

        self.assertEqual(len(rows), 6)
        self.assertEqual(mixer.findChildren(QScrollArea), [])
        self.assertTrue(mixer.view_scroll.isHidden())

        row_spacing = mixer._track_layout.spacing()
        expected_track_height = sum(row.minimumSizeHint().height() for row in rows)
        expected_track_height += row_spacing * (len(rows) - 1)
        self.assertGreaterEqual(
            mixer.track_container.minimumSizeHint().height(),
            expected_track_height,
        )
        self.assertGreaterEqual(
            mixer.minimumSizeHint().height(),
            mixer.track_container.minimumSizeHint().height(),
        )

        mixer.resize(1_400, mixer.minimumSizeHint().height())
        mixer.show()
        self._app.processEvents()
        for row in rows:
            with self.subTest(track=row.track_name):
                top_left = row.mapTo(mixer.track_container, row.rect().topLeft())
                bottom_right = row.mapTo(
                    mixer.track_container,
                    row.rect().bottomRight(),
                )
                self.assertGreaterEqual(top_left.y(), 0)
                self.assertLessEqual(
                    bottom_right.y(),
                    mixer.track_container.rect().bottom(),
                )

    def test_mixer_sliders_and_waveforms_ignore_plain_and_ctrl_wheel(self):
        mixer = self._mixer()
        mixer.resize(1_400, mixer.minimumSizeHint().height())
        mixer.show()
        self._app.processEvents()

        sliders = mixer.findChildren(QSlider)
        expected_sliders = [mixer.zoom_slider, mixer.timeline, mixer.view_scroll]
        for track_name in mixer.track_names:
            row = mixer._backends[track_name].row
            expected_sliders.extend((row.volume_slider, row.offset_slider))
        self.assertCountEqual(sliders, expected_sliders)
        for slider in sliders:
            self.assertIsInstance(slider, NoWheelSlider)
            if slider.minimum() < slider.maximum():
                slider.setValue((slider.minimum() + slider.maximum()) // 2)
            initial_value = slider.value()
            changes = QSignalSpy(slider.valueChanged)
            for modifiers in (
                Qt.KeyboardModifier.NoModifier,
                Qt.KeyboardModifier.ControlModifier,
            ):
                self._send_wheel(slider, modifiers, 120)
                self._send_wheel(slider, modifiers, -120)
            with self.subTest(slider=slider.objectName()):
                self.assertEqual(slider.value(), initial_value)
                self.assertEqual(len(changes), 0)

        timeline_widgets = [mixer.timeline_ruler]
        timeline_widgets.extend(mixer._backends[name].row.waveform for name in mixer.track_names)
        initial_zoom = mixer._zoom_factor
        for timeline_widget in timeline_widgets:
            zooms = QSignalSpy(timeline_widget.zoom_requested)
            for modifiers in (
                Qt.KeyboardModifier.NoModifier,
                Qt.KeyboardModifier.ControlModifier,
            ):
                self._send_wheel(timeline_widget, modifiers, 120)
                self._send_wheel(timeline_widget, modifiers, -120)
            with self.subTest(widget=timeline_widget.objectName()):
                self.assertEqual(len(zooms), 0)
        self.assertEqual(mixer._zoom_factor, initial_zoom)

    def test_combo_box_ignores_plain_and_ctrl_wheel(self):
        panel = TrackPanel()
        self._widgets.append(panel)
        panel.show()
        self._app.processEvents()

        for combo in (
            panel.mode_combo,
            panel.model_combo,
            panel.yourmt3_model_combo,
        ):
            self.assertIsInstance(combo, NoWheelComboBox)
            self.assertGreater(combo.count(), 1)
            combo.setCurrentIndex(1)
            initial_index = combo.currentIndex()
            changes = QSignalSpy(combo.currentIndexChanged)
            for modifiers in (
                Qt.KeyboardModifier.NoModifier,
                Qt.KeyboardModifier.ControlModifier,
            ):
                self._send_wheel(combo, modifiers, 120)
                self._send_wheel(combo, modifiers, -120)
            with self.subTest(combo=combo.objectName()):
                self.assertEqual(combo.currentIndex(), initial_index)
                self.assertEqual(len(changes), 0)

    def test_per_track_midi_model_selectors_ignore_mouse_wheel(self):
        mixer = self._mixer()
        mixer.show()
        self._app.processEvents()

        for track_name in mixer.track_names:
            row = mixer._backends[track_name].row
            combo = row.midi_model_selector
            row.midi_enabled_checkbox.setChecked(True)
            self.assertIsInstance(combo, NoWheelComboBox)
            combo.setCurrentIndex(1)
            initial_index = combo.currentIndex()
            changes = QSignalSpy(combo.currentIndexChanged)
            for modifiers in (
                Qt.KeyboardModifier.NoModifier,
                Qt.KeyboardModifier.ControlModifier,
            ):
                self._send_wheel(combo, modifiers, 120)
                self._send_wheel(combo, modifiers, -120)
            with self.subTest(track=track_name):
                self.assertEqual(combo.currentIndex(), initial_index)
                self.assertEqual(len(changes), 0)


if __name__ == "__main__":
    unittest.main()
