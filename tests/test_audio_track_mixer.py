import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QObject, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from src.gui.widgets.audio_track_mixer import AudioTrackMixerWidget
from src.gui.widgets.audio_waveform import WaveformEnvelope
from src.i18n.translator import set_language


class _FakePlayer(QObject):
    durationChanged = pyqtSignal(int)
    mediaStatusChanged = pyqtSignal(object)
    errorOccurred = pyqtSignal(object, str)

    def __init__(self, parent=None, *, duration_ms=30_000):
        super().__init__(parent)
        self.duration_ms = duration_ms
        self.output = None
        self.source = QUrl()
        self.position_ms = 0
        self.state = QMediaPlayer.PlaybackState.StoppedState
        self.stop_count = 0

    def setAudioOutput(self, output):
        self.output = output

    def setSource(self, source):
        self.source = source
        if source.isEmpty():
            self.position_ms = 0
            self.state = QMediaPlayer.PlaybackState.StoppedState
            return
        self.durationChanged.emit(self.duration_ms)
        self.mediaStatusChanged.emit(QMediaPlayer.MediaStatus.LoadedMedia)

    def errorString(self):
        return ""

    def position(self):
        return self.position_ms

    def setPosition(self, position_ms):
        self.position_ms = int(position_ms)

    def playbackState(self):
        return self.state

    def play(self):
        self.state = QMediaPlayer.PlaybackState.PlayingState

    def pause(self):
        self.state = QMediaPlayer.PlaybackState.PausedState

    def stop(self):
        self.stop_count += 1
        self.state = QMediaPlayer.PlaybackState.StoppedState


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

    def __init__(self, parent=None, *, duration_ms=30_000):
        super().__init__(parent)
        self.duration_ms = duration_ms
        self.path = None
        self.cancel_count = 0

    def start(self, path):
        self.path = Path(path)
        bucket_count = 120
        minimums = np.linspace(-0.1, -0.8, bucket_count, dtype=np.float32)
        maximums = np.linspace(0.2, 0.9, bucket_count, dtype=np.float32)
        self.loaded.emit(
            WaveformEnvelope(
                minimums=minimums,
                maximums=maximums,
                sample_rate=1_000,
                frame_count=self.duration_ms,
                samples_per_bucket=self.duration_ms // bucket_count,
            )
        )

    def cancel(self):
        self.cancel_count += 1


class _BackendHarness:
    def __init__(self):
        self.players = []
        self.outputs = []
        self.waveform_loaders = []

    def player_factory(self, parent):
        player = _FakePlayer(parent)
        self.players.append(player)
        return player

    def output_factory(self, parent):
        output = _FakeAudioOutput(parent)
        self.outputs.append(output)
        return output

    def waveform_factory(self, parent):
        loader = _FakeWaveformLoader(parent)
        self.waveform_loaders.append(loader)
        return loader


class AudioTrackMixerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        set_language("en_US")
        self._temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self._temporary_directory.name)
        self.track_paths = {
            "accompaniment": root / "accompaniment.wav",
            "vocals": root / "vocals.wav",
        }
        for path in self.track_paths.values():
            path.write_bytes(b"test-audio-placeholder")
        self._widgets = []

    def tearDown(self):
        for widget in self._widgets:
            widget.shutdown()
            widget.deleteLater()
        self._app.processEvents()
        self._temporary_directory.cleanup()
        set_language("zh_CN")

    def _mixer(self):
        harness = _BackendHarness()
        mixer = AudioTrackMixerWidget(
            self.track_paths,
            player_factory=harness.player_factory,
            audio_output_factory=harness.output_factory,
            waveform_loader_factory=harness.waveform_factory,
        )
        self._widgets.append(mixer)
        players = dict(zip(mixer.track_names, harness.players, strict=True))
        outputs = dict(zip(mixer.track_names, harness.outputs, strict=True))
        return mixer, players, outputs

    def test_missing_track_file_is_an_explicit_error(self):
        missing = Path(self._temporary_directory.name) / "missing.wav"
        harness = _BackendHarness()

        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            AudioTrackMixerWidget(
                {"vocals": missing},
                player_factory=harness.player_factory,
                audio_output_factory=harness.output_factory,
            )

    def test_global_transport_seek_offset_replay_and_alignment(self):
        mixer, players, _outputs = self._mixer()

        self.assertTrue(mixer.is_ready)
        self.assertEqual(mixer.track_names, ("vocals", "accompaniment"))
        self.assertEqual(mixer.duration_ms, 30_000)

        mixer.seek(5_000)
        self.assertEqual(players["vocals"].position(), 5_000)
        self.assertEqual(players["accompaniment"].position(), 5_000)

        mixer.set_track_offset_ms("vocals", 1_000)
        self.assertEqual(players["vocals"].position(), 4_000)
        self.assertEqual(mixer.duration_ms, 31_000)

        mixer.play()
        self.assertTrue(mixer.is_playing)
        self.assertTrue(
            all(
                player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
                for player in players.values()
            )
        )

        mixer.pause()
        self.assertFalse(mixer.is_playing)
        self.assertTrue(
            all(
                player.playbackState() == QMediaPlayer.PlaybackState.PausedState
                for player in players.values()
            )
        )

        mixer.replay()
        self.assertTrue(mixer.is_playing)
        self.assertEqual(mixer.position_ms, 0)
        self.assertEqual(
            players["vocals"].playbackState(),
            QMediaPlayer.PlaybackState.PausedState,
        )
        self.assertEqual(
            players["accompaniment"].playbackState(),
            QMediaPlayer.PlaybackState.PlayingState,
        )

        mixer.align_tracks()
        self.assertTrue(mixer.is_playing)
        self.assertEqual(mixer.track_state("vocals").offset_ms, 0)
        self.assertEqual(mixer.duration_ms, 30_000)
        self.assertEqual(
            players["vocals"].playbackState(),
            QMediaPlayer.PlaybackState.PlayingState,
        )

    def test_mute_solo_and_decibel_volume_rules(self):
        mixer, _players, outputs = self._mixer()

        mixer.set_track_volume_db("vocals", -6.0)
        self.assertTrue(math.isclose(outputs["vocals"].volume, 10 ** (-6.0 / 20.0)))
        mixer.set_track_volume_db("vocals", -60.0)
        self.assertEqual(outputs["vocals"].volume, 0.0)

        mixer.set_track_solo("vocals", True)
        self.assertFalse(outputs["vocals"].muted)
        self.assertTrue(outputs["accompaniment"].muted)

        mixer.set_track_muted("vocals", True)
        self.assertTrue(outputs["vocals"].muted)
        self.assertTrue(outputs["accompaniment"].muted)

        mixer.set_track_solo("vocals", False)
        self.assertTrue(outputs["vocals"].muted)
        self.assertFalse(outputs["accompaniment"].muted)

    def test_async_decode_error_stops_all_tracks_and_is_visible(self):
        mixer, players, _outputs = self._mixer()
        emitted = []
        mixer.playback_error.connect(lambda *args: emitted.append(args))
        mixer.play()

        players["vocals"].errorOccurred.emit(
            QMediaPlayer.Error.ResourceError,
            "decode failure",
        )

        self.assertTrue(mixer.has_error)
        self.assertFalse(mixer.is_ready)
        self.assertFalse(mixer.is_playing)
        self.assertFalse(mixer.play_button.isEnabled())
        self.assertTrue(mixer.error_label.isVisibleTo(mixer))
        self.assertIn("decode failure", mixer.error_label.text())
        self.assertEqual(len(emitted), 1)
        self.assertTrue(
            all(
                player.playbackState() == QMediaPlayer.PlaybackState.StoppedState
                for player in players.values()
            )
        )
        with self.assertRaisesRegex(RuntimeError, "decode failure"):
            mixer.play()

    def test_shutdown_is_idempotent_and_releases_media(self):
        mixer, players, outputs = self._mixer()
        loaders = [backend.waveform_loader for backend in mixer._backends.values()]

        mixer.shutdown()
        stop_counts = {name: player.stop_count for name, player in players.items()}
        mixer.shutdown()

        self.assertEqual(
            {name: player.stop_count for name, player in players.items()},
            stop_counts,
        )
        self.assertTrue(all(player.source.isEmpty() for player in players.values()))
        self.assertTrue(all(player.output is None for player in players.values()))
        self.assertTrue(all(output.muted for output in outputs.values()))
        self.assertTrue(all(loader.cancel_count == 1 for loader in loaders))

    def test_add_and_remove_local_audio_tracks_updates_the_shared_timeline(self):
        mixer, _players, _outputs = self._mixer()
        extra = Path(self._temporary_directory.name) / "reference.wav"
        extra.write_bytes(b"test-audio-placeholder")

        names = mixer.add_audio_files([extra])

        self.assertEqual(names, ("reference",))
        self.assertIn("reference", mixer.track_names)
        self.assertTrue(mixer.is_ready)
        self.assertIsNotNone(mixer.findChild(QWidget, "audioTrack_reference_waveform"))

        mixer.remove_track("reference")

        self.assertNotIn("reference", mixer.track_names)
        self.assertTrue(mixer.is_ready)

    def test_added_wav_gets_the_same_opt_in_model_and_single_track_start_controls(self):
        mixer, _players, _outputs = self._mixer()
        extra = Path(self._temporary_directory.name) / "reference.wav"
        extra.write_bytes(b"test-audio-placeholder")
        emitted = []
        mixer.midi_conversion_requested.connect(lambda *args: emitted.append(tuple(args)))

        mixer.add_audio_files([extra])
        row = mixer._backends["reference"].row

        self.assertEqual(row.name_label.text(), "♪  reference")
        self.assertFalse(row.midi_enabled_checkbox.isChecked())
        self.assertEqual(row.midi_model_selector.count(), 11)
        self.assertFalse(row.convert_midi_button.isEnabled())
        self.assertIn("not converted", row.midi_status_label.text())

        row.midi_enabled_checkbox.setChecked(True)
        row.midi_model_selector.setCurrentIndex(row.midi_model_selector.findData("piano_transkun"))
        self.assertEqual(emitted, [])

        row.convert_midi_button.click()
        self.assertEqual(
            emitted,
            [("reference", str(extra.resolve()), "piano_transkun")],
        )

    def test_add_track_button_binds_each_selected_file_to_its_own_waveform(self):
        mixer, _players, _outputs = self._mixer()
        selected_paths = [
            Path(self._temporary_directory.name) / "reference.wav",
            Path(self._temporary_directory.name) / "guide.wav",
        ]
        for path in selected_paths:
            path.write_bytes(b"test-audio-placeholder")

        with mock.patch(
            "src.gui.widgets.audio_track_mixer.QFileDialog.getOpenFileNames",
            return_value=([str(path) for path in selected_paths], ""),
        ) as file_picker:
            mixer.add_track_button.click()

        file_picker.assert_called_once()
        for path in selected_paths:
            track_name = path.stem
            backend = mixer._backends[track_name]
            with self.subTest(track=track_name):
                self.assertEqual(backend.path, path.resolve())
                self.assertEqual(backend.waveform_loader.path, path.resolve())
                self.assertEqual(backend.row.waveform.path, path.resolve())
                self.assertIsNotNone(mixer.findChild(QWidget, f"audioTrack_{track_name}_waveform"))
        self.assertIsNot(
            mixer._backends["reference"].waveform_loader,
            mixer._backends["guide"].waveform_loader,
        )

    def test_every_track_has_a_real_waveform_lane_and_shared_zoom(self):
        mixer, _players, _outputs = self._mixer()

        vocals_lane = mixer.findChild(QWidget, "audioTrack_vocals_waveform")
        accompaniment_lane = mixer.findChild(
            QWidget,
            "audioTrack_accompaniment_waveform",
        )
        self.assertIsNotNone(vocals_lane)
        self.assertIsNotNone(accompaniment_lane)

        mixer.zoom_slider.setValue(4)
        mixer.seek(12_000)

        self.assertEqual(mixer.zoom_label.text(), "4×")
        self.assertEqual(vocals_lane._position_ms, 12_000)
        self.assertEqual(accompaniment_lane._position_ms, 12_000)
        self.assertEqual(vocals_lane._view_start_ms, accompaniment_lane._view_start_ms)
        self.assertEqual(vocals_lane._view_end_ms, accompaniment_lane._view_end_ms)

    def test_track_controls_fit_a_narrow_container(self):
        mixer, _players, _outputs = self._mixer()
        mixer.setFixedWidth(280)
        mixer.resize(280, 700)
        mixer.show()
        self._app.processEvents()

        for control in mixer.findChildren((QPushButton, QSlider)):
            with self.subTest(control=control.objectName()):
                bottom_right = control.mapTo(mixer, control.rect().bottomRight())
                self.assertLessEqual(bottom_right.x(), mixer.rect().right())
                self.assertGreaterEqual(control.mapTo(mixer, control.rect().topLeft()).x(), 0)

    def test_six_stem_tracks_keep_the_public_timeline_order(self):
        root = Path(self._temporary_directory.name)
        six_tracks = {}
        for name in ("other", "vocals", "piano", "guitar", "drums", "bass"):
            path = root / f"{name}.wav"
            path.write_bytes(b"test-audio-placeholder")
            six_tracks[name] = path

        harness = _BackendHarness()
        mixer = AudioTrackMixerWidget(
            six_tracks,
            player_factory=harness.player_factory,
            audio_output_factory=harness.output_factory,
            waveform_loader_factory=harness.waveform_factory,
        )
        self._widgets.append(mixer)

        self.assertEqual(
            mixer.track_names,
            ("bass", "drums", "guitar", "piano", "vocals", "other"),
        )
        mixer.setFixedWidth(280)
        mixer.resize(280, 1800)
        mixer.show()
        self._app.processEvents()
        for control in mixer.findChildren((QPushButton, QSlider)):
            with self.subTest(control=control.objectName()):
                self.assertLessEqual(
                    control.mapTo(mixer, control.rect().bottomRight()).x(),
                    mixer.rect().right(),
                )


class MainWindowAudioTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        set_language("en_US")

    def tearDown(self):
        self._app.processEvents()
        set_language("zh_CN")

    def test_track_midi_open_signal_uses_output_folder_handler(self):
        from src.gui.main_window import MainWindow
        from src.models.data_models import Config

        class _FakeMixer(QWidget):
            midi_conversion_requested = pyqtSignal(str, str, str)
            midi_open_requested = pyqtSignal(str)

            def __init__(self, tracks):
                super().__init__()
                self.tracks = dict(tracks)

            def shutdown(self):
                pass

        with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
            window = MainWindow(Config(language="en_US"))
        midi_path = "C:/output/midi/miros/drums.mid"
        try:
            with (
                mock.patch("src.gui.main_window.AudioTrackMixerWidget", _FakeMixer),
                mock.patch.object(MainWindow, "_open_output_folder") as open_folder,
                mock.patch.object(MainWindow, "_open_midi_file") as open_midi,
            ):
                window._set_audio_tracks(
                    {"drums": "C:/output/drums.wav"},
                    show_timeline=False,
                )
                window.audio_mixer.midi_open_requested.emit(midi_path)

            open_folder.assert_called_once_with(midi_path)
            open_midi.assert_not_called()
        finally:
            window.close()

    def test_finished_result_embeds_a_persistent_main_window_timeline(self):
        from src.core.separation_service import SeparationResult
        from src.gui.main_window import MainWindow
        from src.models.data_models import Config, ProcessingMode

        created = []

        class _FakeMixer(QWidget):
            midi_conversion_requested = pyqtSignal(str, str, str)
            midi_open_requested = pyqtSignal(str)

            def __init__(self, tracks):
                super().__init__()
                self.tracks = tracks
                self.shutdown_count = 0
                self.translation_count = 0
                created.append(self)

            def shutdown(self):
                self.shutdown_count += 1

            def update_translations(self):
                self.translation_count += 1

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tracks = {
                "vocals": root / "vocals.wav",
                "accompaniment": root / "accompaniment.wav",
            }
            for path in tracks.values():
                path.write_bytes(b"test-audio-placeholder")
            result = SeparationResult(
                mode=ProcessingMode.VOCAL_SPLIT.value,
                source_path=str(root / "source.wav"),
                output_dir=str(root),
                separated_audio={name: str(path) for name, path in tracks.items()},
                processing_time=1.0,
            )
            with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
                window = MainWindow(Config(language="en_US"))
            try:
                with (
                    mock.patch("src.gui.main_window.AudioTrackMixerWidget", _FakeMixer),
                    mock.patch.object(
                        QDialog,
                        "exec",
                        side_effect=AssertionError("success must not open a dialog"),
                    ),
                ):
                    window._on_separation_finished(result)

                self.assertEqual(len(created), 1)
                self.assertEqual(
                    created[0].tracks,
                    {
                        name: Path(path).resolve()
                        for name, path in result.separated_audio.items()
                    },
                )
                self.assertEqual(created[0].shutdown_count, 0)
                self.assertIs(created[0], window.audio_mixer)
                self.assertIs(created[0].parentWidget(), window.audio_timeline_container)
                self.assertFalse(window.result_panel.isHidden())
                self.assertFalse(window.audio_timeline_container.isHidden())
                self.assertIs(
                    window.audio_timeline_container.parentWidget(),
                    window.result_panel,
                )
                self.assertIs(window.audio_timeline_container.window(), window)

                window._clear_audio_mixer()
                self.assertEqual(created[0].shutdown_count, 1)
                self.assertIsNone(window.audio_mixer)
            finally:
                window.close()

    def test_finished_result_creates_no_top_level_completion_window(self):
        from src.core.separation_service import SeparationResult
        from src.gui.main_window import MainWindow
        from src.models.data_models import Config, ProcessingMode

        class _FakeMixer(QWidget):
            midi_conversion_requested = pyqtSignal(str, str, str)
            midi_open_requested = pyqtSignal(str)

            def __init__(self, tracks):
                super().__init__()
                self.tracks = dict(tracks)

            def shutdown(self):
                pass

            def update_translations(self):
                pass

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            vocals = root / "vocals.wav"
            accompaniment = root / "accompaniment.wav"
            vocals.write_bytes(b"vocals")
            accompaniment.write_bytes(b"accompaniment")
            result = SeparationResult(
                mode=ProcessingMode.VOCAL_SPLIT.value,
                source_path=str(root / "source.wav"),
                output_dir=str(root),
                separated_audio={
                    "vocals": str(vocals),
                    "accompaniment": str(accompaniment),
                },
                processing_time=1.0,
            )
            with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
                window = MainWindow(Config(language="en_US"))
            try:
                window.show()
                self._app.processEvents()
                with (
                    mock.patch("src.gui.main_window.AudioTrackMixerWidget", _FakeMixer),
                    mock.patch.object(
                        QDialog,
                        "exec",
                        side_effect=AssertionError("success must not open a dialog"),
                    ),
                ):
                    window._on_separation_finished(result)
                    self._app.processEvents()

                visible_dialogs = [
                    widget
                    for widget in QApplication.topLevelWidgets()
                    if isinstance(widget, QDialog) and widget.isVisible()
                ]
                self.assertEqual(visible_dialogs, [])
                self.assertFalse(window.audio_timeline_container.isHidden())
                self.assertFalse(window.audio_timeline_container.isWindow())
                self.assertIs(window.audio_timeline_container.window(), window)
                self.assertIs(
                    window.audio_timeline_container.parentWidget(),
                    window.result_panel,
                )
            finally:
                window.close()
    def test_direct_conversion_modes_do_not_create_source_audio_timeline(self):
        from src.gui.main_window import MainWindow
        from src.models.data_models import Config, ProcessingMode, ProcessingResult

        direct_modes = (
            ProcessingMode.SMART,
            ProcessingMode.PIANO_TRANSKUN,
            ProcessingMode.PIANO_TRANSKUN_V2_AUG,
            ProcessingMode.PIANO_ARIA_AMT,
            ProcessingMode.PIANO_BYTEDANCE_PEDAL,
        )
        with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
            window = MainWindow(Config(language="en_US"))
        window.current_file = "C:/input/source.wav"
        try:
            for mode in direct_modes:
                with self.subTest(mode=mode.value):
                    window.config.processing_mode = mode.value
                    result = ProcessingResult(
                        midi_path=f"C:/output/{mode.value}.mid",
                        processing_time=1.0,
                    )
                    with mock.patch("src.gui.main_window.AudioTrackMixerWidget") as mixer_class:
                        window._on_finished(result)

                    mixer_class.assert_not_called()
                    self.assertIsNone(window.audio_mixer)
                    self.assertTrue(window.audio_timeline_container.isHidden())
                    self.assertFalse(window.result_panel.isHidden())
        finally:
            window.close()

    def test_finished_result_real_mixer_shows_waveforms_and_accepts_more_tracks(self):
        from src.core.separation_service import SeparationResult
        from src.gui.main_window import MainWindow
        from src.models.data_models import Config, ProcessingMode

        harness = _BackendHarness()

        def create_mixer(tracks):
            return AudioTrackMixerWidget(
                tracks,
                player_factory=harness.player_factory,
                audio_output_factory=harness.output_factory,
                waveform_loader_factory=harness.waveform_factory,
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            track_paths = {
                "vocals": root / "vocals.wav",
                "accompaniment": root / "accompaniment.wav",
            }
            reference_path = root / "reference.wav"
            for path in (*track_paths.values(), reference_path):
                path.write_bytes(b"test-audio-placeholder")

            with mock.patch.object(
                MainWindow,
                "_start_gpu_detection",
                return_value=None,
            ):
                window = MainWindow(Config(language="en_US"))
            result = SeparationResult(
                mode=ProcessingMode.VOCAL_SPLIT.value,
                source_path=str(root / "source.wav"),
                output_dir=str(root),
                separated_audio={
                    name: str(path) for name, path in track_paths.items()
                },
                processing_time=1.0,
            )
            try:
                with mock.patch(
                    "src.gui.main_window.AudioTrackMixerWidget",
                    side_effect=create_mixer,
                ):
                    window._on_separation_finished(result)

                self.assertIsInstance(window.audio_mixer, AudioTrackMixerWidget)
                for track_name, path in track_paths.items():
                    lane = window.audio_mixer.findChild(
                        QWidget,
                        f"audioTrack_{track_name}_waveform",
                    )
                    with self.subTest(track=track_name):
                        self.assertIsNotNone(lane)
                        self.assertEqual(lane.path, path.resolve())

                with mock.patch(
                    "src.gui.widgets.audio_track_mixer.QFileDialog.getOpenFileNames",
                    return_value=([str(reference_path)], ""),
                ):
                    window.audio_mixer.add_track_button.click()

                reference_lane = window.audio_mixer.findChild(
                    QWidget,
                    "audioTrack_reference_waveform",
                )
                self.assertIsNotNone(reference_lane)
                self.assertEqual(reference_lane.path, reference_path.resolve())
            finally:
                window.close()
    def test_finished_result_displays_timeline_construction_failure_in_the_dock(self):
        from src.core.separation_service import SeparationResult
        from src.gui.main_window import MainWindow
        from src.models.data_models import Config, ProcessingMode

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            vocals = root / "vocals.wav"
            accompaniment = root / "accompaniment.wav"
            vocals.write_bytes(b"vocals")
            accompaniment.write_bytes(b"accompaniment")
            result = SeparationResult(
                mode=ProcessingMode.VOCAL_SPLIT.value,
                source_path=str(root / "source.wav"),
                output_dir=str(root),
                separated_audio={
                    "vocals": str(vocals),
                    "accompaniment": str(accompaniment),
                },
                processing_time=1.0,
            )
            with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
                window = MainWindow(Config(language="en_US"))
            try:
                with (
                    mock.patch(
                        "src.gui.main_window.AudioTrackMixerWidget",
                        side_effect=FileNotFoundError("missing WAV"),
                    ),
                    mock.patch.object(
                        QDialog,
                        "exec",
                        side_effect=AssertionError("success must not open a dialog"),
                    ),
                ):
                    window._on_separation_finished(result)

                error_label = window.audio_timeline_container.findChild(
                    QLabel,
                    "audioMixerUnavailableLabel",
                )
                self.assertIsNotNone(error_label)
                self.assertIn("missing WAV", error_label.text())
                self.assertFalse(window.result_panel.isHidden())
                self.assertFalse(window.audio_timeline_container.isHidden())
            finally:
                window.close()
    def test_new_results_replace_tracks_and_language_updates_preserve_the_mixer(self):
        from src.core.separation_service import SeparationResult
        from src.gui.main_window import MainWindow
        from src.models.data_models import Config, ProcessingMode, ProcessingResult

        created = []

        class _FakeMixer(QWidget):
            midi_conversion_requested = pyqtSignal(str, str, str)
            midi_open_requested = pyqtSignal(str)

            def __init__(self, tracks):
                super().__init__()
                self.tracks = dict(tracks)
                self.shutdown_count = 0
                self.translation_count = 0
                created.append(self)

            def shutdown(self):
                self.shutdown_count += 1

            def update_translations(self):
                self.translation_count += 1

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            vocals = root / "vocals.wav"
            accompaniment = root / "accompaniment.wav"
            vocals.write_bytes(b"vocals")
            accompaniment.write_bytes(b"accompaniment")
            separated_result = SeparationResult(
                mode=ProcessingMode.VOCAL_SPLIT.value,
                source_path=str(root / "source.wav"),
                output_dir=str(root),
                separated_audio={
                    "vocals": str(vocals),
                    "accompaniment": str(accompaniment),
                },
                processing_time=1.0,
            )
            source_result = ProcessingResult(
                midi_path=str(root / "source.mid"),
                processing_time=2.0,
            )
            with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
                window = MainWindow(Config(language="en_US"))
            window.current_file = str(root / "source.wav")
            try:
                with mock.patch("src.gui.main_window.AudioTrackMixerWidget", _FakeMixer):
                    window._on_separation_finished(separated_result)
                    first = window.audio_mixer
                    window._on_finished(source_result)
                    self.assertEqual(first.shutdown_count, 1)
                    self.assertIsNone(window.audio_mixer)
                    self.assertTrue(window.audio_timeline_container.isHidden())

                    set_language("zh_CN")
                    window._update_translations()
                    self.assertIsNone(window.audio_mixer)

                    window._on_file_selected(str(root / "next.wav"))
                    self.assertIsNone(window.audio_mixer)
                    self.assertTrue(window.result_panel.isHidden())
            finally:
                window.close()

    def test_processing_result_with_separated_audio_is_rejected(self):
        from src.gui.main_window import MainWindow
        from src.models.data_models import Config, ProcessingResult

        with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
            window = MainWindow(Config(language="en_US"))
        try:
            result = ProcessingResult(
                midi_path="C:/output/source.mid",
                processing_time=1.0,
                separated_audio={"vocals": "C:/output/vocals.wav"},
            )
            with self.assertRaisesRegex(ValueError, "SeparationResult"):
                window._playback_tracks_for_result(result)
        finally:
            window.close()
if __name__ == "__main__":
    unittest.main()
