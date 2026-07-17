"""Synchronized playback controls for separated audio tracks."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import (
    QElapsedTimer,
    QObject,
    QSignalBlocker,
    QTimer,
    Qt,
    QUrl,
    pyqtSignal,
)
from PyQt6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.gui.layouts import FlowLayout
from src.gui.widgets.audio_waveform import (
    FfmpegWaveformLoader,
    TimelineRuler,
    WaveformEnvelope,
    WaveformLane,
)
from src.gui.widgets.wheel_safe_controls import NoWheelComboBox, NoWheelSlider
from src.core.manual_midi import (
    MIDI_ROUTE_MIROS,
    MIDI_ROUTE_PIANO_ARIA_AMT,
    MIDI_ROUTE_PIANO_BYTEDANCE_PEDAL,
    MIDI_ROUTE_PIANO_TRANSKUN,
    MIDI_ROUTE_PIANO_TRANSKUN_V2_AUG,
    MIDI_ROUTE_YOURMT3_PREFIX,
    YOURMT3_MANUAL_MODELS,
)
from src.i18n.translator import t
from src.utils.yourmt3_downloader import YOURMT3_MODELS

logger = logging.getLogger(__name__)

_LOAD_TIMEOUT_MS = 10_000
_SYNC_INTERVAL_MS = 50
_MAX_PLAYER_DRIFT_MS = 500
_MIN_VOLUME_DB = -60.0
_MAX_VOLUME_DB = 0.0
_MIN_OFFSET_MS = -10_000
_MAX_OFFSET_MS = 10_000
_TRACK_ORDER = (
    "bass",
    "drums",
    "guitar",
    "piano",
    "vocals",
    "accompaniment",
    "other",
    "source",
)
_KNOWN_TRACK_NAMES = frozenset(_TRACK_ORDER)
_TRACK_COLORS = (
    "#5eb1ff",
    "#ff8d66",
    "#7bd88f",
    "#c89bff",
    "#ff70a6",
    "#ffd166",
    "#62d2c3",
    "#9aa8ff",
)

_YOURMT3_MANUAL_MODELS = YOURMT3_MANUAL_MODELS


def midi_route_label(route: str) -> str:
    """Return the localized label for one explicit manual MIDI route."""
    if route.startswith(MIDI_ROUTE_YOURMT3_PREFIX):
        model_name = route.removeprefix(MIDI_ROUTE_YOURMT3_PREFIX)
        info = YOURMT3_MODELS.get(model_name)
        if info is None:
            raise ValueError(f"Unsupported YourMT3 model route: {route!r}")
        model_label = info.get("ui_label") or info.get("name") or model_name
        return f"YourMT3+ · {model_label}"
    labels = {
        MIDI_ROUTE_MIROS: t("dialogs.complete.audio_tracks.manual_midi.models.miros"),
        MIDI_ROUTE_PIANO_TRANSKUN: t(
            "dialogs.complete.audio_tracks.manual_midi.models.piano_transkun"
        ),
        MIDI_ROUTE_PIANO_TRANSKUN_V2_AUG: t(
            "dialogs.complete.audio_tracks.manual_midi.models.piano_transkun_v2_aug"
        ),
        MIDI_ROUTE_PIANO_ARIA_AMT: t(
            "dialogs.complete.audio_tracks.manual_midi.models.piano_aria_amt"
        ),
        MIDI_ROUTE_PIANO_BYTEDANCE_PEDAL: t(
            "dialogs.complete.audio_tracks.manual_midi.models.piano_bytedance_pedal"
        ),
    }
    try:
        return labels[route]
    except KeyError as exc:
        raise ValueError(f"Unsupported manual MIDI route: {route!r}") from exc


PlayerFactory = Callable[[QObject], Any]
AudioOutputFactory = Callable[[QObject], Any]
WaveformLoaderFactory = Callable[[QObject], Any]


def _format_time(position_ms: int) -> str:
    total_seconds = max(0, int(position_ms)) // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def _display_track_name(track_name: str) -> str:
    if track_name in _KNOWN_TRACK_NAMES:
        return t(f"dialogs.complete.audio_tracks.track_names.{track_name}")
    return track_name


def _db_to_linear(volume_db: float) -> float:
    if volume_db <= _MIN_VOLUME_DB:
        return 0.0
    return min(1.0, 10.0 ** (volume_db / 20.0))


def _create_qt_media_player(owner: QObject) -> QMediaPlayer:
    return QMediaPlayer(owner)


def _create_qt_audio_output(owner: QObject) -> QAudioOutput:
    return QAudioOutput(owner)


@dataclass(frozen=True)
class AudioTrackState:
    """Observable state for one audio track."""

    name: str
    path: Path
    muted: bool
    solo: bool
    volume_db: float
    offset_ms: int
    duration_ms: int


class _AudioTrackRow(QFrame):
    mute_changed = pyqtSignal(bool)
    solo_changed = pyqtSignal(bool)
    volume_changed = pyqtSignal(float)
    offset_changed = pyqtSignal(int)
    offset_committed = pyqtSignal()
    remove_requested = pyqtSignal()
    seek_requested = pyqtSignal(int)
    zoom_requested = pyqtSignal(int, int)
    midi_conversion_requested = pyqtSignal(str)
    midi_open_requested = pyqtSignal(str)

    def __init__(
        self,
        track_name: str,
        path: Path,
        color: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.track_name = track_name
        self.path = path
        safe_name = "".join(char if char.isalnum() else "_" for char in track_name)
        self.setObjectName(f"audioTrackRow_{safe_name}")
        self.setProperty("audioTrackRow", True)
        self.setToolTip(str(path))
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(7)

        self.name_label = QLabel()
        self.name_label.setObjectName(f"audioTrack_{safe_name}_name")
        self.name_label.setStyleSheet(f"font-weight: 600; color: {color};")
        self.name_label.setMinimumWidth(0)
        self.name_label.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        self.name_label.setToolTip(str(path))
        layout.addWidget(self.name_label)

        header = FlowLayout(horizontal_spacing=6, vertical_spacing=5)

        self.mute_button = QPushButton(t("dialogs.complete.audio_tracks.mute"))
        self.mute_button.setObjectName(f"audioTrack_{safe_name}_mute")
        self.mute_button.setProperty("audioTrackToggle", True)
        self.mute_button.setCheckable(True)
        self.mute_button.setAccessibleName(
            f"{_display_track_name(track_name)} {t('dialogs.complete.audio_tracks.mute')}"
        )
        header.addWidget(self.mute_button)

        self.solo_button = QPushButton(t("dialogs.complete.audio_tracks.solo"))
        self.solo_button.setObjectName(f"audioTrack_{safe_name}_solo")
        self.solo_button.setProperty("audioTrackToggle", True)
        self.solo_button.setCheckable(True)
        self.solo_button.setAccessibleName(
            f"{_display_track_name(track_name)} {t('dialogs.complete.audio_tracks.solo')}"
        )
        header.addWidget(self.solo_button)

        self.remove_button = QPushButton()
        self.remove_button.setObjectName(f"audioTrack_{safe_name}_remove")
        self.remove_button.setProperty("audioTrackRemove", True)
        header.addWidget(self.remove_button)

        self.midi_enabled_checkbox = QCheckBox()
        self.midi_enabled_checkbox.setObjectName(f"audioTrack_{safe_name}_midiEnabled")
        self.midi_enabled_checkbox.setProperty("audioTrackMidiEnabled", True)
        header.addWidget(self.midi_enabled_checkbox)

        self.midi_model_selector = NoWheelComboBox()
        self.midi_model_selector.setObjectName(f"audioTrack_{safe_name}_midiModel")
        self.midi_model_selector.setProperty("audioTrackMidiModel", True)
        self.midi_model_selector.setMinimumWidth(220)
        self.midi_model_selector.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        header.addWidget(self.midi_model_selector)

        self.convert_midi_button = QPushButton()
        self.convert_midi_button.setObjectName(f"audioTrack_{safe_name}_startMidi")
        self.convert_midi_button.setProperty("audioTrackMidi", True)
        self.start_midi_button = self.convert_midi_button
        header.addWidget(self.convert_midi_button)

        self.open_midi_button = QPushButton()
        self.open_midi_button.setObjectName(f"audioTrack_{safe_name}_openMidi")
        self.open_midi_button.setProperty("audioTrackMidiOpen", True)
        self.open_midi_button.hide()
        header.addWidget(self.open_midi_button)
        layout.addLayout(header)

        self.path_label = QLabel(path.name)
        self.path_label.setObjectName(f"audioTrack_{safe_name}_path")
        self.path_label.setMinimumWidth(0)
        self.path_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.path_label.setToolTip(str(path))
        self.path_label.setStyleSheet("color: #8292aa; font-size: 10px;")
        layout.addWidget(self.path_label)

        self.midi_status_label = QLabel()
        self.midi_status_label.setObjectName(f"audioTrack_{safe_name}_midiStatus")
        self.midi_status_label.setWordWrap(True)
        self.midi_status_label.setMinimumWidth(0)
        self.midi_status_label.setStyleSheet("color: #9fb3d9; font-size: 10px; padding: 1px 0;")
        layout.addWidget(self.midi_status_label)

        self.waveform = WaveformLane(path, color, self)
        self.waveform.setObjectName(f"audioTrack_{safe_name}_waveform")
        layout.addWidget(self.waveform)

        controls = FlowLayout(horizontal_spacing=12, vertical_spacing=6)
        (
            self.volume_title_label,
            self.volume_slider,
            self.volume_value_label,
            volume_control,
        ) = self._slider_control(
            label=t("dialogs.complete.audio_tracks.volume"),
            object_name=f"audioTrack_{safe_name}_volume",
            minimum=-600,
            maximum=0,
            value=0,
        )
        (
            self.offset_title_label,
            self.offset_slider,
            self.offset_value_label,
            offset_control,
        ) = self._slider_control(
            label=t("dialogs.complete.audio_tracks.offset"),
            object_name=f"audioTrack_{safe_name}_offset",
            minimum=_MIN_OFFSET_MS,
            maximum=_MAX_OFFSET_MS,
            value=0,
        )
        self.offset_slider.setSingleStep(100)
        self.offset_slider.setPageStep(1000)
        controls.addWidget(volume_control)
        controls.addWidget(offset_control)
        layout.addLayout(controls)

        self.mute_button.toggled.connect(self.mute_changed)
        self.solo_button.toggled.connect(self.solo_changed)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        self.offset_slider.valueChanged.connect(self._on_offset_changed)
        self.offset_slider.sliderReleased.connect(self.offset_committed)
        self.remove_button.clicked.connect(self.remove_requested)
        self.midi_enabled_checkbox.toggled.connect(self._on_midi_option_changed)
        self.midi_model_selector.currentIndexChanged.connect(self._on_midi_route_selected)
        self.convert_midi_button.clicked.connect(self._request_midi_conversion)
        self.open_midi_button.clicked.connect(self._open_generated_midi)
        self.waveform.seek_requested.connect(self.seek_requested)
        self.waveform.zoom_requested.connect(self.zoom_requested)
        self._midi_state = "idle"
        self._midi_route = ""
        self._midi_message = ""
        self._midi_output_path = ""
        self._selected_midi_route = ""
        self._midi_controls_allowed = True
        self._update_volume_label(0.0)
        self._update_offset_label(0)
        self.update_translations()

    @staticmethod
    def _slider_control(
        *,
        label: str,
        object_name: str,
        minimum: int,
        maximum: int,
        value: int,
    ) -> tuple[QLabel, QSlider, QLabel, QWidget]:
        container = QWidget()
        container.setMinimumWidth(0)
        container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        control_layout = QHBoxLayout(container)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(6)

        title = QLabel(label)
        title.setStyleSheet("color: #aebbd0;")
        control_layout.addWidget(title)

        slider = NoWheelSlider(Qt.Orientation.Horizontal)
        slider.setObjectName(object_name)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setMinimumWidth(64)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        slider.setAccessibleName(label)
        control_layout.addWidget(slider, 1)

        value_label = QLabel()
        value_label.setMinimumWidth(58)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        value_label.setStyleSheet("color: #d8e1ef;")
        control_layout.addWidget(value_label)
        return title, slider, value_label, container

    def _build_midi_menu(self) -> None:
        selected_route = self._selected_midi_route
        multi_label = t("dialogs.complete.audio_tracks.manual_midi.multi_instrument")
        piano_label = t("dialogs.complete.audio_tracks.manual_midi.piano")
        blocker = QSignalBlocker(self.midi_model_selector)
        try:
            self.midi_model_selector.clear()
            self.midi_model_selector.addItem(
                t("dialogs.complete.audio_tracks.manual_midi.select_model"),
                "",
            )
            for model in _YOURMT3_MANUAL_MODELS:
                route = f"{MIDI_ROUTE_YOURMT3_PREFIX}{model.value}"
                self._add_midi_route_action(
                    f"{multi_label} · {midi_route_label(route)}",
                    route,
                )
            self._add_midi_route_action(
                f"{multi_label} · {midi_route_label(MIDI_ROUTE_MIROS)}",
                MIDI_ROUTE_MIROS,
            )
            for route in (
                MIDI_ROUTE_PIANO_TRANSKUN,
                MIDI_ROUTE_PIANO_TRANSKUN_V2_AUG,
                MIDI_ROUTE_PIANO_ARIA_AMT,
                MIDI_ROUTE_PIANO_BYTEDANCE_PEDAL,
            ):
                self._add_midi_route_action(
                    f"{piano_label} · {midi_route_label(route)}",
                    route,
                )
            selected_index = self.midi_model_selector.findData(selected_route)
            self.midi_model_selector.setCurrentIndex(max(0, selected_index))
        finally:
            del blocker
        current_data = self.midi_model_selector.currentData()
        self._selected_midi_route = str(current_data or "")
        self._update_midi_action_state()

    def _add_midi_route_action(
        self,
        label: str,
        route: str,
    ) -> None:
        self.midi_model_selector.addItem(label, route)

    def _on_midi_option_changed(self, _enabled: bool) -> None:
        if self._midi_state != "running":
            self._midi_state = "idle"
            self._midi_message = ""
        self._update_midi_action_state()
        self._refresh_midi_status()

    def _on_midi_route_selected(self, _index: int) -> None:
        selected = self.midi_model_selector.currentData()
        self._selected_midi_route = str(selected or "")
        if self._midi_state != "running":
            self._midi_state = "idle"
            self._midi_message = ""
        self._update_midi_action_state()
        self._refresh_midi_status()

    def _request_midi_conversion(self) -> None:
        if not self._midi_controls_allowed or self._midi_state == "running":
            return
        if not self.midi_enabled_checkbox.isChecked():
            raise RuntimeError("Per-track MIDI conversion is not enabled")
        if not self._selected_midi_route:
            raise RuntimeError("No per-track MIDI transcription model is selected")
        self.midi_conversion_requested.emit(self._selected_midi_route)

    def _update_midi_action_state(self) -> None:
        editable = self._midi_controls_allowed and self._midi_state != "running"
        midi_enabled = self.midi_enabled_checkbox.isChecked()
        self.midi_enabled_checkbox.setEnabled(editable)
        self.midi_model_selector.setEnabled(editable and midi_enabled)
        self.convert_midi_button.setEnabled(
            editable and midi_enabled and bool(self._selected_midi_route)
        )

    def _open_generated_midi(self) -> None:
        if not self._midi_output_path:
            raise RuntimeError("No generated MIDI path is available for this track")
        self.midi_open_requested.emit(self._midi_output_path)

    def set_midi_controls_enabled(self, enabled: bool) -> None:
        self._midi_controls_allowed = bool(enabled)
        self._update_midi_action_state()

    def set_midi_conversion_running(self, route: str) -> None:
        label = midi_route_label(route)
        self._midi_state = "running"
        self._midi_route = route
        self._selected_midi_route = route
        self._midi_message = label
        self._midi_output_path = ""
        self.remove_button.setEnabled(False)
        self.open_midi_button.hide()
        self._update_midi_action_state()
        self._refresh_midi_status()

    def set_midi_conversion_progress(self, message: str) -> None:
        if self._midi_state != "running":
            raise RuntimeError(f"Track {self.track_name!r} has no active MIDI conversion")
        self._midi_message = str(message)
        self._refresh_midi_status()

    def set_midi_conversion_succeeded(self, route: str, midi_path: str) -> None:
        path = Path(midi_path).resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Generated MIDI does not exist or is empty: {path}")
        self._midi_state = "complete"
        self._midi_route = route
        self._midi_message = ""
        self._midi_output_path = str(path)
        self.remove_button.setEnabled(True)
        self.open_midi_button.show()
        self._update_midi_action_state()
        self._refresh_midi_status()

    def set_midi_conversion_failed(self, error: str) -> None:
        self._midi_state = "failed"
        self._midi_message = str(error)
        self._midi_output_path = ""
        self.remove_button.setEnabled(True)
        self.open_midi_button.hide()
        self._update_midi_action_state()
        self._refresh_midi_status()

    def set_midi_conversion_cancelled(self) -> None:
        self._midi_state = "cancelled"
        self._midi_message = ""
        self._midi_output_path = ""
        self.remove_button.setEnabled(True)
        self.open_midi_button.hide()
        self._update_midi_action_state()
        self._refresh_midi_status()

    @staticmethod
    def _midi_error_summary(error: str) -> str:
        lines = [line.strip() for line in str(error).splitlines() if line.strip()]
        summary = lines[-1] if lines else str(error).strip()
        if len(summary) > 240:
            return summary[:237].rstrip() + "…"
        return summary

    def _refresh_midi_status(self) -> None:
        if self._midi_state == "idle":
            if not self.midi_enabled_checkbox.isChecked():
                text = t("dialogs.complete.audio_tracks.manual_midi.not_selected")
            elif not self._selected_midi_route:
                text = t("dialogs.complete.audio_tracks.manual_midi.model_required")
            else:
                text = t(
                    "dialogs.complete.audio_tracks.manual_midi.selected",
                    model=midi_route_label(self._selected_midi_route),
                )
        elif self._midi_state == "running":
            text = t(
                "dialogs.complete.audio_tracks.manual_midi.converting",
                model=midi_route_label(self._midi_route),
                message=self._midi_message,
            )
        elif self._midi_state == "complete":
            text = t(
                "dialogs.complete.audio_tracks.manual_midi.complete",
                file=Path(self._midi_output_path).name,
            )
        elif self._midi_state == "failed":
            text = t(
                "dialogs.complete.audio_tracks.manual_midi.failed",
                error=self._midi_error_summary(self._midi_message),
            )
        elif self._midi_state == "cancelled":
            text = t("dialogs.complete.audio_tracks.manual_midi.cancelled")
        else:
            raise RuntimeError(f"Unknown MIDI conversion state: {self._midi_state}")
        self.midi_status_label.setText(text)
        tooltip = (
            self._midi_message
            if self._midi_state == "failed"
            else self._midi_output_path or self._midi_message
        )
        if not tooltip and self._selected_midi_route:
            tooltip = midi_route_label(self._selected_midi_route)
        self.midi_status_label.setToolTip(tooltip)
        self.midi_status_label.show()

    def update_translations(self) -> None:
        display_name = _display_track_name(self.track_name)
        self.name_label.setText(f"♪  {display_name}")
        self.mute_button.setText(t("dialogs.complete.audio_tracks.mute"))
        self.solo_button.setText(t("dialogs.complete.audio_tracks.solo"))
        self.remove_button.setText(t("dialogs.complete.audio_tracks.remove"))
        self.midi_enabled_checkbox.setText(t("dialogs.complete.audio_tracks.manual_midi.enable"))
        self.convert_midi_button.setText(t("dialogs.complete.audio_tracks.manual_midi.start"))
        self.open_midi_button.setText(t("dialogs.complete.audio_tracks.manual_midi.open"))
        # A hidden QWidget otherwise keeps Qt's 640x480 default geometry until
        # its first visible layout pass, which makes narrow-layout bounds checks
        # report a false horizontal overflow.
        self.open_midi_button.adjustSize()
        self.remove_button.setToolTip(t("dialogs.complete.audio_tracks.remove_tooltip"))
        self.mute_button.setAccessibleName(
            f"{display_name} {t('dialogs.complete.audio_tracks.mute')}"
        )
        self.solo_button.setAccessibleName(
            f"{display_name} {t('dialogs.complete.audio_tracks.solo')}"
        )
        self.remove_button.setAccessibleName(
            f"{display_name} {t('dialogs.complete.audio_tracks.remove')}"
        )
        self.midi_enabled_checkbox.setAccessibleName(
            f"{display_name} " f"{t('dialogs.complete.audio_tracks.manual_midi.enable')}"
        )
        self.midi_model_selector.setAccessibleName(
            f"{display_name} " f"{t('dialogs.complete.audio_tracks.manual_midi.select_model')}"
        )
        self.convert_midi_button.setAccessibleName(
            f"{display_name} " f"{t('dialogs.complete.audio_tracks.manual_midi.start')}"
        )
        self.open_midi_button.setAccessibleName(
            f"{display_name} " f"{t('dialogs.complete.audio_tracks.manual_midi.open')}"
        )
        self.volume_title_label.setText(t("dialogs.complete.audio_tracks.volume"))
        self.offset_title_label.setText(t("dialogs.complete.audio_tracks.offset"))
        self._build_midi_menu()
        self._refresh_midi_status()
        self.waveform.update_translations()

    @property
    def volume_db(self) -> float:
        return self.volume_slider.value() / 10.0

    @property
    def offset_ms(self) -> int:
        return self.offset_slider.value()

    def set_volume_db(self, volume_db: float) -> None:
        if not _MIN_VOLUME_DB <= volume_db <= _MAX_VOLUME_DB:
            raise ValueError(f"volume_db must be between {_MIN_VOLUME_DB} and {_MAX_VOLUME_DB}")
        self.volume_slider.setValue(round(volume_db * 10.0))

    def set_offset_ms(self, offset_ms: int, *, emit: bool = True) -> None:
        if not _MIN_OFFSET_MS <= offset_ms <= _MAX_OFFSET_MS:
            raise ValueError(f"offset_ms must be between {_MIN_OFFSET_MS} and {_MAX_OFFSET_MS}")
        if emit:
            self.offset_slider.setValue(offset_ms)
            return
        with QSignalBlocker(self.offset_slider):
            self.offset_slider.setValue(offset_ms)
        self._update_offset_label(offset_ms)

    def _on_volume_changed(self, slider_value: int) -> None:
        volume_db = slider_value / 10.0
        self._update_volume_label(volume_db)
        self.volume_changed.emit(volume_db)

    def _on_offset_changed(self, offset_ms: int) -> None:
        self._update_offset_label(offset_ms)
        self.offset_changed.emit(offset_ms)

    def _update_volume_label(self, volume_db: float) -> None:
        if volume_db <= _MIN_VOLUME_DB:
            text = "−∞ dB"
        else:
            text = f"{volume_db:.1f} dB"
        self.volume_value_label.setText(text)

    def _update_offset_label(self, offset_ms: int) -> None:
        self.offset_value_label.setText(f"{offset_ms / 1000.0:+.2f}s")


@dataclass
class _TrackBackend:
    path: Path
    row: _AudioTrackRow
    player: Any
    audio_output: Any
    waveform_loader: Any
    duration_ms: int = 0
    waveform_duration_ms: int = 0
    failed: bool = False


class AudioTrackMixerWidget(QWidget):
    """TelkNet-style controls for synchronized preview of separated tracks.

    Synchronization is based on one monotonic logical clock. Multiple
    ``QMediaPlayer`` instances are suitable for preview, but this widget does
    not claim sample-accurate mixing.
    """

    playback_error = pyqtSignal(str, str, str)
    ready_changed = pyqtSignal(bool)
    position_changed = pyqtSignal(int)
    playing_changed = pyqtSignal(bool)
    midi_conversion_requested = pyqtSignal(str, str, str)
    midi_open_requested = pyqtSignal(str)

    def __init__(
        self,
        tracks: Mapping[str, str | Path],
        parent: QWidget | None = None,
        *,
        player_factory: PlayerFactory | None = None,
        audio_output_factory: AudioOutputFactory | None = None,
        waveform_loader_factory: WaveformLoaderFactory | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("audioTrackMixer")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        normalized_tracks = self._validate_tracks(tracks)
        if (player_factory is None) != (audio_output_factory is None):
            raise ValueError("player_factory and audio_output_factory must be provided together")
        if player_factory is None:
            if QMediaDevices.defaultAudioOutput().isNull():
                raise RuntimeError(t("dialogs.complete.audio_tracks.no_output_device"))
            player_factory = _create_qt_media_player
            audio_output_factory = _create_qt_audio_output
        if waveform_loader_factory is None:
            waveform_loader_factory = FfmpegWaveformLoader

        self._player_factory = player_factory
        self._audio_output_factory = audio_output_factory
        self._waveform_loader_factory = waveform_loader_factory
        self._backends: dict[str, _TrackBackend] = {}
        self._pending_media_tracks = set(normalized_tracks)
        self._pending_waveform_tracks = set(normalized_tracks)
        self._ready = False
        self._failed = False
        self._error_message = ""
        self._shutdown = False
        self._is_playing = False
        self._position_ms = 0
        self._duration_ms = 0
        self._play_origin_ms = 0
        self._was_playing_before_seek = False
        self._updating_timeline = False
        self._updating_view_scroll = False
        self._zoom_factor = 1
        self._view_start_ms = 0
        self._view_end_ms = 1
        self._clock = QElapsedTimer()

        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(_SYNC_INTERVAL_MS)
        self._sync_timer.timeout.connect(self._on_sync_tick)
        self._load_timeout_timer = QTimer(self)
        self._load_timeout_timer.setSingleShot(True)
        self._load_timeout_timer.setInterval(_LOAD_TIMEOUT_MS)
        self._load_timeout_timer.timeout.connect(self._on_load_timeout)

        self._setup_ui()
        for track_name, path in normalized_tracks.items():
            self._create_track_backend(track_name, path)
        for backend in self._backends.values():
            if self._failed:
                break
            self._start_track_backend(backend)

        self._apply_mix()
        if (self._pending_media_tracks or self._pending_waveform_tracks) and not self._failed:
            self._load_timeout_timer.start()

    @staticmethod
    def _validate_tracks(tracks: Mapping[str, str | Path]) -> dict[str, Path]:
        if not tracks:
            raise ValueError(t("dialogs.complete.audio_tracks.empty"))

        normalized: dict[str, Path] = {}
        for raw_name, raw_path in tracks.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("Audio track names cannot be empty")
            path = Path(raw_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(t("dialogs.complete.audio_tracks.missing_file", path=path))
            normalized[name] = path.resolve()

        order = {name: index for index, name in enumerate(_TRACK_ORDER)}
        return dict(
            sorted(
                normalized.items(),
                key=lambda item: (order.get(item[0], len(order)), item[0].casefold()),
            )
        )

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(7)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        self.title_label = QLabel()
        self.title_label.setObjectName("audioMixerTitle")
        self.title_label.setStyleSheet("font-size: 15px; font-weight: 700; color: #eef5ff;")
        layout.addWidget(self.title_label)

        self.subtitle_label = QLabel()
        self.subtitle_label.setObjectName("audioMixerSubtitle")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setMinimumWidth(0)
        self.subtitle_label.setStyleSheet("color: #9eacc1;")
        layout.addWidget(self.subtitle_label)

        self.error_label = QLabel()
        self.error_label.setObjectName("audioMixerErrorLabel")
        self.error_label.setWordWrap(True)
        self.error_label.setMinimumWidth(0)
        self.error_label.setStyleSheet(
            "background: #3a1f2a; border: 1px solid #b84d68; "
            "border-radius: 5px; color: #ffb3c3; padding: 8px;"
        )
        self.error_label.hide()
        layout.addWidget(self.error_label)

        transport = FlowLayout(horizontal_spacing=8, vertical_spacing=7)
        self.play_button = QPushButton(t("dialogs.complete.audio_tracks.play"))
        self.play_button.setObjectName("audioMixerPlayButton")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._toggle_playback)
        transport.addWidget(self.play_button)

        self.replay_button = QPushButton(t("dialogs.complete.audio_tracks.replay"))
        self.replay_button.setObjectName("audioMixerReplayButton")
        self.replay_button.setEnabled(False)
        self.replay_button.clicked.connect(self.replay)
        transport.addWidget(self.replay_button)

        self.align_button = QPushButton(t("dialogs.complete.audio_tracks.align"))
        self.align_button.setObjectName("audioMixerAlignButton")
        self.align_button.setEnabled(False)
        self.align_button.clicked.connect(self.align_tracks)
        transport.addWidget(self.align_button)

        self.add_track_button = QPushButton()
        self.add_track_button.setObjectName("audioMixerAddTrackButton")
        self.add_track_button.clicked.connect(self._choose_audio_tracks)
        transport.addWidget(self.add_track_button)

        self.zoom_out_button = QPushButton("−")
        self.zoom_out_button.setObjectName("audioMixerZoomOutButton")
        self.zoom_out_button.clicked.connect(self._zoom_out)
        transport.addWidget(self.zoom_out_button)

        self.zoom_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setObjectName("audioMixerZoomSlider")
        self.zoom_slider.setRange(1, 16)
        self.zoom_slider.setValue(1)
        self.zoom_slider.setFixedWidth(110)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        transport.addWidget(self.zoom_slider)

        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setObjectName("audioMixerZoomInButton")
        self.zoom_in_button.clicked.connect(self._zoom_in)
        transport.addWidget(self.zoom_in_button)

        self.fit_button = QPushButton()
        self.fit_button.setObjectName("audioMixerFitButton")
        self.fit_button.clicked.connect(self.fit_timeline)
        transport.addWidget(self.fit_button)

        self.zoom_label = QLabel("1×")
        self.zoom_label.setObjectName("audioMixerZoomLabel")
        self.zoom_label.setStyleSheet("color: #9eacc1;")
        transport.addWidget(self.zoom_label)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("audioMixerTimeLabel")
        self.time_label.setStyleSheet("color: #f0f5ff;")
        transport.addWidget(self.time_label)

        self.status_label = QLabel(t("dialogs.complete.audio_tracks.loading"))
        self.status_label.setObjectName("audioMixerStatusLabel")
        self.status_label.setStyleSheet("color: #8fa4c2;")
        transport.addWidget(self.status_label)
        layout.addLayout(transport)

        self.timeline_label = QLabel()
        self.timeline_label.setStyleSheet("color: #aebbd0;")
        layout.addWidget(self.timeline_label)

        self.timeline_ruler = TimelineRuler(self)
        self.timeline_ruler.setObjectName("audioMixerTimelineRuler")
        self.timeline_ruler.seek_requested.connect(self.seek)
        self.timeline_ruler.zoom_requested.connect(self._on_waveform_zoom_requested)
        layout.addWidget(self.timeline_ruler)

        self.timeline = NoWheelSlider(Qt.Orientation.Horizontal)
        self.timeline.setObjectName("audioMixerTimeline")
        self.timeline.setRange(0, 0)
        self.timeline.setEnabled(False)
        self.timeline.setMinimumWidth(0)
        self.timeline.setAccessibleName(t("dialogs.complete.audio_tracks.timeline"))
        self.timeline.sliderPressed.connect(self._on_seek_pressed)
        self.timeline.sliderMoved.connect(self._on_seek_moved)
        self.timeline.sliderReleased.connect(self._on_seek_released)
        self.timeline.valueChanged.connect(self._on_timeline_value_changed)
        layout.addWidget(self.timeline)

        self.track_container = QWidget()
        self.track_container.setObjectName("audioMixerTrackContainer")
        self.track_container.setMinimumWidth(0)
        self.track_container.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self._track_layout = QVBoxLayout(self.track_container)
        self._track_layout.setContentsMargins(0, 0, 0, 0)
        self._track_layout.setSpacing(8)
        self._track_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self._track_layout.addStretch(1)
        layout.addWidget(self.track_container)

        # Keep viewport state for button zoom/autofollow without exposing a
        # scrollbar-like control.  The complete timeline is the default view.
        self.view_scroll = NoWheelSlider(Qt.Orientation.Horizontal, self)
        self.view_scroll.setObjectName("audioMixerViewScroll")
        self.view_scroll.setRange(0, 0)
        self.view_scroll.setEnabled(False)
        self.view_scroll.valueChanged.connect(self._on_view_scroll_changed)
        self.view_scroll.hide()

        self.setStyleSheet("""
            QWidget#audioTrackMixer {
                background: #101a32;
                border: 1px solid #38547a;
                border-radius: 9px;
            }
            QWidget#audioTrackMixer QFrame[audioTrackRow="true"] {
                background: #162541;
                border: 1px solid #2f527d;
                border-radius: 7px;
            }
            QWidget#audioTrackMixer QPushButton {
                min-height: 26px;
                padding: 3px 10px;
                color: #e7eef9;
                background: #243a5a;
                border: 1px solid #41658f;
                border-radius: 5px;
            }
            QWidget#audioTrackMixer QPushButton:hover {
                background: #31517a;
                border-color: #62a7f5;
            }
            QWidget#audioTrackMixer QPushButton:disabled {
                color: #718096;
                background: #1d293c;
                border-color: #2b3b52;
            }
            QWidget#audioTrackMixer QPushButton[audioTrackToggle="true"]:checked {
                color: #ffffff;
                background: #3578c7;
                border-color: #74b8ff;
            }
            QWidget#audioTrackMixer QPushButton[audioTrackRemove="true"] {
                color: #ffb3c3;
                background: #332338;
                border-color: #70435a;
            }
            QWidget#audioTrackMixer QWidget#audioMixerTrackContainer {
                background: transparent;
            }
            QWidget#audioTrackMixer QSlider::groove:horizontal {
                height: 5px;
                background: #253958;
                border-radius: 2px;
            }
            QWidget#audioTrackMixer QSlider::sub-page:horizontal {
                background: #4a9eff;
                border-radius: 2px;
            }
            QWidget#audioTrackMixer QSlider::handle:horizontal {
                width: 14px;
                margin: -5px 0;
                background: #dcecff;
                border: 1px solid #4a9eff;
                border-radius: 7px;
            }
            """)

        self.update_translations()

    def _create_track_backend(self, track_name: str, path: Path) -> None:
        color = _TRACK_COLORS[len(self._backends) % len(_TRACK_COLORS)]
        row = _AudioTrackRow(track_name, path, color, self.track_container)
        player = self._player_factory(self)
        audio_output = self._audio_output_factory(self)
        waveform_loader = self._waveform_loader_factory(self)
        player.setAudioOutput(audio_output)
        audio_output.setVolume(1.0)

        backend = _TrackBackend(
            path=path,
            row=row,
            player=player,
            audio_output=audio_output,
            waveform_loader=waveform_loader,
        )
        self._backends[track_name] = backend
        self._track_layout.insertWidget(max(0, self._track_layout.count() - 1), row)

        row.mute_changed.connect(lambda _checked: self._apply_mix())
        row.solo_changed.connect(lambda _checked: self._apply_mix())
        row.volume_changed.connect(
            lambda volume_db, name=track_name: self._set_output_volume(name, volume_db)
        )
        row.offset_changed.connect(
            lambda _offset, name=track_name: self._on_track_offset_changed(name)
        )
        row.offset_committed.connect(lambda name=track_name: self._on_track_offset_committed(name))
        row.remove_requested.connect(lambda name=track_name: self.remove_track(name))
        row.midi_conversion_requested.connect(
            lambda route, name=track_name, track_path=path: (
                self.midi_conversion_requested.emit(
                    name,
                    str(track_path),
                    route,
                )
            )
        )
        row.midi_open_requested.connect(self.midi_open_requested.emit)
        row.seek_requested.connect(self.seek)
        row.zoom_requested.connect(self._on_waveform_zoom_requested)
        player.durationChanged.connect(
            lambda duration, name=track_name: self._on_duration_changed(name, duration)
        )
        player.mediaStatusChanged.connect(
            lambda status, name=track_name: self._on_media_status_changed(name, status)
        )
        player.errorOccurred.connect(
            lambda _error, message, name=track_name: self._on_player_error(name, message)
        )
        waveform_loader.loaded.connect(
            lambda envelope, name=track_name: self._on_waveform_loaded(name, envelope)
        )
        waveform_loader.failed.connect(
            lambda message, name=track_name: self._on_waveform_error(name, message)
        )

    def _start_track_backend(self, backend: _TrackBackend) -> None:
        backend.player.setSource(QUrl.fromLocalFile(str(backend.path)))
        backend.waveform_loader.start(backend.path)

    @property
    def track_names(self) -> tuple[str, ...]:
        return tuple(self._backends)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def has_error(self) -> bool:
        return self._failed

    @property
    def position_ms(self) -> int:
        if self._is_playing:
            return self._clock_position()
        return self._position_ms

    @property
    def duration_ms(self) -> int:
        return self._duration_ms

    def track_state(self, track_name: str) -> AudioTrackState:
        backend = self._backend(track_name)
        return AudioTrackState(
            name=track_name,
            path=backend.path,
            muted=backend.row.mute_button.isChecked(),
            solo=backend.row.solo_button.isChecked(),
            volume_db=backend.row.volume_db,
            offset_ms=backend.row.offset_ms,
            duration_ms=self._track_duration(backend),
        )

    def set_track_muted(self, track_name: str, muted: bool) -> None:
        self._backend(track_name).row.mute_button.setChecked(bool(muted))

    def set_track_solo(self, track_name: str, solo: bool) -> None:
        self._backend(track_name).row.solo_button.setChecked(bool(solo))

    def set_track_volume_db(self, track_name: str, volume_db: float) -> None:
        self._backend(track_name).row.set_volume_db(volume_db)

    def set_track_offset_ms(self, track_name: str, offset_ms: int) -> None:
        self._backend(track_name).row.set_offset_ms(offset_ms)

    def set_midi_controls_enabled(self, enabled: bool) -> None:
        for backend in self._backends.values():
            backend.row.set_midi_controls_enabled(enabled)

    def set_track_midi_running(self, track_name: str, route: str) -> None:
        self._backend(track_name).row.set_midi_conversion_running(route)

    def set_track_midi_progress(self, track_name: str, message: str) -> None:
        self._backend(track_name).row.set_midi_conversion_progress(message)

    def set_track_midi_succeeded(
        self,
        track_name: str,
        route: str,
        midi_path: str,
    ) -> None:
        self._backend(track_name).row.set_midi_conversion_succeeded(
            route,
            midi_path,
        )

    def set_track_midi_failed(self, track_name: str, error: str) -> None:
        self._backend(track_name).row.set_midi_conversion_failed(error)

    def set_track_midi_cancelled(self, track_name: str) -> None:
        self._backend(track_name).row.set_midi_conversion_cancelled()

    def add_audio_files(self, paths: Sequence[str | Path]) -> tuple[str, ...]:
        """Add local audio files and return their generated track names.

        The full selection is validated before any backend is created, so a
        rejected file never leaves a partially-added group behind.
        """
        self._require_operational()
        from src.utils.audio_utils import get_supported_formats

        selected = [Path(path).expanduser() for path in paths]
        if not selected:
            return ()
        supported = {suffix.casefold() for suffix in get_supported_formats()}
        for path in selected:
            if not path.is_file():
                raise FileNotFoundError(t("dialogs.complete.audio_tracks.missing_file", path=path))
            if path.suffix.casefold() not in supported:
                raise ValueError(
                    t(
                        "dialogs.complete.audio_tracks.unsupported_format",
                        path=path,
                        formats=", ".join(sorted(supported)),
                    )
                )

        reserved = set(self._backends)
        additions: dict[str, Path] = {}
        for path in selected:
            base_name = path.stem.strip() or "audio"
            name = base_name
            suffix = 2
            while name in reserved:
                name = f"{base_name} {suffix}"
                suffix += 1
            reserved.add(name)
            additions[name] = path
        self.add_tracks(additions)
        return tuple(additions)

    def add_tracks(self, tracks: Mapping[str, str | Path]) -> None:
        """Add a validated group of tracks to the current timeline."""
        self._require_operational()
        normalized = self._validate_tracks(tracks)
        duplicates = set(normalized).intersection(self._backends)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(f"Audio track names already exist: {names}")

        self.pause()
        self._ready = False
        self.ready_changed.emit(False)
        self.error_label.hide()
        self.status_label.setText(t("dialogs.complete.audio_tracks.loading"))
        self._pending_media_tracks.update(normalized)
        self._pending_waveform_tracks.update(normalized)
        for track_name, path in normalized.items():
            self._create_track_backend(track_name, path)
        for track_name in normalized:
            self._start_track_backend(self._backends[track_name])
        self._apply_mix()
        self._load_timeout_timer.start()
        self._update_track_views()

    def remove_track(self, track_name: str) -> None:
        """Remove one track and release both playback and waveform resources."""
        self._require_not_shutdown()
        backend = self._backend(track_name)
        self.pause()
        self._pending_media_tracks.discard(track_name)
        self._pending_waveform_tracks.discard(track_name)
        del self._backends[track_name]
        self._release_backend(backend)

        self._failed = any(item.failed for item in self._backends.values())
        if not self._failed:
            self._error_message = ""
            self.error_label.clear()
            self.error_label.hide()
        self._recompute_duration()
        self._apply_mix()
        if not self._backends:
            self._set_empty_state()
        elif self._failed:
            self._set_failed_controls()
        else:
            self._check_ready()

    def play(self) -> None:
        self._require_operational()
        if not self._ready:
            raise RuntimeError(t("dialogs.complete.audio_tracks.loading"))
        if self._is_playing:
            return
        if self._duration_ms > 0 and self._position_ms >= self._duration_ms:
            self.seek(0)

        self._play_origin_ms = self._position_ms
        self._clock.start()
        self._is_playing = True
        self._sync_timer.start()
        self._sync_players(force=True)
        self._update_playing_state()

    def pause(self) -> None:
        if not self._is_playing:
            return
        self._position_ms = self._clock_position()
        self._sync_timer.stop()
        self._is_playing = False
        self._pause_players()
        self._update_timeline_ui()
        self._update_playing_state()

    def replay(self) -> None:
        self.seek(0)
        self.play()

    def seek(self, position_ms: int) -> None:
        self._require_operational()
        position = max(0, int(position_ms))
        if self._duration_ms > 0:
            position = min(position, self._duration_ms)
        self._position_ms = position
        if self._is_playing:
            self._play_origin_ms = position
            self._clock.restart()
        self._sync_players(force=True)
        self._update_timeline_ui()
        self.position_changed.emit(position)

    def align_tracks(self) -> None:
        self._require_operational()
        for backend in self._backends.values():
            backend.row.set_offset_ms(0, emit=False)
        self._recompute_duration()
        self.seek(self.position_ms)

    def shutdown(self) -> None:
        """Stop playback and release all media handles. Safe to call repeatedly."""
        if self._shutdown:
            return
        self._shutdown = True
        self._sync_timer.stop()
        self._load_timeout_timer.stop()
        self._sync_timer.timeout.disconnect(self._on_sync_tick)
        self._load_timeout_timer.timeout.disconnect(self._on_load_timeout)
        self._is_playing = False
        for backend in tuple(self._backends.values()):
            self._release_backend(backend)
        self._backends.clear()
        self._pending_media_tracks.clear()
        self._pending_waveform_tracks.clear()
        self.play_button.setEnabled(False)
        self.replay_button.setEnabled(False)
        self.align_button.setEnabled(False)
        self.add_track_button.setEnabled(False)
        self.timeline.setEnabled(False)
        self.view_scroll.setEnabled(False)

    def _backend(self, track_name: str) -> _TrackBackend:
        try:
            return self._backends[track_name]
        except KeyError as exc:
            raise KeyError(f"Unknown audio track: {track_name}") from exc

    def _require_operational(self) -> None:
        self._require_not_shutdown()
        if self._failed:
            raise RuntimeError(self._error_message)

    def _require_not_shutdown(self) -> None:
        if self._shutdown:
            raise RuntimeError(t("dialogs.complete.audio_tracks.closed"))

    def _release_backend(self, backend: _TrackBackend) -> None:
        backend.waveform_loader.cancel()
        backend.player.stop()
        backend.audio_output.setMuted(True)
        backend.player.setSource(QUrl())
        backend.player.setAudioOutput(None)
        backend.row.hide()
        backend.row.setParent(None)
        backend.row.deleteLater()
        backend.player.deleteLater()
        backend.audio_output.deleteLater()
        backend.waveform_loader.deleteLater()

    def _toggle_playback(self) -> None:
        if self._is_playing:
            self.pause()
        else:
            self.play()

    def _on_duration_changed(self, track_name: str, duration_ms: int) -> None:
        backend = self._backend(track_name)
        duration = int(duration_ms)
        if duration > 0 or backend.duration_ms == 0:
            backend.duration_ms = max(0, duration)
        self._recompute_duration()

    def _on_media_status_changed(self, track_name: str, status: QMediaPlayer.MediaStatus) -> None:
        if track_name not in self._backends:
            return
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            backend = self._backend(track_name)
            message = backend.player.errorString() or t(
                "dialogs.complete.audio_tracks.invalid_media"
            )
            self._fail_track(track_name, message)
            return
        if status in {
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
            QMediaPlayer.MediaStatus.EndOfMedia,
        }:
            self._pending_media_tracks.discard(track_name)
            self._check_ready()

    def _on_player_error(self, track_name: str, message: str) -> None:
        if track_name not in self._backends:
            return
        detail = message.strip() or t("dialogs.complete.audio_tracks.invalid_media")
        self._fail_track(track_name, detail)

    def _on_waveform_loaded(
        self,
        track_name: str,
        envelope: WaveformEnvelope,
    ) -> None:
        if track_name not in self._backends or self._shutdown:
            return
        backend = self._backends[track_name]
        backend.waveform_duration_ms = max(0, int(envelope.duration_ms))
        backend.row.waveform.set_envelope(envelope)
        self._pending_waveform_tracks.discard(track_name)
        self._recompute_duration()
        self._check_ready()

    def _on_waveform_error(self, track_name: str, message: str) -> None:
        if track_name not in self._backends or self._shutdown:
            return
        detail = message.strip() or t("dialogs.complete.audio_tracks.waveform_failed")
        self._pending_waveform_tracks.discard(track_name)
        self._backends[track_name].row.waveform.set_error(detail)
        self._fail_track(track_name, detail)

    def _on_load_timeout(self) -> None:
        pending = self._pending_media_tracks | self._pending_waveform_tracks
        if not pending or self._failed or self._shutdown:
            return
        names = ", ".join(_display_track_name(name) for name in sorted(pending))
        self._fail_track(
            sorted(pending)[0],
            t("dialogs.complete.audio_tracks.load_timeout", tracks=names),
        )

    def _check_ready(self) -> None:
        if self._shutdown or self._failed or not self._backends:
            return
        if self._pending_media_tracks or self._pending_waveform_tracks:
            return
        self._mark_ready()

    def _mark_ready(self) -> None:
        self._load_timeout_timer.stop()
        if self._ready:
            self._update_track_views()
            return
        self._ready = True
        self.play_button.setEnabled(True)
        self.replay_button.setEnabled(True)
        self.align_button.setEnabled(True)
        self.timeline.setEnabled(True)
        self.status_label.setText(t("dialogs.complete.audio_tracks.ready"))
        self._recompute_duration()
        self.ready_changed.emit(True)

    def _fail_track(self, track_name: str, message: str) -> None:
        backend = self._backend(track_name)
        detail = t(
            "dialogs.complete.audio_tracks.playback_failed",
            track=_display_track_name(track_name),
            error=message,
        )
        logger.error(
            "Audio track playback failed: track=%s path=%s error=%s",
            track_name,
            backend.path,
            message,
        )
        backend.failed = True
        if self._failed or self._shutdown:
            return

        self._failed = True
        self._ready = False
        self._error_message = detail
        self._sync_timer.stop()
        self._load_timeout_timer.stop()
        self._is_playing = False
        for item in self._backends.values():
            item.player.stop()
        self._set_failed_controls()
        self.status_label.setText(t("dialogs.complete.audio_tracks.failed"))
        self.error_label.setText(detail)
        self.error_label.setToolTip(str(backend.path))
        self.error_label.show()
        self._update_playing_state()
        self.ready_changed.emit(False)
        self.playback_error.emit(track_name, str(backend.path), detail)

    def _set_failed_controls(self) -> None:
        self.play_button.setEnabled(False)
        self.replay_button.setEnabled(False)
        self.align_button.setEnabled(False)
        self.timeline.setEnabled(False)

    def _set_empty_state(self) -> None:
        self._load_timeout_timer.stop()
        self._ready = False
        self._failed = False
        self._duration_ms = 0
        self._position_ms = 0
        self.play_button.setEnabled(False)
        self.replay_button.setEnabled(False)
        self.align_button.setEnabled(False)
        self.timeline.setEnabled(False)
        self.timeline.setRange(0, 0)
        self.view_scroll.setRange(0, 0)
        self.view_scroll.setEnabled(False)
        self.status_label.setText(t("dialogs.complete.audio_tracks.empty_timeline"))
        self._update_timeline_ui()

    def _set_output_volume(self, track_name: str, volume_db: float) -> None:
        self._backend(track_name).audio_output.setVolume(_db_to_linear(volume_db))

    def _apply_mix(self) -> None:
        any_solo = any(backend.row.solo_button.isChecked() for backend in self._backends.values())
        for backend in self._backends.values():
            muted = backend.row.mute_button.isChecked()
            excluded_by_solo = any_solo and not backend.row.solo_button.isChecked()
            backend.audio_output.setMuted(muted or excluded_by_solo)

    def _choose_audio_tracks(self) -> None:
        from src.utils.audio_utils import get_supported_formats

        formats = " ".join(f"*{suffix}" for suffix in get_supported_formats())
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            t("dialogs.complete.audio_tracks.add_track"),
            "",
            t("dialogs.complete.audio_tracks.file_filter", formats=formats),
        )
        if not paths:
            return
        try:
            self.add_audio_files(paths)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            detail = t(
                "dialogs.complete.audio_tracks.add_rejected",
                error=str(exc),
            )
            logger.error("Audio tracks were rejected: %s", exc)
            self.error_label.setText(detail)
            self.error_label.setToolTip("\n".join(paths))
            self.error_label.show()

    def _on_track_offset_changed(self, track_name: str) -> None:
        backend = self._backend(track_name)
        self._recompute_duration()
        if not backend.row.offset_slider.isSliderDown():
            self._sync_players(force=True)

    def _on_track_offset_committed(self, track_name: str) -> None:
        self._backend(track_name)
        self._sync_players(force=True)

    def _recompute_duration(self) -> None:
        self._duration_ms = max(
            (
                max(0, backend.row.offset_ms + self._track_duration(backend))
                for backend in self._backends.values()
            ),
            default=0,
        )
        if self._position_ms > self._duration_ms and self._duration_ms > 0:
            self._position_ms = self._duration_ms
        self.timeline.setRange(0, self._duration_ms)
        self.timeline.setPageStep(max(1000, self._duration_ms // 20))
        self._update_view_range()
        self._update_timeline_ui()

    @staticmethod
    def _track_duration(backend: _TrackBackend) -> int:
        return backend.duration_ms or backend.waveform_duration_ms

    def fit_timeline(self) -> None:
        self._set_zoom_factor(1, anchor_ms=self.position_ms)

    def _zoom_in(self) -> None:
        self._set_zoom_factor(self._zoom_factor + 1, anchor_ms=self.position_ms)

    def _zoom_out(self) -> None:
        self._set_zoom_factor(self._zoom_factor - 1, anchor_ms=self.position_ms)

    def _on_zoom_changed(self, zoom_factor: int) -> None:
        self._set_zoom_factor(zoom_factor, anchor_ms=self.position_ms)

    def _on_waveform_zoom_requested(self, wheel_delta: int, anchor_ms: int) -> None:
        step = 1 if wheel_delta > 0 else -1
        self._set_zoom_factor(self._zoom_factor + step, anchor_ms=anchor_ms)

    def _set_zoom_factor(self, zoom_factor: int, *, anchor_ms: int) -> None:
        factor = max(self.zoom_slider.minimum(), min(self.zoom_slider.maximum(), zoom_factor))
        old_span = max(1, self._view_end_ms - self._view_start_ms)
        old_ratio = (anchor_ms - self._view_start_ms) / old_span
        old_ratio = max(0.0, min(1.0, old_ratio))
        self._zoom_factor = factor
        with QSignalBlocker(self.zoom_slider):
            self.zoom_slider.setValue(factor)
        self.zoom_label.setText(f"{factor}×")

        new_span = self._visible_span_ms()
        new_start = round(anchor_ms - old_ratio * new_span)
        self._set_view_start(new_start)

    def _visible_span_ms(self) -> int:
        if self._duration_ms <= 0:
            return 1
        return max(1, round(self._duration_ms / self._zoom_factor))

    def _update_view_range(self) -> None:
        span = self._visible_span_ms()
        maximum_start = max(0, self._duration_ms - span)
        self._view_start_ms = min(max(0, self._view_start_ms), maximum_start)
        self._view_end_ms = min(
            self._duration_ms if self._duration_ms > 0 else 1,
            self._view_start_ms + span,
        )
        self._updating_view_scroll = True
        try:
            self.view_scroll.setRange(0, maximum_start)
            self.view_scroll.setPageStep(span)
            self.view_scroll.setSingleStep(max(1, span // 20))
            self.view_scroll.setValue(self._view_start_ms)
            self.view_scroll.setEnabled(maximum_start > 0)
        finally:
            self._updating_view_scroll = False

    def _set_view_start(self, start_ms: int) -> None:
        span = self._visible_span_ms()
        maximum_start = max(0, self._duration_ms - span)
        self._view_start_ms = min(max(0, int(start_ms)), maximum_start)
        self._view_end_ms = min(
            self._duration_ms if self._duration_ms > 0 else 1,
            self._view_start_ms + span,
        )
        self._updating_view_scroll = True
        try:
            self.view_scroll.setRange(0, maximum_start)
            self.view_scroll.setPageStep(span)
            self.view_scroll.setValue(self._view_start_ms)
            self.view_scroll.setEnabled(maximum_start > 0)
        finally:
            self._updating_view_scroll = False
        self._update_track_views()

    def _on_view_scroll_changed(self, start_ms: int) -> None:
        if self._updating_view_scroll:
            return
        self._set_view_start(start_ms)

    def _update_track_views(self) -> None:
        position = self._clock_position() if self._is_playing else self._position_ms
        self.timeline_ruler.set_timeline(
            self._duration_ms,
            self._view_start_ms,
            self._view_end_ms,
            position,
        )
        for backend in self._backends.values():
            backend.row.waveform.set_timeline(
                self._duration_ms,
                self._view_start_ms,
                self._view_end_ms,
                position,
                backend.row.offset_ms,
            )

    def _clock_position(self) -> int:
        if not self._is_playing:
            return self._position_ms
        position = self._play_origin_ms + self._clock.elapsed()
        if self._duration_ms > 0:
            return min(position, self._duration_ms)
        return position

    def _on_sync_tick(self) -> None:
        position = self._clock_position()
        if self._duration_ms > 0 and position >= self._duration_ms:
            self._position_ms = self._duration_ms
            self._sync_timer.stop()
            self._is_playing = False
            self._pause_players()
            self._update_timeline_ui()
            self._update_playing_state()
            self.position_changed.emit(self._position_ms)
            return

        self._position_ms = position
        self._sync_players(force=False)
        self._update_timeline_ui()
        self.position_changed.emit(position)

    def _pause_players(self) -> None:
        for backend in self._backends.values():
            if backend.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                backend.player.pause()

    def _sync_players(self, *, force: bool) -> None:
        global_position = self._clock_position() if self._is_playing else self._position_ms
        for backend in self._backends.values():
            local_position = global_position - backend.row.offset_ms
            track_duration = self._track_duration(backend)
            within_track = local_position >= 0 and (
                track_duration <= 0 or local_position < track_duration
            )
            if within_track:
                target = int(local_position)
                playback_state = backend.player.playbackState()
                current_position = backend.player.position()
                if (
                    (force and current_position != target)
                    or abs(current_position - target) > _MAX_PLAYER_DRIFT_MS
                    or (
                        self._is_playing
                        and playback_state != QMediaPlayer.PlaybackState.PlayingState
                        and current_position != target
                    )
                ):
                    backend.player.setPosition(target)
                if self._is_playing:
                    if playback_state != QMediaPlayer.PlaybackState.PlayingState:
                        backend.player.play()
                elif playback_state == QMediaPlayer.PlaybackState.PlayingState:
                    backend.player.pause()
                continue

            target = 0 if local_position < 0 else track_duration
            if force or backend.player.position() != target:
                backend.player.setPosition(target)
            if backend.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                backend.player.pause()

    def _update_timeline_ui(self) -> None:
        position = self._clock_position() if self._is_playing else self._position_ms
        if (
            self._is_playing
            and self._zoom_factor > 1
            and position >= self._view_end_ms
            and self._duration_ms > 0
        ):
            self._set_view_start(position - self._visible_span_ms() // 10)
        self._updating_timeline = True
        try:
            self.timeline.setValue(position)
        finally:
            self._updating_timeline = False
        self.time_label.setText(f"{_format_time(position)} / {_format_time(self._duration_ms)}")
        self._update_track_views()

    def _update_playing_state(self) -> None:
        self.play_button.setText(
            t("dialogs.complete.audio_tracks.pause")
            if self._is_playing
            else t("dialogs.complete.audio_tracks.play")
        )
        self.playing_changed.emit(self._is_playing)

    def update_translations(self) -> None:
        """Refresh visible labels without rebuilding playback state."""
        self.title_label.setText(t("dialogs.complete.audio_tracks.title"))
        self.subtitle_label.setText(t("dialogs.complete.audio_tracks.subtitle"))
        self.play_button.setText(
            t("dialogs.complete.audio_tracks.pause")
            if self._is_playing
            else t("dialogs.complete.audio_tracks.play")
        )
        self.replay_button.setText(t("dialogs.complete.audio_tracks.replay"))
        self.align_button.setText(t("dialogs.complete.audio_tracks.align"))
        self.add_track_button.setText(t("dialogs.complete.audio_tracks.add_track"))
        self.fit_button.setText(t("dialogs.complete.audio_tracks.fit"))
        self.zoom_out_button.setToolTip(t("dialogs.complete.audio_tracks.zoom_out"))
        self.zoom_in_button.setToolTip(t("dialogs.complete.audio_tracks.zoom_in"))
        self.fit_button.setToolTip(t("dialogs.complete.audio_tracks.fit_tooltip"))
        self.zoom_slider.setAccessibleName(t("dialogs.complete.audio_tracks.zoom"))
        self.timeline_label.setText(t("dialogs.complete.audio_tracks.timeline"))
        self.timeline.setAccessibleName(t("dialogs.complete.audio_tracks.timeline"))
        self.timeline_ruler.update_translations()
        for backend in self._backends.values():
            backend.row.update_translations()
        if self._failed:
            self.status_label.setText(t("dialogs.complete.audio_tracks.failed"))
        elif self._ready:
            self.status_label.setText(t("dialogs.complete.audio_tracks.ready"))
        elif self._backends:
            self.status_label.setText(t("dialogs.complete.audio_tracks.loading"))
        else:
            self.status_label.setText(t("dialogs.complete.audio_tracks.empty_timeline"))

    def _on_seek_pressed(self) -> None:
        self._was_playing_before_seek = self._is_playing
        if self._is_playing:
            self.pause()

    def _on_seek_moved(self, position_ms: int) -> None:
        self.seek(position_ms)

    def _on_seek_released(self) -> None:
        self.seek(self.timeline.value())
        if self._was_playing_before_seek:
            self.play()
        self._was_playing_before_seek = False

    def _on_timeline_value_changed(self, position_ms: int) -> None:
        if self._updating_timeline or self.timeline.isSliderDown():
            return
        self.seek(position_ms)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.shutdown()
        super().closeEvent(event)


__all__ = ["AudioTrackMixerWidget", "AudioTrackState"]
