"""Compact persistent controls for chord monitoring and click auditions."""

from __future__ import annotations

import logging

from PyQt6.QtCore import QSettings, QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from src.core.chord_audio import CHORD_DEFAULT_VOLUME, CHORD_MAX_VOLUME
from src.gui.layouts import FlowLayout
from src.gui.widgets.wheel_safe_controls import NoWheelSlider, NoWheelSpinBox
from src.i18n.translator import t


def chord_settings():
    return QSettings(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, "mason369", "MusicToMidi"
    )


class ChordPlaybackControls(QWidget):
    changed = pyqtSignal(bool, int)
    retry_requested = pyqtSignal()

    def __init__(self, parent=None, *, settings=None):
        super().__init__(parent)
        self.settings = settings if settings is not None else chord_settings()
        self._state, self._error = "off", ""
        self._save_error = False
        try:
            volume = int(self.settings.value("chords/volume", CHORD_DEFAULT_VOLUME))
        except (TypeError, ValueError):
            volume = CHORD_DEFAULT_VOLUME
        volume = max(0, min(CHORD_MAX_VOLUME, volume))
        enabled = str(self.settings.value("chords/follow", "false")).lower() == "true"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 4)
        layout.setSpacing(4)
        row = QWidget(self)
        flow = FlowLayout(row, horizontal_spacing=14, vertical_spacing=4)
        flow.setContentsMargins(0, 0, 0, 0)
        self.follow = QCheckBox(row)
        self.follow.setObjectName("chordFollowPlayback")
        self.follow.setChecked(enabled)
        flow.addWidget(self.follow)
        group = QWidget(row)
        group_layout = QHBoxLayout(group)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(6)
        self.volume_label = QLabel(group)
        self.volume_label.setStyleSheet("color: #c8d3e6;")
        self.slider = NoWheelSlider(Qt.Orientation.Horizontal, group)
        self.slider.setObjectName("chordVolumeSlider")
        self.slider.setRange(0, CHORD_MAX_VOLUME)
        self.slider.setValue(volume)
        self.slider.setMinimumWidth(70)
        self.slider.setMaximumWidth(150)
        self.value = NoWheelSpinBox(group)
        self.value.setObjectName("chordVolumeValue")
        self.value.setStyleSheet(
            "background: #172b45; color: #cde5ff; border: 1px solid #365f8d; "
            "border-radius: 4px; padding: 3px 4px;"
        )
        self.value.setRange(0, CHORD_MAX_VOLUME)
        self.value.setSuffix("%")
        self.value.setValue(volume)
        self.volume_label.setBuddy(self.value)
        for widget in (self.volume_label, self.slider, self.value):
            group_layout.addWidget(widget)
        flow.addWidget(group)
        layout.addWidget(row)
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #9bacbf; font-size: 11px;")
        layout.addWidget(self.status)
        self.retry = QPushButton(self)
        self.retry.setObjectName("chordPlaybackRetry")
        self.retry.hide()
        layout.addWidget(self.retry, alignment=Qt.AlignmentFlag.AlignLeft)
        self.follow.toggled.connect(self._changed)
        self.slider.valueChanged.connect(self._volume_changed)
        self.value.valueChanged.connect(self._volume_changed)
        self.slider.sliderReleased.connect(self.save)
        self.value.editingFinished.connect(self.save)
        self.retry.clicked.connect(self.retry_requested)
        self.update_translations()

    def _volume_changed(self, volume):
        with QSignalBlocker(self.slider), QSignalBlocker(self.value):
            self.slider.setValue(volume)
            self.value.setValue(volume)
        self._changed()

    def _changed(self, *_):
        self.settings.setValue("chords/follow", self.follow.isChecked())
        self.settings.setValue("chords/volume", self.value.value())
        self.changed.emit(self.follow.isChecked(), self.value.value())
        if self.sender() is self.follow:
            self.save()

    def save(self):
        self.settings.sync()
        self._save_error = self.settings.status() != QSettings.Status.NoError
        if self._save_error:
            logging.getLogger(__name__).error("Unable to save chord playback preferences")
        self.update_translations()

    def set_state(self, state: str, error: str = ""):
        self._state, self._error = state, error
        self.update_translations()

    def update_translations(self):
        self.follow.setText(t("muscriptor_result.chord_follow"))
        self.follow.setToolTip(t("muscriptor_result.chord_follow_hint"))
        self.volume_label.setText(t("muscriptor_result.chord_volume"))
        hint = t("muscriptor_result.chord_volume_hint")
        for widget in (self.slider, self.value):
            widget.setToolTip(hint)
            widget.setAccessibleName(t("muscriptor_result.chord_volume"))
        state = "save_failed" if self._save_error else self._state
        self.status.setText(t(f"muscriptor_result.chord_monitor_{state}"))
        self.status.setToolTip(self._error)
        self.retry.setText(t("muscriptor_result.chord_monitor_retry"))
        self.retry.setVisible(self._state == "failed")
