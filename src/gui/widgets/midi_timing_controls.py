"""Shared desktop controls for explicit quantization and grid correction."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.i18n.translator import t


class QuantizationControls(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.strength_label = QLabel()
        self.strength = QSpinBox()
        self.strength.setRange(0, 100)
        self.strength.setValue(100)
        self.strength.setSuffix("%")
        self.strength.setKeyboardTracking(False)
        self.triplet = QCheckBox()
        self.automatic = QCheckBox()
        layout.addRow(self.strength_label, self.strength)
        layout.addRow(self.triplet)
        layout.addRow(self.automatic)
        self.update_translations()

    def options(self):
        return dict(
            strength=self.strength.value() / 100,
            triplet=self.triplet.isChecked(),
            auto=self.automatic.isChecked(),
        )

    def update_translations(self):
        self.strength_label.setText(t("muscriptor_result.quantize_strength"))
        self.triplet.setText(t("muscriptor_result.quantize_triplet"))
        self.automatic.setText(t("muscriptor_result.quantize_auto"))


class MidiTimingControls(QPushButton):
    apply_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("midiTimingTools")
        self.clicked.connect(self.open_dialog)
        self.update_translations()

    def update_translations(self):
        self.setText(t("muscriptor_result.timing_tools"))
        self.setToolTip(t("muscriptor_result.timing_notice"))

    def open_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(t("muscriptor_result.timing_title"))
        layout = QVBoxLayout(dialog)
        notice = QLabel(t("muscriptor_result.timing_notice"))
        notice.setWordWrap(True)
        layout.addWidget(notice)
        form = QFormLayout()
        bpm = QDoubleSpinBox()
        bpm.setRange(0, 400)
        bpm.setDecimals(3)
        bpm.setSpecialValueText(t("muscriptor_result.timing_keep"))
        meter = QSpinBox()
        meter.setRange(0, 255)
        meter.setSpecialValueText(t("muscriptor_result.timing_keep"))
        unit = QComboBox()
        for denominator in (1, 2, 4, 8, 16, 32):
            unit.addItem(str(denominator), denominator)
        unit.setCurrentIndex(2)
        meter_row = QWidget()
        meter_layout = QHBoxLayout(meter_row)
        meter_layout.setContentsMargins(0, 0, 0, 0)
        meter_layout.addWidget(meter)
        meter_layout.addWidget(QLabel("/"))
        meter_layout.addWidget(unit)
        first = QDoubleSpinBox()
        first.setRange(-1, 14400)
        first.setDecimals(3)
        first.setValue(-1)
        first.setSpecialValueText(t("muscriptor_result.timing_keep"))
        scale = QComboBox()
        for value in (0.25, 0.5, 1.0, 2.0, 4.0):
            scale.addItem(f"{value:g}×", value)
        scale.setCurrentIndex(2)
        fill = QDoubleSpinBox()
        fill.setRange(0, 4)
        fill.setDecimals(3)
        fill.setSingleStep(0.125)
        trim, smooth = QCheckBox(), QCheckBox()
        smooth.setToolTip(t("muscriptor_result.timing_smooth_help"))
        fields = (
            ("timing_bpm", bpm),
            ("timing_meter", meter_row),
            ("timing_first_downbeat", first),
            ("timing_beatscale", scale),
            ("timing_fill_gaps", fill),
            ("timing_trim", trim),
            ("timing_smooth", smooth),
        )
        for key, field in fields:
            field.setObjectName(key)
            form.addRow(t(f"muscriptor_result.{key}"), field)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).setText(
            t("muscriptor_result.timing_apply")
        )
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(
            t("muscriptor_result.timing_cancel")
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_requested.emit(
                dict(
                    bpm=bpm.value() or None,
                    beats_per_bar=meter.value() or None,
                    beat_unit=unit.currentData(),
                    first_downbeat=None if first.value() < 0 else first.value(),
                    beat_scale=float(scale.currentData()),
                    fill_gaps=fill.value(),
                    trim_leading_bars=trim.isChecked(),
                    smooth_tempo=smooth.isChecked(),
                )
            )
