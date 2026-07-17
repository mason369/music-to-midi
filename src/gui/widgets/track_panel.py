"""
轨道面板组件 - 支持模式选择与多乐器后端选择。
"""
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.i18n.translator import get_translator, t
from src.gui.widgets.wheel_safe_controls import NoWheelComboBox
from src.models.data_models import (
    MidiTrackMode,
    MultiInstrumentModel,
    ProcessingMode,
    TrackLayout,
    YourMT3Model,
)
from src.utils.yourmt3_downloader import YOURMT3_MODELS


class WrappedCheckBox(QWidget):
    """Checkbox with a label that participates in height-for-width wrapping."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.checkbox = QCheckBox()
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.label.installEventFilter(self)
        self.label.setBuddy(self.checkbox)
        self.label.setStyleSheet("color: #c8d3e6;")

        layout.addWidget(self.checkbox, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.label, 1)
        self.setFocusProxy(self.checkbox)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.toggled = self.checkbox.toggled
        self.stateChanged = self.checkbox.stateChanged

    def eventFilter(self, watched, event):
        if (
            watched is self.label
            and event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
            and self.isEnabled()
        ):
            self.checkbox.toggle()
            return True
        return super().eventFilter(watched, event)

    def isChecked(self) -> bool:  # noqa: N802 - mirrors QCheckBox
        return self.checkbox.isChecked()

    def setChecked(self, checked: bool) -> None:  # noqa: N802 - mirrors QCheckBox
        self.checkbox.setChecked(checked)

    def text(self) -> str:
        return self.label.text()

    def setText(self, text: str) -> None:  # noqa: N802 - mirrors QCheckBox
        self.label.setText(text)


class TrackPanel(QGroupBox):
    """轨道面板：处理模式 + 多乐器后端 + 模式说明。"""

    layout_changed = pyqtSignal(object)  # TrackLayout
    mode_changed = pyqtSignal(str)       # processing_mode 字符串
    model_changed = pyqtSignal(str)      # selected backend 字符串
    backend_changed = pyqtSignal(str)    # backward-compatible alias

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_layout = TrackLayout(mode=ProcessingMode.SMART, tracks=[])
        self._controls_enabled = True
        self._setup_ui()

    @staticmethod
    def _combo_style() -> str:
        return """
            QComboBox {
                padding: 4px 10px;
                border: 1px solid #3a4a6a;
                border-radius: 5px;
                background: #16213e;
                color: #e0e0e0;
                font-size: 11px;
                min-width: 0px;
            }
            QComboBox:hover {
                border-color: #4a9eff;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background: #1f2940;
                border: 1px solid #3a4a6a;
                color: #e0e0e0;
                selection-background-color: #3a5a7c;
            }
        """

    @classmethod
    def _configure_combo(cls, combo: QComboBox) -> None:
        combo.setStyleSheet(cls._combo_style())
        combo.setMinimumWidth(0)
        combo.setMinimumContentsLength(0)
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    @staticmethod
    def _selector_layout(parent: QWidget) -> QFormLayout:
        layout = QFormLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return layout

    def _setup_ui(self):
        self.setTitle(t("main.tracks.title"))
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
                color: #e0e0e0;
                border: 1px solid #3a4a6a;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 6px;
                background: #1f2940;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                background: #1f2940;
            }
            QLabel {
                color: #e0e0e0;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 8)
        main_layout.setSpacing(4)

        self._mode_label = QLabel(t("main.mode.label") + ":")
        self._mode_label.setWordWrap(True)
        self._mode_label.setStyleSheet("font-size: 11px; color: #b0b8c8; font-weight: normal;")

        self.mode_combo = NoWheelComboBox()
        self.mode_combo.addItem(t("main.mode.smart"), ProcessingMode.SMART.value)
        self.mode_combo.addItem(t("main.mode.vocal_split"), ProcessingMode.VOCAL_SPLIT.value)
        self.mode_combo.addItem(t("main.mode.six_stem_split"), ProcessingMode.SIX_STEM_SPLIT.value)
        self.mode_combo.addItem(t("main.mode.piano_transkun"), ProcessingMode.PIANO_TRANSKUN.value)
        self.mode_combo.addItem(
            t("main.mode.piano_transkun_v2_aug"),
            ProcessingMode.PIANO_TRANSKUN_V2_AUG.value,
        )
        self.mode_combo.addItem(t("main.mode.piano_aria_amt"), ProcessingMode.PIANO_ARIA_AMT.value)
        self.mode_combo.addItem(
            t("main.mode.piano_bytedance_pedal"),
            ProcessingMode.PIANO_BYTEDANCE_PEDAL.value,
        )
        self._configure_combo(self.mode_combo)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self._mode_row = QWidget()
        mode_row = self._selector_layout(self._mode_row)
        mode_row.addRow(self._mode_label, self.mode_combo)
        main_layout.addWidget(self._mode_row)

        self._model_row = QWidget()
        model_row = self._selector_layout(self._model_row)

        self._model_label = QLabel(t("main.engine.active_label") + ":")
        self._model_label.setWordWrap(True)
        self._model_label.setStyleSheet("font-size: 11px; color: #b0b8c8; font-weight: normal;")

        self.model_combo = NoWheelComboBox()
        self._configure_combo(self.model_combo)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._sync_model_options(MultiInstrumentModel.YOURMT3.value)

        model_row.addRow(self._model_label, self.model_combo)
        main_layout.addWidget(self._model_row)

        self._yourmt3_model_row = QWidget()
        yourmt3_model_row = self._selector_layout(self._yourmt3_model_row)

        self._yourmt3_model_label = QLabel(t("main.engine.yourmt3_model_label") + ":")
        self._yourmt3_model_label.setWordWrap(True)
        self._yourmt3_model_label.setStyleSheet("font-size: 11px; color: #b0b8c8; font-weight: normal;")

        self.yourmt3_model_combo = NoWheelComboBox()
        self._configure_combo(self.yourmt3_model_combo)
        self.yourmt3_model_combo.currentIndexChanged.connect(self._on_yourmt3_model_changed)
        self._sync_yourmt3_model_options(YourMT3Model.YPTF_MOE_MULTI_NOPS.value)

        yourmt3_model_row.addRow(self._yourmt3_model_label, self.yourmt3_model_combo)
        main_layout.addWidget(self._yourmt3_model_row)

        self.yourmt3_model_card = QFrame()
        self.yourmt3_model_card.setObjectName("yourmt3ModelCard")
        self.yourmt3_model_card.setStyleSheet("""
            QFrame#yourmt3ModelCard {
                border: 1px solid #2c4f7c;
                border-radius: 6px;
                background: #17243d;
                margin: 1px 0 2px 0;
            }
        """)
        yourmt3_model_card_layout = QVBoxLayout(self.yourmt3_model_card)
        yourmt3_model_card_layout.setContentsMargins(10, 7, 10, 8)
        yourmt3_model_card_layout.setSpacing(3)

        self.yourmt3_model_title_label = QLabel()
        self.yourmt3_model_title_label.setWordWrap(True)
        self.yourmt3_model_title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.yourmt3_model_title_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #4a9eff;
                font-weight: bold;
                line-height: 135%;
            }
        """)
        yourmt3_model_card_layout.addWidget(self.yourmt3_model_title_label)

        self.yourmt3_model_hint_label = QLabel()
        self.yourmt3_model_hint_label.setWordWrap(True)
        self.yourmt3_model_hint_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.yourmt3_model_hint_label.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #9fb3d9;
                line-height: 135%;
            }
        """)
        yourmt3_model_card_layout.addWidget(self.yourmt3_model_hint_label)
        main_layout.addWidget(self.yourmt3_model_card)

        self.model_hint_label = QLabel()
        self.model_hint_label.setWordWrap(True)
        self.model_hint_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.model_hint_label.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #d2c07a;
                padding: 2px 6px 4px 6px;
                line-height: 135%;
            }
        """)
        main_layout.addWidget(self.model_hint_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background: #3a4a6a; margin: 4px 0;")
        main_layout.addWidget(sep)

        self.model_info_card = QFrame()
        self.model_info_card.setObjectName("modelInfoCard")
        self.model_info_card.setStyleSheet("""
            QFrame#modelInfoCard {
                border: 1px solid #2f4567;
                border-radius: 6px;
                background: #18243a;
                margin: 2px 0 2px 0;
            }
        """)
        model_info_layout = QVBoxLayout(self.model_info_card)
        model_info_layout.setContentsMargins(10, 7, 10, 8)
        model_info_layout.setSpacing(3)

        self.mode_desc_label = QLabel()
        self.mode_desc_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.mode_desc_label.setWordWrap(True)
        self.mode_desc_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #4a9eff;
                font-weight: bold;
                line-height: 135%;
            }
        """)

        self.hint_label = QLabel()
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #9aa6bc;
                line-height: 135%;
            }
        """)

        model_info_layout.addWidget(self.mode_desc_label)
        model_info_layout.addWidget(self.hint_label)
        main_layout.addWidget(self.model_info_card)

        self._vocal_split_options = QWidget()
        vocal_layout = QVBoxLayout(self._vocal_split_options)
        vocal_layout.setContentsMargins(6, 4, 6, 2)
        vocal_layout.setSpacing(4)
        self._vocal_split_merge_check = WrappedCheckBox(t("main.mode.vocal_split_merge_midi"))
        self._vocal_split_merge_check.setChecked(False)
        self._vocal_split_merge_check.setStyleSheet("font-size: 10px; color: #b0b8c8; spacing: 4px;")
        vocal_layout.addWidget(self._vocal_split_merge_check)
        main_layout.addWidget(self._vocal_split_options)

        self.set_processing_mode(ProcessingMode.SMART.value)
        self.set_transcription_backend(MultiInstrumentModel.YOURMT3.value)
        self.set_multi_instrument_model(MultiInstrumentModel.YOURMT3.value)
        self._refresh_labels()
        self._update_mode_option_widgets()

    def _on_mode_changed(self, _index: int):
        self._sync_model_options(self.get_transcription_backend())
        self._refresh_labels()
        self._update_mode_option_widgets()
        self.mode_changed.emit(self.get_processing_mode())

    def _on_model_changed(self, _index: int):
        backend = self.get_transcription_backend()
        self._refresh_labels()
        self._update_mode_option_widgets()
        self.model_changed.emit(backend)
        self.backend_changed.emit(backend)

    def _on_yourmt3_model_changed(self, _index: int):
        self._refresh_labels()

    def _mode_tooltip(self) -> str:
        mode = self.get_processing_mode()
        if mode == ProcessingMode.VOCAL_SPLIT.value:
            return t("main.mode.vocal_split_tooltip")
        if mode == ProcessingMode.SIX_STEM_SPLIT.value:
            return t("main.mode.six_stem_split_tooltip")
        if mode == ProcessingMode.PIANO_TRANSKUN.value:
            return t("main.mode.piano_transkun_tooltip")
        if mode == ProcessingMode.PIANO_TRANSKUN_V2_AUG.value:
            return t("main.mode.piano_transkun_v2_aug_tooltip")
        if mode == ProcessingMode.PIANO_ARIA_AMT.value:
            return t("main.mode.piano_aria_amt_tooltip")
        if mode == ProcessingMode.PIANO_BYTEDANCE_PEDAL.value:
            return t("main.mode.piano_bytedance_pedal_tooltip")
        return t("main.mode.smart_tooltip")

    def _model_tooltip(self) -> str:
        mode = self.get_processing_mode()
        backend = self.get_transcription_backend()
        if backend == MultiInstrumentModel.MIROS.value:
            return t("main.engine.miros_tooltip")
        if mode in {ProcessingMode.VOCAL_SPLIT.value, ProcessingMode.SIX_STEM_SPLIT.value}:
            return t("main.engine.yourmt3_midi_tooltip")
        return t("main.engine.yourmt3_tooltip")

    def _model_label_text(self) -> str:
        mode = self.get_processing_mode()
        if mode == ProcessingMode.VOCAL_SPLIT.value:
            return t("main.engine.midi_backend_label")
        if mode == ProcessingMode.SIX_STEM_SPLIT.value:
            return t("main.engine.stem_midi_backend_label")
        return t("main.engine.active_label")

    def _mode_text(self) -> str:
        mode = self.get_processing_mode()
        if mode == ProcessingMode.VOCAL_SPLIT.value:
            return t("main.mode.vocal_split_desc")
        if mode == ProcessingMode.SIX_STEM_SPLIT.value:
            return t("main.mode.six_stem_split_desc")
        if mode == ProcessingMode.PIANO_TRANSKUN.value:
            return t("main.mode.piano_transkun_desc")
        if mode == ProcessingMode.PIANO_TRANSKUN_V2_AUG.value:
            return t("main.mode.piano_transkun_v2_aug_desc")
        if mode == ProcessingMode.PIANO_ARIA_AMT.value:
            return t("main.mode.piano_aria_amt_desc")
        if mode == ProcessingMode.PIANO_BYTEDANCE_PEDAL.value:
            return t("main.mode.piano_bytedance_pedal_desc")
        if self.get_multi_instrument_model() == MultiInstrumentModel.MIROS.value:
            return t("main.mode.smart_miros_desc")
        if self.get_multi_instrument_model() == MultiInstrumentModel.YOURMT3.value:
            return t("main.mode.smart_yourmt3_desc")
        return t("main.mode.smart_desc")

    def _hint_text(self) -> str:
        mode = self.get_processing_mode()
        if mode == ProcessingMode.VOCAL_SPLIT.value:
            return t("main.mode.vocal_split_hint")
        if mode == ProcessingMode.SIX_STEM_SPLIT.value:
            return t("main.mode.six_stem_split_hint")
        if mode == ProcessingMode.PIANO_TRANSKUN.value:
            return t("main.mode.piano_transkun_hint")
        if mode == ProcessingMode.PIANO_TRANSKUN_V2_AUG.value:
            return t("main.mode.piano_transkun_v2_aug_hint")
        if mode == ProcessingMode.PIANO_ARIA_AMT.value:
            return t("main.mode.piano_aria_amt_hint")
        if mode == ProcessingMode.PIANO_BYTEDANCE_PEDAL.value:
            return t("main.mode.piano_bytedance_pedal_hint")
        if self.get_multi_instrument_model() == MultiInstrumentModel.MIROS.value:
            return t("main.mode.smart_miros_hint")
        if self.get_multi_instrument_model() == MultiInstrumentModel.YOURMT3.value:
            return t("main.mode.smart_yourmt3_hint")
        return t("main.mode.smart_hint")

    def _model_hint_text(self) -> str:
        mode = self.get_processing_mode()
        if mode in {
            ProcessingMode.VOCAL_SPLIT.value,
            ProcessingMode.SIX_STEM_SPLIT.value,
        }:
            return t("main.engine.manual_split_midi_hint")
        multi_model = self.get_multi_instrument_model()

        if mode in {
            ProcessingMode.PIANO_TRANSKUN.value,
            ProcessingMode.PIANO_TRANSKUN_V2_AUG.value,
            ProcessingMode.PIANO_ARIA_AMT.value,
            ProcessingMode.PIANO_BYTEDANCE_PEDAL.value,
        }:
            return t("main.engine.dedicated_mode_hint")
        if multi_model == MultiInstrumentModel.MIROS.value:
            return t("main.engine.miros_general_hint")
        return t("main.engine.yourmt3_general_hint")

    def _yourmt3_model_hint_text(self) -> str:
        model_info = YOURMT3_MODELS.get(self.get_yourmt3_model(), {})
        is_zh = get_translator().get_language().startswith("zh")
        if is_zh:
            description = model_info.get("description") or model_info.get("ui_description") or ""
            feature_items = model_info.get("features_zh") or model_info.get("features") or []
        else:
            description = model_info.get("ui_description") or model_info.get("description") or ""
            feature_items = model_info.get("features_en") or model_info.get("features") or []
        features = "，".join(feature_items) if is_zh else ", ".join(feature_items)
        checkpoint = model_info.get("checkpoint", "")
        if checkpoint:
            separator = "：" if is_zh else ": "
            checkpoint_line = f"{t('main.engine.checkpoint_label')}{separator}{checkpoint}"
        else:
            checkpoint_line = ""
        if features:
            separator = "：" if is_zh else ": "
            feature_line = f"{t('main.engine.model_traits_label')}{separator}{features}"
            return "\n".join(part for part in [description, checkpoint_line, feature_line] if part)
        return "\n".join(part for part in [description, checkpoint_line] if part)

    def _yourmt3_model_title_text(self) -> str:
        model_info = YOURMT3_MODELS.get(self.get_yourmt3_model(), {})
        model_label = model_info.get("ui_label") or model_info.get("name") or self.get_yourmt3_model()
        return f"♪ YourMT3+ — {model_label}"

    def _model_options(self, mode: str | None = None) -> list[tuple[str, str]]:
        mode = mode or self.get_processing_mode()
        if mode == ProcessingMode.SIX_STEM_SPLIT.value:
            return [
                (t("main.engine.yourmt3_midi"), MultiInstrumentModel.YOURMT3.value),
                (t("main.engine.miros_midi"), MultiInstrumentModel.MIROS.value),
            ]
        if mode == ProcessingMode.VOCAL_SPLIT.value:
            return [
                (t("main.engine.yourmt3_midi"), MultiInstrumentModel.YOURMT3.value),
                (t("main.engine.miros_midi"), MultiInstrumentModel.MIROS.value),
            ]
        return [
            (t("main.engine.yourmt3"), MultiInstrumentModel.YOURMT3.value),
            (t("main.engine.miros"), MultiInstrumentModel.MIROS.value),
        ]

    def _sync_model_options(self, preferred_backend: str | None = None) -> None:
        options = self._model_options()
        values = [value for _label, value in options]
        preferred = str(preferred_backend or "").strip().lower()
        if preferred not in values:
            raise ValueError(
                f"Unsupported transcription backend {preferred_backend!r} "
                f"for mode {self.get_processing_mode()!r}"
            )

        previous_blocked = self.model_combo.blockSignals(True)
        try:
            self.model_combo.clear()
            for label, value in options:
                self.model_combo.addItem(label, value)
            index = self.model_combo.findData(preferred)
            if index < 0:
                raise RuntimeError(f"Backend option was not populated: {preferred!r}")
            self.model_combo.setCurrentIndex(index)
        finally:
            self.model_combo.blockSignals(previous_blocked)

    def _sync_yourmt3_model_options(self, preferred_model: str | None = None) -> None:
        valid_values = {model.value for model in YourMT3Model}
        preferred = str(preferred_model or "").strip().lower()
        if preferred not in valid_values:
            raise ValueError(f"Unsupported YourMT3 checkpoint: {preferred_model!r}")

        previous_blocked = self.yourmt3_model_combo.blockSignals(True)
        try:
            self.yourmt3_model_combo.clear()
            for model in (
                YourMT3Model.YMT3_PLUS,
                YourMT3Model.YPTF_SINGLE_NOPS,
                YourMT3Model.YPTF_MULTI_PS,
                YourMT3Model.YPTF_MOE_MULTI_NOPS,
                YourMT3Model.YPTF_MOE_MULTI_PS,
            ):
                info = YOURMT3_MODELS.get(model.value, {})
                self.yourmt3_model_combo.addItem(info.get("ui_label", model.value), model.value)
            index = self.yourmt3_model_combo.findData(preferred)
            if index < 0:
                raise RuntimeError(f"YourMT3 checkpoint option was not populated: {preferred!r}")
            self.yourmt3_model_combo.setCurrentIndex(index)
        finally:
            self.yourmt3_model_combo.blockSignals(previous_blocked)

    def _refresh_labels(self):
        self._model_label.setText(self._model_label_text() + ":")
        self.mode_combo.setToolTip(self._mode_tooltip())
        self.model_combo.setToolTip(self._model_tooltip())
        self.yourmt3_model_combo.setToolTip(t("main.engine.yourmt3_model_tooltip"))
        self.yourmt3_model_title_label.setText(self._yourmt3_model_title_text())
        self.yourmt3_model_hint_label.setText(self._yourmt3_model_hint_text())
        self.mode_desc_label.setText(self._mode_text())
        self.hint_label.setText(self._hint_text())
        self.model_hint_label.setText(self._model_hint_text())

    def get_processing_mode(self) -> str:
        mode = self.mode_combo.currentData()
        if mode is None:
            raise RuntimeError("No processing mode is selected")
        return str(mode)

    def set_processing_mode(self, mode: str):
        index = self.mode_combo.findData(mode)
        if index < 0:
            raise ValueError(f"Unsupported processing mode: {mode!r}")
        self.mode_combo.setCurrentIndex(index)

    def get_transcription_backend(self) -> str:
        backend = self.model_combo.currentData()
        if backend is None:
            raise RuntimeError("No transcription backend is selected")
        return str(backend)

    def set_transcription_backend(self, backend: str):
        self._sync_model_options(backend)
        self._update_mode_option_widgets()

    def get_multi_instrument_model(self) -> str:
        preferred = self.get_transcription_backend()
        if preferred in {MultiInstrumentModel.YOURMT3.value, MultiInstrumentModel.MIROS.value}:
            return preferred
        raise ValueError(f"Unsupported multi-instrument backend: {preferred!r}")

    def set_multi_instrument_model(self, model_name: str):
        if model_name not in {MultiInstrumentModel.YOURMT3.value, MultiInstrumentModel.MIROS.value}:
            raise ValueError(f"Unsupported multi-instrument backend: {model_name!r}")
        self._selected_multi_instrument_model = model_name

    def get_yourmt3_model(self) -> str:
        model_name = self.yourmt3_model_combo.currentData()
        if model_name is None:
            raise RuntimeError("No YourMT3 checkpoint is selected")
        return str(model_name)

    def set_yourmt3_model(self, model_name: str):
        self._sync_yourmt3_model_options(model_name)
        self._refresh_labels()

    def get_midi_track_mode(self) -> str:
        return MidiTrackMode.MULTI_TRACK.value

    def set_midi_track_mode(self, mode: str):
        # Kept as a no-op for older saved Config objects and tests that still call it.
        return None

    def get_vocal_split_merge_midi(self) -> bool:
        return (
            self.get_processing_mode() == ProcessingMode.VOCAL_SPLIT.value
            and self._vocal_split_merge_check.isChecked()
        )

    def _update_mode_option_widgets(self):
        mode = self.get_processing_mode()
        uses_model_selector = mode == ProcessingMode.SMART.value
        shows_yourmt3_model = (
            uses_model_selector
            and self.get_multi_instrument_model() == MultiInstrumentModel.YOURMT3.value
        )

        self._model_row.setVisible(uses_model_selector)
        self._yourmt3_model_row.setVisible(shows_yourmt3_model)
        self.yourmt3_model_card.setVisible(shows_yourmt3_model)
        self.yourmt3_model_title_label.setVisible(shows_yourmt3_model)
        self.yourmt3_model_hint_label.setVisible(shows_yourmt3_model)
        self._vocal_split_options.setVisible(False)

        self.mode_combo.setEnabled(self._controls_enabled)
        self.model_combo.setEnabled(self._controls_enabled and uses_model_selector)
        self.yourmt3_model_combo.setEnabled(self._controls_enabled and shows_yourmt3_model)
        self._vocal_split_merge_check.setEnabled(False)
        self._mode_label.setEnabled(self._controls_enabled)
        self._model_label.setEnabled(self._controls_enabled and uses_model_selector)
        self._yourmt3_model_label.setEnabled(self._controls_enabled and shows_yourmt3_model)
        self.yourmt3_model_card.setEnabled(self._controls_enabled and shows_yourmt3_model)
        self.yourmt3_model_title_label.setEnabled(self._controls_enabled and shows_yourmt3_model)
        self.yourmt3_model_hint_label.setEnabled(self._controls_enabled and shows_yourmt3_model)
        self._refresh_labels()

    def set_processing_controls_enabled(self, enabled: bool):
        self._controls_enabled = enabled
        self._update_mode_option_widgets()

    def get_track_layout(self) -> TrackLayout:
        return self._current_layout

    def set_track_layout(self, layout: TrackLayout):
        self._current_layout = layout

    def get_selected_tracks(self) -> dict:
        return {}

    def update_translations(self):
        self.setTitle(t("main.tracks.title"))
        self._mode_label.setText(t("main.mode.label") + ":")
        self._model_label.setText(self._model_label_text() + ":")
        self._yourmt3_model_label.setText(t("main.engine.yourmt3_model_label") + ":")
        mode_labels = {
            ProcessingMode.SMART.value: "main.mode.smart",
            ProcessingMode.VOCAL_SPLIT.value: "main.mode.vocal_split",
            ProcessingMode.SIX_STEM_SPLIT.value: "main.mode.six_stem_split",
            ProcessingMode.PIANO_TRANSKUN.value: "main.mode.piano_transkun",
            ProcessingMode.PIANO_TRANSKUN_V2_AUG.value: "main.mode.piano_transkun_v2_aug",
            ProcessingMode.PIANO_ARIA_AMT.value: "main.mode.piano_aria_amt",
            ProcessingMode.PIANO_BYTEDANCE_PEDAL.value: "main.mode.piano_bytedance_pedal",
        }
        for mode, translation_key in mode_labels.items():
            index = self.mode_combo.findData(mode)
            if index < 0:
                raise RuntimeError(f"Processing mode option was not populated: {mode!r}")
            self.mode_combo.setItemText(index, t(translation_key))
        self._sync_model_options(self.get_transcription_backend())
        self._sync_yourmt3_model_options(self.get_yourmt3_model())
        self._vocal_split_merge_check.setText(t("main.mode.vocal_split_merge_midi"))
        self._refresh_labels()
