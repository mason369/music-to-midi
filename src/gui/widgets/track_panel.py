"""
轨道面板组件 - 支持模式选择与多乐器后端选择。
"""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.i18n.translator import t
from src.models.data_models import (
    Config,
    MidiTrackMode,
    MultiInstrumentModel,
    ProcessingMode,
    QualityBehavior,
    TrackLayout,
)


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
    def _combo_style(min_width: int = 200) -> str:
        return f"""
            QComboBox {{
                padding: 4px 10px;
                border: 1px solid #3a4a6a;
                border-radius: 5px;
                background: #16213e;
                color: #e0e0e0;
                font-size: 11px;
                min-width: {min_width}px;
            }}
            QComboBox:hover {{
                border-color: #4a9eff;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox QAbstractItemView {{
                background: #1f2940;
                border: 1px solid #3a4a6a;
                color: #e0e0e0;
                selection-background-color: #3a5a7c;
            }}
        """

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
        self._mode_label.setStyleSheet("font-size: 11px; color: #b0b8c8; font-weight: normal;")

        self.mode_combo = QComboBox()
        self.mode_combo.addItem(t("main.mode.smart"), ProcessingMode.SMART.value)
        self.mode_combo.addItem(t("main.mode.vocal_split"), ProcessingMode.VOCAL_SPLIT.value)
        self.mode_combo.setStyleSheet(self._combo_style())
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(10)
        mode_row.addWidget(self._mode_label)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        main_layout.addLayout(mode_row)

        self._model_row = QWidget()
        model_row = QHBoxLayout(self._model_row)
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(10)

        self._model_label = QLabel(t("main.engine.label") + ":")
        self._model_label.setStyleSheet("font-size: 11px; color: #b0b8c8; font-weight: normal;")

        self.model_combo = QComboBox()
        self.model_combo.addItem(t("main.engine.yourmt3"), MultiInstrumentModel.YOURMT3.value)
        self.model_combo.addItem(t("main.engine.miros"), MultiInstrumentModel.MIROS.value)
        self.model_combo.setStyleSheet(self._combo_style(240))
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)

        model_row.addWidget(self._model_label)
        model_row.addWidget(self.model_combo)
        model_row.addStretch()
        main_layout.addWidget(self._model_row)

        self._midi_track_mode_row = QWidget()
        midi_mode_row = QHBoxLayout(self._midi_track_mode_row)
        midi_mode_row.setContentsMargins(0, 0, 0, 0)
        midi_mode_row.setSpacing(10)

        self._midi_track_mode_label = QLabel(t("main.engine.track_mode_label") + ":")
        self._midi_track_mode_label.setStyleSheet("font-size: 11px; color: #b0b8c8; font-weight: normal;")

        self.midi_track_mode_combo = QComboBox()
        self.midi_track_mode_combo.addItem(
            t("main.engine.track_mode_multi"),
            MidiTrackMode.MULTI_TRACK.value,
        )
        self.midi_track_mode_combo.addItem(
            t("main.engine.track_mode_single"),
            MidiTrackMode.SINGLE_TRACK.value,
        )
        self.midi_track_mode_combo.setStyleSheet(self._combo_style(240))

        midi_mode_row.addWidget(self._midi_track_mode_label)
        midi_mode_row.addWidget(self.midi_track_mode_combo)
        midi_mode_row.addStretch()
        main_layout.addWidget(self._midi_track_mode_row)

        self.yourmt3_arch_hint_label = QLabel()
        self.yourmt3_arch_hint_label.setWordWrap(True)
        self.yourmt3_arch_hint_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.yourmt3_arch_hint_label.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #d6c98a;
                padding: 2px 6px 4px 6px;
                line-height: 135%;
            }
        """)
        main_layout.addWidget(self.yourmt3_arch_hint_label)

        self.model_hint_label = QLabel()
        self.model_hint_label.setWordWrap(True)
        self.model_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_hint_label.setStyleSheet("font-size: 10px; color: #d2c07a; padding: 1px 0 2px 0;")
        main_layout.addWidget(self.model_hint_label)

        self.quality_hint_label = QLabel()
        self.quality_hint_label.setWordWrap(True)
        self.quality_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.quality_hint_label.setStyleSheet("font-size: 10px; color: #9fb3d9; padding: 0 0 2px 0;")
        main_layout.addWidget(self.quality_hint_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background: #3a4a6a; margin: 4px 0;")
        main_layout.addWidget(sep)

        self.mode_desc_label = QLabel()
        self.mode_desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_desc_label.setWordWrap(True)
        self.mode_desc_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #4a9eff;
                font-weight: bold;
                padding: 2px 0;
            }
        """)

        self.hint_label = QLabel()
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #8892a0;
                padding: 1px 0;
            }
        """)

        main_layout.addWidget(self.mode_desc_label)
        main_layout.addWidget(self.hint_label)

        self._vocal_split_options = QWidget()
        vocal_layout = QVBoxLayout(self._vocal_split_options)
        vocal_layout.setContentsMargins(6, 4, 6, 2)
        vocal_layout.setSpacing(4)
        self._vocal_split_merge_check = QCheckBox(t("main.mode.vocal_split_merge_midi"))
        self._vocal_split_merge_check.setChecked(False)
        self._vocal_split_merge_check.setStyleSheet("font-size: 10px; color: #b0b8c8; spacing: 4px;")
        vocal_layout.addWidget(self._vocal_split_merge_check)
        main_layout.addWidget(self._vocal_split_options)

        self.set_processing_mode(ProcessingMode.SMART.value)
        self.set_transcription_backend(MultiInstrumentModel.YOURMT3.value)
        self.set_midi_track_mode(MidiTrackMode.MULTI_TRACK.value)
        self._refresh_labels()
        self._update_mode_option_widgets()

    def _on_mode_changed(self, _index: int):
        self._refresh_labels()
        self._update_mode_option_widgets()
        self.mode_changed.emit(self.get_processing_mode())

    def _on_model_changed(self, _index: int):
        backend = self.get_transcription_backend()
        self._refresh_labels()
        self._update_mode_option_widgets()
        self.model_changed.emit(backend)
        self.backend_changed.emit(backend)

    def _mode_tooltip(self) -> str:
        if self.get_processing_mode() == ProcessingMode.VOCAL_SPLIT.value:
            return t("main.mode.vocal_split_tooltip")
        return t("main.mode.smart_tooltip")

    def _model_tooltip(self) -> str:
        if self.get_transcription_backend() == MultiInstrumentModel.MIROS.value:
            return t("main.engine.miros_tooltip")
        return t("main.engine.yourmt3_tooltip")

    def _mode_text(self) -> str:
        if self.get_processing_mode() == ProcessingMode.VOCAL_SPLIT.value:
            return t("main.mode.vocal_split_desc")
        return t("main.mode.smart_desc")

    def _hint_text(self) -> str:
        if self.get_processing_mode() == ProcessingMode.VOCAL_SPLIT.value:
            return t("main.mode.vocal_split_hint")
        return t("main.mode.smart_hint")

    def _model_hint_text(self) -> str:
        if self.get_multi_instrument_model() == MultiInstrumentModel.MIROS.value:
            return t("main.engine.miros_general_hint")
        return t("main.engine.yourmt3_general_hint")

    def get_quality_behavior(self) -> QualityBehavior:
        config = Config(
            processing_mode=self.get_processing_mode(),
            transcription_backend=self.get_transcription_backend(),
            multi_instrument_model=self.get_multi_instrument_model(),
        )
        return config.get_quality_behavior()

    def _quality_hint_text(self) -> str:
        behavior = self.get_quality_behavior()
        if behavior == QualityBehavior.FIXED:
            return t("main.engine.quality_fixed_hint")
        return t("main.engine.quality_configurable_hint")

    def _refresh_labels(self):
        self.mode_combo.setToolTip(self._mode_tooltip())
        self.model_combo.setToolTip(self._model_tooltip())
        self.midi_track_mode_combo.setToolTip(t("main.engine.track_mode_tooltip"))
        self.yourmt3_arch_hint_label.setText(t("main.engine.yourmt3_arch_hint"))
        self.mode_desc_label.setText(self._mode_text())
        self.hint_label.setText(self._hint_text())
        self.model_hint_label.setText(self._model_hint_text())
        self.quality_hint_label.setText(self._quality_hint_text())

    def get_processing_mode(self) -> str:
        return self.mode_combo.currentData() or ProcessingMode.SMART.value

    def set_processing_mode(self, mode: str):
        index = self.mode_combo.findData(mode)
        if index < 0:
            index = self.mode_combo.findData(ProcessingMode.SMART.value)
        self.mode_combo.setCurrentIndex(index)

    def get_transcription_backend(self) -> str:
        return self.model_combo.currentData() or MultiInstrumentModel.YOURMT3.value

    def set_transcription_backend(self, backend: str):
        index = self.model_combo.findData(backend)
        if index < 0:
            index = self.model_combo.findData(MultiInstrumentModel.YOURMT3.value)
        self.model_combo.setCurrentIndex(index)

    def get_multi_instrument_model(self) -> str:
        return self.get_transcription_backend()

    def set_multi_instrument_model(self, model_name: str):
        self.set_transcription_backend(model_name)

    def get_midi_track_mode(self) -> str:
        if self.get_multi_instrument_model() != MultiInstrumentModel.YOURMT3.value:
            return MidiTrackMode.MULTI_TRACK.value
        return self.midi_track_mode_combo.currentData() or MidiTrackMode.MULTI_TRACK.value

    def set_midi_track_mode(self, mode: str):
        index = self.midi_track_mode_combo.findData(mode)
        if index < 0:
            index = self.midi_track_mode_combo.findData(MidiTrackMode.MULTI_TRACK.value)
        self.midi_track_mode_combo.setCurrentIndex(index)

    def get_vocal_split_merge_midi(self) -> bool:
        return (
            self.get_processing_mode() == ProcessingMode.VOCAL_SPLIT.value
            and self._vocal_split_merge_check.isChecked()
        )

    def _update_mode_option_widgets(self):
        is_vocal_mode = self.get_processing_mode() == ProcessingMode.VOCAL_SPLIT.value
        shows_midi_track_mode = self.get_multi_instrument_model() == MultiInstrumentModel.YOURMT3.value

        self._model_row.setVisible(True)
        self._midi_track_mode_row.setVisible(shows_midi_track_mode)
        self.yourmt3_arch_hint_label.setVisible(shows_midi_track_mode)
        self._vocal_split_options.setVisible(is_vocal_mode)

        self.mode_combo.setEnabled(self._controls_enabled)
        self.model_combo.setEnabled(self._controls_enabled)
        self.midi_track_mode_combo.setEnabled(self._controls_enabled and shows_midi_track_mode)
        self._vocal_split_merge_check.setEnabled(self._controls_enabled and is_vocal_mode)
        self._mode_label.setEnabled(self._controls_enabled)
        self._model_label.setEnabled(self._controls_enabled)
        self._midi_track_mode_label.setEnabled(self._controls_enabled and shows_midi_track_mode)
        self.yourmt3_arch_hint_label.setEnabled(self._controls_enabled and shows_midi_track_mode)
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
        self._model_label.setText(t("main.engine.label") + ":")
        self._midi_track_mode_label.setText(t("main.engine.track_mode_label") + ":")
        self.mode_combo.setItemText(0, t("main.mode.smart"))
        self.mode_combo.setItemText(1, t("main.mode.vocal_split"))
        self.model_combo.setItemText(0, t("main.engine.yourmt3"))
        self.model_combo.setItemText(1, t("main.engine.miros"))
        self.midi_track_mode_combo.setItemText(0, t("main.engine.track_mode_multi"))
        self.midi_track_mode_combo.setItemText(1, t("main.engine.track_mode_single"))
        self._vocal_split_merge_check.setText(t("main.mode.vocal_split_merge_midi"))
        self._refresh_labels()
