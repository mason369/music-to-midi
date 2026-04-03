"""
轨道面板组件 - 支持模式选择与后端选择。
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QFrame, QComboBox,
    QCheckBox, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal

from src.i18n.translator import t
from src.models.data_models import (
    Config,
    QualityBehavior,
    TrackLayout,
    ProcessingMode,
    MultiInstrumentModel,
    TranscriptionBackend,
)


class TrackPanel(QGroupBox):
    """轨道面板：模式选择 + 后端选择 + 模式说明。"""

    layout_changed = pyqtSignal(object)  # TrackLayout
    mode_changed = pyqtSignal(str)       # processing_mode 字符串
    model_changed = pyqtSignal(str)      # preferred backend 字符串
    backend_changed = pyqtSignal(str)    # backward-compatible alias

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_layout = TrackLayout(mode=ProcessingMode.SMART, tracks=[])
        self._six_stem_order = ("bass", "drums", "guitar", "piano", "vocals", "other")
        self._six_stem_checks = {}
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
        self.mode_combo.addItem(t("main.mode.six_stem_split"), ProcessingMode.SIX_STEM_SPLIT.value)
        self.mode_combo.addItem(t("main.mode.piano_transkun"), ProcessingMode.PIANO_TRANSKUN.value)
        self.mode_combo.addItem(t("main.mode.piano_aria_amt"), ProcessingMode.PIANO_ARIA_AMT.value)
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
        self.model_combo.addItem(t("main.engine.aria_amt"), TranscriptionBackend.ARIA_AMT.value)
        self.model_combo.addItem(t("main.engine.yourmt3"), MultiInstrumentModel.YOURMT3.value)
        self.model_combo.addItem(t("main.engine.miros"), MultiInstrumentModel.MIROS.value)
        self.model_combo.setStyleSheet(self._combo_style(240))
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)

        model_row.addWidget(self._model_label)
        model_row.addWidget(self.model_combo)
        model_row.addStretch()
        main_layout.addWidget(self._model_row)

        self._multi_model_row = QWidget()
        multi_row = QHBoxLayout(self._multi_model_row)
        multi_row.setContentsMargins(0, 0, 0, 0)
        multi_row.setSpacing(10)

        self._multi_model_label = QLabel(t("main.engine.active_label") + ":")
        self._multi_model_label.setStyleSheet("font-size: 11px; color: #b0b8c8; font-weight: normal;")

        self.multi_model_combo = QComboBox()
        self.multi_model_combo.addItem(t("main.engine.yourmt3"), MultiInstrumentModel.YOURMT3.value)
        self.multi_model_combo.addItem(t("main.engine.miros"), MultiInstrumentModel.MIROS.value)
        self.multi_model_combo.setStyleSheet(self._combo_style(240))
        self.multi_model_combo.currentIndexChanged.connect(self._on_multi_model_changed)

        multi_row.addWidget(self._multi_model_label)
        multi_row.addWidget(self.multi_model_combo)
        multi_row.addStretch()
        main_layout.addWidget(self._multi_model_row)

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

        self._six_stem_options = QWidget()
        six_layout = QVBoxLayout(self._six_stem_options)
        six_layout.setContentsMargins(6, 4, 6, 2)
        six_layout.setSpacing(4)

        self._six_stem_only_selected_check = QCheckBox(t("main.mode.six_stem_only_selected"))
        self._six_stem_only_selected_check.setChecked(False)
        self._six_stem_only_selected_check.setStyleSheet("font-size: 10px; color: #b0b8c8; spacing: 4px;")
        self._six_stem_only_selected_check.toggled.connect(self._update_mode_option_widgets)
        six_layout.addWidget(self._six_stem_only_selected_check)

        self._six_stem_vocal_harmony_check = QCheckBox(t("main.mode.six_stem_vocal_harmony"))
        self._six_stem_vocal_harmony_check.setChecked(False)
        self._six_stem_vocal_harmony_check.setStyleSheet("font-size: 10px; color: #b0b8c8; spacing: 4px;")
        six_layout.addWidget(self._six_stem_vocal_harmony_check)

        stem_grid = QGridLayout()
        stem_grid.setContentsMargins(0, 0, 0, 0)
        stem_grid.setSpacing(8)
        for index, stem in enumerate(self._six_stem_order):
            checkbox = QCheckBox(self._stem_label(stem))
            checkbox.setChecked(True)
            checkbox.setStyleSheet("font-size: 10px; color: #9aa6bc; spacing: 4px;")
            checkbox.toggled.connect(self._ensure_at_least_one_stem_checked)
            stem_grid.addWidget(checkbox, index // 3, index % 3)
            self._six_stem_checks[stem] = checkbox
        six_layout.addLayout(stem_grid)
        main_layout.addWidget(self._six_stem_options)

        self.set_processing_mode(ProcessingMode.SMART.value)
        self.set_transcription_backend(TranscriptionBackend.ARIA_AMT.value)
        self.set_multi_instrument_model(MultiInstrumentModel.YOURMT3.value)
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

    def _on_multi_model_changed(self, _index: int):
        self._refresh_labels()

    def _mode_tooltip(self) -> str:
        mode = self.get_processing_mode()
        if mode == ProcessingMode.VOCAL_SPLIT.value:
            return t("main.mode.vocal_split_tooltip")
        if mode == ProcessingMode.SIX_STEM_SPLIT.value:
            return t("main.mode.six_stem_split_tooltip")
        if mode == ProcessingMode.PIANO_TRANSKUN.value:
            return t("main.mode.piano_transkun_tooltip")
        if mode == ProcessingMode.PIANO_ARIA_AMT.value:
            return t("main.mode.piano_aria_amt_tooltip")
        return t("main.mode.smart_tooltip")

    def _model_tooltip(self) -> str:
        backend = self.get_transcription_backend()
        if backend == TranscriptionBackend.ARIA_AMT.value:
            return t("main.engine.aria_amt_tooltip")
        if backend == MultiInstrumentModel.MIROS.value:
            return t("main.engine.miros_tooltip")
        return t("main.engine.yourmt3_tooltip")

    def _multi_model_tooltip(self) -> str:
        model_name = self.get_multi_instrument_model()
        if model_name == MultiInstrumentModel.MIROS.value:
            return t("main.engine.miros_tooltip")
        return t("main.engine.yourmt3_tooltip")

    def _mode_text(self) -> str:
        mode = self.get_processing_mode()
        if mode == ProcessingMode.VOCAL_SPLIT.value:
            return t("main.mode.vocal_split_desc")
        if mode == ProcessingMode.SIX_STEM_SPLIT.value:
            return t("main.mode.six_stem_split_desc")
        if mode == ProcessingMode.PIANO_TRANSKUN.value:
            return t("main.mode.piano_transkun_desc")
        if mode == ProcessingMode.PIANO_ARIA_AMT.value:
            return t("main.mode.piano_aria_amt_desc")
        return t("main.mode.smart_desc")

    def _hint_text(self) -> str:
        mode = self.get_processing_mode()
        if mode == ProcessingMode.VOCAL_SPLIT.value:
            return t("main.mode.vocal_split_hint")
        if mode == ProcessingMode.SIX_STEM_SPLIT.value:
            return t("main.mode.six_stem_split_hint")
        if mode == ProcessingMode.PIANO_TRANSKUN.value:
            return t("main.mode.piano_transkun_hint")
        if mode == ProcessingMode.PIANO_ARIA_AMT.value:
            return t("main.mode.piano_aria_amt_hint")
        return t("main.mode.smart_hint")

    def _model_hint_text(self) -> str:
        mode = self.get_processing_mode()
        backend = self.get_transcription_backend()
        multi_model = self.get_multi_instrument_model()

        if mode in {ProcessingMode.PIANO_TRANSKUN.value, ProcessingMode.PIANO_ARIA_AMT.value}:
            return t("main.engine.dedicated_mode_hint")
        if backend == TranscriptionBackend.ARIA_AMT.value and mode == ProcessingMode.SIX_STEM_SPLIT.value:
            return t("main.engine.aria_amt_six_stem_hint")
        if backend == TranscriptionBackend.ARIA_AMT.value and multi_model == MultiInstrumentModel.MIROS.value:
            return t("main.engine.aria_amt_with_miros_hint")
        if backend == TranscriptionBackend.ARIA_AMT.value:
            return t("main.engine.aria_amt_general_hint")
        if multi_model == MultiInstrumentModel.MIROS.value and mode == ProcessingMode.SIX_STEM_SPLIT.value:
            return t("main.engine.miros_six_stem_hint")
        if multi_model == MultiInstrumentModel.MIROS.value:
            return t("main.engine.miros_general_hint")
        if mode == ProcessingMode.SIX_STEM_SPLIT.value:
            return t("main.engine.yourmt3_six_stem_hint")
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
        if behavior == QualityBehavior.PARTIAL:
            return t("main.engine.quality_partial_hint")
        if behavior == QualityBehavior.FIXED:
            return t("main.engine.quality_fixed_hint")
        return t("main.engine.quality_configurable_hint")

    def _refresh_labels(self):
        self.mode_combo.setToolTip(self._mode_tooltip())
        self.model_combo.setToolTip(self._model_tooltip())
        self.multi_model_combo.setToolTip(self._multi_model_tooltip())
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
        return self.model_combo.currentData() or TranscriptionBackend.ARIA_AMT.value

    def set_transcription_backend(self, backend: str):
        index = self.model_combo.findData(backend)
        if index < 0:
            index = self.model_combo.findData(TranscriptionBackend.ARIA_AMT.value)
        self.model_combo.setCurrentIndex(index)

    def get_multi_instrument_model(self) -> str:
        preferred = self.get_transcription_backend()
        if preferred in {MultiInstrumentModel.YOURMT3.value, MultiInstrumentModel.MIROS.value}:
            return preferred
        return self.multi_model_combo.currentData() or MultiInstrumentModel.YOURMT3.value

    def set_multi_instrument_model(self, model_name: str):
        index = self.multi_model_combo.findData(model_name)
        if index < 0:
            index = self.multi_model_combo.findData(MultiInstrumentModel.YOURMT3.value)
        self.multi_model_combo.setCurrentIndex(index)

    def get_selected_six_stem_targets(self) -> list[str]:
        if self.get_processing_mode() != ProcessingMode.SIX_STEM_SPLIT.value:
            return []
        if not self._six_stem_only_selected_check.isChecked():
            return []
        return [
            stem
            for stem in self._six_stem_order
            if self._six_stem_checks[stem].isChecked()
        ]

    def get_vocal_split_merge_midi(self) -> bool:
        return (
            self.get_processing_mode() == ProcessingMode.VOCAL_SPLIT.value
            and self._vocal_split_merge_check.isChecked()
        )

    def get_six_stem_vocal_harmony(self) -> bool:
        return (
            self.get_processing_mode() == ProcessingMode.SIX_STEM_SPLIT.value
            and self._six_stem_vocal_harmony_check.isChecked()
        )

    def _stem_label(self, stem: str) -> str:
        labels = {
            "bass": t("main.tracks.bass"),
            "drums": t("main.tracks.drums"),
            "guitar": t("main.tracks.guitar"),
            "piano": t("main.tracks.piano"),
            "vocals": t("main.tracks.vocals"),
            "other": t("main.tracks.other"),
        }
        return labels.get(stem, stem)

    def _ensure_at_least_one_stem_checked(self, _checked: bool):
        if not self._six_stem_only_selected_check.isChecked():
            return
        if any(check.isChecked() for check in self._six_stem_checks.values()):
            return
        sender = self.sender()
        if isinstance(sender, QCheckBox):
            sender.setChecked(True)

    def _update_mode_option_widgets(self):
        mode = self.get_processing_mode()
        uses_model_selector = mode in {
            ProcessingMode.SMART.value,
            ProcessingMode.VOCAL_SPLIT.value,
            ProcessingMode.SIX_STEM_SPLIT.value,
        }
        shows_multi_selector = (
            uses_model_selector
            and self.get_transcription_backend() == TranscriptionBackend.ARIA_AMT.value
        )
        is_vocal_mode = mode == ProcessingMode.VOCAL_SPLIT.value
        is_six_stem_mode = mode == ProcessingMode.SIX_STEM_SPLIT.value

        self._model_row.setVisible(uses_model_selector)
        self._multi_model_row.setVisible(shows_multi_selector)
        self._vocal_split_options.setVisible(is_vocal_mode)
        self._six_stem_options.setVisible(is_six_stem_mode)

        selected_stems_enabled = (
            self._controls_enabled
            and is_six_stem_mode
            and self._six_stem_only_selected_check.isChecked()
        )
        for check in self._six_stem_checks.values():
            check.setEnabled(selected_stems_enabled)

        self.mode_combo.setEnabled(self._controls_enabled)
        self.model_combo.setEnabled(self._controls_enabled and uses_model_selector)
        self.multi_model_combo.setEnabled(self._controls_enabled and shows_multi_selector)
        self._vocal_split_merge_check.setEnabled(self._controls_enabled and is_vocal_mode)
        self._six_stem_only_selected_check.setEnabled(self._controls_enabled and is_six_stem_mode)
        self._six_stem_vocal_harmony_check.setEnabled(self._controls_enabled and is_six_stem_mode)
        self._model_label.setEnabled(self._controls_enabled and uses_model_selector)
        self._multi_model_label.setEnabled(self._controls_enabled and shows_multi_selector)
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
        self._multi_model_label.setText(t("main.engine.active_label") + ":")
        self.mode_combo.setItemText(0, t("main.mode.smart"))
        self.mode_combo.setItemText(1, t("main.mode.vocal_split"))
        self.mode_combo.setItemText(2, t("main.mode.six_stem_split"))
        self.mode_combo.setItemText(3, t("main.mode.piano_transkun"))
        self.mode_combo.setItemText(4, t("main.mode.piano_aria_amt"))
        self.model_combo.setItemText(0, t("main.engine.aria_amt"))
        self.model_combo.setItemText(1, t("main.engine.yourmt3"))
        self.model_combo.setItemText(2, t("main.engine.miros"))
        self.multi_model_combo.setItemText(0, t("main.engine.yourmt3"))
        self.multi_model_combo.setItemText(1, t("main.engine.miros"))
        self._vocal_split_merge_check.setText(t("main.mode.vocal_split_merge_midi"))
        self._six_stem_only_selected_check.setText(t("main.mode.six_stem_only_selected"))
        self._six_stem_vocal_harmony_check.setText(t("main.mode.six_stem_vocal_harmony"))
        for stem, checkbox in self._six_stem_checks.items():
            checkbox.setText(self._stem_label(stem))
        self._refresh_labels()
