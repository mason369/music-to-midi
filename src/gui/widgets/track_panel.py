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

from src.i18n.translator import get_translator, t
from src.models.data_models import (
    MidiTrackMode,
    MultiInstrumentModel,
    ProcessingMode,
    TranscriptionBackend,
    TrackLayout,
    YourMT3Model,
)
from src.utils.yourmt3_downloader import YOURMT3_MODELS


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
        self.mode_combo.addItem(t("main.mode.six_stem_split"), ProcessingMode.SIX_STEM_SPLIT.value)
        self.mode_combo.addItem(t("main.mode.piano_transkun"), ProcessingMode.PIANO_TRANSKUN.value)
        self.mode_combo.addItem(t("main.mode.piano_aria_amt"), ProcessingMode.PIANO_ARIA_AMT.value)
        self.mode_combo.addItem(
            t("main.mode.piano_bytedance_pedal"),
            ProcessingMode.PIANO_BYTEDANCE_PEDAL.value,
        )
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

        self._model_label = QLabel(t("main.engine.active_label") + ":")
        self._model_label.setStyleSheet("font-size: 11px; color: #b0b8c8; font-weight: normal;")

        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet(self._combo_style(240))
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._sync_model_options(MultiInstrumentModel.YOURMT3.value)

        model_row.addWidget(self._model_label)
        model_row.addWidget(self.model_combo)
        model_row.addStretch()
        main_layout.addWidget(self._model_row)

        self._yourmt3_model_row = QWidget()
        yourmt3_model_row = QHBoxLayout(self._yourmt3_model_row)
        yourmt3_model_row.setContentsMargins(0, 0, 0, 0)
        yourmt3_model_row.setSpacing(10)

        self._yourmt3_model_label = QLabel(t("main.engine.yourmt3_model_label") + ":")
        self._yourmt3_model_label.setStyleSheet("font-size: 11px; color: #b0b8c8; font-weight: normal;")

        self.yourmt3_model_combo = QComboBox()
        self.yourmt3_model_combo.setStyleSheet(self._combo_style(260))
        self.yourmt3_model_combo.currentIndexChanged.connect(self._on_yourmt3_model_changed)
        self._sync_yourmt3_model_options(YourMT3Model.YPTF_MOE_MULTI_NOPS.value)

        yourmt3_model_row.addWidget(self._yourmt3_model_label)
        yourmt3_model_row.addWidget(self.yourmt3_model_combo)
        yourmt3_model_row.addStretch()
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
        self._vocal_split_merge_check = QCheckBox(t("main.mode.vocal_split_merge_midi"))
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
        if mode == ProcessingMode.PIANO_ARIA_AMT.value:
            return t("main.mode.piano_aria_amt_tooltip")
        if mode == ProcessingMode.PIANO_BYTEDANCE_PEDAL.value:
            return t("main.mode.piano_bytedance_pedal_tooltip")
        return t("main.mode.smart_tooltip")

    def _model_tooltip(self) -> str:
        mode = self.get_processing_mode()
        backend = self.get_transcription_backend()
        if backend == TranscriptionBackend.ARIA_AMT.value:
            return t("main.engine.aria_amt_tooltip")
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
        backend = self.get_transcription_backend()
        multi_model = self.get_multi_instrument_model()

        if mode in {
            ProcessingMode.PIANO_TRANSKUN.value,
            ProcessingMode.PIANO_ARIA_AMT.value,
            ProcessingMode.PIANO_BYTEDANCE_PEDAL.value,
        }:
            return t("main.engine.dedicated_mode_hint")
        if mode == ProcessingMode.VOCAL_SPLIT.value:
            if multi_model == MultiInstrumentModel.MIROS.value:
                return t("main.engine.vocal_split_miros_midi_hint")
            return t("main.engine.vocal_split_yourmt3_midi_hint")
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
                (t("main.engine.aria_amt_piano_stem"), TranscriptionBackend.ARIA_AMT.value),
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
            preferred = MultiInstrumentModel.YOURMT3.value

        previous_blocked = self.model_combo.blockSignals(True)
        try:
            self.model_combo.clear()
            for label, value in options:
                self.model_combo.addItem(label, value)
            index = self.model_combo.findData(preferred)
            if index < 0:
                index = 0
            self.model_combo.setCurrentIndex(index)
        finally:
            self.model_combo.blockSignals(previous_blocked)

    def _sync_yourmt3_model_options(self, preferred_model: str | None = None) -> None:
        valid_values = {model.value for model in YourMT3Model}
        preferred = str(preferred_model or "").strip().lower()
        if preferred not in valid_values:
            preferred = YourMT3Model.YPTF_MOE_MULTI_NOPS.value

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
                index = self.yourmt3_model_combo.findData(YourMT3Model.YPTF_MOE_MULTI_NOPS.value)
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
        return self.mode_combo.currentData() or ProcessingMode.SMART.value

    def set_processing_mode(self, mode: str):
        index = self.mode_combo.findData(mode)
        if index < 0:
            index = self.mode_combo.findData(ProcessingMode.SMART.value)
        self.mode_combo.setCurrentIndex(index)

    def get_transcription_backend(self) -> str:
        return self.model_combo.currentData() or MultiInstrumentModel.YOURMT3.value

    def set_transcription_backend(self, backend: str):
        self._sync_model_options(backend)
        self._update_mode_option_widgets()

    def get_multi_instrument_model(self) -> str:
        preferred = self.get_transcription_backend()
        if preferred in {MultiInstrumentModel.YOURMT3.value, MultiInstrumentModel.MIROS.value}:
            return preferred
        return MultiInstrumentModel.YOURMT3.value

    def set_multi_instrument_model(self, model_name: str):
        if model_name not in {MultiInstrumentModel.YOURMT3.value, MultiInstrumentModel.MIROS.value}:
            model_name = MultiInstrumentModel.YOURMT3.value
        self._selected_multi_instrument_model = model_name

    def get_yourmt3_model(self) -> str:
        return self.yourmt3_model_combo.currentData() or YourMT3Model.YPTF_MOE_MULTI_NOPS.value

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
        uses_model_selector = mode in {
            ProcessingMode.SMART.value,
            ProcessingMode.VOCAL_SPLIT.value,
            ProcessingMode.SIX_STEM_SPLIT.value,
        }
        is_vocal_mode = mode == ProcessingMode.VOCAL_SPLIT.value
        shows_yourmt3_smart_controls = (
            mode == ProcessingMode.SMART.value
            and self.get_multi_instrument_model() == MultiInstrumentModel.YOURMT3.value
        )
        shows_yourmt3_model = shows_yourmt3_smart_controls

        self._model_row.setVisible(uses_model_selector)
        self._yourmt3_model_row.setVisible(shows_yourmt3_model)
        self.yourmt3_model_card.setVisible(shows_yourmt3_model)
        self.yourmt3_model_title_label.setVisible(shows_yourmt3_model)
        self.yourmt3_model_hint_label.setVisible(shows_yourmt3_model)
        self._vocal_split_options.setVisible(is_vocal_mode)

        self.mode_combo.setEnabled(self._controls_enabled)
        self.model_combo.setEnabled(self._controls_enabled and uses_model_selector)
        self.yourmt3_model_combo.setEnabled(self._controls_enabled and shows_yourmt3_model)
        self._vocal_split_merge_check.setEnabled(self._controls_enabled and is_vocal_mode)
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
        self.mode_combo.setItemText(0, t("main.mode.smart"))
        self.mode_combo.setItemText(1, t("main.mode.vocal_split"))
        self.mode_combo.setItemText(2, t("main.mode.six_stem_split"))
        self.mode_combo.setItemText(3, t("main.mode.piano_transkun"))
        self.mode_combo.setItemText(4, t("main.mode.piano_aria_amt"))
        self.mode_combo.setItemText(5, t("main.mode.piano_bytedance_pedal"))
        self._sync_model_options(self.get_transcription_backend())
        self._sync_yourmt3_model_options(self.get_yourmt3_model())
        self._vocal_split_merge_check.setText(t("main.mode.vocal_split_merge_midi"))
        self._refresh_labels()
