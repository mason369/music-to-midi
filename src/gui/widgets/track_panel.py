"""
轨道面板组件 - 支持模式选择（智能 / 人声分离 / 六声部分离 / 钢琴专用）
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QFrame, QComboBox,
    QCheckBox, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal

from src.i18n.translator import t
from src.models.data_models import TrackLayout, ProcessingMode


class TrackPanel(QGroupBox):
    """轨道面板：模式选择 + 模式说明"""

    layout_changed = pyqtSignal(object)  # TrackLayout
    mode_changed = pyqtSignal(str)       # processing_mode 字符串

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_layout = TrackLayout(mode=ProcessingMode.SMART, tracks=[])
        self._six_stem_order = ("bass", "drums", "guitar", "piano", "vocals", "other")
        self._six_stem_checks = {}
        self._setup_ui()

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

        # 模式选择器
        mode_row = QHBoxLayout()
        mode_row.setSpacing(10)

        mode_label = QLabel(t("main.mode.label") + ":")
        mode_label.setStyleSheet("font-size: 11px; color: #b0b8c8; font-weight: normal;")
        self._mode_label = mode_label

        self.mode_combo = QComboBox()
        self.mode_combo.addItem(t("main.mode.smart"), "smart")
        self.mode_combo.addItem(t("main.mode.vocal_split"), "vocal_split")
        self.mode_combo.addItem(t("main.mode.six_stem_split"), "six_stem_split")
        self.mode_combo.addItem(t("main.mode.piano_aria_amt"), "piano_aria_amt")
        self.mode_combo.setToolTip(t("main.mode.smart_tooltip"))
        self.mode_combo.setStyleSheet("""
            QComboBox {
                padding: 4px 10px;
                border: 1px solid #3a4a6a;
                border-radius: 5px;
                background: #16213e;
                color: #e0e0e0;
                font-size: 11px;
                min-width: 200px;
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
        """)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()

        main_layout.addLayout(mode_row)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background: #3a4a6a; margin: 4px 0;")
        main_layout.addWidget(sep)

        # 模式说明标签
        self.mode_desc_label = QLabel(self._mode_text())
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

        # 说明文字
        self.hint_label = QLabel(self._hint_text())
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

        # vocal-split 输出合并选项
        self._vocal_split_options = QWidget()
        vocal_layout = QVBoxLayout(self._vocal_split_options)
        vocal_layout.setContentsMargins(6, 4, 6, 2)
        vocal_layout.setSpacing(4)
        self._vocal_split_merge_check = QCheckBox(t("main.mode.vocal_split_merge_midi"))
        self._vocal_split_merge_check.setChecked(False)
        self._vocal_split_merge_check.setStyleSheet(
            "font-size: 10px; color: #b0b8c8; spacing: 4px;"
        )
        vocal_layout.addWidget(self._vocal_split_merge_check)
        main_layout.addWidget(self._vocal_split_options)

        # six-stem 指定转写开关 + stem 选择
        self._six_stem_options = QWidget()
        six_layout = QVBoxLayout(self._six_stem_options)
        six_layout.setContentsMargins(6, 4, 6, 2)
        six_layout.setSpacing(4)

        self._six_stem_only_selected_check = QCheckBox(t("main.mode.six_stem_only_selected"))
        self._six_stem_only_selected_check.setChecked(False)
        self._six_stem_only_selected_check.setStyleSheet(
            "font-size: 10px; color: #b0b8c8; spacing: 4px;"
        )
        self._six_stem_only_selected_check.toggled.connect(self._update_mode_option_widgets)
        six_layout.addWidget(self._six_stem_only_selected_check)

        self._six_stem_vocal_harmony_check = QCheckBox(t("main.mode.six_stem_vocal_harmony"))
        self._six_stem_vocal_harmony_check.setChecked(False)
        self._six_stem_vocal_harmony_check.setStyleSheet(
            "font-size: 10px; color: #b0b8c8; spacing: 4px;"
        )
        six_layout.addWidget(self._six_stem_vocal_harmony_check)

        stem_grid = QGridLayout()
        stem_grid.setContentsMargins(0, 0, 0, 0)
        stem_grid.setSpacing(8)
        for index, stem in enumerate(self._six_stem_order):
            checkbox = QCheckBox(self._stem_label(stem))
            checkbox.setChecked(True)
            checkbox.setStyleSheet("font-size: 10px; color: #9aa6bc; spacing: 4px;")
            checkbox.toggled.connect(self._ensure_at_least_one_stem_checked)
            row = index // 3
            col = index % 3
            stem_grid.addWidget(checkbox, row, col)
            self._six_stem_checks[stem] = checkbox
        six_layout.addLayout(stem_grid)

        main_layout.addWidget(self._six_stem_options)
        self._update_mode_option_widgets()

    def _on_mode_changed(self, index: int):
        mode = self.mode_combo.itemData(index)
        if mode == "vocal_split":
            self.mode_combo.setToolTip(t("main.mode.vocal_split_tooltip"))
        elif mode == "six_stem_split":
            self.mode_combo.setToolTip(t("main.mode.six_stem_split_tooltip"))
        elif mode == "piano_aria_amt":
            self.mode_combo.setToolTip(t("main.mode.piano_aria_amt_tooltip"))
        else:
            self.mode_combo.setToolTip(t("main.mode.smart_tooltip"))
        self.mode_desc_label.setText(self._mode_text())
        self.hint_label.setText(self._hint_text())
        self._update_mode_option_widgets()
        self.mode_changed.emit(mode)

    def _mode_text(self) -> str:
        mode = self.get_processing_mode()
        if mode == "vocal_split":
            return t("main.mode.vocal_split_desc")
        if mode == "six_stem_split":
            return t("main.mode.six_stem_split_desc")
        if mode == "piano_aria_amt":
            return t("main.mode.piano_aria_amt_desc")
        return t("main.mode.smart_desc")

    def _hint_text(self) -> str:
        mode = self.get_processing_mode()
        if mode == "vocal_split":
            return t("main.mode.vocal_split_hint")
        if mode == "six_stem_split":
            return t("main.mode.six_stem_split_hint")
        if mode == "piano_aria_amt":
            return t("main.mode.piano_aria_amt_hint")
        return t("main.mode.smart_hint")

    def get_processing_mode(self) -> str:
        """获取当前选择的处理模式字符串"""
        return self.mode_combo.currentData() or "smart"

    def get_selected_six_stem_targets(self) -> list[str]:
        """返回 six_stem_split 模式下需要转写的 stem；空列表表示转写全部。"""
        if self.get_processing_mode() != "six_stem_split":
            return []
        if not self._six_stem_only_selected_check.isChecked():
            return []
        return [
            stem
            for stem in self._six_stem_order
            if self._six_stem_checks[stem].isChecked()
        ]

    def get_vocal_split_merge_midi(self) -> bool:
        """返回 vocal_split 模式下是否输出人声+伴奏合并 MIDI。"""
        return (
            self.get_processing_mode() == "vocal_split"
            and self._vocal_split_merge_check.isChecked()
        )

    def get_six_stem_vocal_harmony(self) -> bool:
        """返回 six_stem_split 模式下是否额外分离主唱/和声。"""
        return (
            self.get_processing_mode() == "six_stem_split"
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
        is_vocal_mode = self.get_processing_mode() == "vocal_split"
        self._vocal_split_options.setVisible(is_vocal_mode)

        is_six_stem_mode = self.get_processing_mode() == "six_stem_split"
        self._six_stem_options.setVisible(is_six_stem_mode)
        enabled = is_six_stem_mode and self._six_stem_only_selected_check.isChecked()
        for check in self._six_stem_checks.values():
            check.setEnabled(enabled)

    def get_track_layout(self) -> TrackLayout:
        return self._current_layout

    def set_track_layout(self, layout: TrackLayout):
        self._current_layout = layout

    def get_selected_tracks(self) -> dict:
        """向后兼容"""
        return {}

    def update_translations(self):
        self.setTitle(t("main.tracks.title"))
        self._mode_label.setText(t("main.mode.label") + ":")
        # 更新 combo 项文本
        self.mode_combo.setItemText(0, t("main.mode.smart"))
        self.mode_combo.setItemText(1, t("main.mode.vocal_split"))
        self.mode_combo.setItemText(2, t("main.mode.six_stem_split"))
        self.mode_combo.setItemText(3, t("main.mode.piano_aria_amt"))
        self._vocal_split_merge_check.setText(t("main.mode.vocal_split_merge_midi"))
        self._six_stem_only_selected_check.setText(t("main.mode.six_stem_only_selected"))
        self._six_stem_vocal_harmony_check.setText(t("main.mode.six_stem_vocal_harmony"))
        for stem, checkbox in self._six_stem_checks.items():
            checkbox.setText(self._stem_label(stem))
        self.mode_desc_label.setText(self._mode_text())
        self.hint_label.setText(self._hint_text())
