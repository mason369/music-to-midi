"""
轨道面板组件 - 支持模式选择（智能模式 / 人声分离模式）
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QFrame, QComboBox
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

    def _on_mode_changed(self, index: int):
        mode = self.mode_combo.itemData(index)
        if mode == "vocal_split":
            self.mode_combo.setToolTip(t("main.mode.vocal_split_tooltip"))
        else:
            self.mode_combo.setToolTip(t("main.mode.smart_tooltip"))
        self.mode_desc_label.setText(self._mode_text())
        self.hint_label.setText(self._hint_text())
        self.mode_changed.emit(mode)

    def _mode_text(self) -> str:
        mode = self.get_processing_mode()
        if mode == "vocal_split":
            return t("main.mode.vocal_split_desc")
        return t("main.mode.smart_desc")

    def _hint_text(self) -> str:
        mode = self.get_processing_mode()
        if mode == "vocal_split":
            return t("main.mode.vocal_split_hint")
        return t("main.mode.smart_hint")

    def get_processing_mode(self) -> str:
        """获取当前选择的处理模式字符串"""
        return self.mode_combo.currentData() or "smart"

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
        self.mode_desc_label.setText(self._mode_text())
        self.hint_label.setText(self._hint_text())
