"""
轨道面板组件 - YourMT3+ MoE 多乐器转写模式
（已移除钢琴模式选择，仅使用 YourMT3+ 进行多乐器转写）
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

from src.i18n.translator import t
from src.models.data_models import TrackLayout, ProcessingMode


class TrackPanel(QGroupBox):
    """轨道面板：显示 YourMT3+ 转写模式说明"""

    layout_changed = pyqtSignal(object)  # TrackLayout

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_layout = TrackLayout(mode=ProcessingMode.SMART, tracks=[])
        self._setup_ui()

    def _setup_ui(self):
        self.setTitle(t("main.tracks.title"))
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                color: #e0e0e0;
                border: 1px solid #3a4a6a;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
                background: #1f2940;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
                background: #1f2940;
            }
            QLabel {
                color: #e0e0e0;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(10)

        # 模式说明标签
        self.mode_label = QLabel(self._mode_text())
        self.mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_label.setWordWrap(True)
        self.mode_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: #4a9eff;
                font-weight: bold;
                padding: 4px 0;
            }
        """)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background: #3a4a6a; margin: 4px 0;")

        # 说明文字
        self.hint_label = QLabel(self._hint_text())
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #8892a0;
                padding: 2px 0;
            }
        """)

        main_layout.addWidget(self.mode_label)
        main_layout.addWidget(sep)
        main_layout.addWidget(self.hint_label)

    def _mode_text(self) -> str:
        return "🎵  YourMT3+ MoE  —  多乐器智能转写"

    def _hint_text(self) -> str:
        return (
            "直接对完整音频进行转写，精确识别 128 种 GM 乐器。\n"
            "轨道数量由模型自动决定，无需手动配置。"
        )

    def get_track_layout(self) -> TrackLayout:
        return self._current_layout

    def set_track_layout(self, layout: TrackLayout):
        self._current_layout = layout

    def get_selected_tracks(self) -> dict:
        """向后兼容"""
        return {}

    def update_translations(self):
        self.setTitle(t("main.tracks.title"))
        self.mode_label.setText(self._mode_text())
        self.hint_label.setText(self._hint_text())
