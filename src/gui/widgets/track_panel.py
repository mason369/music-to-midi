"""
轨道面板组件 - 显示分离的音轨
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QGroupBox
)
from PyQt6.QtCore import Qt

from src.i18n.translator import t
from src.models.data_models import TrackType


class TrackRow(QWidget):
    """单个轨道行，包含图标、名称和选项"""

    ICONS = {
        TrackType.VOCALS: "🎤",
        TrackType.DRUMS: "🥁",
        TrackType.BASS: "🎸",
        TrackType.OTHER: "🎹"
    }

    def __init__(self, track_type: TrackType, parent=None):
        super().__init__(parent)
        self.track_type = track_type
        self._setup_ui()

    def _setup_ui(self):
        """设置用户界面"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        # 图标和名称
        icon = self.ICONS.get(self.track_type, "🎵")
        self.name_label = QLabel(f"{icon} {self._get_track_name()}")
        self.name_label.setMinimumWidth(120)

        # 进度条占位
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("""
            QLabel {
                background: #16213e;
                border-radius: 3px;
                min-height: 20px;
                border: 1px solid #3a4a6a;
            }
        """)

        # 选项
        self.lyrics_check = QCheckBox(t("main.tracks.lyrics"))
        self.midi_check = QCheckBox(t("main.tracks.midi"))
        self.midi_check.setChecked(True)

        # 只为人声显示歌词选项
        if self.track_type != TrackType.VOCALS:
            self.lyrics_check.hide()

        layout.addWidget(self.name_label)
        layout.addWidget(self.progress_label, 1)
        layout.addWidget(self.lyrics_check)
        layout.addWidget(self.midi_check)

    def _get_track_name(self) -> str:
        """获取本地化的轨道名称"""
        names = {
            TrackType.VOCALS: t("main.tracks.vocals"),
            TrackType.DRUMS: t("main.tracks.drums"),
            TrackType.BASS: t("main.tracks.bass"),
            TrackType.OTHER: t("main.tracks.other")
        }
        return names.get(self.track_type, str(self.track_type.value))

    def update_translations(self):
        """更新当前语言的文本"""
        icon = self.ICONS.get(self.track_type, "🎵")
        self.name_label.setText(f"{icon} {self._get_track_name()}")
        self.lyrics_check.setText(t("main.tracks.lyrics"))
        self.midi_check.setText(t("main.tracks.midi"))


class TrackPanel(QGroupBox):
    """显示所有轨道及选项的面板"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.track_rows = {}
        self._setup_ui()

    def _setup_ui(self):
        """设置用户界面"""
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
            QCheckBox {
                color: #b0b8c8;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid #3a4a6a;
                background: #16213e;
            }
            QCheckBox::indicator:checked {
                background: #4a9eff;
                border-color: #4a9eff;
            }
            QCheckBox::indicator:hover {
                border-color: #4a9eff;
            }
        """)

        layout = QVBoxLayout(self)

        # 创建轨道行
        for track_type in TrackType:
            row = TrackRow(track_type)
            self.track_rows[track_type] = row
            layout.addWidget(row)

    def update_translations(self):
        """更新当前语言的文本"""
        self.setTitle(t("main.tracks.title"))
        for row in self.track_rows.values():
            row.update_translations()

    def get_selected_tracks(self) -> dict:
        """获取选中的MIDI和歌词轨道"""
        result = {}
        for track_type, row in self.track_rows.items():
            result[track_type] = {
                "midi": row.midi_check.isChecked(),
                "lyrics": row.lyrics_check.isChecked() if row.lyrics_check.isVisible() else False
            }
        return result
