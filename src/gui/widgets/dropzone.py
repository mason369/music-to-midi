"""
文件拖放区域组件 - 用于文件输入
"""
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from src.i18n.translator import t
from src.utils.audio_utils import get_supported_formats, is_supported_format


class DropZoneWidget(QWidget):
    """
    拖放文件输入组件
    """

    file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 30, 20, 30)

        # 图标
        self.icon_label = QLabel("🎵")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setStyleSheet("font-size: 36px;")

        # 标题
        self.title_label = QLabel(t("main.dropzone.title"))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                font-weight: bold;
                color: #e0e0e0;
            }
        """)

        # 副标题
        self.subtitle_label = QLabel(t("main.dropzone.subtitle"))
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: #8892a0;
            }
        """)

        # 浏览按钮
        self.browse_btn = QPushButton("📂  " + t("main.output.browse"))
        self.browse_btn.setFixedWidth(160)
        self.browse_btn.setFixedHeight(42)
        self.browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_btn.setStyleSheet("""
            QPushButton {
                background: #4a9eff;
                color: white;
                font-size: 13px;
                font-weight: 600;
                border-radius: 8px;
                border: none;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background: #5aafff;
            }
            QPushButton:pressed {
                background: #3a8eef;
            }
        """)
        self.browse_btn.clicked.connect(self._on_browse)

        # 已选文件标签
        self.file_label = QLabel()
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #e0e0e0;
                font-weight: 500;
                padding: 12px 20px;
                background: #2a3f5f;
                border-radius: 8px;
                border: 1px solid #3a5a7c;
            }
        """)
        self.file_label.hide()

        # 支持格式提示
        self.format_label = QLabel("支持格式: MP3, WAV, FLAC, OGG, M4A")
        self.format_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.format_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #6a7a8a;
            }
        """)

        # 添加组件
        layout.addStretch()
        layout.addWidget(self.icon_label)
        layout.addSpacing(8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)
        layout.addSpacing(20)

        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_layout.addWidget(self.browse_btn)
        layout.addWidget(btn_container)

        layout.addSpacing(8)
        layout.addWidget(self.format_label)
        layout.addSpacing(12)
        layout.addWidget(self.file_label)
        layout.addStretch()

        # 样式
        self._apply_default_style()

    def _apply_default_style(self):
        """应用默认样式"""
        self.setStyleSheet("""
            DropZoneWidget {
                background: #1f2940;
                border: 2px dashed #3a4a6a;
                border-radius: 16px;
            }
        """)

    def _apply_hover_style(self):
        """应用悬停样式"""
        self.setStyleSheet("""
            DropZoneWidget {
                background: #2a3f5f;
                border: 2px dashed #4a9eff;
                border-radius: 16px;
            }
        """)

    def _on_browse(self):
        """处理浏览按钮点击"""
        formats = get_supported_formats()
        filter_str = f"{t('dialogs.openFile.filter')} (*{' *'.join(formats)})"

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            t("dialogs.openFile.title"),
            "",
            filter_str
        )

        if file_path:
            self._set_file(file_path)

    def _set_file(self, file_path: str):
        """设置已选文件"""
        self.file_label.setText(f"{t('main.dropzone.selected')}: {Path(file_path).name}")
        self.file_label.show()
        self.file_selected.emit(file_path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """处理拖入事件"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and is_supported_format(urls[0].toLocalFile()):
                event.acceptProposedAction()
                self.setStyleSheet("""
                    DropZoneWidget {
                        background: #1a3a2a;
                        border: 2px dashed #50c878;
                        border-radius: 16px;
                    }
                """)
                self.icon_label.setText("✨")

    def dragLeaveEvent(self, event):
        """处理拖出事件"""
        self._apply_default_style()
        self.icon_label.setText("🎵")

    def dropEvent(self, event: QDropEvent):
        """处理放下事件"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if is_supported_format(file_path):
                self._set_file(file_path)

        self._apply_default_style()
        self.icon_label.setText("🎵")

    def update_translations(self):
        """更新当前语言的文本"""
        self.title_label.setText(t("main.dropzone.title"))
        self.subtitle_label.setText(t("main.dropzone.subtitle"))
        self.browse_btn.setText("📂  " + t("main.output.browse"))
