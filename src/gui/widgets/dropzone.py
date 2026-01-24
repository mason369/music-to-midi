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
        layout.setContentsMargins(30, 50, 30, 50)

        # 图标
        self.icon_label = QLabel("🎵")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setStyleSheet("font-size: 48px;")

        # 标题
        self.title_label = QLabel(t("main.dropzone.title"))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2d3748;
            }
        """)

        # 副标题
        self.subtitle_label = QLabel(t("main.dropzone.subtitle"))
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: #718096;
            }
        """)

        # 浏览按钮
        self.browse_btn = QPushButton("📂  " + t("main.output.browse"))
        self.browse_btn.setFixedWidth(160)
        self.browse_btn.setFixedHeight(42)
        self.browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                color: white;
                font-size: 13px;
                font-weight: 600;
                border-radius: 8px;
                border: none;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #7c8ff0, stop:1 #8b5cb8);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5a6fd6, stop:1 #6a4190);
            }
        """)
        self.browse_btn.clicked.connect(self._on_browse)

        # 已选文件标签
        self.file_label = QLabel()
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #2d3748;
                font-weight: 500;
                padding: 12px 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e0e7ff, stop:1 #c7d2fe);
                border-radius: 8px;
            }
        """)
        self.file_label.hide()

        # 支持格式提示
        self.format_label = QLabel("支持格式: MP3, WAV, FLAC, OGG, M4A")
        self.format_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.format_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #a0aec0;
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
                background: white;
                border: 2px dashed #cbd5e0;
                border-radius: 16px;
            }
        """)

    def _apply_hover_style(self):
        """应用悬停样式"""
        self.setStyleSheet("""
            DropZoneWidget {
                background: #f7fafc;
                border: 2px dashed #667eea;
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
                        background: #f0fff4;
                        border: 2px dashed #48bb78;
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
