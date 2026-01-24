"""
Drag and drop zone widget for file input.
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
    Widget for drag-and-drop file input.
    """

    file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 40, 20, 40)

        # Title
        self.title_label = QLabel(t("main.dropzone.title"))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #666;
            }
        """)

        # Subtitle
        self.subtitle_label = QLabel(t("main.dropzone.subtitle"))
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #999;
            }
        """)

        # Browse button
        self.browse_btn = QPushButton(t("main.output.browse"))
        self.browse_btn.setFixedWidth(120)
        self.browse_btn.clicked.connect(self._on_browse)

        # Selected file label
        self.file_label = QLabel()
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #333;
                padding: 10px;
                background: #f5f5f5;
                border-radius: 5px;
            }
        """)
        self.file_label.hide()

        # Add widgets
        layout.addStretch()
        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)
        layout.addSpacing(20)

        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_layout.addWidget(self.browse_btn)
        layout.addWidget(btn_container)

        layout.addSpacing(10)
        layout.addWidget(self.file_label)
        layout.addStretch()

        # Style
        self.setStyleSheet("""
            DropZoneWidget {
                background: #fafafa;
                border: 2px dashed #ccc;
                border-radius: 10px;
            }
            DropZoneWidget:hover {
                border-color: #999;
                background: #f0f0f0;
            }
        """)

    def _on_browse(self):
        """Handle browse button click."""
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
        """Set selected file."""
        self.file_label.setText(f"{t('main.dropzone.selected')}: {Path(file_path).name}")
        self.file_label.show()
        self.file_selected.emit(file_path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and is_supported_format(urls[0].toLocalFile()):
                event.acceptProposedAction()
                self.setStyleSheet("""
                    DropZoneWidget {
                        background: #e8f4e8;
                        border: 2px dashed #4a9;
                        border-radius: 10px;
                    }
                """)

    def dragLeaveEvent(self, event):
        """Handle drag leave event."""
        self.setStyleSheet("""
            DropZoneWidget {
                background: #fafafa;
                border: 2px dashed #ccc;
                border-radius: 10px;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if is_supported_format(file_path):
                self._set_file(file_path)

        self.setStyleSheet("""
            DropZoneWidget {
                background: #fafafa;
                border: 2px dashed #ccc;
                border-radius: 10px;
            }
        """)

    def update_translations(self):
        """Update text for current language."""
        self.title_label.setText(t("main.dropzone.title"))
        self.subtitle_label.setText(t("main.dropzone.subtitle"))
        self.browse_btn.setText(t("main.output.browse"))
