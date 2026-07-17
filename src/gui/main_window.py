"""
音乐转MIDI应用程序主窗口
"""

import os
import sys
import platform
import logging
import time as _time
from html import escape
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QMenuBar,
    QMenu,
    QToolBar,
    QStatusBar,
    QLabel,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QFrame,
    QSplitter,
    QGraphicsDropShadowEffect,
    QDialog,
    QTextEdit,
    QApplication,
    QSizePolicy,
    QFormLayout,
    QLayout,
    QScrollArea,
)
from PyQt6.QtCore import Qt, QSize, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QFont, QPalette, QColor, QPixmap

from src import __version__
from src.models.data_models import (
    Config,
    ProcessingMode,
    ProcessingProgress,
    ProcessingResult,
    ProcessingStage,
)
from src.core.manual_midi import build_manual_midi_config, manual_midi_output_dir
from src.core.multi_stem_separator import STEM_KEYS
from src.gui.widgets.dropzone import DropZoneWidget
from src.gui.widgets.audio_track_mixer import AudioTrackMixerWidget, midi_route_label
from src.gui.widgets.track_panel import TrackPanel
from src.gui.widgets.progress_widget import ProgressWidget
from src.gui.widgets.wheel_safe_controls import NoWheelComboBox
from src.gui.layouts import FlowLayout
from src.gui.workers.processing_worker import ProcessingWorker
from src.gui.workers.separation_worker import SeparationResult, SeparationWorker
from src.i18n.translator import t, get_translator, set_language, get_resource_path
from src.utils.gpu_utils import get_memory_info

logger = logging.getLogger(__name__)


def _bounded_size(available: QSize, preferred: QSize, minimum: QSize) -> QSize:
    """Return a logical window size that stays inside available screen geometry."""
    if available.width() <= 0 or available.height() <= 0:
        raise ValueError("Screen available geometry must be positive")

    maximum = QSize(
        max(1, int(available.width() * 0.92)),
        max(1, int(available.height() * 0.92)),
    )
    width = min(available.width(), max(minimum.width(), min(preferred.width(), maximum.width())))
    height = min(
        available.height(), max(minimum.height(), min(preferred.height(), maximum.height()))
    )
    return QSize(width, height)


class ElidingLabel(QLabel):
    """Keep full text accessible while eliding it to the allocated width."""

    def __init__(self, text: str = "", parent=None):
        super().__init__("", parent)
        self._full_text = ""
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - mirrors QLabel
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self.setAccessibleName(self._full_text)
        self._sync_elided_text()

    def text(self) -> str:
        return self._full_text

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hint = super().sizeHint()
        hint.setWidth(0)
        return hint

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._sync_elided_text()

    def _sync_elided_text(self) -> None:
        available_width = max(0, self.contentsRect().width())
        displayed = self.fontMetrics().elidedText(
            self._full_text,
            Qt.TextElideMode.ElideRight,
            available_width,
        )
        QLabel.setText(self, displayed)


def get_app_dir() -> Path:
    """
    获取应用程序所在目录（用于输出文件）

    对于打包后的应用，返回可执行文件所在的目录
    对于开发环境，返回项目根目录
    """
    if getattr(sys, "frozen", False):
        # PyInstaller 打包后的环境 - 使用可执行文件所在目录
        return Path(sys.executable).parent
    else:
        # 开发环境 - 使用项目根目录
        return Path(__file__).parent.parent.parent


def get_ui_font(size: int = 10, bold: bool = False) -> "QFont":
    """获取跨平台UI字体，支持中文显示"""
    from PyQt6.QtGui import QFontDatabase

    weight = QFont.Weight.Bold if bold else QFont.Weight.Normal
    if platform.system() == "Windows":
        return QFont("Microsoft YaHei UI", size, weight)
    else:
        # Linux/WSL: Noto Sans CJK 支持中文，回退到 Ubuntu/DejaVu
        available = QFontDatabase.families()
        for family in ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "Ubuntu", "DejaVu Sans"):
            if family in available:
                return QFont(family, size, weight)
        return QFont("Noto Sans CJK SC", size, weight)


def get_monospace_font(size: int = 11) -> "QFont":
    """获取跨平台等宽字体"""
    if platform.system() == "Windows":
        return QFont("Consolas", size)
    else:
        font = QFont("Ubuntu Mono", size)
        if not font.exactMatch():
            font = QFont("DejaVu Sans Mono", size)
        return font


def get_icon_path(name: str) -> str:
    """获取图标文件路径"""
    return str(get_resource_path(f"resources/icons/{name}"))


class MainWindow(QMainWindow):
    """
    应用程序主窗口
    """

    def __init__(self, config: Config = None):
        super().__init__()

        self.config = config or Config()
        set_language(self.config.language)
        self.worker = None
        self.current_file = None
        self._last_memory_update = 0.0  # 内存标签节流时间戳
        self._last_memory_info = None
        self._raw_device_label = ""
        self._stopping = False  # 防止停止后信号竞争
        self._close_pending = False  # 活跃 worker 退出后再真正关闭窗口
        self._manual_midi_context = None

        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()
        self._apply_modern_style()

        self.setWindowTitle(t("app.name"))
        self.setMinimumSize(320, 240)
        self._fit_to_available_screen()

        # 设置应用图标（Linux使用PNG，Windows使用ICO）
        if platform.system() == "Windows":
            icon_path = get_icon_path("app.ico")
        else:
            icon_path = get_icon_path("app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def _fit_to_available_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            raise RuntimeError("No screen is available for the main window")

        available = screen.availableGeometry()
        target = _bounded_size(available.size(), QSize(1100, 820), self.minimumSize())
        self.resize(target)
        self.move(
            available.x() + (available.width() - target.width()) // 2,
            available.y() + (available.height() - target.height()) // 2,
        )

    def _configure_dialog_size(
        self,
        dialog: QDialog,
        preferred: QSize,
        minimum: QSize = QSize(280, 200),
    ) -> None:
        screen = self.screen()
        if screen is None:
            raise RuntimeError("No screen is available for the dialog")

        available = screen.availableGeometry().size()
        dialog.setMinimumSize(
            min(minimum.width(), available.width()),
            min(minimum.height(), available.height()),
        )
        dialog.resize(_bounded_size(available, preferred, dialog.minimumSize()))

    def _create_scrollable_dialog_layout(
        self,
        dialog: QDialog,
        *,
        margins: tuple[int, int, int, int],
        spacing: int,
    ) -> QVBoxLayout:
        dialog.setObjectName("appDialog")
        dialog.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        outer_layout = QVBoxLayout(dialog)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(dialog)
        scroll.setObjectName("dialogScrollArea")
        scroll.viewport().setObjectName("dialogScrollViewport")
        scroll.viewport().setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setAlignment(Qt.AlignmentFlag.AlignTop)

        content = QWidget()
        content.setObjectName("dialogScrollContent")
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(*margins)
        layout.setSpacing(spacing)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll.setWidget(content)
        outer_layout.addWidget(scroll)
        dialog.setStyleSheet("""
            QDialog#appDialog,
            QScrollArea#dialogScrollArea,
            QWidget#dialogScrollViewport,
            QWidget#dialogScrollContent {
                background: #1a1a2e;
                color: #e0e0e0;
            }
            QScrollArea#dialogScrollArea {
                border: none;
            }
            QScrollArea#dialogScrollArea QScrollBar:vertical {
                background: #16213e;
                width: 12px;
                margin: 0;
            }
            QScrollArea#dialogScrollArea QScrollBar::handle:vertical {
                background: #3a4a6a;
                min-height: 28px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollArea#dialogScrollArea QScrollBar::handle:vertical:hover {
                background: #4a5f80;
            }
            QScrollArea#dialogScrollArea QScrollBar::add-line:vertical,
            QScrollArea#dialogScrollArea QScrollBar::sub-line:vertical {
                height: 0;
                background: transparent;
            }
            QScrollArea#dialogScrollArea QScrollBar::add-page:vertical,
            QScrollArea#dialogScrollArea QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)
        return layout

    def _setup_ui(self):
        """设置主用户界面"""
        self.content_scroll = QScrollArea()
        self.content_scroll.setObjectName("mainContentScroll")
        self.content_scroll.viewport().setObjectName("mainContentViewport")
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.content_scroll.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setCentralWidget(self.content_scroll)

        central_widget = QWidget()
        central_widget.setObjectName("mainContent")
        central_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.content_scroll.setWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 6, 12, 6)
        main_layout.setSpacing(6)
        main_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        # 顶部标题区域
        header_widget = self._create_header()
        main_layout.addWidget(header_widget)

        # 文件输入拖放区域
        self.dropzone = DropZoneWidget()
        self._add_shadow(self.dropzone)

        # 轨道面板
        self.track_panel = TrackPanel()
        self.track_panel.set_processing_mode(self.config.processing_mode)
        self.track_panel.set_transcription_backend(
            getattr(
                self.config,
                "transcription_backend",
                getattr(self.config, "multi_instrument_model", "yourmt3"),
            )
        )
        self.track_panel.set_multi_instrument_model(
            getattr(self.config, "multi_instrument_model", "yourmt3")
        )
        self.track_panel.set_midi_track_mode(getattr(self.config, "midi_track_mode", "multi_track"))
        self.track_panel.set_yourmt3_model(
            getattr(self.config, "yourmt3_model", "yptf_moe_multi_nops")
        )
        self._add_shadow(self.track_panel)

        # 进度组件
        self.progress_widget = ProgressWidget()
        self._add_shadow(self.progress_widget)

        # 输出设置
        self.output_group = self._create_output_settings()
        self._add_shadow(self.output_group)

        # 操作按钮
        self.action_layout = self._create_action_buttons()

        # 完成结果面板：仅在一次处理成功后显示，隐藏时不占据主内容空间。
        self._last_result = None
        self._last_separation_result = None
        self.result_panel = QFrame()
        self.result_panel.setObjectName("successResultPanel")
        self.result_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.result_panel.setMinimumWidth(0)
        self.result_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        result_layout = QVBoxLayout(self.result_panel)
        result_layout.setContentsMargins(14, 12, 14, 12)
        result_layout.setSpacing(10)

        self.result_title_label = QLabel()
        self.result_title_label.setObjectName("successResultTitle")
        self.result_title_label.setWordWrap(True)
        self.result_title_label.setMinimumWidth(0)
        self.result_title_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #7ee2a8;")
        result_layout.addWidget(self.result_title_label)

        self.result_info_label = QLabel()
        self.result_info_label.setObjectName("successResultInfo")
        self.result_info_label.setWordWrap(True)
        self.result_info_label.setMinimumWidth(0)
        self.result_info_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.result_info_label.setStyleSheet(
            "background: #16213e; border: 1px solid #3a4a6a; "
            "border-radius: 8px; color: #b0b8c8; padding: 12px;"
        )
        result_layout.addWidget(self.result_info_label)

        self.audio_mixer = None
        self._audio_mixer_tracks = None
        self.audio_mixer_error_label = None
        self.audio_timeline_container = QWidget(self.result_panel)
        self.audio_timeline_container.setObjectName("audioTimelineContainer")
        self.audio_timeline_container.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )
        self.audio_timeline_container.setMinimumWidth(0)
        self.audio_timeline_container.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.audio_timeline_container.setStyleSheet("""
            QWidget#audioTimelineContainer {
                background: #1a1a2e;
                border: 1px solid #2a3f5f;
                border-radius: 8px;
            }
        """)
        self.audio_timeline_layout = QVBoxLayout(self.audio_timeline_container)
        self.audio_timeline_layout.setContentsMargins(8, 8, 8, 8)
        self.audio_timeline_layout.setSpacing(0)
        self.audio_timeline_container.hide()
        result_layout.addWidget(self.audio_timeline_container)

        self.result_actions_layout = FlowLayout(
            horizontal_spacing=12,
            vertical_spacing=8,
        )
        result_layout.addLayout(self.result_actions_layout)
        self.result_panel.setStyleSheet(
            "QFrame#successResultPanel { background: #101a32; "
            "border: 1px solid #38547a; border-radius: 9px; }"
        )
        self._add_shadow(self.result_panel)
        self.result_panel.hide()

        # 添加组件
        main_layout.addWidget(self.dropzone)
        main_layout.addWidget(self.track_panel)
        main_layout.addWidget(self.progress_widget)
        main_layout.addWidget(self.output_group)
        main_layout.addLayout(self.action_layout)
        main_layout.addWidget(self.result_panel)

    def _create_header(self) -> QWidget:
        """创建顶部标题区域"""
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 4)

        # 图标
        icon_label = QLabel()
        icon_path = get_icon_path("icon_48.png")
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            icon_label.setPixmap(
                pixmap.scaled(
                    32,
                    32,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        icon_label.setFixedSize(36, 36)

        # 标题文字
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)

        self.title_label = QLabel(t("app.name"))
        self.title_label.setFont(get_ui_font(13, bold=True))
        self.title_label.setStyleSheet("color: #e0e0e0;")

        self.subtitle_label = QLabel(t("app.subtitle"))
        self.subtitle_label.setFont(get_ui_font(8))
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.subtitle_label.setStyleSheet("color: #8892a0;")

        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.subtitle_label)

        layout.addWidget(icon_label)
        layout.addSpacing(8)
        layout.addLayout(title_layout)
        layout.addStretch()

        return header

    def _add_shadow(self, widget: QWidget):
        """为组件添加阴影效果"""
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 80))
        widget.setGraphicsEffect(shadow)

    def _create_output_settings(self) -> QGroupBox:
        """创建输出设置组"""
        group = QGroupBox(t("main.output.title"))
        group.setStyleSheet("""
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
        """)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 12, 10, 8)
        layout.setSpacing(6)

        # 输出目录。QFormLayout 会在宽度不足时把标签放到输入框上方。
        dir_layout = QFormLayout()
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_layout.setSpacing(6)
        dir_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        dir_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.output_dir_label = QLabel(t("main.output.directory") + ":")
        self.output_dir_label.setStyleSheet("font-weight: normal; color: #b0b8c8;")
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setMinimumWidth(0)
        self.output_dir_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.output_dir_edit.setText(str(get_app_dir() / "MidiOutput"))
        self.output_dir_edit.setStyleSheet("""
            QLineEdit {
                padding: 5px 10px;
                border: 1px solid #3a4a6a;
                border-radius: 5px;
                background: #16213e;
                color: #e0e0e0;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #4a9eff;
                background: #1f2940;
            }
        """)
        self.browse_dir_btn = QPushButton(t("main.output.browse"))
        self.browse_dir_btn.setStyleSheet("""
            QPushButton {
                padding: 5px 12px;
                background: #2a3f5f;
                border: 1px solid #3a4a6a;
                border-radius: 5px;
                color: #e0e0e0;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #3a5a7c;
                border-color: #4a9eff;
            }
        """)
        self.browse_dir_btn.clicked.connect(self._browse_output_dir)

        dir_field = QWidget()
        dir_field_layout = QHBoxLayout(dir_field)
        dir_field_layout.setContentsMargins(0, 0, 0, 0)
        dir_field_layout.setSpacing(6)
        dir_field_layout.addWidget(self.output_dir_edit, 1)
        dir_field_layout.addWidget(self.browse_dir_btn)
        dir_layout.addRow(self.output_dir_label, dir_field)

        # 选项在窄窗口下自动换行。
        options_layout = FlowLayout(horizontal_spacing=20, vertical_spacing=6)

        checkbox_style = """
            QCheckBox {
                font-size: 11px;
                color: #b0b8c8;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
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
        """

        self.tracks_check = QCheckBox(t("main.output.options.saveTracks"))
        self.tracks_check.setChecked(True)
        self.tracks_check.setStyleSheet(checkbox_style)

        options_layout.addWidget(self.tracks_check)

        layout.addLayout(dir_layout)
        layout.addLayout(options_layout)

        return group

    def _create_action_buttons(self) -> QHBoxLayout:
        """创建操作按钮布局"""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)

        self.start_btn = QPushButton("▶  " + t("toolbar.start"))
        self.start_btn.setMinimumHeight(34)
        self.start_btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.start_btn.setFont(get_ui_font(10, bold=True))
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #4a9eff;
                color: white;
                font-weight: bold;
                padding: 8px 8px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background: #5aafff;
            }
            QPushButton:pressed {
                background: #3a8eef;
            }
            QPushButton:disabled {
                background: #2a3f5f;
                color: #5a6a8a;
            }
        """)

        self.stop_btn = QPushButton("■  " + t("toolbar.stop"))
        self.stop_btn.setMinimumHeight(34)
        self.stop_btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.stop_btn.setFont(get_ui_font(9))
        self.stop_btn.setEnabled(False)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #e05050;
                color: white;
                padding: 8px 8px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background: #f06060;
            }
            QPushButton:pressed {
                background: #d04040;
            }
            QPushButton:disabled {
                background: #2a3f5f;
                color: #5a6a8a;
            }
        """)

        layout.addStretch()
        layout.addWidget(self.start_btn)
        layout.addSpacing(16)
        layout.addWidget(self.stop_btn)
        layout.addStretch()

        return layout

    def _apply_modern_style(self):
        """应用现代化样式 - 专业音频软件风格"""
        self.setStyleSheet("""
            QMainWindow {
                background: #1a1a2e;
            }
            QMainWindow QLabel,
            QMainWindow QCheckBox {
                color: #c8d3e6;
            }
            QScrollArea#mainContentScroll,
            QWidget#mainContentViewport,
            QWidget#mainContent {
                background: #1a1a2e;
            }
            QScrollArea#mainContentScroll QScrollBar:vertical {
                background: #16213e;
                width: 12px;
                margin: 0;
            }
            QScrollArea#mainContentScroll QScrollBar::handle:vertical {
                background: #3a4a6a;
                min-height: 28px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollArea#mainContentScroll QScrollBar::handle:vertical:hover {
                background: #4a5f80;
            }
            QScrollArea#mainContentScroll QScrollBar::add-line:vertical,
            QScrollArea#mainContentScroll QScrollBar::sub-line:vertical {
                height: 0;
                background: transparent;
            }
            QScrollArea#mainContentScroll QScrollBar::add-page:vertical,
            QScrollArea#mainContentScroll QScrollBar::sub-page:vertical {
                background: transparent;
            }
            QMenuBar {
                background: #16213e;
                border-bottom: 1px solid #2a2a4a;
                padding: 4px;
                color: #e0e0e0;
            }
            QMenuBar::item {
                padding: 6px 12px;
                border-radius: 4px;
                color: #e0e0e0;
            }
            QMenuBar::item:selected {
                background: #2a3f5f;
            }
            QMenu {
                background: #1f2940;
                border: 1px solid #2a2a4a;
                border-radius: 6px;
                padding: 4px;
                color: #e0e0e0;
            }
            QMenu::item {
                padding: 8px 24px;
                border-radius: 4px;
                color: #e0e0e0;
            }
            QMenu::item:selected {
                background: #3a5a7c;
            }
            QToolBar {
                background: #16213e;
                border-bottom: 1px solid #2a2a4a;
                spacing: 8px;
                padding: 4px 8px;
            }
            QToolBar QLabel {
                color: #c8d3e6;
                background: transparent;
            }
            QStatusBar {
                background: #16213e;
                border-top: 1px solid #2a2a4a;
                color: #8892a0;
                font-size: 11px;
            }
            QStatusBar QWidget,
            QStatusBar QLabel {
                background: transparent;
                color: #c8d3e6;
            }
            QStatusBar::item {
                border: none;
            }
            QComboBox {
                padding: 6px 12px;
                border: 1px solid #3a4a6a;
                border-radius: 6px;
                background: #1f2940;
                color: #e0e0e0;
                min-width: 100px;
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

    def _setup_menu(self):
        """设置菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        self.file_menu = menubar.addMenu(t("menu.file"))

        self.open_action = QAction(t("menu.open"), self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self._open_file)
        self.file_menu.addAction(self.open_action)

        self.save_action = QAction(t("menu.save"), self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.setEnabled(False)
        self.file_menu.addAction(self.save_action)

        self.file_menu.addSeparator()

        self.exit_action = QAction(t("menu.exit"), self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)

        # 编辑菜单
        self.edit_menu = menubar.addMenu(t("menu.edit"))

        self.settings_action = QAction(t("menu.settings"), self)
        self.settings_action.triggered.connect(self._open_settings)
        self.edit_menu.addAction(self.settings_action)

        # 视图菜单
        self.view_menu = menubar.addMenu(t("menu.view"))

        # 语言子菜单
        self.lang_menu = QMenu(t("settings.general.language"), self)
        self.view_menu.addMenu(self.lang_menu)

        for code, name in get_translator().AVAILABLE_LANGUAGES.items():
            action = QAction(name, self)
            action.setData(code)
            action.triggered.connect(lambda checked, c=code: self._change_language(c))
            self.lang_menu.addAction(action)

        # 帮助菜单
        self.help_menu = menubar.addMenu(t("menu.help"))

        self.about_action = QAction(t("menu.about"), self)
        self.about_action.triggered.connect(self._show_about)
        self.help_menu.addAction(self.about_action)

    def _setup_toolbar(self):
        """设置工具栏"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # 语言选择器放在右侧
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.lang_combo = NoWheelComboBox()
        for code, name in get_translator().AVAILABLE_LANGUAGES.items():
            self.lang_combo.addItem(name, code)

        # 设置当前语言
        current_lang = get_translator().get_language()
        index = self.lang_combo.findData(current_lang)
        if index >= 0:
            self.lang_combo.setCurrentIndex(index)

        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        toolbar.addWidget(QLabel("  "))
        toolbar.addWidget(self.lang_combo)

    def _setup_statusbar(self):
        """设置状态栏"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # 三段信息共享状态栏宽度。每段保留完整 tooltip，显示文本按宽度省略。
        status_container = QWidget()
        status_container.setObjectName("statusInfoContainer")
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(8)

        self.status_label = ElidingLabel(t("status.ready"))
        self.device_label = ElidingLabel("...")
        self.memory_label = ElidingLabel()
        status_label_style = "background: transparent; color: #c8d3e6;"
        for label in (self.status_label, self.device_label, self.memory_label):
            label.setStyleSheet(status_label_style)

        status_layout.addWidget(self.status_label, 3)
        status_layout.addWidget(self.device_label, 2)
        status_layout.addWidget(self.memory_label, 1)
        self.statusbar.addWidget(status_container, 1)

        # 在后台线程中检测 GPU，避免同步 import torch 阻塞 UI 初始化
        self._start_gpu_detection()

    def _start_gpu_detection(self):
        """启动后台 GPU 检测线程"""

        class _GpuDetector(QThread):
            detected = pyqtSignal(str, object)  # (device_label, memory_info)

            def run(self):
                try:
                    from src.utils.gpu_utils import (
                        is_gpu_available,
                        get_accelerator_label,
                        get_memory_info,
                    )

                    device_text = get_accelerator_label() if is_gpu_available() else "CPU"
                    mem_info = get_memory_info()
                    if mem_info:
                        used, total = mem_info
                        memory_info = (used, total)
                    else:
                        memory_info = None
                    self.detected.emit(device_text, memory_info)
                except Exception:
                    self.detected.emit("CPU", None)

        self._gpu_detector = _GpuDetector()
        self._gpu_detector.detected.connect(self._on_gpu_detected)
        self._gpu_detector.start()

    def _format_device_label(self, dev: str) -> str:
        return t("status.cpu") if dev == "CPU" else dev

    def _render_memory_label(self):
        if self._last_memory_info:
            used, total = self._last_memory_info
            self.memory_label.setText(f"{t('status.video_memory')}: {used:.1f}/{total:.1f}GB")
        else:
            self.memory_label.setText("")

    def _on_gpu_detected(self, dev: str, mem_info):
        """处理 GPU 检测结果"""
        self._raw_device_label = dev
        self._last_memory_info = mem_info
        self.device_label.setText(self._format_device_label(dev))
        self._render_memory_label()

    def _connect_signals(self):
        """连接信号与槽"""
        self.dropzone.file_selected.connect(self._on_file_selected)
        self.start_btn.clicked.connect(self._start_processing)
        self.stop_btn.clicked.connect(self._stop_processing)
        self.track_panel.mode_changed.connect(self.progress_widget.set_mode)
        self.track_panel.mode_changed.connect(self._update_output_option_visibility)
        self.track_panel.mode_changed.connect(self._update_start_button_label)
        self._update_output_option_visibility(self.track_panel.get_processing_mode())
        self._update_start_button_label(self.track_panel.get_processing_mode())

    def _update_start_button_label(self, mode: str):
        """Split modes only separate WAVs; direct modes convert to MIDI.

        The button text mirrors the Space and Colab action labels so every
        platform presents the same workflow wording.
        """
        if mode in {
            ProcessingMode.VOCAL_SPLIT.value,
            ProcessingMode.SIX_STEM_SPLIT.value,
        }:
            self.start_btn.setText("▶  " + t("toolbar.start_separation"))
        else:
            self.start_btn.setText("▶  " + t("toolbar.start_convert"))

    def _update_output_option_visibility(self, mode: str):
        """Separated WAV files are mandatory inputs for manual per-track AMT."""
        if mode in {
            ProcessingMode.VOCAL_SPLIT.value,
            ProcessingMode.SIX_STEM_SPLIT.value,
        }:
            self.tracks_check.setChecked(True)
        self.tracks_check.setVisible(False)

    @staticmethod
    def _clear_widget_layout(layout: QLayout) -> None:
        """Remove and schedule deletion for every widget owned by a result layout."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.hide()
                widget.deleteLater()

    def _clear_audio_mixer(self) -> None:
        """Stop playback, release media handles, and hide the inline timeline."""
        self.audio_timeline_container.hide()

        if self.audio_mixer is not None:
            mixer = self.audio_mixer
            self.audio_mixer = None
            mixer.shutdown()
            self.audio_timeline_layout.removeWidget(mixer)
            mixer.hide()
            mixer.deleteLater()

        if self.audio_mixer_error_label is not None:
            error_label = self.audio_mixer_error_label
            self.audio_mixer_error_label = None
            self.audio_timeline_layout.removeWidget(error_label)
            error_label.hide()
            error_label.deleteLater()

        self._audio_mixer_tracks = None

    def _set_audio_tracks(self, tracks, *, show_timeline: bool = True) -> None:
        """Replace the inline mixer with a fresh owner for the provided tracks."""
        normalized_tracks = dict(tracks or {})
        self._clear_audio_mixer()
        if not normalized_tracks:
            return

        self._audio_mixer_tracks = normalized_tracks
        try:
            mixer = AudioTrackMixerWidget(normalized_tracks)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("无法创建音轨播放器: %s", exc)
            error_label = QLabel(t("dialogs.complete.audio_tracks.unavailable", error=exc))
            error_label.setObjectName("audioMixerUnavailableLabel")
            error_label.setWordWrap(True)
            error_label.setMinimumWidth(0)
            error_label.setStyleSheet(
                "background: #3a1f2a; border: 1px solid #b84d68; "
                "border-radius: 6px; color: #ffb3c3; padding: 10px;"
            )
            self.audio_mixer_error_label = error_label
            self.audio_timeline_layout.addWidget(error_label)
        else:
            self.audio_mixer = mixer
            mixer.midi_conversion_requested.connect(self._start_track_midi_conversion)
            mixer.midi_open_requested.connect(self._open_output_folder)
            self.audio_timeline_layout.addWidget(mixer)

        if show_timeline:
            self.audio_timeline_container.show()

    def _playback_tracks_for_result(self, result: ProcessingResult) -> dict:
        """Direct conversion results never own the separated-track timeline."""
        if result.separated_audio:
            raise ValueError(
                "ProcessingResult contains separated audio; split workflows must "
                "render a verified SeparationResult instead"
            )
        return {}

    def _clear_completed_result(self) -> None:
        """Hide stale success output before another input or processing run."""
        self._last_result = None
        self._last_separation_result = None
        self._manual_midi_context = None
        self.result_panel.hide()
        self.result_title_label.clear()
        self.result_info_label.clear()
        self._clear_widget_layout(self.result_actions_layout)
        if hasattr(self, "save_action"):
            self.save_action.setEnabled(False)
        self._clear_audio_mixer()

    @staticmethod
    def _result_button_style(*, primary: bool = False) -> str:
        background = "#4a9eff" if primary else "#2a3f5f"
        hover = "#5aafff" if primary else "#3a5a7c"
        border = "none" if primary else "1px solid #3a4a6a"
        return f"""
            QPushButton {{
                padding: 9px 18px;
                background: {background};
                border: {border};
                border-radius: 6px;
                color: white;
                font-weight: 600;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: {hover};
                border-color: #4a9eff;
            }}
        """

    def _add_result_button(
        self,
        text: str,
        object_name: str,
        callback,
        *,
        primary: bool = False,
    ) -> None:
        button = QPushButton(text)
        button.setObjectName(object_name)
        button.setStyleSheet(self._result_button_style(primary=primary))
        button.clicked.connect(callback)
        self.result_actions_layout.addWidget(button)

    def _show_success_result(
        self,
        result: ProcessingResult,
        *,
        reveal: bool = True,
        show_timeline: bool = True,
        replace_timeline: bool = True,
    ) -> None:
        """Render the completed-output summary and replace the persistent timeline."""
        self._last_result = result
        self._last_separation_result = None
        is_six_stem = bool(result.stem_midi_paths)
        is_vocal_split = bool(result.vocal_midi_path and result.accompaniment_midi_path)
        has_vocal_merged = bool(
            is_vocal_split
            and result.merged_midi_path
            and result.merged_midi_path != result.accompaniment_midi_path
        )

        def _format_named_paths(paths) -> str:
            return "<br>".join(
                f"&nbsp;&nbsp;&bull; {escape(str(name))}: " f"{escape(Path(path).name)}"
                for name, path in sorted((paths or {}).items())
                if path
            )

        if is_vocal_split:
            merged_line = ""
            if has_vocal_merged:
                merged_line = (
                    f"<b>{t('dialogs.complete.merged_midi')}:</b> "
                    f"{escape(str(result.merged_midi_path))}<br>"
                )
            separated_audio_lines = _format_named_paths(result.separated_audio)
            info_text = f"""
            <p style="color: #b0b8c8; line-height: 1.6;">
            <b>{t('dialogs.complete.accompaniment_midi')}:</b> {escape(str(result.accompaniment_midi_path))}<br>
            <b>{t('dialogs.complete.vocal_midi')}:</b> {escape(str(result.vocal_midi_path))}<br>
            {merged_line}
            <b>{t('dialogs.complete.separated_audio')}:</b><br>{separated_audio_lines}<br>
            <b>{t('dialogs.complete.processing_time')}:</b> {result.processing_time:.1f}{t('dialogs.complete.seconds_suffix')}
            </p>
            """
        elif is_six_stem:
            stem_midi_lines = _format_named_paths(result.stem_midi_paths)
            stem_audio_lines = _format_named_paths(result.separated_audio)
            info_text = f"""
            <p style="color: #b0b8c8; line-height: 1.6;">
            <b>{t('dialogs.complete.merged_midi')}:</b> {escape(str(result.midi_path))}<br>
            <b>{t('dialogs.complete.stem_midi_count')}:</b> {len(result.stem_midi_paths)}<br>
            <b>{t('dialogs.complete.stem_audio_count')}:</b> {len(result.separated_audio or {})}<br>
            <b>{t('dialogs.complete.stem_audio_files')}:</b><br>{stem_audio_lines}<br>
            <b>{t('dialogs.complete.stem_midis')}:</b><br>{stem_midi_lines}<br>
            <b>{t('dialogs.complete.processing_time')}:</b> {result.processing_time:.1f}{t('dialogs.complete.seconds_suffix')}
            </p>
            """
        else:
            bpm_text = f"{result.beat_info.bpm:.1f}" if result.beat_info else "N/A"
            info_text = f"""
            <p style="color: #b0b8c8; line-height: 1.6;">
            <b>{t('dialogs.complete.midi_file')}:</b> {escape(str(result.midi_path))}<br>
            <b>{t('dialogs.complete.track_count')}:</b> {len(result.tracks)}<br>
            <b>{t('dialogs.complete.note_count')}:</b> {result.total_notes}<br>
            <b>{t('dialogs.complete.bpm')}:</b> {escape(str(bpm_text))}<br>
            <b>{t('dialogs.complete.device')}:</b> {escape(self._format_device_label(self._raw_device_label))}<br>
            <b>{t('dialogs.complete.processing_time')}:</b> {result.processing_time:.1f}{t('dialogs.complete.seconds_suffix')}
            </p>
            """

        self.result_title_label.setText(
            "✓  " + t("dialogs.complete.audio_tracks.result_panel_title")
        )
        self.result_info_label.setText(info_text)
        self._clear_widget_layout(self.result_actions_layout)
        self._add_result_button(
            "📁  " + t("dialogs.complete.openFolder"),
            "resultOpenFolderButton",
            lambda _checked=False, path=result.midi_path: self._open_output_folder(path),
        )

        if is_vocal_split:
            self._add_result_button(
                "🎵  " + t("dialogs.complete.accompaniment_midi"),
                "resultOpenAccompanimentButton",
                lambda _checked=False, path=result.accompaniment_midi_path: self._open_midi_file(
                    path
                ),
            )
            self._add_result_button(
                "🎤  " + t("dialogs.complete.vocal_midi"),
                "resultOpenVocalButton",
                lambda _checked=False, path=result.vocal_midi_path: self._open_midi_file(path),
            )
            if has_vocal_merged:
                self._add_result_button(
                    "🎼  " + t("dialogs.complete.merged_midi"),
                    "resultOpenMergedButton",
                    lambda _checked=False, path=result.merged_midi_path: self._open_midi_file(path),
                    primary=True,
                )
        elif is_six_stem:
            self._add_result_button(
                "🎼  " + t("dialogs.complete.openMergedFile"),
                "resultOpenMergedButton",
                lambda _checked=False, path=result.midi_path: self._open_midi_file(path),
                primary=True,
            )
        else:
            self._add_result_button(
                "🎵  " + t("dialogs.complete.openFile"),
                "resultOpenMidiButton",
                lambda _checked=False, path=result.midi_path: self._open_midi_file(path),
                primary=True,
            )

        self.result_panel.show()
        self.save_action.setEnabled(True)
        if replace_timeline:
            self._set_audio_tracks(
                self._playback_tracks_for_result(result),
                show_timeline=show_timeline,
            )
        if reveal:
            QTimer.singleShot(
                0,
                lambda: self.content_scroll.ensureWidgetVisible(
                    self.result_panel,
                    0,
                    16,
                ),
            )

    def _show_separation_result(
        self,
        result: SeparationResult,
        *,
        reveal: bool = True,
        replace_timeline: bool = True,
    ) -> None:
        """Render real WAV outputs and let each row choose its own AMT model."""
        if result.mode == ProcessingMode.VOCAL_SPLIT.value:
            expected_names = ("vocals", "accompaniment")
            mode_key = "main.mode.vocal_split"
        elif result.mode == ProcessingMode.SIX_STEM_SPLIT.value:
            expected_names = tuple(STEM_KEYS)
            mode_key = "main.mode.six_stem_split"
        else:
            raise ValueError(f"Unsupported separation result mode: {result.mode!r}")

        actual_names = set(result.separated_audio)
        expected_set = set(expected_names)
        if actual_names != expected_set:
            missing = sorted(expected_set - actual_names)
            unexpected = sorted(actual_names - expected_set)
            raise ValueError(
                "Separation result track set does not match its mode: "
                f"missing={missing or 'none'}, unexpected={unexpected or 'none'}"
            )

        ordered_paths = [
            (name, Path(result.separated_audio[name]).resolve())
            for name in expected_names
        ]
        for name, path in ordered_paths:
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(
                    "Separated audio does not exist or is empty: "
                    f"track={name}, path={path}"
                )

        self._last_result = None
        self._last_separation_result = result
        track_lines = "<br>".join(
            f"&nbsp;&nbsp;&bull; {escape(str(name))}: {escape(path.name)}"
            for name, path in ordered_paths
        )
        info_text = f"""
        <p style="color: #b0b8c8; line-height: 1.6;">
        <b>{t('dialogs.complete.audio_tracks.separation_mode')}:</b>
        {escape(t(mode_key))}<br>
        <b>{t('dialogs.complete.stem_audio_count')}:</b>
        {len(ordered_paths)}<br>
        <b>{t('dialogs.complete.separated_audio')}:</b><br>
        {track_lines}<br>
        <b>{t('dialogs.complete.processing_time')}:</b>
        {result.processing_time:.1f}{t('dialogs.complete.seconds_suffix')}<br>
        <span style="color: #78b9ff;">
        {escape(t('dialogs.complete.audio_tracks.separation_manual_hint'))}
        </span>
        </p>
        """
        self.result_title_label.setText(
            "✓  " + t("dialogs.complete.audio_tracks.separation_result_title")
        )
        self.result_info_label.setText(info_text)
        self._clear_widget_layout(self.result_actions_layout)
        self._add_result_button(
            "📁  " + t("dialogs.complete.openFolder"),
            "resultOpenFolderButton",
            lambda _checked=False, path=str(ordered_paths[0][1]): (self._open_output_folder(path)),
        )
        self.result_panel.show()
        self.save_action.setEnabled(False)
        if replace_timeline:
            self._set_audio_tracks(dict(ordered_paths))
        if reveal:
            QTimer.singleShot(
                0,
                lambda: self.content_scroll.ensureWidgetVisible(
                    self.result_panel,
                    0,
                    16,
                ),
            )

    def _on_file_selected(self, file_path: str):
        """处理文件选择"""
        if self.worker and self.worker.isRunning():
            logger.warning("处理仍在运行，忽略新的文件选择")
            return
        self._clear_completed_result()
        self.current_file = file_path
        self.start_btn.setEnabled(True)
        self.status_label.setText(f"{t('status.ready')} - {Path(file_path).name}")
        logger.info(f"已选择文件: {file_path}")

    def _open_file(self):
        """打开文件对话框"""
        from src.utils.audio_utils import get_supported_formats

        formats = get_supported_formats()
        filter_str = f"{t('dialogs.openFile.filter')} (*{' *'.join(formats)})"

        file_path, _ = QFileDialog.getOpenFileName(
            self, t("dialogs.openFile.title"), "", filter_str
        )

        if file_path:
            self._on_file_selected(file_path)

    def _browse_output_dir(self):
        """浏览输出目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, t("dialogs.selectDir.title"), self.output_dir_edit.text()
        )

        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _start_processing(self):
        """开始处理音频文件"""
        if not self.current_file:
            return

        # 检查旧 worker 是否仍在运行
        if self.worker and self.worker.isRunning():
            logger.warning("上一个工作线程仍在运行，忽略重复启动")
            return

        self._clear_completed_result()
        self._stopping = False

        # 从UI更新配置
        self.config.output_dir = self.output_dir_edit.text()
        selected_mode = self.track_panel.get_processing_mode()
        self.config.processing_mode = selected_mode
        self.config.save_separated_tracks = (
            True
            if selected_mode
            in {
                ProcessingMode.VOCAL_SPLIT.value,
                ProcessingMode.SIX_STEM_SPLIT.value,
            }
            else self.tracks_check.isChecked()
        )
        self.config.transcription_backend = self.track_panel.get_transcription_backend()
        self.config.multi_instrument_model = self.track_panel.get_multi_instrument_model()
        self.config.midi_track_mode = self.track_panel.get_midi_track_mode()
        self.config.yourmt3_model = self.track_panel.get_yourmt3_model()
        self.config.vocal_split_merge_midi = self.track_panel.get_vocal_split_merge_midi()

        # 创建以音乐名命名的子文件夹（如果已存在则添加数字后缀）
        music_name = Path(self.current_file).stem
        output_dir_with_music_name = os.path.join(self.config.output_dir, music_name)

        if os.path.exists(output_dir_with_music_name):
            counter = 2
            while True:
                new_dir = os.path.join(self.config.output_dir, f"{music_name}_{counter}")
                if not os.path.exists(new_dir):
                    output_dir_with_music_name = new_dir
                    logger.info(f"输出目录已存在，使用新目录: {new_dir}")
                    break
                counter += 1

        os.makedirs(output_dir_with_music_name, exist_ok=True)

        if selected_mode in {
            ProcessingMode.VOCAL_SPLIT.value,
            ProcessingMode.SIX_STEM_SPLIT.value,
        }:
            self.worker = SeparationWorker(
                self.current_file,
                output_dir_with_music_name,
                self.config,
                self,
            )
            self.worker.progress_updated.connect(self._on_progress)
            self.worker.separation_finished.connect(self._on_separation_finished)
            self.worker.error_occurred.connect(self._on_error)
        else:
            self.worker = ProcessingWorker(
                self.current_file,
                output_dir_with_music_name,
                self.config,
                self,
            )
            self.worker.progress_updated.connect(self._on_progress)
            self.worker.processing_finished.connect(self._on_finished)
            self.worker.error_occurred.connect(self._on_error)
        worker = self.worker
        self.worker.finished.connect(lambda worker=worker: self._on_worker_thread_finished(worker))

        # 更新UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.track_panel.set_processing_controls_enabled(False)
        self.progress_widget.reset()
        self.status_label.setText(t("status.processing"))

        # 开始处理
        self.worker.start()
        logger.info("处理已开始")

    @staticmethod
    def _manual_midi_config(base_config: Config, route: str) -> Config:
        """Build an isolated one-file config from the shared route contract."""
        return build_manual_midi_config(base_config, route)

    def _start_track_midi_conversion(
        self,
        track_name: str,
        audio_path: str,
        route: str,
    ) -> None:
        """Transcribe only the clicked timeline row with its explicit model."""
        mixer = self.audio_mixer
        if mixer is None:
            raise RuntimeError("The audio timeline is not available")
        if self.worker and self.worker.isRunning():
            mixer.set_track_midi_failed(
                track_name,
                t("dialogs.complete.audio_tracks.manual_midi.busy"),
            )
            return

        resolved_audio = Path(audio_path).resolve()
        state = mixer.track_state(track_name)
        if state.path != resolved_audio:
            raise RuntimeError(
                "Stale track conversion request: "
                f"track={track_name}, expected={state.path}, actual={resolved_audio}"
            )
        if not resolved_audio.is_file() or resolved_audio.stat().st_size <= 0:
            mixer.set_track_midi_failed(
                track_name,
                f"Audio track does not exist or is empty: {resolved_audio}",
            )
            return

        try:
            manual_config = self._manual_midi_config(self.config, route)
            output_dir = manual_midi_output_dir(resolved_audio, route)
            output_dir.mkdir(parents=True, exist_ok=True)
            manual_config.output_dir = str(output_dir)
            worker = ProcessingWorker(
                str(resolved_audio),
                str(output_dir),
                manual_config,
                self,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error(
                "无法启动逐轨 MIDI 转写: track=%s route=%s error=%s",
                track_name,
                route,
                exc,
            )
            mixer.set_track_midi_failed(track_name, str(exc))
            return

        self.worker = worker
        self._manual_midi_context = (
            track_name,
            str(resolved_audio),
            route,
        )
        self._stopping = False
        mixer.set_midi_controls_enabled(False)
        mixer.set_track_midi_running(track_name, route)
        worker.progress_updated.connect(
            lambda progress, name=track_name, selected_route=route: (
                self._on_track_midi_progress(name, selected_route, progress)
            )
        )
        worker.processing_finished.connect(
            lambda result, name=track_name, selected_route=route: (
                self._on_track_midi_finished(name, selected_route, result)
            )
        )
        worker.error_occurred.connect(
            lambda message, name=track_name, selected_route=route: (
                self._on_track_midi_error(name, selected_route, message)
            )
        )
        worker.finished.connect(lambda worker=worker: self._on_worker_thread_finished(worker))

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.track_panel.set_processing_controls_enabled(False)
        self.progress_widget.set_mode(manual_config.processing_mode)
        self.progress_widget.reset()
        self.status_label.setText(
            t(
                "dialogs.complete.audio_tracks.manual_midi.converting_short",
                track=track_name,
                model=midi_route_label(route),
            )
        )
        worker.start()
        logger.info(
            "逐轨 MIDI 转写已开始: track=%s path=%s route=%s",
            track_name,
            resolved_audio,
            route,
        )

    def _on_track_midi_progress(
        self,
        track_name: str,
        route: str,
        progress: ProcessingProgress,
    ) -> None:
        context = self._manual_midi_context
        if context is None or context[0] != track_name or context[2] != route:
            return
        if self._stopping:
            return
        self._on_progress(progress)
        if self.audio_mixer is not None:
            self.audio_mixer.set_track_midi_progress(
                track_name,
                progress.message,
            )
        self.status_label.setText(f"{midi_route_label(route)} · {progress.message}")

    def _on_track_midi_finished(
        self,
        track_name: str,
        route: str,
        result: ProcessingResult,
    ) -> None:
        context = self._manual_midi_context
        if context is None or context[0] != track_name or context[2] != route:
            logger.warning("忽略过期的逐轨 MIDI 成功信号")
            return
        if self._stopping:
            return
        if not result.midi_path:
            self._on_track_midi_error(
                track_name,
                route,
                "转写后端没有返回 MIDI 文件路径",
            )
            return
        if self.audio_mixer is None:
            logger.error("逐轨 MIDI 已完成，但音轨时间线已被销毁")
            return
        self.audio_mixer.set_track_midi_succeeded(
            track_name,
            route,
            result.midi_path,
        )
        self.status_label.setText(f"{t('status.complete')} - {Path(result.midi_path).name}")
        logger.info(
            "逐轨 MIDI 转写完成: track=%s route=%s output=%s",
            track_name,
            route,
            result.midi_path,
        )

    def _on_track_midi_error(
        self,
        track_name: str,
        route: str,
        error_message: str,
    ) -> None:
        context = self._manual_midi_context
        if context is None or context[0] != track_name or context[2] != route:
            logger.warning("忽略过期的逐轨 MIDI 错误信号")
            return
        if self.audio_mixer is not None:
            if self._stopping:
                self.audio_mixer.set_track_midi_cancelled(track_name)
            else:
                self.audio_mixer.set_track_midi_failed(
                    track_name,
                    error_message,
                )
        if not self._stopping:
            self.status_label.setText(t("status.error"))
            logger.error(
                "逐轨 MIDI 转写失败: track=%s route=%s error=%s",
                track_name,
                route,
                error_message,
            )

    def _stop_processing(self):
        """请求停止处理，并保持停止态直到工作线程真正退出。"""
        if not self.worker:
            return

        self._stopping = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.track_panel.set_processing_controls_enabled(False)
        self.status_label.setText(t("status.cancelling"))
        self.worker.cancel()
        logger.info("已请求取消；等待工作线程完成清理")

    def _on_worker_thread_finished(self, finished_worker):
        """仅由预期 worker 的 finished 信号解锁 UI 或完成延迟关闭。"""
        if finished_worker is not self.worker:
            logger.warning("忽略非当前工作线程的 finished 信号")
            finished_worker.deleteLater()
            return

        worker = finished_worker
        manual_context = self._manual_midi_context
        self.worker = None
        worker.deleteLater()
        if manual_context is not None and self.audio_mixer is not None:
            if self._stopping:
                self.audio_mixer.set_track_midi_cancelled(manual_context[0])
            self.audio_mixer.set_midi_controls_enabled(True)
        self._manual_midi_context = None

        if getattr(self, "_close_pending", False):
            self._close_pending = False
            self.close()
            return

        if not self._stopping:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.track_panel.set_processing_controls_enabled(True)
            return

        self._stopping = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.track_panel.set_processing_controls_enabled(True)
        self.status_label.setText(t("status.cancelled"))
        logger.info("取消后的工作线程已退出，界面恢复就绪")

    def _on_progress(self, progress: ProcessingProgress):
        """处理进度更新"""
        if self._stopping:
            return
        self.progress_widget.update_progress(progress)
        # 节流内存标签更新（最多每 2 秒一次），避免阻塞 GUI 线程
        now = _time.monotonic()
        if now - self._last_memory_update >= 2.0:
            self._last_memory_update = now
            self._update_memory_label()

    def _on_separation_finished(self, result: SeparationResult):
        """Show separated WAV tracks without starting any MIDI backend."""
        if self._stopping:
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.track_panel.set_processing_controls_enabled(False)
        self.status_label.setText(
            f"{t('status.complete')} - {result.processing_time:.1f}"
            f"{t('dialogs.complete.seconds_suffix')}"
        )
        self._show_separation_result(result)
        logger.info(
            "分离完成，等待逐轨手动转 MIDI: mode=%s tracks=%s",
            result.mode,
            sorted(result.separated_audio),
        )

    def _on_finished(self, result: ProcessingResult):
        """处理完成"""
        if self._stopping:
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.track_panel.set_processing_controls_enabled(False)

        self.status_label.setText(
            f"{t('status.complete')} - {result.processing_time:.1f}"
            f"{t('dialogs.complete.seconds_suffix')}"
        )
        self._show_success_result(result)
        logger.info(f"处理完成: {result.midi_path}")

    def _open_output_folder(self, file_path: str):
        """打开输出文件夹"""
        import subprocess

        folder_path = os.path.dirname(file_path)

        if sys.platform == "win32":
            # Windows: 使用 explorer 打开并选中文件
            subprocess.run(["explorer", "/select,", file_path.replace("/", "\\")])
        elif sys.platform == "darwin":
            # macOS: 使用 Finder 打开
            subprocess.run(["open", "-R", file_path])
        else:
            # Linux: 使用默认文件管理器
            subprocess.run(["xdg-open", folder_path])

    def _open_midi_file(self, file_path: str):
        """使用默认程序打开MIDI文件"""
        import subprocess

        if sys.platform == "win32":
            os.startfile(file_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", file_path])
        else:
            subprocess.run(["xdg-open", file_path])

    def _on_error(self, error_msg: str):
        """处理错误"""
        if self._stopping:
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.track_panel.set_processing_controls_enabled(False)
        self.status_label.setText(t("status.error"))

        # 创建自定义错误对话框
        dialog = QDialog(self)
        dialog.setWindowTitle(t("dialogs.error.title"))
        self._configure_dialog_size(dialog, QSize(560, 360))

        layout = self._create_scrollable_dialog_layout(
            dialog,
            margins=(16, 16, 16, 16),
            spacing=12,
        )

        # 错误提示标签
        error_label = QLabel(t("dialogs.error.processingFailed"))
        error_label.setStyleSheet("font-weight: bold; color: #e05050; font-size: 14px;")
        layout.addWidget(error_label)

        # 错误详情文本框随对话框伸缩，内容自身保持可滚动。
        error_text = QTextEdit()
        error_text.setPlainText(error_msg)
        error_text.setReadOnly(True)
        error_text.setMinimumHeight(120)
        error_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        error_text.setStyleSheet("""
            QTextEdit {
                background: #16213e;
                border: 1px solid #3a4a6a;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Ubuntu Mono', 'DejaVu Sans Mono', Consolas, monospace;
                font-size: 11px;
                color: #e0e0e0;
            }
        """)
        layout.addWidget(error_text)

        # 按钮布局
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        # 复制按钮
        copy_btn = QPushButton(t("dialogs.error.copy"))
        copy_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                background: #2a3f5f;
                border: 1px solid #3a4a6a;
                border-radius: 6px;
                color: #e0e0e0;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #3a5a7c;
            }
        """)
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(error_msg))

        # 确定按钮
        ok_btn = QPushButton(t("dialogs.error.ok"))
        ok_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 20px;
                background: #4a9eff;
                border: none;
                border-radius: 6px;
                color: white;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #5aafff;
            }
        """)
        ok_btn.clicked.connect(dialog.accept)

        btn_layout.addStretch()
        btn_layout.addWidget(copy_btn)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

        dialog.exec()

        logger.error(f"处理错误: {error_msg}")

    def _update_memory_label(self):
        """更新内存使用显示"""
        mem_info = get_memory_info()
        if mem_info:
            self._last_memory_info = mem_info
        else:
            self._last_memory_info = None
        self._render_memory_label()

    def _open_settings(self):
        """打开设置对话框"""
        # TODO: 实现设置对话框
        pass

    def _show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            t("dialogs.about.title"),
            f"{t('app.name')} v{__version__}\n\n"
            f"{t('dialogs.about.description')}\n\n"
            f"{t('dialogs.about.author')}: mason369\n"
            f"{t('dialogs.about.license')}: MIT",
        )

    def _change_language(self, lang_code: str):
        """更改应用程序语言"""
        set_language(lang_code)
        self._update_translations()

    def _on_language_changed(self, index: int):
        """处理语言下拉框更改"""
        lang_code = self.lang_combo.itemData(index)
        if lang_code:
            set_language(lang_code)
            self._update_translations()

    def _update_translations(self):
        """更新当前语言的所有UI文本"""
        self.setWindowTitle(t("app.name"))
        app = QApplication.instance()
        if app:
            app.setApplicationName(t("app.name"))
        self.config.language = get_translator().get_language()

        # 菜单
        self.file_menu.menuAction().setText(t("menu.file"))
        self.edit_menu.menuAction().setText(t("menu.edit"))
        self.view_menu.menuAction().setText(t("menu.view"))
        self.help_menu.menuAction().setText(t("menu.help"))
        self.lang_menu.setTitle(t("settings.general.language"))
        self.open_action.setText(t("menu.open"))
        self.save_action.setText(t("menu.save"))
        self.exit_action.setText(t("menu.exit"))
        self.settings_action.setText(t("menu.settings"))
        self.about_action.setText(t("menu.about"))

        self.title_label.setText(t("app.name"))
        self.subtitle_label.setText(t("app.subtitle"))

        # 输出设置
        self.output_group.setTitle(t("main.output.title"))
        self.output_dir_label.setText(t("main.output.directory") + ":")
        self.browse_dir_btn.setText(t("main.output.browse"))
        self.tracks_check.setText(t("main.output.options.saveTracks"))

        # 按钮
        self._update_start_button_label(self.track_panel.get_processing_mode())
        self.stop_btn.setText("■  " + t("toolbar.stop"))

        # 状态
        self.status_label.setText(t("status.ready"))
        self.device_label.setText(self._format_device_label(self._raw_device_label))
        self._render_memory_label()

        # 组件
        self.dropzone.update_translations()
        self.track_panel.update_translations()
        self.progress_widget.update_translations()

        if self._last_result is not None:
            self._show_success_result(
                self._last_result,
                reveal=False,
                replace_timeline=False,
            )
            if self.audio_mixer is not None:
                self.audio_mixer.update_translations()
            self.status_label.setText(
                f"{t('status.complete')} - {self._last_result.processing_time:.1f}"
                f"{t('dialogs.complete.seconds_suffix')}"
            )
        elif self._last_separation_result is not None:
            self._show_separation_result(
                self._last_separation_result,
                reveal=False,
                replace_timeline=False,
            )
            if self.audio_mixer is not None:
                self.audio_mixer.update_translations()
            self.status_label.setText(
                f"{t('status.complete')} - "
                f"{self._last_separation_result.processing_time:.1f}"
                f"{t('dialogs.complete.seconds_suffix')}"
            )

        logger.info(f"语言已更新为: {get_translator().get_language()}")

    def closeEvent(self, event):
        """窗口关闭时清理后台线程"""
        self._clear_audio_mixer()
        if self.worker and self.worker.isRunning():
            self._close_pending = True
            self._stopping = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.track_panel.set_processing_controls_enabled(False)
            self.status_label.setText(t("status.cancelling"))
            self.worker.cancel()
            event.ignore()
            logger.info("关闭请求已延后；等待工作线程真正退出")
            return
        if hasattr(self, "_gpu_detector") and self._gpu_detector.isRunning():
            self._gpu_detector.wait(1000)
        super().closeEvent(event)
