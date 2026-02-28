"""
音乐转MIDI应用程序主窗口
"""
import os
import sys
import platform
import logging
import time as _time
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QToolBar, QStatusBar,
    QLabel, QPushButton, QCheckBox, QLineEdit,
    QGroupBox, QFileDialog, QMessageBox, QComboBox,
    QFrame, QSplitter, QGraphicsDropShadowEffect,
    QDialog, QTextEdit, QApplication, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QFont, QPalette, QColor, QPixmap

from src.models.data_models import (
    Config, ProcessingProgress, ProcessingResult, ProcessingStage,
)
from src.gui.widgets.dropzone import DropZoneWidget
from src.gui.widgets.track_panel import TrackPanel
from src.gui.widgets.progress_widget import ProgressWidget
from src.gui.workers.processing_worker import ProcessingWorker
from src.i18n.translator import t, get_translator, set_language, get_resource_path
from src.utils.gpu_utils import get_memory_info

logger = logging.getLogger(__name__)


def get_app_dir() -> Path:
    """
    获取应用程序所在目录（用于输出文件）

    对于打包后的应用，返回可执行文件所在的目录
    对于开发环境，返回项目根目录
    """
    if getattr(sys, 'frozen', False):
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
        self.worker = None
        self.current_file = None
        self._last_memory_update = 0.0  # 内存标签节流时间戳
        self._stopping = False  # 防止停止后信号竞争

        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()
        self._apply_modern_style()

        self.setWindowTitle(t("app.name"))
        self.setMinimumSize(900, 700)
        self.showMaximized()

        # 设置应用图标（Linux使用PNG，Windows使用ICO）
        if platform.system() == "Windows":
            icon_path = get_icon_path("app.ico")
        else:
            icon_path = get_icon_path("app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def _setup_ui(self):
        """设置主用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 6, 12, 6)
        main_layout.setSpacing(6)

        # 顶部标题区域
        header_widget = self._create_header()
        main_layout.addWidget(header_widget)

        # 文件输入拖放区域
        self.dropzone = DropZoneWidget()
        self.dropzone.setMinimumHeight(90)
        self._add_shadow(self.dropzone)

        # 轨道面板
        self.track_panel = TrackPanel()
        self._add_shadow(self.track_panel)

        # 进度组件
        self.progress_widget = ProgressWidget()
        self._add_shadow(self.progress_widget)

        # 输出设置
        self.output_group = self._create_output_settings()
        self._add_shadow(self.output_group)

        # 操作按钮
        self.action_layout = self._create_action_buttons()

        # 添加组件
        main_layout.addWidget(self.dropzone)
        main_layout.addWidget(self.track_panel)
        main_layout.addWidget(self.progress_widget)
        main_layout.addWidget(self.output_group)
        main_layout.addLayout(self.action_layout)

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
            icon_label.setPixmap(pixmap.scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation))
        icon_label.setFixedSize(36, 36)

        # 标题文字
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)

        title_label = QLabel(t("app.name"))
        title_label.setFont(get_ui_font(13, bold=True))
        title_label.setStyleSheet("color: #e0e0e0;")

        subtitle_label = QLabel(t("app.subtitle"))
        subtitle_label.setFont(get_ui_font(8))
        subtitle_label.setStyleSheet("color: #8892a0;")

        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)

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

        # 输出目录
        dir_layout = QHBoxLayout()
        self.output_dir_label = QLabel(t("main.output.directory") + ":")
        self.output_dir_label.setStyleSheet("font-weight: normal; color: #b0b8c8;")
        self.output_dir_edit = QLineEdit()
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

        dir_layout.addWidget(self.output_dir_label)
        dir_layout.addWidget(self.output_dir_edit, 1)
        dir_layout.addWidget(self.browse_dir_btn)

        # 选项
        options_layout = QHBoxLayout()
        options_layout.setSpacing(20)

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

        self.midi_check = QCheckBox(t("main.output.options.generateMidi"))
        self.midi_check.setChecked(True)
        self.midi_check.setStyleSheet(checkbox_style)

        self.tracks_check = QCheckBox(t("main.output.options.saveTracks"))
        self.tracks_check.setChecked(True)
        self.tracks_check.setStyleSheet(checkbox_style)

        options_layout.addWidget(self.midi_check)
        options_layout.addWidget(self.tracks_check)
        options_layout.addStretch()

        layout.addLayout(dir_layout)
        layout.addLayout(options_layout)

        return group

    def _create_action_buttons(self) -> QHBoxLayout:
        """创建操作按钮布局"""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)

        self.start_btn = QPushButton("▶  " + t("toolbar.start"))
        self.start_btn.setFixedSize(120, 34)
        self.start_btn.setFont(get_ui_font(10, bold=True))
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #4a9eff;
                color: white;
                font-weight: bold;
                padding: 8px 18px;
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
        self.stop_btn.setFixedSize(90, 34)
        self.stop_btn.setFont(get_ui_font(9))
        self.stop_btn.setEnabled(False)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #e05050;
                color: white;
                padding: 8px 14px;
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
            QStatusBar {
                background: #16213e;
                border-top: 1px solid #2a2a4a;
                color: #8892a0;
                font-size: 11px;
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
        file_menu = menubar.addMenu(t("menu.file"))

        self.open_action = QAction(t("menu.open"), self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self._open_file)
        file_menu.addAction(self.open_action)

        self.save_action = QAction(t("menu.save"), self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.setEnabled(False)
        file_menu.addAction(self.save_action)

        file_menu.addSeparator()

        self.exit_action = QAction(t("menu.exit"), self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)

        # 编辑菜单
        edit_menu = menubar.addMenu(t("menu.edit"))

        self.settings_action = QAction(t("menu.settings"), self)
        self.settings_action.triggered.connect(self._open_settings)
        edit_menu.addAction(self.settings_action)

        # 视图菜单
        view_menu = menubar.addMenu(t("menu.view"))

        # 语言子菜单
        self.lang_menu = QMenu(t("settings.general.language"), self)
        view_menu.addMenu(self.lang_menu)

        for code, name in get_translator().AVAILABLE_LANGUAGES.items():
            action = QAction(name, self)
            action.setData(code)
            action.triggered.connect(lambda checked, c=code: self._change_language(c))
            self.lang_menu.addAction(action)

        # 帮助菜单
        help_menu = menubar.addMenu(t("menu.help"))

        self.about_action = QAction(t("menu.about"), self)
        self.about_action.triggered.connect(self._show_about)
        help_menu.addAction(self.about_action)

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

        self.lang_combo = QComboBox()
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

        # 状态标签
        self.status_label = QLabel(t("status.ready"))

        # GPU/CPU 指示器（先显示占位符，后台检测后更新）
        self.device_label = QLabel("...")

        # 内存指示器
        self.memory_label = QLabel()

        self.statusbar.addWidget(self.status_label, 1)
        self.statusbar.addPermanentWidget(self.device_label)
        self.statusbar.addPermanentWidget(self.memory_label)

        # 在后台线程中检测 GPU，避免同步 import torch 阻塞 UI 初始化
        self._start_gpu_detection()

    def _start_gpu_detection(self):
        """启动后台 GPU 检测线程"""
        class _GpuDetector(QThread):
            detected = pyqtSignal(str, str)  # (device_label, memory_text)

            def run(self):
                try:
                    from src.utils.gpu_utils import (
                        is_gpu_available, get_accelerator_label, get_memory_info
                    )
                    device_text = get_accelerator_label() if is_gpu_available() else "CPU"
                    mem_info = get_memory_info()
                    if mem_info:
                        used, total = mem_info
                        mem_text = f"显存: {used:.1f}/{total:.1f}GB"
                    else:
                        mem_text = ""
                    self.detected.emit(device_text, mem_text)
                except Exception:
                    self.detected.emit("CPU", "")

        self._gpu_detector = _GpuDetector()
        self._gpu_detector.detected.connect(self._on_gpu_detected)
        self._gpu_detector.start()

    def _on_gpu_detected(self, dev: str, mem: str):
        """处理 GPU 检测结果"""
        self.device_label.setText(dev)
        self.memory_label.setText(mem)

    def _connect_signals(self):
        """连接信号与槽"""
        self.dropzone.file_selected.connect(self._on_file_selected)
        self.start_btn.clicked.connect(self._start_processing)
        self.stop_btn.clicked.connect(self._stop_processing)
        self.track_panel.mode_changed.connect(self.progress_widget.set_mode)

    def _on_file_selected(self, file_path: str):
        """处理文件选择"""
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
            self,
            t("dialogs.openFile.title"),
            "",
            filter_str
        )

        if file_path:
            self._on_file_selected(file_path)

    def _browse_output_dir(self):
        """浏览输出目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            t("dialogs.selectDir.title"),
            self.output_dir_edit.text()
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

        self._stopping = False

        # 从UI更新配置
        self.config.output_dir = self.output_dir_edit.text()
        self.config.save_separated_tracks = self.tracks_check.isChecked()
        self.config.processing_mode = self.track_panel.get_processing_mode()

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

        self.worker = ProcessingWorker(
            self.current_file,
            output_dir_with_music_name,
            self.config,
            self,
        )

        self.worker.progress_updated.connect(self._on_progress)
        self.worker.processing_finished.connect(self._on_finished)
        self.worker.error_occurred.connect(self._on_error)

        # 更新UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.track_panel.mode_combo.setEnabled(False)
        self.progress_widget.reset()
        self.status_label.setText(t("status.processing"))

        # 开始处理
        self.worker.start()
        logger.info("处理已开始")

    def _stop_processing(self):
        """停止处理"""
        if self.worker:
            self._stopping = True
            # 断开信号连接，防止 worker 后续信号导致重复状态变更
            try:
                self.worker.progress_updated.disconnect(self._on_progress)
                self.worker.processing_finished.disconnect(self._on_finished)
                self.worker.error_occurred.disconnect(self._on_error)
            except (TypeError, RuntimeError):
                pass
            self.worker.cancel()
            # 使用短超时避免阻塞 GUI，线程结束后资源由 GC 回收
            if not self.worker.wait(500):
                logger.warning("工作线程未在超时内停止，等待后台结束")

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.track_panel.mode_combo.setEnabled(True)
        self.status_label.setText(t("status.ready"))

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

    def _on_finished(self, result: ProcessingResult):
        """处理完成"""
        if self._stopping:
            return
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.track_panel.mode_combo.setEnabled(True)
        self.save_action.setEnabled(True)

        self.status_label.setText(
            f"{t('status.complete')} - {result.processing_time:.1f}秒"
        )

        # 创建自定义完成对话框
        dialog = QDialog(self)
        dialog.setWindowTitle(t("status.complete"))
        dialog.setMinimumWidth(450)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # 成功图标和标题
        header_layout = QHBoxLayout()
        success_icon = QLabel("✓")
        success_icon.setStyleSheet("""
            QLabel {
                color: #4ade80;
                font-size: 32px;
                font-weight: bold;
            }
        """)
        success_icon.setFixedWidth(50)

        title_label = QLabel(t("dialogs.complete.title"))
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #e0e0e0;")

        header_layout.addWidget(success_icon)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # 结果信息
        if result.vocal_midi_path:
            # 人声分离模式：显示两个 MIDI 文件
            info_text = f"""
            <p style="color: #b0b8c8; line-height: 1.6;">
            <b>{t('dialogs.complete.accompaniment_midi')}:</b> {result.accompaniment_midi_path}<br>
            <b>{t('dialogs.complete.vocal_midi')}:</b> {result.vocal_midi_path}<br>
            <b>处理时间:</b> {result.processing_time:.1f}秒
            </p>
            """
        else:
            info_text = f"""
            <p style="color: #b0b8c8; line-height: 1.6;">
            <b>MIDI文件:</b> {result.midi_path}<br>
            <b>轨道数:</b> {len(result.tracks)}<br>
            <b>音符数:</b> {result.total_notes}<br>
            <b>处理时间:</b> {result.processing_time:.1f}秒
            </p>
            """
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("""
            QLabel {
                background: #16213e;
                border: 1px solid #3a4a6a;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        layout.addWidget(info_label)

        # 按钮布局
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        # 打开文件夹按钮
        open_folder_btn = QPushButton("📁  " + t("dialogs.complete.openFolder"))
        open_folder_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                background: #2a3f5f;
                border: 1px solid #3a4a6a;
                border-radius: 6px;
                color: #e0e0e0;
                font-weight: 500;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #3a5a7c;
                border-color: #4a9eff;
            }
        """)
        open_folder_btn.clicked.connect(lambda: self._open_output_folder(result.midi_path))

        midi_btn_style = """
            QPushButton {
                padding: 10px 20px;
                background: #2a3f5f;
                border: 1px solid #3a4a6a;
                border-radius: 6px;
                color: #e0e0e0;
                font-weight: 500;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #3a5a7c;
                border-color: #4a9eff;
            }
        """

        if result.vocal_midi_path:
            # 人声分离模式：两个打开按钮
            open_acc_btn = QPushButton("🎵  " + t("dialogs.complete.accompaniment_midi"))
            open_acc_btn.setStyleSheet(midi_btn_style)
            open_acc_btn.clicked.connect(lambda: self._open_midi_file(result.accompaniment_midi_path))

            open_vocal_btn = QPushButton("🎤  " + t("dialogs.complete.vocal_midi"))
            open_vocal_btn.setStyleSheet(midi_btn_style)
            open_vocal_btn.clicked.connect(lambda: self._open_midi_file(result.vocal_midi_path))

            btn_layout.addWidget(open_folder_btn)
            btn_layout.addWidget(open_acc_btn)
            btn_layout.addWidget(open_vocal_btn)
        else:
            # 智能模式：单个打开按钮
            open_file_btn = QPushButton("🎵  " + t("dialogs.complete.openFile"))
            open_file_btn.setStyleSheet(midi_btn_style)
            open_file_btn.clicked.connect(lambda: self._open_midi_file(result.midi_path))

            btn_layout.addWidget(open_folder_btn)
            btn_layout.addWidget(open_file_btn)

        btn_layout.addStretch()
        ok_btn = QPushButton(t("dialogs.complete.ok"))
        ok_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 24px;
                background: #4a9eff;
                border: none;
                border-radius: 6px;
                color: white;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #5aafff;
            }
        """)
        ok_btn.clicked.connect(dialog.accept)

        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

        dialog.exec()

        logger.info(f"处理完成: {result.midi_path}")

    def _open_output_folder(self, file_path: str):
        """打开输出文件夹"""
        import subprocess
        folder_path = os.path.dirname(file_path)

        if sys.platform == 'win32':
            # Windows: 使用 explorer 打开并选中文件
            subprocess.run(['explorer', '/select,', file_path.replace('/', '\\')])
        elif sys.platform == 'darwin':
            # macOS: 使用 Finder 打开
            subprocess.run(['open', '-R', file_path])
        else:
            # Linux: 使用默认文件管理器
            subprocess.run(['xdg-open', folder_path])

    def _open_midi_file(self, file_path: str):
        """使用默认程序打开MIDI文件"""
        import subprocess

        if sys.platform == 'win32':
            os.startfile(file_path)
        elif sys.platform == 'darwin':
            subprocess.run(['open', file_path])
        else:
            subprocess.run(['xdg-open', file_path])

    def _on_error(self, error_msg: str):
        """处理错误"""
        if self._stopping:
            return
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.track_panel.mode_combo.setEnabled(True)
        self.status_label.setText(t("status.error"))

        # 创建自定义错误对话框
        dialog = QDialog(self)
        dialog.setWindowTitle(t("dialogs.error.title"))
        dialog.setMinimumWidth(500)
        dialog.setMaximumHeight(400)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 错误提示标签
        error_label = QLabel(t("dialogs.error.processingFailed"))
        error_label.setStyleSheet("font-weight: bold; color: #e05050; font-size: 14px;")
        layout.addWidget(error_label)

        # 错误详情文本框（可滚动，限制高度）
        error_text = QTextEdit()
        error_text.setPlainText(error_msg)
        error_text.setReadOnly(True)
        error_text.setMaximumHeight(200)
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
        copy_text = t("dialogs.error.copy")
        copy_btn = QPushButton(copy_text if copy_text != "dialogs.error.copy" else "复制错误信息")
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
        ok_text = t("dialogs.error.ok")
        ok_btn = QPushButton(ok_text if ok_text != "dialogs.error.ok" else "确定")
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
            used, total = mem_info
            self.memory_label.setText(f"{t('status.memory')}: {used:.1f}/{total:.1f}GB")
        else:
            self.memory_label.setText("")

    def _open_settings(self):
        """打开设置对话框"""
        # TODO: 实现设置对话框
        pass

    def _show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            t("dialogs.about.title"),
            f"{t('app.name')} v1.0.0\n\n"
            f"{t('dialogs.about.description')}\n\n"
            f"{t('dialogs.about.author')}: mason369\n"
            f"{t('dialogs.about.license')}: MIT"
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

        # 菜单
        self.open_action.setText(t("menu.open"))
        self.save_action.setText(t("menu.save"))
        self.exit_action.setText(t("menu.exit"))
        self.settings_action.setText(t("menu.settings"))
        self.about_action.setText(t("menu.about"))

        # 输出设置
        self.output_group.setTitle(t("main.output.title"))
        self.output_dir_label.setText(t("main.output.directory") + ":")
        self.browse_dir_btn.setText(t("main.output.browse"))
        self.midi_check.setText(t("main.output.options.generateMidi"))
        self.tracks_check.setText(t("main.output.options.saveTracks"))

        # 按钮
        self.start_btn.setText(t("toolbar.start"))
        self.stop_btn.setText(t("toolbar.stop"))

        # 状态
        self.status_label.setText(t("status.ready"))

        # 组件
        self.dropzone.update_translations()
        self.track_panel.update_translations()
        self.progress_widget.update_translations()

        logger.info(f"语言已更新为: {get_translator().get_language()}")

    def closeEvent(self, event):
        """窗口关闭时清理后台线程"""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        if hasattr(self, '_gpu_detector') and self._gpu_detector.isRunning():
            self._gpu_detector.wait(1000)
        super().closeEvent(event)
