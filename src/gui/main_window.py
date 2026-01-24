"""
音乐转MIDI应用程序主窗口
"""
import os
import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QToolBar, QStatusBar,
    QLabel, QPushButton, QCheckBox, QLineEdit,
    QGroupBox, QFileDialog, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon

from src.models.data_models import Config, ProcessingProgress, ProcessingResult, ProcessingStage
from src.gui.widgets.dropzone import DropZoneWidget
from src.gui.widgets.track_panel import TrackPanel
from src.gui.widgets.progress_widget import ProgressWidget
from src.gui.workers.processing_worker import ProcessingWorker
from src.i18n.translator import t, get_translator, set_language
from src.utils.gpu_utils import is_cuda_available, get_gpu_info, get_memory_info

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    应用程序主窗口
    """

    def __init__(self, config: Config = None):
        super().__init__()

        self.config = config or Config()
        self.worker = None
        self.current_file = None

        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()

        self.setWindowTitle(t("app.name"))
        self.setMinimumSize(800, 700)
        self.resize(900, 750)

    def _setup_ui(self):
        """设置主用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 10, 20, 10)
        main_layout.setSpacing(15)

        # 文件输入拖放区域
        self.dropzone = DropZoneWidget()
        self.dropzone.setMinimumHeight(150)

        # 轨道面板
        self.track_panel = TrackPanel()

        # 进度组件
        self.progress_widget = ProgressWidget()

        # 输出设置
        self.output_group = self._create_output_settings()

        # 操作按钮
        self.action_layout = self._create_action_buttons()

        # 添加组件
        main_layout.addWidget(self.dropzone)
        main_layout.addWidget(self.track_panel)
        main_layout.addWidget(self.progress_widget)
        main_layout.addWidget(self.output_group)
        main_layout.addLayout(self.action_layout)

    def _create_output_settings(self) -> QGroupBox:
        """创建输出设置组"""
        group = QGroupBox(t("main.output.title"))
        layout = QVBoxLayout(group)

        # 输出目录
        dir_layout = QHBoxLayout()
        self.output_dir_label = QLabel(t("main.output.directory") + ":")
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(str(Path.home() / "Music" / "MidiOutput"))
        self.browse_dir_btn = QPushButton(t("main.output.browse"))
        self.browse_dir_btn.clicked.connect(self._browse_output_dir)

        dir_layout.addWidget(self.output_dir_label)
        dir_layout.addWidget(self.output_dir_edit, 1)
        dir_layout.addWidget(self.browse_dir_btn)

        # 选项
        options_layout = QHBoxLayout()

        self.midi_check = QCheckBox(t("main.output.options.generateMidi"))
        self.midi_check.setChecked(True)

        self.lyrics_check = QCheckBox(t("main.output.options.embedLyrics"))
        self.lyrics_check.setChecked(True)

        self.lrc_check = QCheckBox(t("main.output.options.exportLrc"))
        self.lrc_check.setChecked(True)

        self.tracks_check = QCheckBox(t("main.output.options.saveTracks"))
        self.tracks_check.setChecked(True)

        options_layout.addWidget(self.midi_check)
        options_layout.addWidget(self.lyrics_check)
        options_layout.addWidget(self.lrc_check)
        options_layout.addWidget(self.tracks_check)
        options_layout.addStretch()

        layout.addLayout(dir_layout)
        layout.addLayout(options_layout)

        return group

    def _create_action_buttons(self) -> QHBoxLayout:
        """创建操作按钮布局"""
        layout = QHBoxLayout()

        self.start_btn = QPushButton(t("toolbar.start"))
        self.start_btn.setFixedWidth(120)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #4a9;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background: #5ba;
            }
            QPushButton:disabled {
                background: #ccc;
            }
        """)

        self.stop_btn = QPushButton(t("toolbar.stop"))
        self.stop_btn.setFixedWidth(100)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #e55;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background: #f66;
            }
            QPushButton:disabled {
                background: #ccc;
            }
        """)

        layout.addStretch()
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addStretch()

        return layout

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
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding,
                            spacer.sizePolicy().verticalPolicy().Preferred)
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

        # GPU/CPU指示器
        if is_cuda_available():
            gpu_info = get_gpu_info()
            gpu_name = gpu_info[0]["name"] if gpu_info else "GPU"
            self.device_label = QLabel(f"{t('status.gpu')}: {gpu_name}")
        else:
            self.device_label = QLabel(f"{t('status.cpu')}")

        # 内存指示器
        self.memory_label = QLabel()
        self._update_memory_label()

        self.statusbar.addWidget(self.status_label, 1)
        self.statusbar.addPermanentWidget(self.device_label)
        self.statusbar.addPermanentWidget(self.memory_label)

    def _connect_signals(self):
        """连接信号与槽"""
        self.dropzone.file_selected.connect(self._on_file_selected)
        self.start_btn.clicked.connect(self._start_processing)
        self.stop_btn.clicked.connect(self._stop_processing)

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

        # 从UI更新配置
        self.config.output_dir = self.output_dir_edit.text()
        self.config.embed_lyrics = self.lyrics_check.isChecked()
        self.config.export_lrc = self.lrc_check.isChecked()
        self.config.save_separated_tracks = self.tracks_check.isChecked()

        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)

        # 创建工作线程
        self.worker = ProcessingWorker(
            self.current_file,
            self.config.output_dir,
            self.config,
            self
        )

        self.worker.progress_updated.connect(self._on_progress)
        self.worker.processing_finished.connect(self._on_finished)
        self.worker.error_occurred.connect(self._on_error)

        # 更新UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_widget.reset()
        self.status_label.setText(t("status.processing"))

        # 开始处理
        self.worker.start()
        logger.info("处理已开始")

    def _stop_processing(self):
        """停止处理"""
        if self.worker:
            self.worker.cancel()
            self.worker.wait()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(t("status.ready"))

    def _on_progress(self, progress: ProcessingProgress):
        """处理进度更新"""
        self.progress_widget.update_progress(progress)
        self._update_memory_label()

    def _on_finished(self, result: ProcessingResult):
        """处理完成"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_action.setEnabled(True)

        self.status_label.setText(
            f"{t('status.complete')} - {result.processing_time:.1f}秒"
        )

        # 显示成功消息
        QMessageBox.information(
            self,
            t("status.complete"),
            f"MIDI: {result.midi_path}\n"
            f"轨道数: {len(result.tracks)}\n"
            f"音符数: {sum(len(track.notes) for track in result.tracks)}\n"
            f"歌词数: {len(result.lyrics)}"
        )

        logger.info(f"处理完成: {result.midi_path}")

    def _on_error(self, error_msg: str):
        """处理错误"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(t("status.error"))

        QMessageBox.critical(
            self,
            t("dialogs.error.title"),
            f"{t('dialogs.error.processingFailed')}\n\n{error_msg}"
        )

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
        self.lyrics_check.setText(t("main.output.options.embedLyrics"))
        self.lrc_check.setText(t("main.output.options.exportLrc"))
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
