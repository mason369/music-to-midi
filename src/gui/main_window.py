"""
Main window for the Music to MIDI application.
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
    Main application window.
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
        """Set up the main user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 10, 20, 10)
        main_layout.setSpacing(15)

        # Drop zone for file input
        self.dropzone = DropZoneWidget()
        self.dropzone.setMinimumHeight(150)

        # Track panel
        self.track_panel = TrackPanel()

        # Progress widget
        self.progress_widget = ProgressWidget()

        # Output settings
        self.output_group = self._create_output_settings()

        # Action buttons
        self.action_layout = self._create_action_buttons()

        # Add widgets
        main_layout.addWidget(self.dropzone)
        main_layout.addWidget(self.track_panel)
        main_layout.addWidget(self.progress_widget)
        main_layout.addWidget(self.output_group)
        main_layout.addLayout(self.action_layout)

    def _create_output_settings(self) -> QGroupBox:
        """Create output settings group."""
        group = QGroupBox(t("main.output.title"))
        layout = QVBoxLayout(group)

        # Output directory
        dir_layout = QHBoxLayout()
        self.output_dir_label = QLabel(t("main.output.directory") + ":")
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(str(Path.home() / "Music" / "MidiOutput"))
        self.browse_dir_btn = QPushButton(t("main.output.browse"))
        self.browse_dir_btn.clicked.connect(self._browse_output_dir)

        dir_layout.addWidget(self.output_dir_label)
        dir_layout.addWidget(self.output_dir_edit, 1)
        dir_layout.addWidget(self.browse_dir_btn)

        # Options
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
        """Create action buttons layout."""
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
        """Set up the menu bar."""
        menubar = self.menuBar()

        # File menu
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

        # Edit menu
        edit_menu = menubar.addMenu(t("menu.edit"))

        self.settings_action = QAction(t("menu.settings"), self)
        self.settings_action.triggered.connect(self._open_settings)
        edit_menu.addAction(self.settings_action)

        # View menu
        view_menu = menubar.addMenu(t("menu.view"))

        # Language submenu
        self.lang_menu = QMenu(t("settings.general.language"), self)
        view_menu.addMenu(self.lang_menu)

        for code, name in get_translator().AVAILABLE_LANGUAGES.items():
            action = QAction(name, self)
            action.setData(code)
            action.triggered.connect(lambda checked, c=code: self._change_language(c))
            self.lang_menu.addAction(action)

        # Help menu
        help_menu = menubar.addMenu(t("menu.help"))

        self.about_action = QAction(t("menu.about"), self)
        self.about_action.triggered.connect(self._show_about)
        help_menu.addAction(self.about_action)

    def _setup_toolbar(self):
        """Set up the toolbar."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Language selector on the right
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding,
                            spacer.sizePolicy().verticalPolicy().Preferred)
        toolbar.addWidget(spacer)

        self.lang_combo = QComboBox()
        for code, name in get_translator().AVAILABLE_LANGUAGES.items():
            self.lang_combo.addItem(name, code)

        # Set current language
        current_lang = get_translator().get_language()
        index = self.lang_combo.findData(current_lang)
        if index >= 0:
            self.lang_combo.setCurrentIndex(index)

        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        toolbar.addWidget(QLabel("  "))
        toolbar.addWidget(self.lang_combo)

    def _setup_statusbar(self):
        """Set up the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # Status label
        self.status_label = QLabel(t("status.ready"))

        # GPU/CPU indicator
        if is_cuda_available():
            gpu_info = get_gpu_info()
            gpu_name = gpu_info[0]["name"] if gpu_info else "GPU"
            self.device_label = QLabel(f"{t('status.gpu')}: {gpu_name}")
        else:
            self.device_label = QLabel(f"{t('status.cpu')}")

        # Memory indicator
        self.memory_label = QLabel()
        self._update_memory_label()

        self.statusbar.addWidget(self.status_label, 1)
        self.statusbar.addPermanentWidget(self.device_label)
        self.statusbar.addPermanentWidget(self.memory_label)

    def _connect_signals(self):
        """Connect signals and slots."""
        self.dropzone.file_selected.connect(self._on_file_selected)
        self.start_btn.clicked.connect(self._start_processing)
        self.stop_btn.clicked.connect(self._stop_processing)

    def _on_file_selected(self, file_path: str):
        """Handle file selection."""
        self.current_file = file_path
        self.start_btn.setEnabled(True)
        self.status_label.setText(f"{t('status.ready')} - {Path(file_path).name}")
        logger.info(f"File selected: {file_path}")

    def _open_file(self):
        """Open file dialog."""
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
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            t("dialogs.selectDir.title"),
            self.output_dir_edit.text()
        )

        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _start_processing(self):
        """Start processing the audio file."""
        if not self.current_file:
            return

        # Update config from UI
        self.config.output_dir = self.output_dir_edit.text()
        self.config.embed_lyrics = self.lyrics_check.isChecked()
        self.config.export_lrc = self.lrc_check.isChecked()
        self.config.save_separated_tracks = self.tracks_check.isChecked()

        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Create worker
        self.worker = ProcessingWorker(
            self.current_file,
            self.config.output_dir,
            self.config,
            self
        )

        self.worker.progress_updated.connect(self._on_progress)
        self.worker.processing_finished.connect(self._on_finished)
        self.worker.error_occurred.connect(self._on_error)

        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_widget.reset()
        self.status_label.setText(t("status.processing"))

        # Start processing
        self.worker.start()
        logger.info("Processing started")

    def _stop_processing(self):
        """Stop processing."""
        if self.worker:
            self.worker.cancel()
            self.worker.wait()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(t("status.ready"))

    def _on_progress(self, progress: ProcessingProgress):
        """Handle progress update."""
        self.progress_widget.update_progress(progress)
        self._update_memory_label()

    def _on_finished(self, result: ProcessingResult):
        """Handle processing completion."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_action.setEnabled(True)

        self.status_label.setText(
            f"{t('status.complete')} - {result.processing_time:.1f}s"
        )

        # Show success message
        QMessageBox.information(
            self,
            t("status.complete"),
            f"MIDI: {result.midi_path}\n"
            f"Tracks: {len(result.tracks)}\n"
            f"Notes: {sum(len(track.notes) for track in result.tracks)}\n"
            f"Lyrics: {len(result.lyrics)}"
        )

        logger.info(f"Processing complete: {result.midi_path}")

    def _on_error(self, error_msg: str):
        """Handle processing error."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(t("status.error"))

        QMessageBox.critical(
            self,
            t("dialogs.error.title"),
            f"{t('dialogs.error.processingFailed')}\n\n{error_msg}"
        )

        logger.error(f"Processing error: {error_msg}")

    def _update_memory_label(self):
        """Update memory usage display."""
        mem_info = get_memory_info()
        if mem_info:
            used, total = mem_info
            self.memory_label.setText(f"{t('status.memory')}: {used:.1f}/{total:.1f}GB")
        else:
            self.memory_label.setText("")

    def _open_settings(self):
        """Open settings dialog."""
        # TODO: Implement settings dialog
        pass

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            t("dialogs.about.title"),
            f"{t('app.name')} v1.0.0\n\n"
            f"{t('dialogs.about.description')}\n\n"
            f"{t('dialogs.about.author')}: mason369\n"
            f"{t('dialogs.about.license')}: MIT"
        )

    def _change_language(self, lang_code: str):
        """Change application language."""
        set_language(lang_code)
        self._update_translations()

    def _on_language_changed(self, index: int):
        """Handle language combo box change."""
        lang_code = self.lang_combo.itemData(index)
        if lang_code:
            set_language(lang_code)
            self._update_translations()

    def _update_translations(self):
        """Update all UI text for current language."""
        self.setWindowTitle(t("app.name"))

        # Menu
        self.open_action.setText(t("menu.open"))
        self.save_action.setText(t("menu.save"))
        self.exit_action.setText(t("menu.exit"))
        self.settings_action.setText(t("menu.settings"))
        self.about_action.setText(t("menu.about"))

        # Output settings
        self.output_group.setTitle(t("main.output.title"))
        self.output_dir_label.setText(t("main.output.directory") + ":")
        self.browse_dir_btn.setText(t("main.output.browse"))
        self.midi_check.setText(t("main.output.options.generateMidi"))
        self.lyrics_check.setText(t("main.output.options.embedLyrics"))
        self.lrc_check.setText(t("main.output.options.exportLrc"))
        self.tracks_check.setText(t("main.output.options.saveTracks"))

        # Buttons
        self.start_btn.setText(t("toolbar.start"))
        self.stop_btn.setText(t("toolbar.stop"))

        # Status
        self.status_label.setText(t("status.ready"))

        # Widgets
        self.dropzone.update_translations()
        self.track_panel.update_translations()
        self.progress_widget.update_translations()

        logger.info(f"Language updated to: {get_translator().get_language()}")
