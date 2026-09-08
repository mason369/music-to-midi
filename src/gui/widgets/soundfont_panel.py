"""Result-scoped, explicit SoundFont selection with background file import."""

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QGroupBox, QLabel, QPushButton, QVBoxLayout

from src.core.soundfont_library import import_soundfont, soundfont_selection
from src.gui.layouts import FlowLayout
from src.gui.widgets.wheel_safe_controls import NoWheelComboBox
from src.i18n.translator import t


class _ImportWorker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path, directory, parent):
        super().__init__(parent)
        self.path, self.directory = path, directory

    def run(self):
        try:
            self.succeeded.emit(import_soundfont(self.path, self.directory))
        except Exception as exc:
            self.failed.emit(str(exc))


class SoundFontPanel(QGroupBox):
    apply_requested = pyqtSignal(object)

    def __init__(self, directory: Path, parent=None, *, instrument_label=str):
        super().__init__(parent)
        self.directory = directory
        self.instrument_label = instrument_label
        self.setStyleSheet(
            "QGroupBox { color:#dbeaff; border:1px solid #365f8d; border-radius:6px; margin-top:8px; padding-top:8px; }"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; }"
            "QLabel { color:#9fb3d9; }"
            "QComboBox { color:#e0e0e0; background:#16213e; border:1px solid #3a4a6a; border-radius:4px; padding:5px 7px; }"
            "QComboBox:disabled { color:#73829a; }"
            "QComboBox QAbstractItemView { color:#e0e0e0; background:#16213e; selection-background-color:#365f8d; }"
        )
        self._worker = None
        self._sources = {}
        self._assignments = {}
        self._libraries = {}
        self.library = NoWheelComboBox()
        self.library.addItem("", None)
        self.source = NoWheelComboBox()
        self.preset = NoWheelComboBox()
        for combo in (self.library, self.source, self.preset):
            combo.setMinimumWidth(0)
            combo.setSizeAdjustPolicy(combo.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(14)
        self.import_button = QPushButton()
        self.apply_button = QPushButton()
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.notice)
        row = FlowLayout(horizontal_spacing=8, vertical_spacing=6)
        for widget in (
            self.library,
            self.import_button,
            self.source,
            self.preset,
            self.apply_button,
        ):
            row.addWidget(widget)
        layout.addLayout(row)
        layout.addWidget(self.status)
        self.library.currentIndexChanged.connect(self._library_changed)
        self.source.currentIndexChanged.connect(self._show_preset)
        self.preset.activated.connect(self._remember_preset)
        self.import_button.clicked.connect(self._choose_file)
        self.apply_button.clicked.connect(self._apply)
        self.update_translations()
        self._library_changed()

    def set_notes(self, notes):
        sources = {(note.program, note.is_drum): note.instrument for note in notes}
        if sources == self._sources:
            return
        self._sources = sources
        self.source.clear()
        for key, label in sources.items():
            self.source.addItem(self.instrument_label(label), key)
        self._show_preset()

    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, t("soundfont.import"), "", "SoundFont (*.sf2 *.sf3)"
        )
        if path:
            self.import_file(path)

    def import_file(self, path):
        if self._worker is not None:
            return
        self.import_button.setEnabled(False)
        self.status.setText(t("soundfont.importing"))
        worker = _ImportWorker(path, self.directory, self)
        self._worker = worker
        worker.succeeded.connect(self._imported)
        worker.failed.connect(self._failed)
        worker.finished.connect(self._import_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _imported(self, library):
        if library.sha256 not in self._libraries:
            self._libraries[library.sha256] = library
            self.library.addItem(library.name, library.sha256)
        self.library.setCurrentIndex(self.library.findData(library.sha256))
        self.status.setText(t("soundfont.imported", count=len(library.presets)))

    def _failed(self, error):
        self.status.setText(t("soundfont.failed", error=error))

    def _import_finished(self):
        self._worker = None
        self.import_button.setEnabled(True)

    def _library_changed(self):
        self._assignments = {}
        enabled = self.library.currentData() is not None
        self.source.setEnabled(enabled)
        self.preset.setEnabled(enabled)
        self._show_preset()

    def _show_preset(self):
        self.preset.clear()
        self.preset.addItem(t("soundfont.choose"), None)
        library = self._libraries.get(self.library.currentData())
        source = self.source.currentData()
        if library is None or source is None:
            return
        for preset in library.presets:
            if (preset.bank == 128) == source[1]:
                self.preset.addItem(
                    f"{preset.name} · {preset.bank}:{preset.program}", (preset.bank, preset.program)
                )
        selected = self._assignments.get(source, (128 if source[1] else 0, source[0]))
        # QVariant's Python-object equality can depend on tuple identity.
        index = next(
            (i for i in range(self.preset.count()) if self.preset.itemData(i) == selected), -1
        )
        self.preset.setCurrentIndex(max(0, index))

    def _remember_preset(self):
        source, preset = self.source.currentData(), self.preset.currentData()
        if source is not None:
            if preset is None:
                self._assignments.pop(source, None)
            else:
                self._assignments[source] = preset

    def selection(self):
        library = self._libraries.get(self.library.currentData())
        if library is None:
            return None
        assignments = []
        for source in self._sources:
            bank, program = self._assignments.get(source, (128 if source[1] else 0, source[0]))
            assignments.append(
                dict(source_program=source[0], is_drum=source[1], bank=bank, program=program)
            )
        return soundfont_selection(library, assignments)

    def _apply(self):
        try:
            selection = self.selection()
            self.status.clear()
            self.apply_requested.emit(selection)
        except Exception as exc:
            self._failed(str(exc))

    def update_translations(self):
        self.setTitle(t("soundfont.title"))
        self.library.setItemText(0, t("soundfont.default"))
        self.library.setAccessibleName(t("soundfont.library"))
        self.source.setAccessibleName(t("soundfont.source"))
        self.preset.setAccessibleName(t("soundfont.preset"))
        self.notice.setText(t("soundfont.notice"))
        self.import_button.setText(t("soundfont.import"))
        self.apply_button.setText(t("soundfont.apply"))
        for index, label in enumerate(self._sources.values()):
            self.source.setItemText(index, self.instrument_label(label))

    def shutdown(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()
