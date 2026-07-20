"""Project-native searchable tag selector for MuScriptor constraints."""

from __future__ import annotations

from PyQt6.QtCore import (
    QAbstractListModel,
    QEvent,
    QModelIndex,
    QPoint,
    QRect,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtWidgets import (
    QCompleter,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from src.i18n.translator import get_translator, t
from src.models.muscriptor_instruments import (
    MUSCRIPTOR_INSTRUMENTS,
    muscriptor_instrument_label,
    validate_muscriptor_instruments,
)


class _FlowLayout(QLayout):
    """Small wrapping layout used for removable instrument tags."""

    def __init__(self, parent=None, spacing: int = 6):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802
        self._items.append(item)

    def insertWidget(self, index: int, widget: QWidget) -> None:  # noqa: N802
        self.addChildWidget(widget)
        self._items.insert(index, QWidgetItem(widget))
        self.invalidate()

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._arrange(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _arrange(self, rect: QRect, *, test_only: bool) -> int:
        x = rect.x()
        y = rect.y()
        line_height = 0
        spacing = self.spacing()
        right = rect.x() + max(0, rect.width())
        for item in self._items:
            hint = item.sizeHint()
            if x > rect.x() and x + hint.width() > right:
                x = rect.x()
                y += line_height + spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(x, y, hint.width(), hint.height()))
            x += hint.width() + spacing
            line_height = max(line_height, hint.height())
        return y - rect.y() + line_height


class _InstrumentListModel(QAbstractListModel):
    def __init__(self, selector: "MuscriptorInstrumentSelector"):
        super().__init__(selector)
        self._selector = selector
        self._items: list[str] = []

    def set_items(self, items: list[str]) -> None:
        self.beginResetModel()
        self._items = items
        self.endResetModel()

    def rowCount(self, _parent=QModelIndex()) -> int:  # noqa: N802
        return len(self._items)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._items):
            return None
        name = self._items[index.row()]
        if role in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole}:
            return self._selector._label(name)
        if role == Qt.ItemDataRole.UserRole:
            return name
        return None

    def canonical_at(self, row: int) -> str:
        return self._items[row]


class _InstrumentTag(QFrame):
    remove_requested = pyqtSignal(str)

    def __init__(self, name: str, label: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.setObjectName("muscriptorInstrumentTag")
        self.setStyleSheet(
            "QFrame#muscriptorInstrumentTag { background: #2a3f5f; "
            "border: 1px solid #4a6d96; border-radius: 4px; } "
            "QLabel { border: none; color: #e0e0e0; font-size: 11px; } "
            "QToolButton { border: none; color: #b0b8c8; background: transparent; "
            "padding: 0; border-radius: 3px; } "
            "QToolButton:hover { color: white; background: #3a5a7c; }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.setSpacing(4)
        self.label = QLabel(label)
        self.remove_button = QToolButton()
        self.remove_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton)
        )
        self.remove_button.setIconSize(QSize(12, 12))
        self.remove_button.setFixedSize(20, 20)
        self.remove_button.setAccessibleName(f"Remove {label}")
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self.name))
        layout.addWidget(self.label)
        layout.addWidget(self.remove_button)


class MuscriptorInstrumentSelector(QFrame):
    """Searchable multi-select whose values are official canonical names."""

    selection_changed = pyqtSignal(list)
    MAX_VISIBLE_ITEMS = 7
    MAX_POPUP_HEIGHT = 224

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected: list[str] = []
        self._tags: dict[str, _InstrumentTag] = {}
        self.setObjectName("muscriptorInstrumentSelector")
        self.setStyleSheet(
            "QFrame#muscriptorInstrumentSelector { background: #17243d; "
            "border: 1px solid #2c4f7c; border-radius: 6px; }"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 7, 10, 9)
        outer.setSpacing(6)
        header = QHBoxLayout()
        header.setSpacing(6)
        heading = QVBoxLayout()
        heading.setSpacing(2)
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #4a9eff;")
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.description_label.setStyleSheet("font-size: 10px; color: #9fb3d9;")
        heading.addWidget(self.title_label)
        heading.addWidget(self.description_label)
        self.clear_button = QPushButton()
        self.clear_button.setStyleSheet(
            "QPushButton { background: #2a3f5f; border: 1px solid #3a4a6a; "
            "border-radius: 4px; color: #d9e4f5; padding: 4px 10px; font-size: 11px; } "
            "QPushButton:hover { background: #3a5a7c; border-color: #4a9eff; } "
            "QPushButton:disabled { background: #202e49; color: #65738a; "
            "border-color: #33425d; }"
        )
        self.clear_button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.clear_button.clicked.connect(lambda: self.set_selected_instruments([]))
        header.addLayout(heading, 1)
        header.addWidget(self.clear_button, 0, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(header)

        self.input_surface = QFrame()
        self.input_surface.setObjectName("muscriptorInputSurface")
        self.input_surface.setStyleSheet(
            "QFrame#muscriptorInputSurface { background: #16213e; border: 1px solid #3a4a6a; "
            "border-radius: 5px; } QLineEdit { border: none; background: transparent; "
            "color: #e0e0e0; selection-background-color: #4a9eff; "
            "font-size: 11px; padding: 5px 4px; min-width: 100px; }"
        )
        self.flow = _FlowLayout(self.input_surface, spacing=6)
        self.flow.setContentsMargins(7, 5, 7, 5)
        self.search_edit = QLineEdit()
        self.search_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.search_edit.setClearButtonEnabled(False)
        self.search_edit.installEventFilter(self)
        self.flow.addWidget(self.search_edit)
        outer.addWidget(self.input_surface)

        self._model = _InstrumentListModel(self)
        self.completer = QCompleter(self._model, self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.completer.setMaxVisibleItems(self.MAX_VISIBLE_ITEMS)
        popup = self.completer.popup()
        popup.setUniformItemSizes(True)
        popup.setStyleSheet(
            "QListView { background: #1f2940; color: #e0e0e0; border: 1px solid #3a4a6a; "
            "outline: none; padding: 2px; font-size: 11px; } "
            "QListView::item { min-height: 30px; padding: 2px 10px; border-radius: 3px; } "
            "QListView::item:selected { background: #4a9eff; color: white; } "
            "QScrollBar:vertical { background: #16213e; width: 10px; margin: 0; } "
            "QScrollBar::handle:vertical { background: #4b6388; min-height: 28px; "
            "border-radius: 4px; margin: 2px; } "
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )
        self.search_edit.setCompleter(self.completer)
        self.completer.activated[QModelIndex].connect(self._on_completion_index)
        self.search_edit.returnPressed.connect(self._add_exact_search_match)
        self.search_edit.textEdited.connect(lambda _text: self._show_completion_popup())
        self.update_translations()
        self._refresh_model()

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self.search_edit and event.type() == QEvent.Type.MouseButtonPress:
            QTimer.singleShot(0, self._show_completion_popup)
        return super().eventFilter(watched, event)

    def _show_completion_popup(self) -> None:
        if not self.isEnabled():
            return
        self.completer.setCompletionPrefix(self.search_edit.text())
        if self.completer.completionModel().rowCount() <= 0:
            self.completer.popup().hide()
            return
        input_offset = self.search_edit.mapTo(self.input_surface, QPoint(0, 0))
        available = self.screen().availableGeometry()
        popup_width = min(self.input_surface.width(), max(160, available.width() - 16))
        popup_rect = QRect(
            -input_offset.x(),
            0,
            popup_width,
            self.search_edit.height(),
        )
        popup = self.completer.popup()
        popup.setMinimumWidth(popup_width)
        popup.setMaximumWidth(popup_width)

        row_height = max(30, popup.sizeHintForRow(0))
        row_count = min(
            self.MAX_VISIBLE_ITEMS,
            self.completer.completionModel().rowCount(),
        )
        desired_height = min(
            self.MAX_POPUP_HEIGHT,
            row_count * row_height + popup.frameWidth() * 2 + 4,
        )
        input_top = self.input_surface.mapToGlobal(QPoint(0, 0)).y()
        input_bottom = self.input_surface.mapToGlobal(QPoint(0, self.input_surface.height())).y()
        available_above = max(0, input_top - available.top() - 8)
        available_below = max(0, available.bottom() - input_bottom - 8)
        available_height = max(available_above, available_below)
        popup_height = min(
            desired_height,
            max(row_height + popup.frameWidth() * 2, available_height),
        )
        popup.setMinimumHeight(popup_height)
        popup.setMaximumHeight(popup_height)
        self.completer.complete(popup_rect)

    def _language(self) -> str:
        return get_translator().get_language()

    def _label(self, name: str) -> str:
        return muscriptor_instrument_label(name, self._language())

    def selected_instruments(self) -> list[str]:
        return list(self._selected)

    def set_selected_instruments(self, values: list[str]) -> None:
        selected = validate_muscriptor_instruments(values)
        if selected == self._selected:
            return
        self._selected = selected
        self._rebuild_tags()
        self._refresh_model()
        self.clear_button.setEnabled(bool(self._selected) and self.isEnabled())
        self.selection_changed.emit(list(self._selected))

    def _refresh_model(self) -> None:
        self._model.set_items(
            [name for name in MUSCRIPTOR_INSTRUMENTS if name not in self._selected]
        )

    def _rebuild_tags(self) -> None:
        while self.flow.count() > 1:
            item = self.flow.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        self._tags.clear()
        for name in self._selected:
            tag = _InstrumentTag(name, self._label(name), self.input_surface)
            tag.remove_requested.connect(self._remove)
            self._tags[name] = tag
            self.flow.insertWidget(self.flow.count() - 1, tag)

    def _remove(self, name: str) -> None:
        self.set_selected_instruments([item for item in self._selected if item != name])

    def _on_completion_index(self, index: QModelIndex) -> None:
        if not index.isValid():
            return
        # ``index`` belongs to QCompleter's filtered proxy model, so its row
        # cannot be applied to the unfiltered source model. QCompleter does not
        # forward custom roles, therefore resolve its displayed localized label
        # back to the one remaining canonical instrument.
        label = index.data(Qt.ItemDataRole.DisplayRole)
        matches = [name for name in self._model._items if self._label(name) == label]
        if len(matches) != 1:
            raise RuntimeError(
                "MuScriptor completion label is not uniquely identifiable: "
                f"{label!r} ({len(matches)} matches)"
            )
        self._add(matches[0])

    def _add_exact_search_match(self) -> None:
        query = self.search_edit.text().strip().lower()
        if not query:
            return
        candidates = [
            name
            for name in MUSCRIPTOR_INSTRUMENTS
            if name not in self._selected and query in {name.lower(), self._label(name).lower()}
        ]
        if len(candidates) == 1:
            self._add(candidates[0])

    def _add(self, name: str) -> None:
        self.set_selected_instruments([*self._selected, name])
        self.search_edit.clear()
        # QCompleter writes the activated display text back into its QLineEdit
        # after emitting activated(). Clear that stale filter on the next event
        # turn, once the completer has finished its own activation handling.
        QTimer.singleShot(0, self._reset_search_after_add)

    def _reset_search_after_add(self) -> None:
        self.search_edit.clear()
        self.completer.setCompletionPrefix("")
        self.search_edit.setFocus()
        # Keep the remaining choices open so selecting several distinct
        # instrument groups is a continuous multi-select interaction.
        self._show_completion_popup()

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        super().setEnabled(enabled)
        self.search_edit.setEnabled(enabled)
        self.clear_button.setEnabled(enabled and bool(self._selected))
        for tag in self._tags.values():
            tag.remove_button.setEnabled(enabled)

    def update_translations(self) -> None:
        self.title_label.setText(t("main.engine.muscriptor_instruments_title"))
        self.description_label.setText(t("main.engine.muscriptor_instruments_desc"))
        self.clear_button.setText(t("main.engine.muscriptor_instruments_clear"))
        # This selector can begin hidden in per-track rows. Give its only
        # QPushButton a compact geometry so narrow-layout bounds checks do not
        # see Qt's unlaid-out 640x480 default rectangle.
        self.clear_button.adjustSize()
        self.search_edit.setPlaceholderText(t("main.engine.muscriptor_instruments_placeholder"))
        self._rebuild_tags()
        self._refresh_model()
