"""A height-for-width layout that wraps child widgets onto new rows."""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QLayout, QLayoutItem, QSizePolicy


class FlowLayout(QLayout):
    """Lay widgets left-to-right and wrap them when the row is full."""

    def __init__(
        self,
        parent=None,
        *,
        horizontal_spacing: int = 8,
        vertical_spacing: int = 6,
    ) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._horizontal_spacing = horizontal_spacing
        self._vertical_spacing = vertical_spacing

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt API
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802 - Qt API
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802 - Qt API
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802 - Qt API
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt API
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt API
        minimum_height = max(
            (item.minimumSize().height() for item in self._items),
            default=0,
        )
        margins = self.contentsMargins()
        return QSize(
            margins.left() + margins.right(),
            minimum_height + margins.top() + margins.bottom(),
        )

    def _smart_spacing(
        self, policy: QSizePolicy.ControlType, orientation: Qt.Orientation
    ) -> int:
        parent = self.parentWidget()
        if parent is None:
            return (
                self._horizontal_spacing
                if orientation == Qt.Orientation.Horizontal
                else self._vertical_spacing
            )
        return parent.style().layoutSpacing(
            policy,
            policy,
            orientation,
            None,
            parent,
        )

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(),
            margins.top(),
            -margins.right(),
            -margins.bottom(),
        )
        x = effective.x()
        y = effective.y()
        line_height = 0

        for item in self._items:
            widget = item.widget()
            if widget is not None and widget.isHidden():
                continue

            control_type = (
                widget.sizePolicy().controlType()
                if widget is not None
                else QSizePolicy.ControlType.DefaultType
            )
            space_x = self._horizontal_spacing
            space_y = self._vertical_spacing
            if space_x < 0:
                space_x = self._smart_spacing(control_type, Qt.Orientation.Horizontal)
            if space_y < 0:
                space_y = self._smart_spacing(control_type, Qt.Orientation.Vertical)
            space_x = max(0, space_x)
            space_y = max(0, space_y)

            hint = item.sizeHint()
            item_width = min(hint.width(), max(0, effective.width()))
            item_height = hint.height()
            next_x = x + item_width + space_x
            if line_height > 0 and next_x - space_x > effective.right() + 1:
                x = effective.x()
                y += line_height + space_y
                next_x = x + item_width + space_x
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), QSize(item_width, item_height)))

            x = next_x
            line_height = max(line_height, item_height)

        return y + line_height - rect.y() + margins.bottom()
