"""Qt controls whose values never change from mouse-wheel events."""

from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QSlider, QSpinBox


class NoWheelComboBox(QComboBox):
    """Require an explicit click or keyboard action to change the selection."""

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        event.ignore()


class NoWheelSlider(QSlider):
    """Keep page scrolling from accidentally changing a slider value."""

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """Keep scrolling from modifying selected notes' velocity."""

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Require deliberate input for tempo, rate and zoom changes."""

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        event.ignore()
