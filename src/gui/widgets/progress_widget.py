"""
Progress widget showing processing stages.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QGroupBox
)
from PyQt6.QtCore import Qt

from src.i18n.translator import t
from src.models.data_models import ProcessingStage, ProcessingProgress


class StageIndicator(QWidget):
    """Small indicator for a processing stage."""

    def __init__(self, stage: ProcessingStage, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.status = "pending"  # pending, current, done
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.icon_label = QLabel("○")
        self.icon_label.setFixedWidth(20)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.name_label = QLabel(self._get_stage_name())
        self.name_label.setStyleSheet("font-size: 11px;")

        layout.addWidget(self.icon_label)
        layout.addWidget(self.name_label)

        self._update_style()

    def _get_stage_name(self) -> str:
        """Get localized stage name."""
        stage_keys = {
            ProcessingStage.PREPROCESSING: "preprocessing",
            ProcessingStage.SEPARATION: "separation",
            ProcessingStage.TRANSCRIPTION: "transcription",
            ProcessingStage.LYRICS: "lyrics",
            ProcessingStage.SYNTHESIS: "synthesis",
            ProcessingStage.COMPLETE: "complete"
        }
        key = stage_keys.get(self.stage, "")
        return t(f"main.progress.stages.{key}")

    def set_status(self, status: str):
        """Set stage status (pending, current, done)."""
        self.status = status
        self._update_style()

    def _update_style(self):
        """Update visual style based on status."""
        if self.status == "done":
            self.icon_label.setText("✓")
            self.icon_label.setStyleSheet("color: #4a9; font-weight: bold;")
            self.name_label.setStyleSheet("font-size: 11px; color: #4a9;")
        elif self.status == "current":
            self.icon_label.setText("◉")
            self.icon_label.setStyleSheet("color: #49f; font-weight: bold;")
            self.name_label.setStyleSheet("font-size: 11px; color: #49f; font-weight: bold;")
        else:
            self.icon_label.setText("○")
            self.icon_label.setStyleSheet("color: #999;")
            self.name_label.setStyleSheet("font-size: 11px; color: #999;")

    def update_translations(self):
        """Update text for current language."""
        self.name_label.setText(self._get_stage_name())


class ProgressWidget(QGroupBox):
    """Widget showing overall progress and stage indicators."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stage_indicators = {}
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        self.setTitle(t("main.progress.title"))

        layout = QVBoxLayout(self)

        # Current stage label
        self.current_label = QLabel(f"{t('main.progress.current')}: --")

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        # Stage indicators
        stages_layout = QHBoxLayout()
        stages_layout.setSpacing(5)

        stages = [
            ProcessingStage.PREPROCESSING,
            ProcessingStage.SEPARATION,
            ProcessingStage.TRANSCRIPTION,
            ProcessingStage.LYRICS,
            ProcessingStage.SYNTHESIS
        ]

        for i, stage in enumerate(stages):
            indicator = StageIndicator(stage)
            self.stage_indicators[stage] = indicator
            stages_layout.addWidget(indicator)

            # Add arrow between stages
            if i < len(stages) - 1:
                arrow = QLabel("→")
                arrow.setStyleSheet("color: #ccc;")
                stages_layout.addWidget(arrow)

        stages_layout.addStretch()

        layout.addWidget(self.current_label)
        layout.addWidget(self.progress_bar)
        layout.addSpacing(10)
        layout.addLayout(stages_layout)

    def update_progress(self, progress: ProcessingProgress):
        """Update progress display."""
        # Update progress bar
        self.progress_bar.setValue(int(progress.overall_progress * 100))

        # Update current label
        self.current_label.setText(f"{t('main.progress.current')}: {progress.message}")

        # Update stage indicators
        current_stage = progress.stage
        stage_order = [
            ProcessingStage.PREPROCESSING,
            ProcessingStage.SEPARATION,
            ProcessingStage.TRANSCRIPTION,
            ProcessingStage.LYRICS,
            ProcessingStage.SYNTHESIS,
            ProcessingStage.COMPLETE
        ]

        current_idx = stage_order.index(current_stage) if current_stage in stage_order else 0

        for stage, indicator in self.stage_indicators.items():
            stage_idx = stage_order.index(stage) if stage in stage_order else 0

            if stage_idx < current_idx:
                indicator.set_status("done")
            elif stage_idx == current_idx:
                indicator.set_status("current")
            else:
                indicator.set_status("pending")

    def reset(self):
        """Reset progress to initial state."""
        self.progress_bar.setValue(0)
        self.current_label.setText(f"{t('main.progress.current')}: --")

        for indicator in self.stage_indicators.values():
            indicator.set_status("pending")

    def update_translations(self):
        """Update text for current language."""
        self.setTitle(t("main.progress.title"))
        self.current_label.setText(f"{t('main.progress.current')}: --")

        for indicator in self.stage_indicators.values():
            indicator.update_translations()
