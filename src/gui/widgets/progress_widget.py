"""
进度组件 - 显示处理阶段
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QGroupBox
)
from PyQt6.QtCore import Qt

from src.i18n.translator import t
from src.models.data_models import ProcessingStage, ProcessingProgress


class StageIndicator(QWidget):
    """处理阶段的小型指示器"""

    def __init__(self, stage: ProcessingStage, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.status = "pending"  # pending（待处理）, current（当前）, done（完成）
        self._setup_ui()

    def _setup_ui(self):
        """设置用户界面"""
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
        """获取本地化的阶段名称"""
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
        """设置阶段状态（pending待处理, current当前, done完成）"""
        self.status = status
        self._update_style()

    def _update_style(self):
        """根据状态更新视觉样式"""
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
        """更新当前语言的文本"""
        self.name_label.setText(self._get_stage_name())


class ProgressWidget(QGroupBox):
    """显示总体进度和阶段指示器的组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stage_indicators = {}
        self._setup_ui()

    def _setup_ui(self):
        """设置用户界面"""
        self.setTitle(t("main.progress.title"))

        layout = QVBoxLayout(self)

        # 当前阶段标签
        self.current_label = QLabel(f"{t('main.progress.current')}: --")

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        # 阶段指示器
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

            # 阶段之间添加箭头
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
        """更新进度显示"""
        # 更新进度条
        self.progress_bar.setValue(int(progress.overall_progress * 100))

        # 更新当前标签
        self.current_label.setText(f"{t('main.progress.current')}: {progress.message}")

        # 更新阶段指示器
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
        """重置进度到初始状态"""
        self.progress_bar.setValue(0)
        self.current_label.setText(f"{t('main.progress.current')}: --")

        for indicator in self.stage_indicators.values():
            indicator.set_status("pending")

    def update_translations(self):
        """更新当前语言的文本"""
        self.setTitle(t("main.progress.title"))
        self.current_label.setText(f"{t('main.progress.current')}: --")

        for indicator in self.stage_indicators.values():
            indicator.update_translations()
