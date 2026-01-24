"""
轨道面板组件 - 支持钢琴模式和智能识别模式
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QGroupBox,
    QRadioButton, QButtonGroup, QSpinBox, QScrollArea, QFrame, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal

from src.i18n.translator import t
from src.models.data_models import (
    TrackType, InstrumentType, TrackConfig, TrackLayout, ProcessingMode
)


class TrackRowV2(QWidget):
    """单个轨道行（新版），支持 TrackConfig"""

    # 乐器类型对应的图标
    ICONS = {
        InstrumentType.PIANO: "🎹",
        InstrumentType.DRUMS: "🥁",
        InstrumentType.BASS: "🎸",
        InstrumentType.GUITAR: "🎸",
        InstrumentType.VOCALS: "🎤",
        InstrumentType.STRINGS: "🎻",
        InstrumentType.OTHER: "🎵",
    }

    # 信号：轨道配置变更
    track_changed = pyqtSignal(str, dict)  # track_id, config_dict

    def __init__(self, track_config: TrackConfig, parent=None):
        super().__init__(parent)
        self.track_config = track_config
        self._setup_ui()

    def _setup_ui(self):
        """设置用户界面"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)

        # 启用复选框
        self.enabled_check = QCheckBox()
        self.enabled_check.setChecked(self.track_config.enabled)
        self.enabled_check.toggled.connect(self._on_enabled_changed)

        # 图标
        icon = self.ICONS.get(self.track_config.instrument, "🎵")
        self.icon_label = QLabel(icon)
        self.icon_label.setFixedWidth(24)

        # 名称
        self.name_label = QLabel(self.track_config.name)
        self.name_label.setMinimumWidth(80)

        # 通道选择
        self.channel_label = QLabel(t("main.tracks.channel"))
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(0, 15)
        self.channel_spin.setValue(self.track_config.midi_channel)
        self.channel_spin.setFixedWidth(50)
        self.channel_spin.valueChanged.connect(self._on_channel_changed)

        # 音色选择
        self.program_label = QLabel(t("main.tracks.program"))
        self.program_combo = QComboBox()
        self._populate_programs()
        self.program_combo.setCurrentIndex(self.track_config.program)
        self.program_combo.currentIndexChanged.connect(self._on_program_changed)

        layout.addWidget(self.enabled_check)
        layout.addWidget(self.icon_label)
        layout.addWidget(self.name_label)
        layout.addStretch()
        layout.addWidget(self.channel_label)
        layout.addWidget(self.channel_spin)
        layout.addWidget(self.program_label)
        layout.addWidget(self.program_combo)

    def _populate_programs(self):
        """填充 GM 音色列表"""
        # 常用音色的简短列表
        programs = [
            (0, t("main.programs.piano")),
            (24, t("main.programs.acoustic_guitar")),
            (25, t("main.programs.steel_guitar")),
            (32, t("main.programs.acoustic_bass")),
            (33, t("main.programs.electric_bass")),
            (48, t("main.programs.strings")),
            (52, t("main.programs.choir")),
            (56, t("main.programs.trumpet")),
            (73, t("main.programs.flute")),
        ]

        for program_num, name in programs:
            self.program_combo.addItem(name, program_num)

        # 设置当前值
        for i in range(self.program_combo.count()):
            if self.program_combo.itemData(i) == self.track_config.program:
                self.program_combo.setCurrentIndex(i)
                break

    def _on_enabled_changed(self, enabled: bool):
        """启用状态变更"""
        self.track_config.enabled = enabled
        self._emit_change()

    def _on_channel_changed(self, channel: int):
        """通道变更"""
        self.track_config.midi_channel = channel
        self._emit_change()

    def _on_program_changed(self, index: int):
        """音色变更"""
        program = self.program_combo.itemData(index)
        if program is not None:
            self.track_config.program = program
            self._emit_change()

    def _emit_change(self):
        """发送变更信号"""
        self.track_changed.emit(self.track_config.id, {
            "enabled": self.track_config.enabled,
            "midi_channel": self.track_config.midi_channel,
            "program": self.track_config.program,
        })

    def update_translations(self):
        """更新翻译"""
        self.channel_label.setText(t("main.tracks.channel"))
        self.program_label.setText(t("main.tracks.program"))


class TrackRow(QWidget):
    """单个轨道行（旧版，保持向后兼容）"""

    ICONS = {
        TrackType.VOCALS: "🎤",
        TrackType.DRUMS: "🥁",
        TrackType.BASS: "🎸",
        TrackType.OTHER: "🎹"
    }

    def __init__(self, track_type: TrackType, parent=None):
        super().__init__(parent)
        self.track_type = track_type
        self._setup_ui()

    def _setup_ui(self):
        """设置用户界面"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        # 图标和名称
        icon = self.ICONS.get(self.track_type, "🎵")
        self.name_label = QLabel(f"{icon} {self._get_track_name()}")
        self.name_label.setMinimumWidth(120)

        # 进度条占位
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("""
            QLabel {
                background: #16213e;
                border-radius: 3px;
                min-height: 20px;
                border: 1px solid #3a4a6a;
            }
        """)

        # 选项
        self.lyrics_check = QCheckBox(t("main.tracks.lyrics"))
        self.midi_check = QCheckBox(t("main.tracks.midi"))
        self.midi_check.setChecked(True)

        # 只为人声显示歌词选项
        if self.track_type != TrackType.VOCALS:
            self.lyrics_check.hide()

        layout.addWidget(self.name_label)
        layout.addWidget(self.progress_label, 1)
        layout.addWidget(self.lyrics_check)
        layout.addWidget(self.midi_check)

    def _get_track_name(self) -> str:
        """获取本地化的轨道名称"""
        names = {
            TrackType.VOCALS: t("main.tracks.vocals"),
            TrackType.DRUMS: t("main.tracks.drums"),
            TrackType.BASS: t("main.tracks.bass"),
            TrackType.OTHER: t("main.tracks.other")
        }
        return names.get(self.track_type, str(self.track_type.value))

    def update_translations(self):
        """更新当前语言的文本"""
        icon = self.ICONS.get(self.track_type, "🎵")
        self.name_label.setText(f"{icon} {self._get_track_name()}")
        self.lyrics_check.setText(t("main.tracks.lyrics"))
        self.midi_check.setText(t("main.tracks.midi"))


class ModeSelector(QWidget):
    """处理模式选择器"""

    mode_changed = pyqtSignal(str)  # "piano" 或 "smart"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """设置用户界面"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.button_group = QButtonGroup(self)

        # 钢琴模式
        self.piano_radio = QRadioButton(t("main.mode.piano"))
        self.piano_radio.setChecked(True)
        self.piano_radio.setToolTip(t("main.mode.piano_tooltip"))
        self.button_group.addButton(self.piano_radio)

        # 智能识别模式
        self.smart_radio = QRadioButton(t("main.mode.smart"))
        self.smart_radio.setToolTip(t("main.mode.smart_tooltip"))
        self.button_group.addButton(self.smart_radio)

        layout.addWidget(self.piano_radio)
        layout.addWidget(self.smart_radio)
        layout.addStretch()

        self.button_group.buttonClicked.connect(self._on_mode_changed)

    def _on_mode_changed(self, button):
        """模式变更"""
        if button == self.piano_radio:
            self.mode_changed.emit("piano")
        else:
            self.mode_changed.emit("smart")

    def get_mode(self) -> str:
        """获取当前模式"""
        return "piano" if self.piano_radio.isChecked() else "smart"

    def set_mode(self, mode: str):
        """设置模式"""
        if mode == "piano":
            self.piano_radio.setChecked(True)
        else:
            self.smart_radio.setChecked(True)

    def update_translations(self):
        """更新翻译"""
        self.piano_radio.setText(t("main.mode.piano"))
        self.piano_radio.setToolTip(t("main.mode.piano_tooltip"))
        self.smart_radio.setText(t("main.mode.smart"))
        self.smart_radio.setToolTip(t("main.mode.smart_tooltip"))


class PianoTrackConfig(QWidget):
    """钢琴轨道数量配置"""

    count_changed = pyqtSignal(int)  # -1 表示自动

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """设置用户界面"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(t("main.piano.track_count"))

        # 自动检测复选框
        self.auto_check = QCheckBox(t("main.piano.auto"))
        self.auto_check.setChecked(True)  # 默认自动
        self.auto_check.setToolTip(t("main.piano.auto_tooltip"))
        self.auto_check.toggled.connect(self._on_auto_toggled)

        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 8)
        self.count_spin.setValue(2)
        self.count_spin.setFixedWidth(60)
        self.count_spin.setEnabled(False)  # 自动模式下禁用
        self.count_spin.valueChanged.connect(lambda v: self.count_changed.emit(v))

        self.hint_label = QLabel(t("main.piano.hint"))
        self.hint_label.setStyleSheet("color: #808080; font-size: 11px;")

        layout.addWidget(self.label)
        layout.addWidget(self.auto_check)
        layout.addWidget(self.count_spin)
        layout.addWidget(self.hint_label)
        layout.addStretch()

    def _on_auto_toggled(self, checked: bool):
        """自动模式切换"""
        self.count_spin.setEnabled(not checked)
        if checked:
            self.count_changed.emit(-1)  # -1 表示自动
        else:
            self.count_changed.emit(self.count_spin.value())

    def get_count(self) -> int:
        """获取钢琴轨道数量，-1 表示自动检测"""
        if self.auto_check.isChecked():
            return -1
        return self.count_spin.value()

    def set_count(self, count: int):
        """设置钢琴轨道数量，-1 表示自动"""
        if count == -1:
            self.auto_check.setChecked(True)
        else:
            self.auto_check.setChecked(False)
            self.count_spin.setValue(count)

    def is_auto(self) -> bool:
        """是否为自动模式"""
        return self.auto_check.isChecked()

    def update_translations(self):
        """更新翻译"""
        self.label.setText(t("main.piano.track_count"))
        self.auto_check.setText(t("main.piano.auto"))
        self.auto_check.setToolTip(t("main.piano.auto_tooltip"))
        self.hint_label.setText(t("main.piano.hint"))


class TrackPanel(QGroupBox):
    """显示所有轨道及选项的面板（新版）"""

    layout_changed = pyqtSignal(object)  # TrackLayout

    def __init__(self, parent=None):
        super().__init__(parent)
        self.track_rows = {}
        self._current_layout: TrackLayout = None
        self._setup_ui()
        self._update_tracks()

    def _setup_ui(self):
        """设置用户界面"""
        self.setTitle(t("main.tracks.title"))
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                color: #e0e0e0;
                border: 1px solid #3a4a6a;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
                background: #1f2940;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
                background: #1f2940;
            }
            QLabel {
                color: #e0e0e0;
            }
            QCheckBox {
                color: #b0b8c8;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
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
            QRadioButton {
                color: #e0e0e0;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 2px solid #3a4a6a;
                background: #16213e;
            }
            QRadioButton::indicator:checked {
                background: #4a9eff;
                border-color: #4a9eff;
            }
            QSpinBox, QComboBox {
                background: #16213e;
                border: 1px solid #3a4a6a;
                border-radius: 4px;
                padding: 4px 8px;
                color: #e0e0e0;
            }
            QSpinBox:focus, QComboBox:focus {
                border-color: #4a9eff;
            }
        """)

        main_layout = QVBoxLayout(self)

        # 模式选择器
        self.mode_selector = ModeSelector()
        self.mode_selector.mode_changed.connect(self._on_mode_changed)
        main_layout.addWidget(self.mode_selector)

        # 钢琴轨道配置
        self.piano_config = PianoTrackConfig()
        self.piano_config.count_changed.connect(self._on_piano_count_changed)
        main_layout.addWidget(self.piano_config)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background: #3a4a6a;")
        main_layout.addWidget(separator)

        # 轨道列表容器
        self.tracks_container = QWidget()
        self.tracks_layout = QVBoxLayout(self.tracks_container)
        self.tracks_layout.setContentsMargins(0, 0, 0, 0)
        self.tracks_layout.setSpacing(2)

        main_layout.addWidget(self.tracks_container)

    def _on_mode_changed(self, mode: str):
        """处理模式变更"""
        # 显示/隐藏钢琴配置
        self.piano_config.setVisible(mode == "piano")

        # 更新轨道列表
        self._update_tracks()

        # 发送变更信号
        if self._current_layout:
            self.layout_changed.emit(self._current_layout)

    def _on_piano_count_changed(self, count: int):
        """处理钢琴轨道数量变更"""
        self._update_tracks()
        if self._current_layout:
            self.layout_changed.emit(self._current_layout)

    def _update_tracks(self):
        """根据当前模式更新轨道列表"""
        # 清除现有轨道行
        for row in self.track_rows.values():
            row.setParent(None)
            row.deleteLater()
        self.track_rows.clear()

        # 清除占位提示
        while self.tracks_layout.count():
            item = self.tracks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        mode = self.mode_selector.get_mode()

        if mode == "piano":
            count = self.piano_config.get_count()
            if count == -1:
                # 自动模式：显示提示，实际轨道数在处理时确定
                self._current_layout = TrackLayout(
                    mode=ProcessingMode.PIANO,
                    tracks=[]  # 空轨道表示自动检测
                )
                hint = QLabel(t("main.piano.auto_hint"))
                hint.setStyleSheet("color: #808080; font-style: italic; padding: 10px;")
                hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.tracks_layout.addWidget(hint)
            else:
                # 固定数量模式
                self._current_layout = TrackLayout.default_piano(count)
        else:
            # 智能模式：暂时显示占位提示
            self._current_layout = TrackLayout(
                mode=ProcessingMode.SMART,
                tracks=[]
            )

        # 添加轨道行
        for track_config in self._current_layout.tracks:
            row = TrackRowV2(track_config)
            row.track_changed.connect(self._on_track_changed)
            self.track_rows[track_config.id] = row
            self.tracks_layout.addWidget(row)

        # 如果是智能模式且没有轨道，显示提示
        if mode == "smart" and not self._current_layout.tracks:
            hint = QLabel(t("main.smart.hint"))
            hint.setStyleSheet("color: #808080; font-style: italic; padding: 10px;")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tracks_layout.addWidget(hint)

    def _on_track_changed(self, track_id: str, config: dict):
        """处理轨道配置变更"""
        # 更新 TrackLayout 中的配置
        for track in self._current_layout.tracks:
            if track.id == track_id:
                track.enabled = config.get("enabled", track.enabled)
                track.midi_channel = config.get("midi_channel", track.midi_channel)
                track.program = config.get("program", track.program)
                break

        self.layout_changed.emit(self._current_layout)

    def get_track_layout(self) -> TrackLayout:
        """获取当前轨道布局"""
        return self._current_layout

    def set_track_layout(self, layout: TrackLayout):
        """设置轨道布局（通常由智能识别后调用）"""
        self._current_layout = layout

        # 清除现有轨道行
        for row in self.track_rows.values():
            row.setParent(None)
            row.deleteLater()
        self.track_rows.clear()

        # 添加新轨道行
        for track_config in layout.tracks:
            row = TrackRowV2(track_config)
            row.track_changed.connect(self._on_track_changed)
            self.track_rows[track_config.id] = row
            self.tracks_layout.addWidget(row)

    def update_translations(self):
        """更新当前语言的文本"""
        self.setTitle(t("main.tracks.title"))
        self.mode_selector.update_translations()
        self.piano_config.update_translations()
        for row in self.track_rows.values():
            if hasattr(row, 'update_translations'):
                row.update_translations()

    def get_selected_tracks(self) -> dict:
        """获取选中的MIDI和歌词轨道（向后兼容）"""
        result = {}
        for track_type in TrackType:
            result[track_type] = {
                "midi": True,
                "lyrics": False
            }
        return result
