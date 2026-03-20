# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置文件

用法: pyinstaller MusicToMidi.spec
"""

import os
import sys
from pathlib import Path
import importlib.util

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(SPEC))
USER_HOME = str(Path.home())


def _collect_tree(source_dir, target_root):
    items = []
    if not source_dir or not os.path.exists(source_dir):
        return items
    source_dir = os.path.abspath(source_dir)
    for current_root, _dirs, files in os.walk(source_dir):
        rel = os.path.relpath(current_root, source_dir)
        dest = target_root if rel == "." else os.path.join(target_root, rel)
        for name in files:
            items.append((os.path.join(current_root, name), dest))
    return items


def _resolve_existing_dir(*candidates):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


audio_separator_models_dir = _resolve_existing_dir(
    os.environ.get("MUSIC_TO_MIDI_BUNDLE_AUDIO_SEPARATOR_DIR"),
    os.path.join(USER_HOME, ".music-to-midi", "models", "audio-separator"),
)
yourmt3_models_dir = _resolve_existing_dir(
    os.environ.get("MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR"),
    os.path.join(USER_HOME, ".cache", "music_ai_models", "yourmt3_all"),
)
aria_amt_models_dir = _resolve_existing_dir(
    os.environ.get("MUSIC_TO_MIDI_BUNDLE_ARIA_DIR"),
    os.path.join(USER_HOME, ".cache", "music_ai_models", "aria_amt"),
)
ffmpeg_dir = _resolve_existing_dir(
    os.environ.get("MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR"),
    os.path.join(ROOT_DIR, "tools", "ffmpeg"),
    os.path.join(ROOT_DIR, "ffmpeg"),
)

datas = [
    # 翻译文件
    ('src/i18n/zh_CN.json', 'src/i18n'),
    ('src/i18n/en_US.json', 'src/i18n'),
    # 资源文件（图标等）
    ('resources/icons', 'resources/icons'),
]
datas += _collect_tree(os.path.join(ROOT_DIR, "YourMT3", "amt", "src"), "YourMT3/amt/src")
datas += _collect_tree(audio_separator_models_dir, "models/audio-separator")
datas += _collect_tree(yourmt3_models_dir, "models/yourmt3_all")
datas += _collect_tree(aria_amt_models_dir, "models/aria_amt")
datas += _collect_tree(ffmpeg_dir, "tools/ffmpeg")

hiddenimports = [
    # PyQt6 相关
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    # 音频处理
    'librosa',
    'soundfile',
    'audioread',
    # 人声分离（audio-separator + BS-RoFormer）
    'audio_separator',
    'audio_separator.separator',
    'onnxruntime',
    'onnx2torch',
    'rotary_embedding_torch',
    'beartype',
    'julius',
    'ml_collections',
    'pydub',
    'samplerate',
    # 数值计算
    'numpy',
    'scipy',
    # 其他
    'mido',
]
if importlib.util.find_spec("torch_directml") is not None:
    hiddenimports.append("torch_directml")

# 分析主入口
a = Analysis(
    ['src/main.py'],
    pathex=[ROOT_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除不需要的大型模块以减小体积
        'matplotlib',
        'tkinter',
        'PIL',
        'cv2',
        'tensorflow',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MusicToMidi',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # 无控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/icons/app.ico',  # 应用图标
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MusicToMidi',
)
