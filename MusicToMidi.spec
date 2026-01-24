# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置文件

用法: pyinstaller MusicToMidi.spec
"""

import os
import sys
from pathlib import Path

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(SPEC))

# 分析主入口
a = Analysis(
    ['src/main.py'],
    pathex=[ROOT_DIR],
    binaries=[],
    datas=[
        # 翻译文件
        ('src/i18n/zh_CN.json', 'src/i18n'),
        ('src/i18n/en_US.json', 'src/i18n'),
        # 资源文件（图标等）
        ('resources/icons', 'resources/icons'),
    ],
    hiddenimports=[
        # PyQt6 相关
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        # 音频处理
        'librosa',
        'soundfile',
        'audioread',
        # 数值计算
        'numpy',
        'scipy',
        # 其他
        'mido',
    ],
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
