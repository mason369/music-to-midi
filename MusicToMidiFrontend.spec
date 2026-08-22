# -*- mode: python ; coding: utf-8 -*-
"""Small, inference-free executable host for the standalone web frontend."""

import os


ROOT_DIR = os.path.dirname(os.path.abspath(SPEC))

frontend_analysis = Analysis(
    ["src/web_frontend/__main__.py"],
    pathex=[ROOT_DIR],
    binaries=[],
    datas=[
        ("web", "web"),
        ("config/web-frontend.json", "config"),
        ("LICENSE", "."),
        ("THIRD_PARTY_NOTICES.md", "."),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchaudio",
        "tensorflow",
        "PyQt6",
        "gradio",
    ],
    noarchive=False,
    optimize=0,
)

frontend_pyz = PYZ(frontend_analysis.pure)

frontend_exe = EXE(
    frontend_pyz,
    frontend_analysis.scripts,
    [],
    exclude_binaries=True,
    name="MusicToMidiFrontend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="resources/icons/app.ico",
)

frontend_collect = COLLECT(
    frontend_exe,
    frontend_analysis.binaries,
    frontend_analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="MusicToMidiFrontend",
)
