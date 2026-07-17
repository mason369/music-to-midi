# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置文件

用法: pyinstaller MusicToMidi.spec
"""

import os
import sys
from pathlib import Path
import importlib.util
from PyInstaller.utils.hooks import copy_metadata

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(SPEC))
USER_HOME = str(Path.home())


def _collect_tree(source_dir, target_root):
    items = []
    if not source_dir or not os.path.isdir(source_dir):
        raise FileNotFoundError(
            f"Required portable bundle directory is missing for {target_root}: {source_dir}"
        )
    source_dir = os.path.abspath(source_dir)
    for current_root, dirs, files in os.walk(source_dir):
        dirs[:] = [name for name in dirs if name not in {".git", "__pycache__", ".pytest_cache"}]
        rel = os.path.relpath(current_root, source_dir)
        dest = target_root if rel == "." else os.path.join(target_root, rel)
        for name in files:
            items.append((os.path.join(current_root, name), dest))
    if not items:
        raise RuntimeError(
            f"Required portable bundle directory is empty for {target_root}: {source_dir}"
        )
    return items


def _resolve_existing_dir(*candidates):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def _collect_aria_amt_config_datas():
    amt_spec = importlib.util.find_spec("amt")
    if not amt_spec or not amt_spec.origin:
        raise RuntimeError("Required aria-amt package is not installed")

    config_dir = os.path.abspath(os.path.join(os.path.dirname(amt_spec.origin), "..", "config"))
    config_file = os.path.join(config_dir, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Required Aria-AMT config is missing: {config_file}")

    return _collect_tree(config_dir, "config")


audio_separator_models_dir = _resolve_existing_dir(
    os.environ.get("MUSIC_TO_MIDI_BUNDLE_AUDIO_SEPARATOR_DIR"),
    os.path.join(USER_HOME, ".music-to-midi", "models", "audio-separator"),
)
aria_amt_models_dir = _resolve_existing_dir(
    os.environ.get("MUSIC_TO_MIDI_BUNDLE_ARIA_AMT_DIR"),
    os.path.join(USER_HOME, ".cache", "music_ai_models", "aria_amt"),
)
bytedance_piano_models_dir = _resolve_existing_dir(
    os.environ.get("MUSIC_TO_MIDI_BUNDLE_BYTEDANCE_PIANO_DIR"),
    os.path.join(USER_HOME, ".cache", "music_ai_models", "bytedance_piano"),
)
transkun_v2_aug_models_dir = _resolve_existing_dir(
    os.environ.get("MUSIC_TO_MIDI_BUNDLE_TRANSKUN_V2_AUG_DIR"),
    os.path.join(USER_HOME, ".cache", "music_ai_models", "transkun_v2_aug"),
)
yourmt3_models_dir = _resolve_existing_dir(
    os.environ.get("MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR"),
    os.path.join(USER_HOME, ".cache", "music_ai_models", "yourmt3_all"),
)
miros_source_dir = _resolve_existing_dir(
    os.environ.get("MUSIC_TO_MIDI_BUNDLE_MIROS_DIR"),
    os.path.join(ROOT_DIR, "external", "ai4m-miros"),
    os.path.join(ROOT_DIR, "ai4m-miros"),
    os.path.join(ROOT_DIR, ".tmp", "ai4m-miros"),
)
ffmpeg_dir = _resolve_existing_dir(
    os.environ.get("MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR"),
    os.path.join(ROOT_DIR, "tools", "ffmpeg"),
    os.path.join(ROOT_DIR, "ffmpeg"),
)
torch_lib_dir = None
torch_spec = importlib.util.find_spec("torch")
if torch_spec and torch_spec.origin:
    torch_lib_dir = _resolve_existing_dir(
        os.path.join(os.path.dirname(torch_spec.origin), "lib"),
    )


def _require_ffmpeg_tools(source_dir):
    if not source_dir:
        raise FileNotFoundError("Required FFmpeg bundle directory is missing")
    executable_suffix = ".exe" if os.name == "nt" else ""
    for tool_name in ("ffmpeg", "ffprobe"):
        executable_name = tool_name + executable_suffix
        candidates = (
            os.path.join(source_dir, "bin", executable_name),
            os.path.join(source_dir, executable_name),
        )
        if not any(os.path.isfile(candidate) and os.path.getsize(candidate) > 0 for candidate in candidates):
            raise FileNotFoundError(
                f"Required FFmpeg executable is missing: {executable_name} under {source_dir}"
            )


_require_ffmpeg_tools(ffmpeg_dir)

datas = [
    # 翻译文件
    ('src/i18n/zh_CN.json', 'src/i18n'),
    ('src/i18n/en_US.json', 'src/i18n'),
    # 资源文件（图标等）
    ('resources/icons', 'resources/icons'),
    # Project and embedded third-party license notices.
    ('LICENSE', '.'),
    ('THIRD_PARTY_NOTICES.md', '.'),
]
aria_amt_config_datas = _collect_aria_amt_config_datas()
datas += _collect_tree(os.path.join(ROOT_DIR, "YourMT3", "amt", "src"), "YourMT3/amt/src")
datas += _collect_tree(audio_separator_models_dir, "models/audio-separator")
datas += _collect_tree(aria_amt_models_dir, "models/aria_amt")
datas += _collect_tree(bytedance_piano_models_dir, "models/bytedance_piano")
datas += _collect_tree(transkun_v2_aug_models_dir, "models/transkun_v2_aug")
datas += _collect_tree(yourmt3_models_dir, "models/yourmt3_all")
datas += _collect_tree(miros_source_dir, "external/ai4m-miros")
datas += _collect_tree(ffmpeg_dir, "tools/ffmpeg")
datas += aria_amt_config_datas
datas += copy_metadata('audio-separator')
datas += copy_metadata('aria-amt')
datas += copy_metadata('piano-transcription-inference')
datas += copy_metadata('transkun')
datas += copy_metadata('torchlibrosa')

hiddenimports = [
    # PyQt6 相关
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    # 音频处理
    'librosa',
    'soundfile',
    'audioread',
    # 人声分离（Leap XE vocals + PolarFormer accompaniment）与六声部 BS-RoFormer SW Fixed
    'audio_separator',
    'audio_separator.separator',
    'onnxruntime',
    'PIL',
    'onnx2torch',
    'rotary_embedding_torch',
    'beartype',
    'julius',
    'ml_collections',
    'pydub',
    'samplerate',
    'mir_eval',
    # 钢琴专用转写
    'transkun',
    'transkun.transcribe',
    'amt',
    'amt.run',
    'piano_transcription_inference',
    'torchlibrosa',
    'matplotlib',
    'matplotlib.pyplot',
    # 数值计算
    'numpy',
    'scipy',
    # 其他
    'mido',
    'wandb',
    'pytorch_lightning',
    'lightning_fabric',
    'lightning_utilities',
    'torchmetrics',
]
if importlib.util.find_spec("torch_directml") is not None:
    hiddenimports.append("torch_directml")

# 收集 PyTorch 完整包（含 CUDA 运行时 DLL）
from PyInstaller.utils.hooks import collect_all

audio_sep_datas, audio_sep_binaries, audio_sep_hiddenimports = collect_all('audio_separator')
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
torchaudio_datas, torchaudio_binaries, torchaudio_hiddenimports = collect_all('torchaudio')
torchvision_datas, torchvision_binaries, torchvision_hiddenimports = collect_all('torchvision')
onnxruntime_datas, onnxruntime_binaries, onnxruntime_hiddenimports = collect_all('onnxruntime')
pil_datas, pil_binaries, pil_hiddenimports = collect_all('PIL')
mir_eval_datas, mir_eval_binaries, mir_eval_hiddenimports = collect_all('mir_eval')
transkun_datas, transkun_binaries, transkun_hiddenimports = collect_all('transkun')
aria_amt_datas, aria_amt_binaries, aria_amt_hiddenimports = collect_all('amt')
bytedance_piano_datas, bytedance_piano_binaries, bytedance_piano_hiddenimports = collect_all('piano_transcription_inference')
torchlibrosa_datas, torchlibrosa_binaries, torchlibrosa_hiddenimports = collect_all('torchlibrosa')
matplotlib_datas, matplotlib_binaries, matplotlib_hiddenimports = collect_all('matplotlib')
wandb_datas, wandb_binaries, wandb_hiddenimports = collect_all('wandb')
smart_open_datas, smart_open_binaries, smart_open_hiddenimports = collect_all('smart_open')
einops_datas, einops_binaries, einops_hiddenimports = collect_all('einops')
soundfile_datas, soundfile_binaries, soundfile_hiddenimports = collect_all('soundfile')
pretty_midi_datas, pretty_midi_binaries, pretty_midi_hiddenimports = collect_all('pretty_midi')
soxr_datas, soxr_binaries, soxr_hiddenimports = collect_all('soxr')
mido_datas, mido_binaries, mido_hiddenimports = collect_all('mido')
lightning_datas, lightning_binaries, lightning_hiddenimports = collect_all('pytorch_lightning')
fabric_datas, fabric_binaries, fabric_hiddenimports = collect_all('lightning_fabric')
utilities_datas, utilities_binaries, utilities_hiddenimports = collect_all('lightning_utilities')
torchmetrics_datas, torchmetrics_binaries, torchmetrics_hiddenimports = collect_all('torchmetrics')
datas += (
    audio_sep_datas
    + torch_datas
    + torchaudio_datas
    + torchvision_datas
    + onnxruntime_datas
    + pil_datas
    + mir_eval_datas
    + transkun_datas
    + aria_amt_datas
    + bytedance_piano_datas
    + torchlibrosa_datas
    + matplotlib_datas
    + wandb_datas
    + smart_open_datas
    + einops_datas
    + soundfile_datas
    + pretty_midi_datas
    + soxr_datas
    + mido_datas
    + lightning_datas
    + fabric_datas
    + utilities_datas
    + torchmetrics_datas
)
hiddenimports += (
    audio_sep_hiddenimports
    + torch_hiddenimports
    + torchaudio_hiddenimports
    + torchvision_hiddenimports
    + onnxruntime_hiddenimports
    + pil_hiddenimports
    + mir_eval_hiddenimports
    + transkun_hiddenimports
    + aria_amt_hiddenimports
    + bytedance_piano_hiddenimports
    + torchlibrosa_hiddenimports
    + matplotlib_hiddenimports
    + wandb_hiddenimports
    + smart_open_hiddenimports
    + einops_hiddenimports
    + soundfile_hiddenimports
    + pretty_midi_hiddenimports
    + soxr_hiddenimports
    + mido_hiddenimports
    + lightning_hiddenimports
    + fabric_hiddenimports
    + utilities_hiddenimports
    + torchmetrics_hiddenimports
)

all_binaries = (
    audio_sep_binaries
    + torch_binaries
    + torchaudio_binaries
    + torchvision_binaries
    + onnxruntime_binaries
    + pil_binaries
    + mir_eval_binaries
    + transkun_binaries
    + aria_amt_binaries
    + bytedance_piano_binaries
    + torchlibrosa_binaries
    + matplotlib_binaries
    + wandb_binaries
    + smart_open_binaries
    + einops_binaries
    + soundfile_binaries
    + pretty_midi_binaries
    + soxr_binaries
    + mido_binaries
    + lightning_binaries
    + fabric_binaries
    + utilities_binaries
    + torchmetrics_binaries
)
if torch_lib_dir:
    libomp_dll = os.path.join(torch_lib_dir, "libomp140.x86_64.dll")
    if os.path.exists(libomp_dll):
        all_binaries.append((libomp_dll, "torch/lib"))

# 分析主入口
a = Analysis(
    ['src/main.py'],
    pathex=[ROOT_DIR],
    binaries=all_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除不需要的大型模块以减小体积
        'tkinter',
        'cv2',
        'tensorflow',
        # onnx.reference 在 CUDA + Windows 下导入崩溃（DLL 冲突），运行时不需要
        'onnx.reference',
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
