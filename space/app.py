"""
Music to MIDI - Gradio Web 界面
视觉风格对齐 PyQt6 桌面版暗色主题
"""
import os
import sys
import tempfile
import logging
from pathlib import Path

# ── 增加递归限制（Python 3.12 循环导入问题）──
sys.setrecursionlimit(3000)

# ── 环境变量（必须在 import torch 前设置）──
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

# ── 修复 torchvision 循环导入问题（必须在任何其他导入前）──
# Python 3.12 + torch 2.10 环境下，torchvision._meta_registrations 会在初始化时
# 访问 torchvision.extension，导致 AttributeError。解决方案：
# 1. 增加递归限制到 3000（默认 1000）
# 2. 使用 import hook 拦截所有 torchvision 导入，返回假模块
# 3. 项目不依赖 torchvision，此方案不影响核心功能
import warnings
import types
import importlib.abc
import importlib.machinery

warnings.filterwarnings("ignore", message=".*partially initialized module.*torchvision.*")
warnings.filterwarnings("ignore", message=".*torchvision.*")

# 创建假的 torchvision 模块
_fake_torchvision = types.ModuleType("torchvision")
_fake_torchvision.extension = types.ModuleType("extension")
_fake_torchvision.extension._has_ops = lambda: False
_fake_torchvision._meta_registrations = types.ModuleType("_meta_registrations")
_fake_torchvision._meta_registrations.register_meta = lambda *a, **k: lambda fn: fn

# 自定义 import hook，拦截所有 torchvision 导入
class TorchvisionBlocker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("torchvision"):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        if spec.name == "torchvision":
            return _fake_torchvision
        elif spec.name == "torchvision.extension":
            return _fake_torchvision.extension
        elif spec.name == "torchvision._meta_registrations":
            return _fake_torchvision._meta_registrations
        # 其他 torchvision 子模块返回空模块
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        pass  # 不执行任何初始化代码

# 注册 import hook（必须在导入任何其他库之前）
sys.meta_path.insert(0, TorchvisionBlocker())

try:
    # 现在可以安全导入 torch
    import torch  # noqa: F401

except Exception as e:
    # 导入失败不影响后续流程，但记录日志
    logging.getLogger(__name__).warning(f"torch/torchvision setup warning (non-critical): {e}")

# ── 实时日志文件（供前端轮询读取）──
LOG_FILE = "/tmp/midi_process.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("music-to-midi-web")

# ── 健壮的日志文件 Handler（每次写入重新打开文件，不依赖持久化 stream）──
# 原因：标准 FileHandler 的 stream 会被 spaces/Gradio 关闭（stream=None），
# 导致 logging 框架的写入全部丢失。此 Handler 每次 emit 都 open→write→close，
# 虽然性能略低，但完全不受 stream 生命周期影响。
class _RobustFileHandler(logging.Handler):
    def __init__(self, filename, encoding="utf-8"):
        super().__init__()
        self.filename = filename
        self.encoding = encoding

    def emit(self, record):
        try:
            msg = self.format(record)
            with open(self.filename, "a", encoding=self.encoding) as f:
                f.write(msg + "\n")
        except Exception:
            pass


_file_handler = _RobustFileHandler(LOG_FILE)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
))
_file_handler.setLevel(logging.INFO)
for _name in ("music-to-midi-web", "src.core", "src.utils"):
    logging.getLogger(_name).addHandler(_file_handler)


def clear_logs():
    """清空日志文件"""
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass


def read_logs():
    """读取日志文件的最新内容（供前端轮询）"""
    try:
        file_size = os.path.getsize(LOG_FILE) if os.path.exists(LOG_FILE) else 0
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        content = content.replace("\x00", "")
        lines = content.strip().split("\n")

        if lines and lines[0]:
            return "\n".join(lines[-50:])
        return ""
    except Exception as e:
        return f"[read_logs error] {e}"


# ── 确保 YourMT3 源码可用 ──
def ensure_yourmt3_code():
    """从 HF Space 仓库下载 YourMT3 源代码（GitHub 仓库不含源码）"""
    yourmt3_dir = os.path.join(APP_DIR, "YourMT3")
    amt_src = os.path.join(yourmt3_dir, "amt", "src")

    if not os.path.exists(os.path.join(amt_src, "model", "ymt3.py")):
        logger.info("Downloading YourMT3 source code from HF Space...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            "mimbres/YourMT3",
            repo_type="space",
            local_dir=yourmt3_dir,
            allow_patterns=["amt/src/**"],
            ignore_patterns=["*.ckpt", "*.bin", "*.safetensors",
                             "amt/logs/**", "*.DS_Store"],
        )
        logger.info("YourMT3 source code downloaded")
    else:
        logger.info("YourMT3 source code already present")

    # 关键：在导入任何 src 模块前，将 YourMT3/amt/src 加入 sys.path
    if os.path.exists(amt_src) and amt_src not in sys.path:
        sys.path.insert(0, amt_src)
        logger.info(f"Added YourMT3 path: {amt_src}")

    # 验证导入可用
    try:
        from model.ymt3 import YourMT3  # noqa: F401
        logger.info("YourMT3 code import OK")
    except Exception as e:
        logger.warning(f"YourMT3 code import failed: {e}")


try:
    ensure_yourmt3_code()
except Exception as e:
    logger.warning(f"Failed to setup YourMT3: {e}")


# ── Monkey-patch: 修复 gradio_client schema 解析 bug ──
try:
    import gradio_client.utils as _gcu
    _original_json_schema = _gcu._json_schema_to_python_type

    def _patched_json_schema(schema, defs=None):
        if isinstance(schema, bool):
            return "bool"
        return _original_json_schema(schema, defs)

    _gcu._json_schema_to_python_type = _patched_json_schema
except Exception:
    pass


# ── ZeroGPU 支持 ──
try:
    import spaces
    ZERO_GPU = True
    logger.info("ZeroGPU (spaces) available")
except ImportError:
    ZERO_GPU = False
    logger.info("Running without ZeroGPU")


import gradio as gr


# ── 延迟导入核心模块（在 YourMT3 路径就绪后）──
from src.models.data_models import Config, ProcessingStage
from src.core.yourmt3_transcriber import YourMT3Transcriber


# ── 模型预下载（仅下载权重，不加载到内存）──
def ensure_model_weights():
    """确保模型权重已下载（不检查 is_available，只检查文件）"""
    try:
        from src.utils.yourmt3_downloader import get_model_path, DEFAULT_MODEL
        model_path = get_model_path(DEFAULT_MODEL)
        if model_path and model_path.exists():
            logger.info(f"Model weights found: {model_path}")
            return
    except Exception:
        pass

    logger.info("Model weights not found, downloading...")
    try:
        from download_sota_models import download_ultimate_moe
        download_ultimate_moe()
        logger.info("Model weights downloaded")
    except Exception as e:
        logger.warning(f"Model download failed (will retry on first use): {e}")


try:
    ensure_model_weights()
except Exception as e:
    logger.warning(f"Model preload failed: {e}")


def ensure_aria_amt_weights():
    try:
        from download_aria_amt_model import is_aria_model_available, download_aria_model
    except Exception as exc:
        logger.warning(f"Aria-AMT downloader unavailable: {exc}")
        return

    if is_aria_model_available():
        logger.info("Aria-AMT checkpoint found")
        return

    logger.info("Aria-AMT checkpoint not found, downloading...")
    try:
        download_aria_model()
        logger.info("Aria-AMT checkpoint downloaded")
    except Exception as e:
        logger.warning(f"Aria-AMT checkpoint download failed: {e}")

# 启动完成后清空日志，避免启动日志残留在用户界面
clear_logs()


# ── 设备信息 ──
def get_device_label():
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU ({torch.cuda.get_device_name(0)})"
        return "CPU"
    except Exception:
        return "CPU"


# ── 核心转换函数 ──
def _convert_impl(
    audio_path,
    mode,
    quality,
    vocal_split_merge_midi=False,
    six_stem_only_selected=False,
    six_stem_targets=None,
    six_stem_vocal_harmony=False,
    progress=gr.Progress(),
):
    """实际执行转换的内部函数"""
    import threading
    from src.core.pipeline import MusicToMidiPipeline
    from src.utils.gpu_utils import clear_gpu_memory

    if audio_path is None:
        raise gr.Error("请先上传音频文件")

    # 诊断：直接写入日志文件，绕过 logging 框架
    def _write_log(msg):
        """直接写入日志文件（不依赖 logging handler）"""
        try:
            import datetime
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{ts} [INFO] {msg}\n")
                f.flush()
        except Exception:
            pass

    _write_log(f"[diag] pid={os.getpid()} thread={threading.current_thread().name}")
    _write_log(f"[diag] LOG_FILE={LOG_FILE} exists={os.path.exists(LOG_FILE)}")

    # 清空旧日志，开始新的处理
    clear_logs()
    _write_log("=" * 40)
    _write_log("开始音频转 MIDI 处理")
    _write_log(f"音频文件: {Path(audio_path).name}")
    _write_log(f"处理模式: {mode}")
    _write_log(f"转写质量: {quality}")
    _write_log("=" * 40)

    # 同时用 logging 框架写入（对比测试）
    logger.info("[logging framework] 开始处理")

    config = Config()
    mode_mapping = {
        "YourMT3+ 多乐器转写": "smart",
        "人声分离 + 分别转写": "vocal_split",
        "六声部分离 + 分别转写": "six_stem_split",
        "钢琴专用转写 (Aria-AMT)": "piano_aria_amt",
    }
    config.processing_mode = mode_mapping.get(mode, "smart")
    config.transcription_quality = quality
    config.vocal_split_merge_midi = bool(
        config.processing_mode == "vocal_split" and vocal_split_merge_midi
    )
    if config.processing_mode == "six_stem_split" and six_stem_only_selected:
        config.six_stem_targets = [str(stem).lower() for stem in (six_stem_targets or [])]
    else:
        config.six_stem_targets = []
    config.six_stem_split_vocal_harmony = bool(
        config.processing_mode == "six_stem_split" and six_stem_vocal_harmony
    )

    # 按模式准备所需模型
    if config.processing_mode == "piano_aria_amt":
        ensure_aria_amt_weights()
    else:
        ensure_model_weights()

    pipeline = MusicToMidiPipeline(config)
    output_dir = tempfile.mkdtemp(prefix="midi_output_")

    def on_progress(p):
        stage_name = {
            ProcessingStage.PREPROCESSING: "预处理",
            ProcessingStage.SEPARATION: "音源分离",
            ProcessingStage.TRANSCRIPTION: "音频转写",
            ProcessingStage.VOCAL_TRANSCRIPTION: "人声转写",
            ProcessingStage.SYNTHESIS: "MIDI合成",
            ProcessingStage.COMPLETE: "完成",
        }.get(p.stage, str(p.stage))
        progress(p.overall_progress, desc=f"[{stage_name}] {p.message}")

    try:
        result = pipeline.process(
            audio_path=audio_path,
            output_dir=output_dir,
            progress_callback=on_progress,
        )
    except Exception as e:
        logger.error(f"转换失败: {e}")
        raise gr.Error(f"转换失败: {e}")
    finally:
        try:
            clear_gpu_memory()
        except Exception:
            pass

    # 收集输出文件
    output_files = []
    if result.midi_path and Path(result.midi_path).exists():
        output_files.append(result.midi_path)
    if result.stem_midi_paths:
        for stem_name, stem_midi_path in result.stem_midi_paths.items():
            if stem_midi_path and Path(stem_midi_path).exists() and stem_midi_path not in output_files:
                output_files.append(stem_midi_path)
    if result.vocal_midi_path and Path(result.vocal_midi_path).exists():
        output_files.append(result.vocal_midi_path)
    if result.accompaniment_midi_path and Path(result.accompaniment_midi_path).exists():
        if result.accompaniment_midi_path != result.midi_path:
            output_files.append(result.accompaniment_midi_path)
    # 分离后的音频文件（人声 + 伴奏 WAV）
    if result.separated_audio:
        for audio_key, audio_file in result.separated_audio.items():
            if audio_file and Path(audio_file).exists():
                output_files.append(audio_file)

    # 构建状态文本
    device_label = get_device_label()
    bpm_str = f"{result.beat_info.bpm:.1f}" if result.beat_info else "N/A"
    status_lines = [
        "--- 转换完成 ---",
        f"耗时: {result.processing_time:.1f} 秒",
        f"总音符数: {result.total_notes}",
        f"BPM: {bpm_str}",
        f"设备: {device_label}",
    ]
    if result.stem_midi_paths:
        status_lines.append(f"合并 MIDI: {Path(result.midi_path).name}")
        status_lines.append(f"分 stem MIDI: {len(result.stem_midi_paths)} 个")
    elif result.vocal_midi_path:
        status_lines.append(f"伴奏 MIDI: {Path(result.accompaniment_midi_path).name}")
        status_lines.append(f"人声 MIDI: {Path(result.vocal_midi_path).name}")
        if result.merged_midi_path:
            status_lines.append(f"合并 MIDI: {Path(result.merged_midi_path).name}")
    else:
        status_lines.append(f"MIDI 文件: {Path(result.midi_path).name}")

    logger.info("转换完成!")
    return output_files, "\n".join(status_lines)


# ── 根据环境包装 GPU 装饰器 ──
if ZERO_GPU:
    @spaces.GPU
    def convert_audio_to_midi(
        audio_path,
        mode,
        quality,
        vocal_split_merge_midi=False,
        six_stem_only_selected=False,
        six_stem_targets=None,
        six_stem_vocal_harmony=False,
        progress=gr.Progress(),
    ):
        return _convert_impl(
            audio_path,
            mode,
            quality,
            vocal_split_merge_midi,
            six_stem_only_selected,
            six_stem_targets,
            six_stem_vocal_harmony,
            progress,
        )
else:
    def convert_audio_to_midi(
        audio_path,
        mode,
        quality,
        vocal_split_merge_midi=False,
        six_stem_only_selected=False,
        six_stem_targets=None,
        six_stem_vocal_harmony=False,
        progress=gr.Progress(),
    ):
        return _convert_impl(
            audio_path,
            mode,
            quality,
            vocal_split_merge_midi,
            six_stem_only_selected,
            six_stem_targets,
            six_stem_vocal_harmony,
            progress,
        )


# ── 模式切换时更新说明 ──
def update_mode_info(mode):
    if mode == "人声分离 + 分别转写":
        return (
            "**BS-RoFormer + YourMT3+** — 先用 BS-RoFormer 分离人声与伴奏，"
            "再分别用 YourMT3+ 转写；默认输出两个独立 MIDI，可选额外输出 1 个合并 MIDI。"
        )
    if mode == "六声部分离 + 分别转写":
        return (
            "**BS Roformer SW + YourMT3+** — 分离 bass/drums/guitar/piano/vocals/other 六个 stem，"
            "默认转写 6 个 stem（可切换为仅转写选中 stem），并额外生成 1 个合并 MIDI。"
        )
    if mode == "钢琴专用转写 (Aria-AMT)":
        return (
            "**Aria-AMT (Piano)** — 针对独奏钢琴场景的专用转写模型，"
            "直接输出钢琴 MIDI（本地权重可下载）。"
        )
    return (
        "**YourMT3+ MoE** — 直接对完整音频进行多乐器转写，"
        "精确识别 128 种 GM 乐器，轨道数量由模型自动决定。"
    )


def update_mode_controls(mode, only_selected):
    is_vocal = mode == "人声分离 + 分别转写"
    is_six = mode == "六声部分离 + 分别转写"
    return (
        update_mode_info(mode),
        gr.update(visible=is_vocal),
        gr.update(visible=is_six),
        gr.update(visible=(is_six and bool(only_selected))),
        gr.update(visible=is_six),
    )


def update_six_stem_targets_visibility(mode, only_selected):
    return gr.update(visible=(mode == "六声部分离 + 分别转写" and bool(only_selected)))


# ── 自定义 CSS（对齐 PyQt6 暗色主题）──
CUSTOM_CSS = """
.gradio-container {
    background: #1a1a2e !important;
    max-width: 1100px !important;
}
.app-header {
    background: #16213e;
    border-bottom: 2px solid #2a2a4a;
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 12px;
}
.app-header h1 {
    color: #e0e0e0 !important;
    font-size: 22px !important;
    margin: 0 !important;
}
.app-header p {
    color: #8892a0 !important;
    font-size: 13px !important;
    margin: 4px 0 0 0 !important;
}
.upload-zone {
    background: #1f2940 !important;
    border: 2px dashed #3a4a6a !important;
    border-radius: 16px !important;
    min-height: 120px !important;
}
.upload-zone:hover {
    border-color: #4a9eff !important;
    background: #2a3f5f !important;
}
.convert-btn {
    background: #4a9eff !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 15px !important;
    padding: 12px 32px !important;
    border-radius: 10px !important;
    border: none !important;
    min-height: 48px !important;
}
.convert-btn:hover {
    background: #5aafff !important;
}
.result-box textarea {
    background: #16213e !important;
    color: #e0e0e0 !important;
    border: 1px solid #3a4a6a !important;
    border-radius: 8px !important;
    font-family: 'Consolas', 'Ubuntu Mono', monospace !important;
    font-size: 13px !important;
}
.log-box textarea {
    background: #0d1117 !important;
    color: #8dc891 !important;
    border: 1px solid #2a3a4a !important;
    border-radius: 8px !important;
    font-family: 'Consolas', 'Ubuntu Mono', monospace !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
}
.section-title {
    color: #e0e0e0 !important;
    font-weight: bold !important;
    font-size: 13px !important;
    border-bottom: 1px solid #3a4a6a;
    padding-bottom: 6px;
    margin-bottom: 10px;
}
.mode-info {
    background: #16213e;
    border: 1px solid #3a4a6a;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 8px;
}
.mode-info p {
    color: #4a9eff !important;
    font-size: 12px !important;
    margin: 0 !important;
}
.device-badge {
    background: #16213e;
    border: 1px solid #3a4a6a;
    border-radius: 6px;
    padding: 6px 12px;
    text-align: center;
}
.device-badge p {
    color: #8892a0 !important;
    font-size: 11px !important;
    margin: 0 !important;
}
.footer-info {
    text-align: center;
    color: #6a7a8a !important;
    font-size: 12px;
    border-top: 1px solid #2a2a4a;
    padding-top: 12px;
    margin-top: 16px;
}
.footer-info a {
    color: #4a9eff !important;
}
"""

# ── JavaScript: 独立日志轮询（通过 head 参数注入 <script> 标签）──
LOG_POLL_HEAD = """<script>
(function() {
    var pollCount = 0;
    var _pollTimer = setInterval(function() {
        pollCount++;
        var ta = document.querySelector('.log-box textarea');
        if (!ta) return;
        var setter = Object.getOwnPropertyDescriptor(
            HTMLTextAreaElement.prototype, 'value'
        ).set;
        fetch('./api/read_logs', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({data: []})
        })
        .then(function(r) { return r.json(); })
        .then(function(json) {
            var logText = (json.data && json.data[0]) ? json.data[0] : '';
            if (logText) {
                setter.call(ta, logText);
            } else {
                setter.call(ta, '[poll #' + pollCount + '] waiting for logs... (' + new Date().toLocaleTimeString() + ')');
            }
            ta.dispatchEvent(new Event('input', {bubbles: true}));
            ta.scrollTop = ta.scrollHeight;
        })
        .catch(function(err) {
            setter.call(ta, '[poll error] ' + err.message);
            ta.dispatchEvent(new Event('input', {bubbles: true}));
        });
    }, 2000);
})();
</script>"""


# ── 构建 Gradio 界面 ──
DEVICE_LABEL = get_device_label()

with gr.Blocks(
    title="Music to MIDI",
    css=CUSTOM_CSS,
    head=LOG_POLL_HEAD,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=["system-ui", "Noto Sans SC", "sans-serif"],
    ).set(
        body_background_fill="#1a1a2e",
        block_background_fill="#1f2940",
        block_border_color="#3a4a6a",
        block_label_text_color="#b0b8c8",
        block_title_text_color="#e0e0e0",
        input_background_fill="#16213e",
        input_border_color="#3a4a6a",
        button_primary_background_fill="#4a9eff",
        button_primary_text_color="white",
        button_secondary_background_fill="#2a3f5f",
        button_secondary_text_color="#e0e0e0",
    ),
) as demo:

    # ── 标题 ──
    with gr.Group(elem_classes="app-header"):
        gr.Markdown(
            "# 🎵 音乐转MIDI\n"
            "将音乐智能转换为多轨道 MIDI 文件 — 基于 YourMT3+ MoE 深度学习模型"
        )

    with gr.Row(equal_height=False):
        # ── 左栏：输入 ──
        with gr.Column(scale=5):
            gr.Markdown("**音频输入**", elem_classes="section-title")
            audio_input = gr.Audio(
                label="拖拽音频文件到此处，或点击选择",
                type="filepath",
                elem_classes="upload-zone",
            )
            gr.Markdown(
                "<small style='color:#6a7a8a'>支持 MP3, WAV, FLAC, OGG, M4A</small>"
            )

            gr.Markdown("**轨道设置**", elem_classes="section-title")
            mode_radio = gr.Radio(
                choices=[
                    "YourMT3+ 多乐器转写",
                    "人声分离 + 分别转写",
                    "六声部分离 + 分别转写",
                    "钢琴专用转写 (Aria-AMT)",
                ],
                value="YourMT3+ 多乐器转写",
                label="处理模式",
            )
            mode_info = gr.Markdown(
                "**YourMT3+ MoE** — 直接对完整音频进行多乐器转写，"
                "精确识别 128 种 GM 乐器，轨道数量由模型自动决定。",
                elem_classes="mode-info",
            )
            vocal_split_merge_midi = gr.Checkbox(
                value=False,
                label="输出 1 个人声+伴奏合并 MIDI（人声分离模式）",
                visible=False,
            )
            six_stem_only_selected = gr.Checkbox(
                value=False,
                label="仅转写选中的 stem（六声部模式）",
                visible=False,
            )
            six_stem_vocal_harmony = gr.Checkbox(
                value=False,
                label="将 vocals 进一步分离为主唱 + 和声（实验近似）",
                visible=False,
            )
            six_stem_targets = gr.CheckboxGroup(
                choices=["bass", "drums", "guitar", "piano", "vocals", "other"],
                value=["bass", "drums", "guitar", "piano", "vocals", "other"],
                label="六声部转写目标",
                info="仅在勾选“仅转写选中的 stem”且模式为六声部分离时生效",
                visible=False,
            )

            mode_radio.change(
                fn=update_mode_controls,
                inputs=[mode_radio, six_stem_only_selected],
                outputs=[
                    mode_info,
                    vocal_split_merge_midi,
                    six_stem_only_selected,
                    six_stem_targets,
                    six_stem_vocal_harmony,
                ],
                api_name=False,
            )
            six_stem_only_selected.change(
                fn=update_six_stem_targets_visibility,
                inputs=[mode_radio, six_stem_only_selected],
                outputs=[six_stem_targets],
                api_name=False,
            )

            quality_radio = gr.Radio(
                choices=["fast", "balanced", "best"],
                value="balanced",
                label="转写质量",
                info="fast = 快速  |  balanced = 均衡  |  best = 最佳",
            )

            convert_btn = gr.Button(
                "▶  开始转换",
                variant="primary",
                elem_classes="convert-btn",
                size="lg",
            )

            # 设备信息
            gpu_note = " (ZeroGPU: 转换时自动分配)" if ZERO_GPU else ""
            gr.Markdown(
                f"当前设备: **{DEVICE_LABEL}**{gpu_note}",
                elem_classes="device-badge",
            )

        # ── 右栏：输出 ──
        with gr.Column(scale=5):
            gr.Markdown("**处理结果**", elem_classes="section-title")

            status_output = gr.Textbox(
                label="状态",
                interactive=False,
                lines=7,
                placeholder="等待转换...",
                elem_classes="result-box",
            )

            gr.Markdown("**下载 MIDI 文件**", elem_classes="section-title")
            file_output = gr.File(
                label="转换完成后点击下载",
                file_count="multiple",
            )

            gr.Markdown("**控制台日志**", elem_classes="section-title")
            log_output = gr.Textbox(
                label="实时处理日志",
                interactive=False,
                lines=12,
                max_lines=20,
                placeholder="日志输出将在此处显示...",
                elem_classes="log-box",
            )

    # ── 按钮事件 ──
    convert_btn.click(
        fn=convert_audio_to_midi,
        inputs=[
            audio_input,
            mode_radio,
            quality_radio,
            vocal_split_merge_midi,
            six_stem_only_selected,
            six_stem_targets,
            six_stem_vocal_harmony,
        ],
        outputs=[file_output, status_output],
        api_name="convert",
    )

    # ── 日志 API 端点（供 JavaScript 轮询调用）──
    # 用隐藏按钮注册 api_name="read_logs"，JS 通过 POST ./api/read_logs 调用
    # queue=False 确保不被转换任务阻塞
    _log_poll_btn = gr.Button(visible=False)
    _log_poll_btn.click(
        fn=read_logs,
        inputs=[],
        outputs=[log_output],
        api_name="read_logs",
        queue=False,
    )

    # ── 底部 ──
    gr.Markdown(
        '<div class="footer-info">'
        "基于 <a href='https://github.com/mimbres/YourMT3'>YourMT3+</a> MoE 模型 | "
        "<a href='https://github.com/mason369/music-to-midi'>GitHub</a> | "
        "MIT License"
        "</div>"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
else:
    demo.launch(server_name="0.0.0.0")
