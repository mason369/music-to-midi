"""
Music to MIDI - Gradio Web 界面
视觉风格对齐 PyQt6 桌面版暗色主题
"""
import os
import sys
import subprocess
import tempfile
import logging
from pathlib import Path

# ── 环境变量（必须在 import torch 前设置）──
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# ── Monkey-patch: 修复 gradio_client schema 解析 bug ──
# gradio_client.utils.get_type() 对 additionalProperties=true（布尔值）
# 执行 `"const" in schema` 导致 TypeError: argument of type 'bool' is not iterable
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

import gradio as gr

# 确保 src 可以被 import
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("music-to-midi-web")



# ── 确保 YourMT3 源码可用 ──
def ensure_yourmt3_code():
    """克隆 YourMT3 仓库（如果不存在）"""
    yourmt3_dir = os.path.join(APP_DIR, "YourMT3")
    if os.path.exists(os.path.join(yourmt3_dir, "amt", "src", "model", "ymt3.py")):
        logger.info("YourMT3 source code already present")
        return
    logger.info("Cloning YourMT3 repository...")
    subprocess.run(
        ["git", "clone", "--depth=1", "https://github.com/mimbres/YourMT3.git", yourmt3_dir],
        check=True,
        timeout=300,
    )
    logger.info("YourMT3 source code cloned successfully")


try:
    ensure_yourmt3_code()
except Exception as e:
    logger.warning(f"Failed to clone YourMT3: {e}")


# ── 延迟导入核心模块（在 YourMT3 代码就绪后）──
from src.models.data_models import Config, ProcessingStage
from src.core.yourmt3_transcriber import YourMT3Transcriber


# ── 模型预加载 ──
def ensure_model():
    """启动时检查并下载 YourMT3+ MoE 模型权重"""
    if not YourMT3Transcriber.is_available():
        logger.info("Model not found, downloading YourMT3+ MoE ...")
        from download_sota_models import download_ultimate_moe
        download_ultimate_moe()
        logger.info("Model download complete")
    else:
        logger.info("YourMT3+ model is ready")


try:
    ensure_model()
except Exception as e:
    logger.warning(f"Model preload failed (will retry on first use): {e}")


# ── 设备信息 ──
def get_device_label():
    """获取当前设备标签"""
    try:
        from src.utils.gpu_utils import get_accelerator_label, is_gpu_available
        if is_gpu_available():
            return get_accelerator_label()
        return "CPU"
    except Exception:
        return "CPU"


DEVICE_LABEL = get_device_label()


# ── 核心转换函数 ──
def convert_audio_to_midi(audio_path, mode, quality, progress=gr.Progress()):
    """执行音频到 MIDI 的转换"""
    from src.core.pipeline import MusicToMidiPipeline
    from src.utils.gpu_utils import clear_gpu_memory

    if audio_path is None:
        raise gr.Error("请先上传音频文件")

    config = Config()
    config.processing_mode = "vocal_split" if mode == "人声分离 + 分别转写" else "smart"
    config.transcription_quality = quality

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
        logger.error(f"Conversion failed: {e}", exc_info=True)
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
    if result.vocal_midi_path and Path(result.vocal_midi_path).exists():
        output_files.append(result.vocal_midi_path)
    if result.accompaniment_midi_path and Path(result.accompaniment_midi_path).exists():
        if result.accompaniment_midi_path != result.midi_path:
            output_files.append(result.accompaniment_midi_path)

    # 构建状态文本
    bpm_str = f"{result.beat_info.bpm:.1f}" if result.beat_info else "N/A"
    status_lines = [
        "--- 转换完成 ---",
        f"耗时: {result.processing_time:.1f} 秒",
        f"总音符数: {result.total_notes}",
        f"BPM: {bpm_str}",
        f"设备: {DEVICE_LABEL}",
    ]
    if result.vocal_midi_path:
        status_lines.append(f"伴奏 MIDI: {Path(result.accompaniment_midi_path).name}")
        status_lines.append(f"人声 MIDI: {Path(result.vocal_midi_path).name}")
    else:
        status_lines.append(f"MIDI 文件: {Path(result.midi_path).name}")

    return output_files, "\n".join(status_lines)


# ── 模式切换时更新说明 ──
def update_mode_info(mode):
    if mode == "人声分离 + 分别转写":
        return (
            "**Demucs + YourMT3+** — 先用 Demucs 分离人声与伴奏，"
            "再分别用 YourMT3+ 转写，输出两个独立 MIDI 文件。"
        )
    return (
        "**YourMT3+ MoE** — 直接对完整音频进行多乐器转写，"
        "精确识别 128 种 GM 乐器，轨道数量由模型自动决定。"
    )


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


# ── 构建 Gradio 界面 ──
with gr.Blocks(
    title="Music to MIDI",
    css=CUSTOM_CSS,
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
                choices=["YourMT3+ 多乐器转写", "人声分离 + 分别转写"],
                value="YourMT3+ 多乐器转写",
                label="处理模式",
            )
            mode_info = gr.Markdown(
                "**YourMT3+ MoE** — 直接对完整音频进行多乐器转写，"
                "精确识别 128 种 GM 乐器，轨道数量由模型自动决定。",
                elem_classes="mode-info",
            )
            mode_radio.change(
                fn=update_mode_info,
                inputs=[mode_radio],
                outputs=[mode_info],
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

            # 设备信息（模拟桌面版状态栏）
            gr.Markdown(
                f"当前设备: **{DEVICE_LABEL}**",
                elem_classes="device-badge",
            )

        # ── 右栏：输出 ──
        with gr.Column(scale=5):
            gr.Markdown("**处理结果**", elem_classes="section-title")

            status_output = gr.Textbox(
                label="状态",
                interactive=False,
                lines=8,
                placeholder="等待转换...",
                elem_classes="result-box",
            )

            gr.Markdown("**下载 MIDI 文件**", elem_classes="section-title")
            file_output = gr.File(
                label="转换完成后点击下载",
                file_count="multiple",
            )

    # ── 按钮事件 ──
    convert_btn.click(
        fn=convert_audio_to_midi,
        inputs=[audio_input, mode_radio, quality_radio],
        outputs=[file_output, status_output],
        api_name="convert",
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
