"""
Music to MIDI - Gradio Web 界面
视觉风格对齐 PyQt6 桌面版暗色主题。
"""
import logging
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

sys.setrecursionlimit(3000)

APP_TEMP_DIR = tempfile.gettempdir()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["NUMBA_CACHE_DIR"] = os.path.join(APP_TEMP_DIR, "numba_cache")
os.environ["MPLCONFIGDIR"] = os.path.join(APP_TEMP_DIR, "matplotlib")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

LOG_FILE = os.path.join(APP_TEMP_DIR, "midi_process.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("music-to-midi-web")


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
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass


def read_logs():
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().replace("\x00", "")
        lines = content.strip().split("\n")
        return "\n".join(lines[-50:]) if lines and lines[0] else ""
    except Exception as exc:
        return f"[read_logs error] {exc}"


def ensure_yourmt3_code():
    """从 HF Space 仓库下载 YourMT3 源代码。"""
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
            ignore_patterns=["*.ckpt", "*.bin", "*.safetensors", "amt/logs/**", "*.DS_Store"],
        )
        logger.info("YourMT3 source code downloaded")
    else:
        logger.info("YourMT3 source code already present")

    if os.path.exists(amt_src) and amt_src not in sys.path:
        sys.path.insert(0, amt_src)
        logger.info("Added YourMT3 path: %s", amt_src)


try:
    ensure_yourmt3_code()
except Exception as exc:
    logger.warning("Failed to setup YourMT3: %s", exc)


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


try:
    import spaces
    ZERO_GPU = True
    logger.info("ZeroGPU (spaces) available")
except ImportError:
    ZERO_GPU = False
    logger.info("Running without ZeroGPU")


import gradio as gr
from gradio.components.base import Component

from src.i18n.translator import Translator
from src.models.data_models import Config, ProcessingStage

SPACE_LANGUAGE = os.environ.get("MUSIC_TO_MIDI_LANGUAGE", "zh_CN")
if SPACE_LANGUAGE not in Translator.AVAILABLE_LANGUAGES:
    raise RuntimeError(f"Unsupported MUSIC_TO_MIDI_LANGUAGE: {SPACE_LANGUAGE}")
SPACE_TRANSLATOR = Translator(SPACE_LANGUAGE)


def st(key: str, **kwargs) -> str:
    return SPACE_TRANSLATOR.t(key, **kwargs)


def _normalize_json_schema_bool_nodes(schema):
    if isinstance(schema, dict):
        for key, value in list(schema.items()):
            if key == "additionalProperties" and isinstance(value, bool):
                schema[key] = {}
            else:
                _normalize_json_schema_bool_nodes(value)
    elif isinstance(schema, list):
        for item in schema:
            _normalize_json_schema_bool_nodes(item)
    return schema


_original_component_api_info = Component.api_info


def _patched_component_api_info(self):
    return _normalize_json_schema_bool_nodes(deepcopy(_original_component_api_info(self)))


Component.api_info = _patched_component_api_info


MODE_IDS = (
    "smart",
    "vocal_split",
    "six_stem_split",
    "piano_transkun",
    "piano_aria_amt",
    "piano_bytedance_pedal",
)
MODE_LABELS = {mode_id: st(f"space.mode.{mode_id}") for mode_id in MODE_IDS}
MODE_CHOICES = [(MODE_LABELS[mode_id], mode_id) for mode_id in MODE_IDS]
STAGE_LABEL_KEYS = {
    ProcessingStage.PREPROCESSING: "preprocessing",
    ProcessingStage.SEPARATION: "separation",
    ProcessingStage.TRANSCRIPTION: "transcription",
    ProcessingStage.VOCAL_TRANSCRIPTION: "vocal_transcription",
    ProcessingStage.SYNTHESIS: "synthesis",
    ProcessingStage.COMPLETE: "complete",
}


def ensure_model_weights():
    """确保 YourMT3+ 官方模式权重已下载。"""
    ensure_yourmt3_code()

    from src.utils.yourmt3_downloader import OFFICIAL_YOURMT3_MODEL_KEYS, get_model_path

    missing = []
    for model_key in OFFICIAL_YOURMT3_MODEL_KEYS:
        model_path = get_model_path(model_key)
        if model_path and model_path.exists():
            logger.info("YourMT3+ model found: %s", model_path)
        else:
            missing.append(model_key)

    if not missing:
        return

    logger.info("YourMT3+ model weights missing, downloading: %s", ", ".join(missing))
    from download_sota_models import download_official_yourmt3_models

    download_official_yourmt3_models()
    logger.info("YourMT3+ model weights downloaded")


def ensure_multistem_weights():
    """确保 BS-RoFormer SW 六轨资源已下载并通过校验。"""
    from download_multistem_model import download_multistem_model

    model_path, config_path = download_multistem_model(printer=logger.info)
    logger.info("BS-RoFormer SW checkpoint ready: %s", model_path)
    logger.info("BS-RoFormer SW config ready: %s", config_path)


def ensure_vocal_split_weights():
    """确保 RoFormer vocal_rvc/karaoke 人声分离 ensemble 已下载。"""
    from download_vocal_harmony_model import download_chorus_model
    from download_vocal_model import download_vocal_model

    vocal_model = download_vocal_model(printer=logger.info)
    karaoke_model = download_chorus_model(printer=logger.info)
    logger.info("RoFormer vocal_rvc ensemble ready: %s", vocal_model)
    logger.info("RoFormer karaoke ensemble ready: %s", karaoke_model)


try:
    ensure_model_weights()
except Exception as exc:
    logger.warning("Model preload failed: %s", exc)


def ensure_aria_amt_weights():
    """确保 Aria-AMT 钢琴 checkpoint 已下载。"""
    from download_aria_amt_model import download_aria_model, is_aria_model_available

    if is_aria_model_available():
        logger.info("Aria-AMT checkpoint found")
        return

    logger.info("Aria-AMT checkpoint not found, downloading...")
    download_aria_model()
    logger.info("Aria-AMT checkpoint downloaded")


def ensure_bytedance_piano_weights():
    """确保 ByteDance Piano 带踏板 checkpoint 已下载。"""
    from download_bytedance_piano_model import (
        download_bytedance_piano_model,
        is_bytedance_piano_model_available,
    )

    if is_bytedance_piano_model_available():
        logger.info("ByteDance Piano checkpoint found")
        return

    logger.info("ByteDance Piano checkpoint not found, downloading...")
    download_bytedance_piano_model()
    logger.info("ByteDance Piano checkpoint downloaded")

clear_logs()


def get_device_label():
    try:
        import torch

        if torch.cuda.is_available():
            return f"GPU ({torch.cuda.get_device_name(0)})"
        return "CPU"
    except Exception:
        return "CPU"


def _convert_impl(
    audio_path,
    mode,
    vocal_split_merge_midi=False,
    progress=gr.Progress(),
):
    import datetime

    from src.core.pipeline import MusicToMidiPipeline
    from src.utils.gpu_utils import clear_gpu_memory

    if audio_path is None:
        raise gr.Error(st("space.error.upload_required"))

    def _write_log(msg):
        try:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{ts} [INFO] {msg}\n")
        except Exception:
            pass

    clear_logs()
    _write_log("=" * 40)
    _write_log(st("space.log.start"))
    _write_log(f"{st('space.log.audio_file')}: {Path(audio_path).name}")
    _write_log(f"{st('space.log.processing_mode')}: {MODE_LABELS.get(mode, mode)}")
    _write_log("=" * 40)

    config = Config()
    if mode not in MODE_IDS:
        raise RuntimeError(f"Unsupported processing mode: {mode}")
    config.processing_mode = mode
    config.language = SPACE_LANGUAGE
    config.vocal_split_merge_midi = bool(
        config.processing_mode == "vocal_split" and vocal_split_merge_midi
    )

    if config.processing_mode == "piano_aria_amt":
        ensure_aria_amt_weights()
    elif config.processing_mode == "piano_bytedance_pedal":
        ensure_bytedance_piano_weights()
    elif config.processing_mode != "piano_transkun":
        ensure_model_weights()
        if config.processing_mode == "six_stem_split":
            ensure_multistem_weights()
        elif config.processing_mode == "vocal_split":
            ensure_vocal_split_weights()

    pipeline = MusicToMidiPipeline(config)
    output_dir = tempfile.mkdtemp(prefix="midi_output_")

    def on_progress(p):
        stage_key = STAGE_LABEL_KEYS.get(p.stage)
        stage_name = (
            st(f"main.progress.stages.{stage_key}") if stage_key else str(p.stage)
        )
        progress(p.overall_progress, desc=f"[{stage_name}] {p.message}")

    try:
        result = pipeline.process(
            audio_path=audio_path,
            output_dir=output_dir,
            progress_callback=on_progress,
        )
    except Exception as exc:
        logger.error("转换失败: %s", exc)
        raise gr.Error(st("space.error.conversion_failed", error=exc)) from exc
    finally:
        try:
            clear_gpu_memory()
        except Exception:
            pass

    output_files = []
    if result.midi_path and Path(result.midi_path).exists():
        output_files.append(result.midi_path)
    if result.stem_midi_paths:
        for stem_midi_path in result.stem_midi_paths.values():
            if (
                stem_midi_path
                and Path(stem_midi_path).exists()
                and stem_midi_path not in output_files
            ):
                output_files.append(stem_midi_path)
    if result.vocal_midi_path and Path(result.vocal_midi_path).exists():
        output_files.append(result.vocal_midi_path)
    if result.accompaniment_midi_path and Path(result.accompaniment_midi_path).exists():
        if result.accompaniment_midi_path != result.midi_path:
            output_files.append(result.accompaniment_midi_path)
    if result.separated_audio:
        for audio_file in result.separated_audio.values():
            if audio_file and Path(audio_file).exists():
                output_files.append(audio_file)

    device_label = get_device_label()
    bpm_str = f"{result.beat_info.bpm:.1f}" if result.beat_info else "N/A"
    status_lines = [
        st("space.status.complete_header"),
        f"{st('space.status.elapsed')}: {result.processing_time:.1f} {st('space.status.seconds')}",
        f"{st('space.status.total_notes')}: {result.total_notes}",
        f"BPM: {bpm_str}",
        f"{st('space.status.device')}: {device_label}",
    ]
    if result.stem_midi_paths:
        status_lines.append(f"{st('space.status.merged_midi')}: {Path(result.midi_path).name}")
        status_lines.append(
            f"{st('space.status.stem_midi_count')}: "
            f"{len(result.stem_midi_paths)}{st('space.status.stem_midi_count_suffix')}"
        )
    elif result.vocal_midi_path:
        status_lines.append(
            f"{st('space.status.accompaniment_midi')}: {Path(result.accompaniment_midi_path).name}"
        )
        status_lines.append(f"{st('space.status.vocal_midi')}: {Path(result.vocal_midi_path).name}")
        if result.merged_midi_path:
            status_lines.append(f"{st('space.status.merged_midi')}: {Path(result.merged_midi_path).name}")
    else:
        status_lines.append(f"{st('space.status.midi_file')}: {Path(result.midi_path).name}")

    logger.info(st("space.log.complete"))
    return output_files, "\n".join(status_lines)


if ZERO_GPU:
    @spaces.GPU
    def convert_audio_to_midi(
        audio_path,
        mode,
        vocal_split_merge_midi=False,
        progress=gr.Progress(),
    ):
        return _convert_impl(
            audio_path,
            mode,
            vocal_split_merge_midi,
            progress,
        )
else:
    def convert_audio_to_midi(
        audio_path,
        mode,
        vocal_split_merge_midi=False,
        progress=gr.Progress(),
    ):
        return _convert_impl(
            audio_path,
            mode,
            vocal_split_merge_midi,
            progress,
        )


def update_mode_info(mode):
    if mode not in MODE_IDS:
        raise RuntimeError(f"Unsupported processing mode: {mode}")
    return st(f"space.mode.{mode}_info")


def update_mode_controls(mode):
    is_vocal = mode == "vocal_split"
    return (
        update_mode_info(mode),
        gr.update(visible=is_vocal),
    )


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
.device-badge {
    background: #16213e;
    border: 1px solid #3a4a6a;
    border-radius: 6px;
    padding: 6px 12px;
    text-align: center;
}
.footer-info {
    text-align: center;
    color: #6a7a8a !important;
    font-size: 12px;
    border-top: 1px solid #2a2a4a;
    padding-top: 12px;
    margin-top: 16px;
}
"""

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
            setter.call(ta, logText || '[poll #' + pollCount + '] waiting for logs...');
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

DEVICE_LABEL = get_device_label()
ZERO_GPU_NOTE = st("space.ui.zerogpu_note") if ZERO_GPU else ""

with gr.Blocks(
    title=st("space.app.title"),
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
    with gr.Group(elem_classes="app-header"):
        gr.Markdown(
            f"# 🎵 {st('space.app.title')}\n"
            f"{st('space.app.subtitle')}"
        )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            gr.Markdown(f"**{st('space.ui.audio_section')}**", elem_classes="section-title")
            audio_input = gr.Audio(
                label=st("space.ui.audio_input"),
                type="filepath",
                elem_classes="upload-zone",
            )
            gr.Markdown(
                f"<small style='color:#6a7a8a'>{st('space.ui.audio_hint')}</small>"
            )

            gr.Markdown(f"**{st('space.ui.track_section')}**", elem_classes="section-title")
            mode_radio = gr.Radio(
                choices=MODE_CHOICES,
                value="smart",
                label=st("space.ui.mode_label"),
            )
            mode_info = gr.Markdown(update_mode_info("smart"), elem_classes="mode-info")
            vocal_split_merge_midi = gr.Checkbox(
                value=False,
                label=st("space.ui.vocal_split_merge_midi"),
                visible=False,
            )
            mode_radio.change(
                fn=update_mode_controls,
                inputs=[mode_radio],
                outputs=[
                    mode_info,
                    vocal_split_merge_midi,
                ],
                api_name=False,
            )

            convert_btn = gr.Button(
                st("space.ui.convert_button"),
                variant="primary",
                elem_classes="convert-btn",
                size="lg",
            )

            gr.Markdown(
                f"{st('space.ui.device')}: **{DEVICE_LABEL}**{ZERO_GPU_NOTE}",
                elem_classes="device-badge",
            )

        with gr.Column(scale=5):
            gr.Markdown(f"**{st('space.ui.result_section')}**", elem_classes="section-title")
            status_output = gr.Textbox(
                label=st("space.ui.status_label"),
                interactive=False,
                lines=7,
                placeholder=st("space.ui.status_placeholder"),
                elem_classes="result-box",
            )

            gr.Markdown(f"**{st('space.ui.download_section')}**", elem_classes="section-title")
            file_output = gr.File(label=st("space.ui.download_label"), file_count="multiple")

            gr.Markdown(f"**{st('space.ui.logs_section')}**", elem_classes="section-title")
            log_output = gr.Textbox(
                label=st("space.ui.logs_label"),
                interactive=False,
                lines=12,
                max_lines=20,
                placeholder=st("space.ui.logs_placeholder"),
                elem_classes="log-box",
            )

    convert_btn.click(
        fn=convert_audio_to_midi,
        inputs=[
            audio_input,
            mode_radio,
            vocal_split_merge_midi,
        ],
        outputs=[file_output, status_output],
        api_name="convert",
    )

    _log_poll_btn = gr.Button(visible=False)
    _log_poll_btn.click(
        fn=read_logs,
        inputs=[],
        outputs=[log_output],
        api_name="read_logs",
        queue=False,
    )

    gr.Markdown(
        '<div class="footer-info">'
        f"{st('space.ui.footer_powered_by')} "
        "<a href='https://github.com/mimbres/YourMT3'>YourMT3+</a> | "
        "<a href='https://github.com/mason369/music-to-midi'>GitHub</a> | "
        "MIT License"
        "</div>"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
