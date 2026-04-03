"""
YourMT3+ 转写器模块

使用 YourMT3+（2025 AMT Challenge 获奖架构）进行最先进的多乐器音频转写。

优势:
- 层次化注意力 Transformer 架构
- 混合专家 (MoE) 针对不同乐器
- 直接多乐器转写（无需分离）
- 精确识别 128 种 GM 乐器
- PyTorch 原生，完美支持 GPU 加速
"""
import logging
import os
import sys
import threading
import types
import importlib
from importlib.machinery import ModuleSpec
from io import StringIO
from pathlib import Path


def _load_audio(audio_path):
    """用 soundfile 加载音频，绕过 torchaudio 2.9+ 强制使用 torchcodec 的问题"""
    import torch
    import soundfile as sf
    data, sr = sf.read(audio_path, dtype='float32')
    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T  # (samples, channels) -> (channels, samples)
    return waveform, sr


from typing import Any, List, Optional, Callable, Dict, Tuple, Union
from contextlib import contextmanager
import numpy as np

from src.models.data_models import Config, NoteEvent, InstrumentType, PedalEvent, TranscriptionQuality
from src.models.gm_instruments import get_instrument_name
from src.utils.gpu_utils import (
    get_device,
    get_optimal_batch_size,
    _fix_torch_dll_path,
    get_optimal_thread_count,
    ensure_cuda_runtime_compatibility,
    rewrite_cuda_runtime_error,
)
from src.utils.runtime_paths import get_resource_path, get_yourmt3_source_dir

# 在任何 import torch 之前修复 Windows 特殊路径 DLL 加载问题
_fix_torch_dll_path()

logger = logging.getLogger(__name__)

_YOURMT3_TRANSIENT_MODULE_PREFIXES = (
    "model",
    "utils",
    "config",
)


@contextmanager
def _temporary_onnxruntime_stub():
    existing = sys.modules.get("onnxruntime")
    if existing is not None:
        yield
        return

    stub = types.ModuleType("onnxruntime")
    stub.__dict__.update(
        {
            "__version__": "0.0",
            "__path__": [],
            "get_available_providers": lambda: [],
            "SessionOptions": type("SessionOptions", (), {}),
            "InferenceSession": type("InferenceSession", (), {}),
        }
    )
    stub.__spec__ = ModuleSpec("onnxruntime", loader=None, is_package=True)
    sys.modules["onnxruntime"] = stub
    try:
        yield
    finally:
        if sys.modules.get("onnxruntime") is stub:
            sys.modules.pop("onnxruntime", None)


def _iter_yourmt3_base_paths() -> list[str]:
    candidates = [
        "YourMT3",
        os.path.join(os.getcwd(), "YourMT3"),
        "external/YourMT3",
        os.path.join(os.getcwd(), "external/YourMT3"),
        os.path.join(os.path.dirname(__file__), "../../YourMT3"),
        os.path.join(os.path.dirname(__file__), "../../external/YourMT3"),
        str(get_resource_path("YourMT3")),
        str(get_resource_path("external/YourMT3")),
    ]

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm not in seen:
            unique.append(norm)
            seen.add(norm)
    return unique


def _is_yourmt3_amt_src_dir(path: Union[str, Path]) -> bool:
    root = Path(path)
    return all(
        (root / relative_path).exists()
        for relative_path in (
            Path("model/ymt3.py"),
            Path("utils/task_manager.py"),
            Path("config/config.py"),
        )
    )


def _get_yourmt3_amt_src_path() -> Optional[str]:
    direct = get_yourmt3_source_dir()
    if direct is not None and _is_yourmt3_amt_src_dir(direct):
        return str(direct)

    for base_path in _iter_yourmt3_base_paths():
        if os.path.exists(base_path):
            potential_path = os.path.join(base_path, "amt/src")
            if _is_yourmt3_amt_src_dir(potential_path):
                return potential_path

    for entry in sys.path:
        if not entry:
            continue
        if _is_yourmt3_amt_src_dir(entry):
            return os.path.normpath(entry)
    return None


def _clear_yourmt3_import_state() -> None:
    """Drop cached top-level YourMT3 modules before a fresh import in frozen apps."""
    module_names = [
        name
        for name in sys.modules
        if name in _YOURMT3_TRANSIENT_MODULE_PREFIXES
        or any(name.startswith(prefix + ".") for prefix in _YOURMT3_TRANSIENT_MODULE_PREFIXES)
    ]
    for name in module_names:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _import_torch():
    """
    安全导入 torch，失败时给出清晰的错误提示而非原始 OSError。
    所有需要 import torch 的地方应调用此函数。
    """
    try:
        import torch
        return torch
    except OSError as e:
        import re
        err_str = str(e)
        path_match = re.search(r'Error loading "([^"]+)"', err_str)
        dll_path = path_match.group(1) if path_match else ""
        has_special = bool(re.search(r'[\s\(\)\[\]{}]|[^\x00-\x7F]', dll_path))

        msg = "PyTorch DLL 加载失败，无法启动转写引擎。\n\n"
        if has_special:
            msg += (
                "检测到路径含特殊字符（空格/括号/中文等），这会导致 PyTorch 无法加载。\n"
                "请将项目移动到纯英文且无空格的路径，如 C:\\MusicToMidi\n"
            )
        else:
            msg += (
                "可能原因及解决方法：\n"
                "1. 未安装 Visual C++ Redistributable 2022\n"
                "   下载安装: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                "2. PyTorch 安装不完整\n"
                "   执行: venv\\Scripts\\pip install --force-reinstall torch\n"
                "3. libomp140.x86_64.dll 缺失\n"
                "   重新运行 install.bat 可自动修复\n"
            )
        msg += f"\n原始错误: {e}"
        raise RuntimeError(msg) from e


@contextmanager
def suppress_output():
    """临时抑制标准输出和标准错误"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def program_to_instrument_type(program: int, is_drum: bool = False) -> InstrumentType:
    """
    将 GM 程序号映射到 InstrumentType

    使用精确的程序号到乐器类型映射，支持更多乐器类型。

    参数:
        program: GM 程序号 (0-127)
        is_drum: 是否为鼓轨道

    返回:
        对应的 InstrumentType
    """
    # 鼓轨道特殊处理
    if is_drum:
        return InstrumentType.DRUMS

    if program < 0 or program > 127:
        return InstrumentType.OTHER

    # 精确的 GM 程序号到乐器类型映射
    # Piano (0-7)
    if 0 <= program <= 7:
        return InstrumentType.PIANO

    # Chromatic Percussion (8-15) - 半音阶打击乐
    if 8 <= program <= 15:
        return InstrumentType.PERCUSSION

    # Organ (16-23)
    if 16 <= program <= 23:
        return InstrumentType.ORGAN

    # Guitar (24-31)
    if 24 <= program <= 31:
        return InstrumentType.GUITAR

    # Bass (32-39)
    if 32 <= program <= 39:
        return InstrumentType.BASS

    # Strings (40-47)
    if 40 <= program <= 47:
        return InstrumentType.STRINGS

    # Ensemble (48-55) - 合奏/合唱
    if 48 <= program <= 55:
        # 区分人声和弦乐合奏
        if program in [52, 53, 54]:  # Choir Aahs, Voice Oohs, Synth Voice
            return InstrumentType.CHOIR
        return InstrumentType.STRINGS

    # Brass (56-63)
    if 56 <= program <= 63:
        return InstrumentType.BRASS

    # Reed (64-71) - 簧管乐器
    if 64 <= program <= 71:
        return InstrumentType.WOODWIND

    # Pipe (72-79) - 哨笛类
    if 72 <= program <= 79:
        return InstrumentType.WOODWIND

    # Synth Lead (80-87)
    if 80 <= program <= 87:
        return InstrumentType.LEAD_SYNTH

    # Synth Pad (88-95)
    if 88 <= program <= 95:
        return InstrumentType.PAD_SYNTH

    # Synth Effects (96-103)
    if 96 <= program <= 103:
        return InstrumentType.SYNTH

    # Ethnic (104-111)
    if 104 <= program <= 111:
        # 民族乐器中的一些可以分类
        if program == 105:  # Banjo
            return InstrumentType.GUITAR
        if program == 106:  # Shamisen
            return InstrumentType.STRINGS
        if program == 107:  # Koto
            return InstrumentType.HARP
        if program == 110:  # Fiddle (小提琴)
            return InstrumentType.STRINGS
        return InstrumentType.OTHER

    # Percussive (112-119)
    if 112 <= program <= 119:
        return InstrumentType.PERCUSSION

    # Sound Effects (120-127)
    if 120 <= program <= 127:
        return InstrumentType.OTHER

    return InstrumentType.OTHER


class YourMT3Transcriber:
    """
    使用 YourMT3+ 进行最先进的多乐器转写

    优势:
    - 2025 AMT Challenge 获奖架构
    - 层次化注意力 Transformer + 混合专家 (MoE)
    - 直接多乐器转写（可输出超过6轨道）
    - PyTorch 原生，完美 GPU 加速
    """

    # 类级别的模型缓存
    _model = None
    _model_name = None
    _device = None
    _model_lock = threading.Lock()
    _audio_cfg = None
    _task_manager = None
    _last_unavailable_reason = None

    def __init__(self, config: Config):
        """
        初始化 YourMT3+ 转写器

        参数:
            config: 应用配置
        """
        self.config = config
        self.device = get_device(config.use_gpu, config.gpu_device)

        # DirectML 不支持 YourMT3 所需的高级操作（ComplexFloat STFT、RoPE 等），回退 CPU
        if "privateuseone" in self.device:
            logger.warning("DirectML 不兼容 YourMT3 模型（STFT/RoPE 算子不支持），回退 CPU 模式")
            self.device = "cpu"

        # CPU 模式下动态设置线程数，充分利用多核 CPU 同时保留 2 核给 GUI
        if self.device == "cpu":
            torch = _import_torch()
            optimal_threads = get_optimal_thread_count()
            # 注意：os.environ 设置仅对子进程生效（当前进程的 OpenMP 在 import torch 时已初始化）
            # torch.set_num_threads() 才是运行时生效的 API
            os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
            os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
            torch.set_num_threads(optimal_threads)
            logger.debug(f"CPU 模式：已设置 PyTorch 线程数为 {optimal_threads}")

        self._cancelled = False
        self._cancel_check_callback = None

    def _inference_with_oom_retry(self, model, bsz, audio_segments, progress_callback=None):
        """执行推理，逐批处理并输出实时进度，VRAM 不足时自动减半 batch size 重试"""
        import torch
        import time as _time
        from src.utils.gpu_utils import clear_gpu_memory

        # Keep official full precision during inference.

        n_segments = audio_segments.shape[0]

        while bsz >= 1:
            n_batches = (n_segments + bsz - 1) // bsz
            logger.info(
                f"YourMT3 推理: bsz={bsz}, segments={n_segments}, "
                f"batches={n_batches}, device={self.device}, precision=float32"
            )

            if progress_callback:
                progress_callback(0.5, f"正在进行神经网络推理（共 {n_batches} 批）...")

            pred_token_array_file = []
            _infer_start = _time.time()

            try:
                with torch.no_grad():
                    for batch_idx, i in enumerate(range(0, n_segments, bsz)):
                        self._check_cancelled()

                        end = min(i + bsz, n_segments)
                        x = audio_segments[i:end]

                        preds = model.inference(x, None).detach().cpu().numpy()

                        pred_token_array_file.append(preds)

                        # 每批完成后输出实时进度
                        elapsed = _time.time() - _infer_start
                        done = batch_idx + 1
                        speed = done / elapsed  # batches/s
                        remaining = (n_batches - done) / speed if speed > 0 else 0
                        pct = done / n_batches * 100

                        logger.info(
                            f"[推理进度] 批次 {done}/{n_batches} | "
                            f"{elapsed:.1f}s 已用 | 剩余 ~{remaining:.0f}s | {pct:.0f}%"
                        )

                        if progress_callback:
                            # 推理阶段映射到 0.5~0.8
                            cb_progress = 0.5 + (pct / 100) * 0.3
                            progress_callback(
                                cb_progress,
                                f"神经网络推理 {done}/{n_batches} | "
                                f"剩余 ~{remaining:.0f}s"
                            )

                _infer_elapsed = _time.time() - _infer_start
                speed = n_segments / _infer_elapsed
                logger.info(
                    f"YourMT3 推理完成: 耗时={_infer_elapsed:.1f}s, "
                    f"速度={speed:.1f} segments/s ({1/speed:.2f}s/segment)"
                )
                return pred_token_array_file

            except InterruptedError:
                raise
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bsz > 1:
                    old_bsz = bsz
                    bsz = max(1, bsz // 2)
                    logger.warning(f"VRAM 不足 (bsz={old_bsz})，自动回退到 bsz={bsz}")
                    clear_gpu_memory()
                    pred_token_array_file.clear()
                    if progress_callback:
                        progress_callback(0.5, f"显存不足，降低批处理到 {bsz} 重试...")
                else:
                    raise
        raise RuntimeError("YourMT3 推理失败：即使 bsz=1 也无法完成")

    @classmethod
    def _mark_unavailable(cls, reason: str, *, info: Optional[str] = None) -> bool:
        cls._last_unavailable_reason = reason
        logger.warning(reason)
        if info:
            logger.info(info)
        return False

    @classmethod
    def get_unavailable_reason(cls) -> str:
        return cls._last_unavailable_reason or (
            "YourMT3+ 不可用。\n\n"
            "请先下载模型权重：\n"
            "  python download_sota_models.py\n\n"
            "详见 README.md 中的安装说明。"
        )

    @classmethod
    def is_available(cls) -> bool:
        """???? YourMT3+ ???????????????????"""
        cls._last_unavailable_reason = None
        try:
            _import_torch()
            logger.debug("PyTorch available")

            try:
                with _temporary_onnxruntime_stub():
                    import pytorch_lightning
                logger.debug("pytorch-lightning available")
            except ImportError as e:
                logger.debug("Failed to import pytorch_lightning", exc_info=True)
                missing_name = getattr(e, "name", "") or ""
                is_missing_lightning = (
                    missing_name == "pytorch_lightning" or "pytorch_lightning" in str(e)
                )
                if is_missing_lightning:
                    return cls._mark_unavailable(
                        "YourMT3+ ?????? pytorch-lightning?\n"
                        "??????????????pip install pytorch-lightning\n"
                        "?????????????????????????"
                    )
                return cls._mark_unavailable(
                    "YourMT3+ ????pytorch-lightning ?????????????????\n"
                    f"????: {e}\n"
                    "????????????????????"
                    "????????????????????"
                )

            amt_src_path = _get_yourmt3_amt_src_path()
            if not amt_src_path:
                return cls._mark_unavailable(
                    "YourMT3+ ??????? YourMT3 ?????\n"
                    "???????????????????????????? exe?\n"
                    "???????????????????? YourMT3/ ???",
                    info="???????????? python download_sota_models.py",
                )
            logger.debug("YourMT3 source tree available: %s", amt_src_path)

            try:
                from src.utils.yourmt3_downloader import DEFAULT_MODEL, get_model_path

                model_path = get_model_path(DEFAULT_MODEL)
                if not model_path or not model_path.exists():
                    return cls._mark_unavailable(
                        "YourMT3+ ????????????\n\n"
                        "?????????\n"
                        "  python download_sota_models.py\n\n"
                        "?? README.md ???????"
                    )
                logger.debug("Found YourMT3+ model: %s", model_path)
            except ImportError:
                logger.debug("yourmt3_downloader unavailable, skipping model path check")

            cls._last_unavailable_reason = None
            logger.info("YourMT3+ fully available")
            return True

        except ImportError as e:
            return cls._mark_unavailable(f"YourMT3+ ????{e}")

    def set_cancel_check(self, callback) -> None:
        """设置取消检查回调"""
        self._cancel_check_callback = callback

    def cancel(self) -> None:
        """取消正在进行的处理"""
        self._cancelled = True
        logger.info("YourMT3+ 转写器：处理已取消")

    def reset_cancel(self) -> None:
        """重置取消标志"""
        self._cancelled = False

    def _check_cancelled(self) -> None:
        """检查是否已取消"""
        if self._cancelled:
            raise InterruptedError("YourMT3+ 转写处理已取消")
        if self._cancel_check_callback and self._cancel_check_callback():
            raise InterruptedError("YourMT3+ 转写处理已取消")

    def _load_model(
        self,
        model_name: str = "yptf_moe_multi_ps",  # 默认使用 MoE 专家版
        progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """
        加载 YourMT3 MoE 模型

        参数:
            model_name: 模型名称（只支持 MoE 版本）
            progress_callback: 进度回调
        """
        # 双重检查锁定：整个加载过程在锁内执行，防止并发加载
        with YourMT3Transcriber._model_lock:
            # 检查是否已加载（同时验证设备，避免设备切换后使用旧缓存）
            if (YourMT3Transcriber._model is not None and
                    YourMT3Transcriber._model_name == model_name and
                    YourMT3Transcriber._device == self.device):
                logger.debug("模型已加载，跳过重新加载")
                return

            # 解析友好模型名称
            from src.utils.yourmt3_downloader import YOURMT3_MODELS
            model_info = YOURMT3_MODELS.get(model_name, {})
            friendly_name = model_info.get("name", model_name)
            features = ", ".join(model_info.get("features", []))

            if progress_callback:
                progress_callback(0.1, f"正在加载 {friendly_name}...")

            logger.info(f"加载模型: {friendly_name}")
            if features:
                logger.info(f"模型特性: {features}")

            try:
                # 1. 添加 YourMT3 路径到 sys.path
                import sys
                logger.info("正在查找 YourMT3 代码库路径...")
                amt_src_path = _get_yourmt3_amt_src_path()

                if not amt_src_path:
                    raise FileNotFoundError(
                        "未找到 YourMT3 代码库\n"
                        "请确保 YourMT3/ 目录已随程序一起分发"
                    )

                if amt_src_path not in sys.path:
                    sys.path.insert(0, amt_src_path)
                    logger.info(f"添加路径到 sys.path: {amt_src_path}")

                # 2. 导入依赖
                logger.info("正在导入 YourMT3 依赖模块...")
                _clear_yourmt3_import_state()
                import torch
                if self.device.startswith("cuda"):
                    ensure_cuda_runtime_compatibility(self.device)
                from utils.task_manager import TaskManager
                from model.ymt3 import YourMT3
                from config.config import shared_cfg as default_shared_cfg
                from config.config import audio_cfg as default_audio_cfg
                from config.config import model_cfg as default_model_cfg
                logger.debug("YourMT3 import origin | model=%s", getattr(sys.modules.get("model"), "__path__", None))
                logger.debug("YourMT3 import origin | utils=%s", getattr(sys.modules.get("utils"), "__path__", None))
                logger.debug("YourMT3 import origin | config=%s", getattr(sys.modules.get("config"), "__path__", None))
                logger.info("YourMT3 依赖模块导入完成")

                if progress_callback:
                    progress_callback(0.2, "正在获取模型路径...")

                # 3. 获取模型 checkpoint 路径
                from src.utils.yourmt3_downloader import get_model_path

                model_path = get_model_path(model_name)
                if not model_path or not model_path.exists():
                    raise FileNotFoundError(
                        f"YourMT3 MoE 模型未找到: {model_name}\n"
                        f"请先运行: python download_sota_models.py"
                    )

                logger.info(f"使用 checkpoint: {model_path}")

                if progress_callback:
                    progress_callback(0.3, "正在构建配置...")

                # 4. 构建配置参数（参考 model_helper.py）
                import argparse

                # 从 checkpoint 路径提取 exp_id 并使用 @ 扩展指定checkpoint文件
                # 例如: "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt"
                checkpoint_dir = model_path.parent  # checkpoints/
                exp_dir = checkpoint_dir.parent      # mc13_256.../
                exp_id = exp_dir.name                # mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2
                checkpoint_file = model_path.name    # model.ckpt 或 last.ckpt

                # 使用 @ 语法指定checkpoint文件
                exp_id_with_checkpoint = f"{exp_id}@{checkpoint_file}"

                # 需要创建一个符号链接或临时目录结构
                # 因为 initialize_trainer 期望的路径是 amt/logs/ymt3/{exp_id}/checkpoints/{checkpoint}
                # 但实际路径是 ~/.cache/music_ai_models/yourmt3_all/logs/2024/{exp_id}/checkpoints/{checkpoint}

                # 方案：在 YourMT3 目录下创建符号链接
                import tempfile
                yourmt3_logs_dir = os.path.join(amt_src_path, "../logs/ymt3")
                os.makedirs(yourmt3_logs_dir, exist_ok=True)

                # 创建到实际checkpoint目录的符号链接
                symlink_path = os.path.join(yourmt3_logs_dir, exp_id)
                if os.path.islink(symlink_path):
                    os.unlink(symlink_path)
                elif os.path.exists(symlink_path):
                    # 如果是真实目录，不要覆盖
                    pass
                else:
                    import sys as _sys
                    if _sys.platform == "win32":
                        # Windows 普通用户无 symlink 权限；模型通过绝对路径直接加载，无需符号链接
                        logger.debug("Windows 环境跳过符号链接创建，使用绝对路径加载模型")
                    else:
                        try:
                            os.symlink(str(exp_dir), symlink_path)
                            logger.debug(f"创建符号链接: {symlink_path} -> {exp_dir}")
                        except OSError as e:
                            logger.warning(f"无法创建符号链接: {e}，将直接使用绝对路径")

                args = argparse.Namespace(
                    exp_id=exp_id_with_checkpoint,
                    project='ymt3',
                    audio_codec=None,
                    hop_length=None,
                    n_mels=None,
                    input_frames=None,
                    sca_use_query_residual=None,
                    encoder_type='perceiver-tf',
                    decoder_type='multi-t5',
                    pre_encoder_type='default',
                    pre_decoder_type='default',
                    conv_out_channels=None,
                    task_cond_encoder=True,
                    task_cond_decoder=True,
                    d_feat=None,
                    pretrained=False,
                    base_name="google/t5-v1_1-small",
                    encoder_position_encoding_type='rope',
                    decoder_position_encoding_type='default',
                    tie_word_embedding=None,
                    event_length=None,
                    d_latent=None,
                    num_latents=26,
                    perceiver_tf_d_model=None,
                    num_perceiver_tf_blocks=3,
                    num_perceiver_tf_local_transformers_per_block=2,
                    num_perceiver_tf_temporal_transformers_per_block=2,
                    attention_to_channel=True,
                    layer_norm_type='rms_norm',
                    ff_layer_type='moe',
                    ff_widening_factor=4,
                    moe_num_experts=8,
                    moe_topk=2,
                    hidden_act='silu',
                    rotary_type=None,
                    rope_apply_to_keys=None,
                    rope_partial_pe=True,
                    decoder_ff_layer_type=None,
                    decoder_ff_widening_factor=None,
                    task='mt3_full_plus',
                    eval_program_vocab=None,
                    eval_drum_vocab=None,
                    eval_subtask_key='default',
                    onset_tolerance=0.05,
                    test_octave_shift=False,
                    write_model_output=False,
                    precision="32-true",
                    strategy='auto',
                    num_nodes=1,
                    num_gpus='auto',
                    wandb_mode="disabled",
                    debug_mode=False,
                    test_pitch_shift=None,
                    epochs=None
                )

                if progress_callback:
                    progress_callback(0.5, "正在从 checkpoint 加载配置...")

                # 5. 从 checkpoint 加载超参数（最可靠的方法）
                logger.info(f"正在加载 checkpoint: {model_path} ...")
                checkpoint = None
                try:
                    try:
                        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
                    except Exception as e:
                        raise RuntimeError(f"Checkpoint 文件加载失败（可能已损坏）: {e}") from e

                    if 'hyper_parameters' not in checkpoint:
                        raise RuntimeError("Checkpoint 格式无效: 缺少 'hyper_parameters' 键")
                    hparams = checkpoint['hyper_parameters']
                    logger.info("Checkpoint 加载完成，正在提取配置...")

                    # 提取配置
                    audio_cfg = hparams['audio_cfg']
                    model_cfg = hparams['model_cfg']
                    shared_cfg = hparams['shared_cfg']

                    # task_manager 是一个对象，提取其 task_name
                    task_manager_obj = hparams.get('task_manager')
                    if task_manager_obj and hasattr(task_manager_obj, 'task_name'):
                        task_name = task_manager_obj.task_name
                    else:
                        task_name = 'mt3_full_plus'  # 默认任务

                    logger.info(f"从 checkpoint 加载的配置: task={task_name}, encoder={model_cfg['encoder_type']}, decoder={model_cfg['decoder_type']}")

                    if progress_callback:
                        progress_callback(0.6, "正在创建任务管理器...")

                    # 6. 创建任务管理器
                    max_shift_steps_value = shared_cfg["TOKENIZER"]["max_shift_steps"]
                    if isinstance(max_shift_steps_value, str) and max_shift_steps_value == "auto":
                        max_shift_steps = 127  # 默认值
                    else:
                        max_shift_steps = int(max_shift_steps_value)

                    tm = TaskManager(
                        task_name=task_name,
                        max_shift_steps=max_shift_steps,
                        debug_mode=args.debug_mode
                    )
                    logger.info(f"任务: {tm.task_name}, 最大偏移步数: {tm.max_shift_steps}")

                    if progress_callback:
                        progress_callback(0.7, "正在创建模型实例...")

                    # 7. 创建 YourMT3 MoE 模型实例
                    logger.info(f"正在创建 YourMT3 MoE 模型实例并移至 {self.device}...")
                    model = YourMT3(
                        audio_cfg=audio_cfg,
                        model_cfg=model_cfg,
                        shared_cfg=shared_cfg,
                        optimizer=None,
                        task_manager=tm,
                        eval_subtask_key=args.eval_subtask_key,
                        write_output_dir=None
                    ).to(self.device)
                    logger.info("模型实例创建完成")

                    if progress_callback:
                        progress_callback(0.8, "正在加载 checkpoint 权重...")

                    # 8. 加载权重（checkpoint已经在步骤5加载）
                    logger.info("正在加载 checkpoint 权重到模型...")
                    state_dict = checkpoint['state_dict']
                    # 移除 pitch shift 相关权重（不需要）
                    new_state_dict = {k: v for k, v in state_dict.items() if 'pitchshift' not in k}
                    model.load_state_dict(new_state_dict, strict=False)
                    model.eval()
                    logger.info(f"权重加载完成, 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
                finally:
                    del checkpoint  # 确保异常时也释放 checkpoint 内存
                    import gc
                    gc.collect()

                # 9. 缓存模型和配置（已在 _model_lock 内，无需再次加锁）
                YourMT3Transcriber._model = model
                YourMT3Transcriber._model_name = model_name
                YourMT3Transcriber._device = self.device
                YourMT3Transcriber._audio_cfg = audio_cfg
                YourMT3Transcriber._task_manager = tm

                if progress_callback:
                    progress_callback(1.0, "模型加载完成")

                logger.info(f"模型加载完成: {friendly_name}, 设备: {self.device}")

            except Exception as e:
                logger.error(f"加载模型 {friendly_name} 失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise

    def _prepare_and_infer(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        load_progress_weight: float = 0.3,
    ) -> tuple:
        """
        公共推理流程：加载模型 → 获取快照 → 加载音频 → 重采样 → 分段 → 推理

        参数:
            audio_path: 音频文件路径
            progress_callback: 进度回调
            load_progress_weight: 模型加载阶段占总进度的比例

        返回:
            (pred_token_arr, n_segments, audio_cfg, task_manager, slice_hop, onset_threshold)
        """
        if self.device.startswith("cuda"):
            ensure_cuda_runtime_compatibility(self.device)

        # 加载模型
        self._load_model(
            progress_callback=(
                lambda p, m: progress_callback(p * load_progress_weight, m)
                if progress_callback else None
            )
        )

        self._check_cancelled()

        # 获取模型快照
        with YourMT3Transcriber._model_lock:
            model = YourMT3Transcriber._model
            audio_cfg = YourMT3Transcriber._audio_cfg
            task_manager = YourMT3Transcriber._task_manager

        if model is None or audio_cfg is None or task_manager is None:
            raise RuntimeError("YourMT3+ 模型未正确加载，请重试")

        # 加载音频
        import torch
        import torchaudio

        waveform, sr = _load_audio(audio_path)
        logger.info(f"音频加载完成: shape={waveform.shape}, sr={sr}")

        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            logger.info("已转为单声道")

        # 重采样到模型要求的采样率
        target_sr = audio_cfg['sample_rate']
        if sr != target_sr:
            logger.info(f"正在重采样: {sr} -> {target_sr}")
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            logger.info("重采样完成")

        self._check_cancelled()

        # 分段处理
        from utils.audio import slice_padded_array

        audio_np = waveform.numpy()
        input_frames = audio_cfg['input_frames']

        # 根据质量配置选择重叠策略
        quality = getattr(self.config, 'transcription_quality', 'best')
        ultra_quality = getattr(self.config, 'ultra_quality_mode', False)

        if quality == "best" or ultra_quality:
            slice_hop = input_frames * 3 // 4  # 25% 重叠
            onset_threshold = 0.010  # 10ms
            logger.info("使用极致质量模式：25% 重叠 + 10ms 去重阈值")
        elif quality == "balanced":
            slice_hop = input_frames // 2  # 50% 重叠
            onset_threshold = 0.025  # 25ms
            logger.info("使用平衡模式：50% 重叠 + 25ms 去重阈值")
        else:
            slice_hop = input_frames  # 无重叠
            onset_threshold = 0.050  # 50ms
            logger.info("使用快速模式：无重叠 + 50ms 去重阈值")

        audio_segments = slice_padded_array(audio_np, input_frames, slice_hop, pad=True)
        audio_segments = torch.from_numpy(audio_segments.astype('float32')).to(self.device).unsqueeze(1)
        n_segments = audio_segments.shape[0]
        overlap_pct = 1 - slice_hop / input_frames
        logger.info(f"音频分段数: {n_segments}, 每段帧数: {input_frames}, 重叠: {overlap_pct:.0%}")

        self._check_cancelled()

        # 推理
        bsz = get_optimal_batch_size(n_segments, quality, self.device, ultra_quality)
        logger.info(f"推理批处理大小: {bsz}, 预计处理 {(n_segments + bsz - 1) // bsz} 个批次")

        pred_token_arr = self._inference_with_oom_retry(model, bsz, audio_segments, progress_callback)

        self._check_cancelled()

        return pred_token_arr, n_segments, audio_cfg, task_manager, slice_hop, onset_threshold

    def transcribe_full_mix(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[InstrumentType, List[NoteEvent]]:
        """
        直接转写完整音频为多乐器 MIDI（SMART_DIRECT 模式）

        这是 YourMT3+ 的核心功能，可以直接从混音中识别并转写多个乐器，
        无需先进行音源分离。

        参数:
            audio_path: 音频文件路径
            progress_callback: 进度回调

        返回:
            乐器类型到音符列表的字典
        """
        logger.info(f"YourMT3+ 直接转写: {audio_path}")

        self._check_cancelled()

        if progress_callback:
            progress_callback(0.0, "正在准备 YourMT3+ 转写...")

        if not self.is_available():
            raise RuntimeError(self.get_unavailable_reason())

        try:
            pred_token_arr, n_segments, audio_cfg, task_manager, slice_hop, onset_threshold = \
                self._prepare_and_infer(audio_path, progress_callback, load_progress_weight=0.3)

            if progress_callback:
                progress_callback(0.8, "正在处理转写结果...")

            # 解析输出并按乐器类型分类
            result = self._parse_yourmt3_output_from_tokens(
                pred_token_arr,
                n_segments,
                audio_cfg,
                task_manager,
                slice_hop=slice_hop,
                onset_threshold=onset_threshold
            )

            if progress_callback:
                total_notes = sum(len(notes) for notes in result.values())
                progress_callback(1.0, f"发现 {len(result)} 种乐器，共 {total_notes} 个音符")

            return result

        except Exception as e:
            friendly_message = rewrite_cuda_runtime_error(e, self.device)
            logger.error(f"YourMT3+ 转写失败: {friendly_message}")
            raise

    def transcribe_single_stem(
        self,
        audio_path: str,
        instrument_hint: Optional[InstrumentType] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[NoteEvent]:
        """
        转写单个分离轨道

        用于 SMART_SEPARATED 模式，转写人声分离后的单个轨道。

        参数:
            audio_path: 音频文件路径
            instrument_hint: 乐器类型提示
            progress_callback: 进度回调

        返回:
            音符事件列表
        """
        logger.info(f"YourMT3+ 单轨转写: {audio_path} (乐器提示: {instrument_hint})")

        self._check_cancelled()

        if not self.is_available():
            raise RuntimeError("YourMT3+ 不可用，无法转写")

        try:
            pred_token_arr, n_segments, audio_cfg, task_manager, slice_hop, onset_threshold = \
                self._prepare_and_infer(audio_path, progress_callback, load_progress_weight=0.2)

            if progress_callback:
                progress_callback(0.8, "正在处理结果...")

            # 解析输出
            all_notes = self._parse_yourmt3_output_from_tokens(
                pred_token_arr,
                n_segments,
                audio_cfg,
                task_manager,
                slice_hop=slice_hop,
                onset_threshold=onset_threshold
            )

            # 如果有乐器提示，优先返回该乐器的音符
            if instrument_hint and instrument_hint in all_notes:
                notes = all_notes[instrument_hint]
            else:
                # 合并所有音符
                notes = []
                for note_list in all_notes.values():
                    notes.extend(note_list)
                notes.sort(key=lambda n: n.start_time)

            if progress_callback:
                progress_callback(1.0, f"发现 {len(notes)} 个音符")

            return notes

        except Exception as e:
            friendly_message = rewrite_cuda_runtime_error(e, self.device)
            logger.error(f"YourMT3+ 单轨转写失败: {friendly_message}")
            raise RuntimeError(friendly_message) from e

    def _parse_yourmt3_output_from_tokens(
        self,
        pred_token_arr: list,
        n_items: int,
        audio_cfg: dict,
        task_manager: Any,
        slice_hop: Optional[int] = None,
        onset_threshold: float = 0.025
    ) -> Dict[InstrumentType, List[NoteEvent]]:
        """
        从 YourMT3 token 输出解析音符事件（参考 model_helper.py）

        参数:
            pred_token_arr: 预测的 token 数组
            n_items: 音频段数量
            audio_cfg: 音频配置
            task_manager: 任务管理器
            slice_hop: 分段步长（用于计算重叠分段的起始时间）
            onset_threshold: 去重阈值（秒）

        返回:
            乐器类型到音符列表的字典
        """
        from collections import Counter
        from utils.event2note import merge_zipped_note_events_and_ties_to_notes
        from utils.note2event import mix_notes

        result: Dict[InstrumentType, List[NoteEvent]] = {}

        try:
            # 计算每段的起始时间（考虑重叠分段）
            input_frames = audio_cfg['input_frames']
            if slice_hop is None:
                slice_hop = input_frames  # 无重叠
            start_secs_file = [
                slice_hop * i / audio_cfg['sample_rate']
                for i in range(n_items)
            ]

            # 逐通道解析
            num_channels = task_manager.num_decoding_channels
            pred_notes_in_file = []
            n_err_cnt = Counter()

            # 调试：检查 pred_token_arr 的结构
            if pred_token_arr and len(pred_token_arr) > 0:
                first_arr = pred_token_arr[0]
                logger.debug(f"pred_token_arr 长度: {len(pred_token_arr)}, 第一个元素形状: {first_arr.shape}")
                # 检查实际通道数
                actual_channels = first_arr.shape[1] if first_arr.ndim >= 2 else 1
                if actual_channels != num_channels:
                    logger.warning(f"实际通道数 ({actual_channels}) 与期望通道数 ({num_channels}) 不匹配，使用实际值")
                    num_channels = actual_channels

            for ch in range(num_channels):
                logger.info(f"正在解码通道 {ch+1}/{num_channels}...")
                # 提取该通道的 token
                # 确保数组是 3 维的 (B, C, L)
                pred_token_arr_ch = []
                for arr in pred_token_arr:
                    if arr.ndim == 3:
                        pred_token_arr_ch.append(arr[:, ch, :])
                    elif arr.ndim == 2:
                        # 如果是 2 维 (B, L)，只有一个通道
                        if ch == 0:
                            pred_token_arr_ch.append(arr)
                        else:
                            continue
                    else:
                        logger.warning(f"意外的数组维度: {arr.ndim}")
                        continue

                # 跳过空通道
                if not pred_token_arr_ch:
                    logger.debug(f"通道 {ch} 没有数据，跳过")
                    continue

                # 解码 token 为音符事件
                zipped_note_events_and_tie, list_events, ne_err_cnt = task_manager.detokenize_list_batches(
                    pred_token_arr_ch, start_secs_file, return_events=True
                )

                # 合并音符和连音符
                pred_notes_ch, n_err_cnt_ch = merge_zipped_note_events_and_ties_to_notes(zipped_note_events_and_tie)
                pred_notes_in_file.append(pred_notes_ch)
                n_err_cnt += n_err_cnt_ch

            # 混合所有通道的音符
            mixed_notes = mix_notes(pred_notes_in_file)
            logger.info(f"通道合并完成: 共 {len(mixed_notes)} 个原始音符")

            # 使用智能去重：处理重叠分段产生的重复音符
            logger.info(f"正在智能去重 (阈值={onset_threshold*1000:.0f}ms)...")
            mixed_notes = self._deduplicate_overlapping_notes_smart(
                mixed_notes,
                onset_threshold=onset_threshold,
                preserve_longer=True
            )

            # 转换为 NoteEvent 格式并按乐器分类
            # YourMT3 返回的是 Note dataclass 对象列表
            for note in mixed_notes:
                # Note 是 dataclass: is_drum, program, onset, offset, pitch, velocity
                try:
                    # 尝试作为 dataclass 访问
                    if hasattr(note, 'pitch'):
                        is_drum = note.is_drum
                        program = note.program
                        onset = note.onset
                        offset = note.offset
                        pitch = note.pitch
                        velocity = note.velocity
                    # 或者作为 tuple/list 访问（旧版本兼容）
                    elif hasattr(note, '__len__') and len(note) >= 6:
                        is_drum, program, onset, offset, pitch, velocity = note[:6]
                    else:
                        continue
                except (AttributeError, TypeError, ValueError):
                    continue

                # 确保值有效
                pitch = int(pitch)
                program = int(program)
                if pitch < 0 or pitch > 127:
                    continue
                if program < 0 or program > 127:
                    program = 0

                # 保留 YourMT3 人声程序号 (100/101)，不再强制映射为钢琴
                # 下游 pipeline 需要区分人声与钢琴以正确路由到 stem
                # MIDI 生成器会在写入时将 100/101 映射为 GM 52 (Choir Aahs)

                # YourMT3 在 ignore_velocity=True 时只返回 0 或 1
                # 需要将其设置为合理的默认力度值
                if velocity <= 1:
                    velocity = self.config.default_velocity
                else:
                    velocity = int(np.clip(velocity, 1, 127))

                # 使用精确的程序号到乐器类型映射（传入 is_drum 参数）
                instrument = program_to_instrument_type(program, is_drum=bool(is_drum))

                # 跳过无效音符（onset >= offset）
                if float(onset) >= float(offset):
                    continue

                # 创建音符事件
                note_event = NoteEvent(
                    pitch=pitch,
                    start_time=float(onset),
                    end_time=float(offset),
                    velocity=velocity,
                    program=program  # 保存精确的 GM 程序号
                )

                if instrument not in result:
                    result[instrument] = []
                result[instrument].append(note_event)

            # 按开始时间排序每个乐器的音符
            for notes in result.values():
                notes.sort(key=lambda n: n.start_time)

            # 输出统计信息
            logger.info(f"YourMT3 解析完成: {len(result)} 种乐器类型")
            for inst, notes in result.items():
                logger.info(f"  {inst.get_display_name()}: {len(notes)} 个音符")

            if n_err_cnt:
                logger.warning(f"解析错误计数: {dict(n_err_cnt)}")

        except Exception as e:
            logger.error(f"解析 YourMT3 输出失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

        return result

    def _decode_channels_to_notes(
        self,
        pred_token_arr: list,
        n_segments: int,
        audio_cfg: dict,
        task_manager,
        slice_hop: int,
        onset_threshold: float,
    ) -> list:
        """
        公共通道解码流程：多通道 token → 合并音符 → 智能去重

        参数:
            pred_token_arr: 推理输出的 token 数组
            n_segments: 音频分段数
            audio_cfg: 音频配置
            task_manager: YourMT3 任务管理器
            slice_hop: 分段步长
            onset_threshold: 去重阈值

        返回:
            去重后的 mixed_notes 列表（YourMT3 Note 对象）
        """
        from utils.event2note import merge_zipped_note_events_and_ties_to_notes
        from utils.note2event import mix_notes

        start_secs_file = [
            slice_hop * i / audio_cfg['sample_rate']
            for i in range(n_segments)
        ]

        num_channels = task_manager.num_decoding_channels
        pred_notes_in_file = []

        # 检查实际通道数
        if pred_token_arr and len(pred_token_arr) > 0:
            first_arr = pred_token_arr[0]
            actual_channels = first_arr.shape[1] if first_arr.ndim >= 2 else 1
            if actual_channels != num_channels:
                logger.warning(
                    f"实际通道数 ({actual_channels}) 与期望通道数 ({num_channels}) 不匹配，使用实际值"
                )
                num_channels = actual_channels

        for ch in range(num_channels):
            logger.info(f"正在解码通道 {ch+1}/{num_channels}...")
            pred_token_arr_ch = []
            for arr in pred_token_arr:
                if arr.ndim == 3:
                    pred_token_arr_ch.append(arr[:, ch, :])
                elif arr.ndim == 2:
                    if ch == 0:
                        pred_token_arr_ch.append(arr)
                    else:
                        continue
                else:
                    continue

            if not pred_token_arr_ch:
                continue

            zipped_note_events_and_tie, _, _ = task_manager.detokenize_list_batches(
                pred_token_arr_ch, start_secs_file, return_events=True
            )
            pred_notes_ch, _ = merge_zipped_note_events_and_ties_to_notes(zipped_note_events_and_tie)
            pred_notes_in_file.append(pred_notes_ch)

        mixed_notes = mix_notes(pred_notes_in_file)
        logger.info(f"通道合并完成: 共 {len(mixed_notes)} 个原始音符")

        # 智能去重
        logger.info(f"正在智能去重 (阈值={onset_threshold*1000:.0f}ms)...")
        mixed_notes = self._deduplicate_overlapping_notes_smart(
            mixed_notes,
            onset_threshold=onset_threshold,
            preserve_longer=True
        )

        return mixed_notes

    def _notes_to_gm_groups(
        self,
        mixed_notes: list,
        separate_drums: bool = False,
    ) -> Union[Dict[int, List[NoteEvent]], Tuple[Dict[int, List[NoteEvent]], Dict[int, List[NoteEvent]]]]:
        """
        将 YourMT3 Note 对象按 GM 程序号分组为 NoteEvent

        参数:
            mixed_notes: YourMT3 Note 对象列表
            separate_drums: 是否将鼓单独分组

        返回:
            separate_drums=False: Dict[program, List[NoteEvent]]
            separate_drums=True: (instrument_notes, drum_notes)
        """
        instrument_notes: Dict[int, List[NoteEvent]] = {}
        drum_notes: Dict[int, List[NoteEvent]] = {}

        for note in mixed_notes:
            try:
                if hasattr(note, 'pitch'):
                    is_drum = note.is_drum
                    program = note.program
                    onset = note.onset
                    offset = note.offset
                    pitch = note.pitch
                    velocity = note.velocity
                elif hasattr(note, '__len__') and len(note) >= 6:
                    is_drum, program, onset, offset, pitch, velocity = note[:6]
                else:
                    continue
            except (AttributeError, TypeError, ValueError):
                continue

            pitch = int(pitch)
            program = int(program)
            if pitch < 0 or pitch > 127:
                continue
            if program < 0 or program > 127:
                program = 0

            # 跳过无效音符（零时长或负时长）
            if float(onset) >= float(offset):
                continue

            # 保留 YourMT3 人声程序号 (100/101)，不再强制映射为钢琴
            # 下游 pipeline 需要区分人声与钢琴以正确路由到 stem
            # MIDI 生成器会在写入时将 100/101 映射为 GM 0 (钢琴音色)

            # YourMT3 在 ignore_velocity=True 时只返回 0 或 1
            if velocity <= 1:
                velocity = self.config.default_velocity
            else:
                velocity = int(np.clip(velocity, 1, 127))

            note_event = NoteEvent(
                pitch=pitch,
                start_time=float(onset),
                end_time=float(offset),
                velocity=velocity,
                program=program
            )

            if separate_drums and is_drum:
                if pitch not in drum_notes:
                    drum_notes[pitch] = []
                drum_notes[pitch].append(note_event)
            else:
                if program not in instrument_notes:
                    instrument_notes[program] = []
                instrument_notes[program].append(note_event)

        # 排序
        for notes in instrument_notes.values():
            notes.sort(key=lambda n: n.start_time)
        if separate_drums:
            for notes in drum_notes.values():
                notes.sort(key=lambda n: n.start_time)
            return instrument_notes, drum_notes

        return instrument_notes

    def transcribe_full_mix_precise(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[int, List[NoteEvent]]:
        """
        精确转写：直接输出按 GM 程序号分组的音符

        返回的字典键是 GM 程序号 (0-127)，可精确识别 128 种乐器。

        参数:
            audio_path: 音频文件路径
            progress_callback: 进度回调

        返回:
            GM程序号到音符列表的字典 (精确到每种GM乐器)
        """
        logger.info(f"YourMT3+ 精确转写: {audio_path}")

        self._check_cancelled()

        if progress_callback:
            progress_callback(0.0, "正在准备精确转写...")

        if not self.is_available():
            raise RuntimeError(self.get_unavailable_reason())

        try:
            pred_token_arr, n_segments, audio_cfg, task_manager, slice_hop, onset_threshold = \
                self._prepare_and_infer(audio_path, progress_callback, load_progress_weight=0.3)

            if progress_callback:
                progress_callback(0.8, "正在解析精确乐器...")

            mixed_notes = self._decode_channels_to_notes(
                pred_token_arr, n_segments, audio_cfg, task_manager, slice_hop, onset_threshold
            )

            result = self._notes_to_gm_groups(mixed_notes, separate_drums=False)

            if progress_callback:
                total_notes = sum(len(notes) for notes in result.values())
                progress_callback(1.0, f"发现 {len(result)} 种精确乐器，共 {total_notes} 个音符")

            # 输出详细日志
            for program, notes in result.items():
                inst_name = get_instrument_name(program)
                logger.info(f"  GM {program}: {inst_name} - {len(notes)} 个音符")

            return result

        except Exception as e:
            friendly_message = rewrite_cuda_runtime_error(e, self.device)
            logger.error(f"精确转写失败: {friendly_message}", exc_info=True)
            raise RuntimeError(f"精确转写失败: {friendly_message}") from e

    def unload_model(self) -> None:
        """卸载模型以释放 GPU 内存"""
        with YourMT3Transcriber._model_lock:
            if YourMT3Transcriber._model is not None:
                logger.info("正在卸载 YourMT3+ 模型")
                YourMT3Transcriber._model = None
                YourMT3Transcriber._model_name = None
                YourMT3Transcriber._device = None
                # 同步清理配置缓存，避免设备切换后使用过期数据
                YourMT3Transcriber._audio_cfg = None
                YourMT3Transcriber._task_manager = None

                # 清理 GPU 缓存
                try:
                    import torch
                    from src.utils.gpu_utils import clear_gpu_memory
                    clear_gpu_memory()
                except Exception as e:
                    logger.debug("GPU 缓存清理失败: %s", e)

                logger.info("YourMT3+ 模型已卸载")

    def transcribe_precise(
        self,
        audio_path: str,
        quality: str = "best",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[Dict[int, List[NoteEvent]], Dict[int, List[NoteEvent]]]:
        """
        极致精度转写：按 GM 程序号分组输出

        优化策略:
        1. 使用优化的分段策略减少边界错误
        2. 智能去重算法处理重叠分段
        3. 保留精确的 GM 程序号（0-127）

        参数:
            audio_path: 音频文件路径
            quality: 质量模式 ("fast", "balanced", "best")
            progress_callback: 进度回调

        返回:
            (instrument_notes, drum_notes) 元组:
            - instrument_notes: Dict[program 0-127, List[NoteEvent]]
            - drum_notes: Dict[drum_pitch 35-81, List[NoteEvent]]
        """
        logger.info(f"YourMT3+ 极致精度转写: {audio_path} (质量模式: {quality})")

        self._check_cancelled()

        if progress_callback:
            progress_callback(0.0, "正在准备极致精度转写...")

        if not self.is_available():
            raise RuntimeError(self.get_unavailable_reason())

        try:
            pred_token_arr, n_segments, audio_cfg, task_manager, slice_hop, onset_threshold = \
                self._prepare_and_infer(audio_path, progress_callback, load_progress_weight=0.3)

            if progress_callback:
                progress_callback(0.8, "正在解析精确乐器...")

            logger.info("正在解析 token 输出为音符事件...")

            mixed_notes = self._decode_channels_to_notes(
                pred_token_arr, n_segments, audio_cfg, task_manager, slice_hop, onset_threshold
            )

            total_raw_notes = len(mixed_notes)

            instrument_notes, drum_notes = self._notes_to_gm_groups(mixed_notes, separate_drums=True)

            # 过滤误识别乐器：音符数过少的乐器轨道很可能是模型幻觉
            total_instrument_notes = sum(len(notes) for notes in instrument_notes.values())
            min_note_ratio = 0.005  # 占总音符数 0.5% 以下
            min_note_abs = 15       # 绝对数量低于 15
            filtered_programs = []
            for program, notes in list(instrument_notes.items()):
                ratio = len(notes) / max(total_instrument_notes, 1)
                if len(notes) < min_note_abs and ratio < min_note_ratio:
                    filtered_programs.append((program, len(notes)))
                    del instrument_notes[program]

            if filtered_programs:
                for prog, cnt in filtered_programs:
                    logger.info(f"  过滤误识别乐器: GM {prog:03d} ({get_instrument_name(prog)}) - 仅 {cnt} 个音符")

            # 重新统计
            total_instrument_notes = sum(len(notes) for notes in instrument_notes.values())
            total_drum_notes = sum(len(notes) for notes in drum_notes.values())

            if progress_callback:
                progress_callback(1.0, f"发现 {len(instrument_notes)} 种乐器 + {len(drum_notes)} 种鼓音色")

            logger.info(f"极致精度转写完成:")
            logger.info(f"  原始音符: {total_raw_notes}")
            logger.info(f"  乐器音符: {total_instrument_notes} ({len(instrument_notes)} 种 GM 音色)")
            logger.info(f"  鼓音符: {total_drum_notes} ({len(drum_notes)} 种音高)")
            logger.info(f"  保留率: {(total_instrument_notes + total_drum_notes) / max(total_raw_notes, 1):.1%}")

            # 详细乐器日志
            for program, notes in sorted(instrument_notes.items()):
                inst_name = get_instrument_name(program)
                logger.info(f"  GM {program:03d}: {inst_name} - {len(notes)} 个音符")

            return instrument_notes, drum_notes

        except Exception as e:
            friendly_message = rewrite_cuda_runtime_error(e, self.device)
            logger.error(f"极致精度转写失败: {friendly_message}", exc_info=True)
            raise RuntimeError(f"极致精度转写失败: {friendly_message}") from e

    def _deduplicate_overlapping_notes_smart(
        self,
        notes: list,
        onset_threshold: float = 0.025,
        preserve_longer: bool = True
    ) -> list:
        """
        智能去重：处理重叠分段产生的重复音符

        改进策略:
        1. 按 (pitch, program, is_drum) 分组
        2. 在每组内按 onset 排序
        3. 使用动态阈值合并相近的音符
        4. 保留持续时间更长或力度更大的音符

        参数:
            notes: YourMT3 Note 对象列表
            onset_threshold: 判断重复的时间阈值（秒）
            preserve_longer: True = 保留更长音符, False = 保留更早音符

        返回:
            去重后的音符列表
        """
        if not notes:
            return notes

        from collections import defaultdict

        # 按 (pitch, program, is_drum) 分组
        groups = defaultdict(list)
        for note in notes:
            try:
                if hasattr(note, 'pitch'):
                    key = (note.pitch, note.program, note.is_drum)
                elif hasattr(note, '__len__') and len(note) >= 6:
                    is_drum, program, onset, offset, pitch, velocity = note[:6]
                    key = (pitch, program, is_drum)
                else:
                    continue
                groups[key].append(note)
            except (AttributeError, TypeError, ValueError):
                continue

        result = []
        dedup_count = 0

        # 辅助函数提到循环外，避免每次迭代重新定义
        def get_onset(n):
            if hasattr(n, 'onset'):
                return n.onset
            elif hasattr(n, '__len__') and len(n) >= 4:
                return n[2]
            return 0

        def get_offset(n):
            if hasattr(n, 'offset'):
                return n.offset
            elif hasattr(n, '__len__') and len(n) >= 4:
                return n[3]
            return 0

        def get_velocity(n):
            if hasattr(n, 'velocity'):
                return n.velocity
            elif hasattr(n, '__len__') and len(n) >= 6:
                return n[5]
            return 80

        for key, group_notes in groups.items():
            # 按 onset 排序
            group_notes.sort(key=get_onset)

            # 智能去重
            filtered = []
            i = 0
            while i < len(group_notes):
                current = group_notes[i]
                current_onset = get_onset(current)
                current_offset = get_offset(current)
                current_velocity = get_velocity(current)

                # 收集所有在阈值内的音符
                cluster = [current]
                j = i + 1
                while j < len(group_notes):
                    next_note = group_notes[j]
                    next_onset = get_onset(next_note)

                    if next_onset - current_onset <= onset_threshold:
                        cluster.append(next_note)
                        j += 1
                    else:
                        break

                if len(cluster) == 1:
                    # 单个音符，直接保留
                    filtered.append(current)
                else:
                    # 多个音符需要合并
                    dedup_count += len(cluster) - 1

                    if preserve_longer:
                        # 选择持续时间最长的
                        best = max(cluster, key=lambda n: get_offset(n) - get_onset(n))
                    else:
                        # 选择力度最大的
                        best = max(cluster, key=get_velocity)

                    filtered.append(best)

                i = j

            result.extend(filtered)

        # 按 onset 排序返回
        result.sort(key=get_onset)

        if dedup_count > 0:
            logger.info(f"智能去重: 移除 {dedup_count} 个重复音符 (阈值: {onset_threshold*1000:.0f}ms)")

        return result
