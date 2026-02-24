"""
警告过滤器 - 抑制第三方库的英文警告并提供中文替代信息
"""
import os
import sys
import logging
import warnings
import builtins
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

# 警告消息的中文翻译
WARNING_TRANSLATIONS = {
    "Coremltools is not installed": "CoreML 未安装（可选，仅 macOS 需要）",
    "tflite-runtime is not installed": "TFLite 运行时未安装（可选，使用默认模型）",
    "oneDNN custom operations are on": "已启用 oneDNN 优化",
    "This TensorFlow binary is optimized": "TensorFlow 已针对 CPU 指令集优化",
}

# 英文消息到中文的翻译映射
PRINT_TRANSLATIONS = {
    "Predicting MIDI for": "正在分析音频:",
    "Loading model": "正在加载模型",
    "Model loaded": "模型已加载",
    "Processing": "正在处理",
    "Saving": "正在保存",
    "Done": "完成",
}

# 保存原始 print 函数
_original_print = builtins.print


def suppress_third_party_warnings():
    """
    抑制第三方库的英文警告

    在应用程序启动时调用此函数
    """
    # 抑制 TensorFlow 警告
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # 抑制 absl 日志（TensorFlow 使用）
    os.environ['ABSL_MIN_LOG_LEVEL'] = '2'

    # 设置日志级别
    for logger_name in [
        'tensorflow',
        'tensorflow.python',
        'keras',
        'absl',
        'h5py',
        'numba',
        'librosa',
        'audioread',
        'basic_pitch',
        'root',
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # 抑制 Python 警告
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # 抑制特定模块的警告
    warnings.filterwarnings('ignore', module='tensorflow')
    warnings.filterwarnings('ignore', module='keras')
    warnings.filterwarnings('ignore', module='basic_pitch')
    warnings.filterwarnings('ignore', module='librosa')


def translate_message(message: str) -> str:
    """
    将英文消息翻译为中文

    参数:
        message: 英文消息

    返回:
        中文翻译，如果没有找到翻译则返回原消息
    """
    for eng, chn in PRINT_TRANSLATIONS.items():
        if eng in message:
            # 替换英文部分
            translated = message.replace(eng, chn)
            # 清理结尾的 "..."
            if translated.endswith("...") and chn.endswith(":"):
                translated = translated.rstrip("...")
            return translated
    return message


def _patched_print(*args, **kwargs):
    """
    修补后的 print 函数，翻译英文输出
    """
    if args:
        message = str(args[0])

        # 跳过 TensorFlow 的技术性信息
        skip_patterns = [
            "oneDNN custom operations",
            "TF_ENABLE_ONEDNN_OPTS",
            "This TensorFlow binary",
            "SSE SSE2",
            "AVX2 AVX512",
            "rebuild TensorFlow",
            "appropriate compiler flags",
            "tensorflow/core",
            "cpu_feature_guard",
        ]
        for pattern in skip_patterns:
            if pattern in message:
                return  # 完全抑制这些信息

        # 翻译消息
        translated = translate_message(message)
        if translated != message:
            _original_print(translated, *args[1:], **kwargs)
            return

    _original_print(*args, **kwargs)


def patch_print_output():
    """
    修补 print 输出，使其显示中文

    某些库（如 Basic Pitch）直接使用 print 输出进度信息
    """
    builtins.print = _patched_print


def patch_logging_output():
    """
    修补 logging 的 root 日志处理器，翻译警告消息
    """
    class TranslatingFilter(logging.Filter):
        def filter(self, record):
            # 抑制包含特定英文关键词的警告
            skip_keywords = [
                "Coremltools is not installed",
                "tflite-runtime is not installed",
                "oneDNN",
                "TensorFlow binary is optimized",
            ]
            for keyword in skip_keywords:
                if keyword in record.getMessage():
                    return False
            return True

    # 添加过滤器到 root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(TranslatingFilter())

    # 同时添加到警告模块的 logger
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.addFilter(TranslatingFilter())


@contextmanager
def suppress_output():
    """
    上下文管理器：临时抑制标准输出和错误输出
    """
    import io

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class FilteredStderr:
    """
    过滤后的 stderr，抑制特定英文消息
    """

    def __init__(self, original_stderr):
        self.original = original_stderr
        self.skip_patterns = [
            "oneDNN custom operations",
            "TF_ENABLE_ONEDNN_OPTS",
            "This TensorFlow binary",
            "SSE SSE2",
            "AVX2 AVX512",
            "rebuild TensorFlow",
            "appropriate compiler flags",
            "tensorflow/core",
            "cpu_feature_guard",
            "Coremltools is not installed",
            "tflite-runtime is not installed",
            "Could not find platform independent libraries",
        ]

    def write(self, message):
        # 检查是否应该抑制
        for pattern in self.skip_patterns:
            if pattern in message:
                return  # 抑制此消息
        self.original.write(message)

    def flush(self):
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


def patch_stderr_output():
    """
    修补 stderr 输出，过滤英文技术消息
    """
    sys.stderr = FilteredStderr(sys.stderr)


def setup_chinese_environment():
    """
    设置中文环境

    调用此函数来：
    1. 抑制第三方库的英文警告
    2. 修补输出以显示中文
    """
    suppress_third_party_warnings()
    patch_print_output()
    patch_logging_output()
    patch_stderr_output()


# 自定义警告处理器
def _custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    """
    自定义警告处理器，抑制或翻译警告
    """
    msg_str = str(message)

    # 抑制特定警告
    skip_keywords = [
        "Coremltools",
        "tflite-runtime",
        "oneDNN",
        "TensorFlow",
    ]
    for keyword in skip_keywords:
        if keyword in msg_str:
            return  # 完全抑制

    # 显示其他警告
    _original_print(f"警告: {msg_str}", file=sys.stderr)


# 替换默认警告显示函数
warnings.showwarning = _custom_warning_handler
