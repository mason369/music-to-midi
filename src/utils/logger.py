"""
应用程序日志配置
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# ANSI 颜色码
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_LEVEL_COLORS = {
    logging.DEBUG:    "\033[36m",    # 青色
    logging.INFO:     "\033[32m",    # 绿色
    logging.WARNING:  "\033[33m",    # 黄色
    logging.ERROR:    "\033[31m",    # 红色
    logging.CRITICAL: "\033[35m",    # 品红（粗体）
}

_LEVEL_LABELS = {
    logging.DEBUG:    "DEBUG   ",
    logging.INFO:     "INFO    ",
    logging.WARNING:  "WARNING ",
    logging.ERROR:    "ERROR   ",
    logging.CRITICAL: "CRITICAL",
}


class ColoredFormatter(logging.Formatter):
    """带 ANSI 颜色的控制台格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        label = _LEVEL_LABELS.get(record.levelno, record.levelname.ljust(8))

        # 时间戳：暗色
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        ts_str = f"{_DIM}{ts}{_RESET}"

        # 级别：对应颜色 + 粗体
        bold = _BOLD if record.levelno >= logging.WARNING else ""
        level_str = f"{bold}{color}{label}{_RESET}"

        # 模块名：暗色
        name_str = f"{_DIM}{record.name}:{record.lineno}{_RESET}"

        # 消息：ERROR/CRITICAL 用颜色高亮，其余正常
        if record.levelno >= logging.ERROR:
            msg_str = f"{color}{record.getMessage()}{_RESET}"
        else:
            msg_str = record.getMessage()

        return f"{ts_str} | {level_str} | {name_str} | {msg_str}"


class PlainFormatter(logging.Formatter):
    """写入文件时不含颜色的纯文本格式化器"""
    pass


def setup_logger(
    name: str = "music_to_midi",
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    设置并配置日志记录器

    参数:
        name: 日志记录器名称
        level: 日志级别
        log_dir: 日志文件目录（None = 不记录到文件）
        console: 是否输出到控制台

    返回:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # 禁止向 root logger 传播，避免第三方库（如 intel_extension_for_pytorch）
    # 通过 basicConfig 安装的处理器产生重复输出
    logger.propagate = False

    # 清除现有处理器
    logger.handlers.clear()

    # 控制台处理器（带颜色）
    # PyInstaller --windowed 模式下 sys.stdout/sys.stderr 为 None，跳过控制台输出
    if console and sys.stdout is not None:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        # 仅在 TTY 终端启用颜色；重定向到文件时回退到纯文本
        if getattr(sys.stdout, 'isatty', lambda: False)():
            console_handler.setFormatter(ColoredFormatter())
        else:
            console_handler.setFormatter(PlainFormatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
        logger.addHandler(console_handler)

    # 文件处理器（纯文本，无颜色）
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        log_file = log_path / f"music_to_midi_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(PlainFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "music_to_midi") -> logging.Logger:
    """获取现有日志记录器或创建新的"""
    return logging.getLogger(name)
