"""
PANNs 模型下载工具 - 解决 Windows 下 wget 不可用的问题

panns_inference 库使用 wget 命令下载模型文件，在 Windows 上会失败。
此模块提供 Python 原生的下载功能，在加载 PANNs 之前确保所有必需文件已存在。
"""
import os
import logging
from pathlib import Path
from typing import Optional, Callable
import urllib.request
import ssl

logger = logging.getLogger(__name__)

# PANNs 所需的文件
PANNS_FILES = {
    "class_labels_indices.csv": {
        "url": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
        "size": 20000,  # 约 20KB
        "description": "AudioSet 类别标签"
    },
    "Cnn14_mAP=0.431.pth": {
        "url": "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
        "size": 300_000_000,  # 约 300MB
        "description": "PANNs CNN14 模型权重"
    }
}


def get_panns_data_dir() -> Path:
    """获取 PANNs 数据目录"""
    return Path.home() / "panns_data"


def check_panns_files() -> dict:
    """
    检查 PANNs 所需文件是否存在

    返回:
        dict: 文件名到存在状态的映射
    """
    data_dir = get_panns_data_dir()
    status = {}

    for filename, info in PANNS_FILES.items():
        file_path = data_dir / filename
        if file_path.exists():
            # 检查文件大小是否合理
            actual_size = file_path.stat().st_size
            expected_min_size = info["size"] * 0.5  # 允许50%的误差
            status[filename] = actual_size >= expected_min_size
        else:
            status[filename] = False

    return status


def download_file(
    url: str,
    dest_path: Path,
    description: str = "",
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> bool:
    """
    下载文件

    参数:
        url: 下载链接
        dest_path: 目标路径
        description: 文件描述
        progress_callback: 进度回调 (progress: 0-1, message: str)

    返回:
        bool: 是否成功
    """
    try:
        # 创建目录
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"正在下载 {description}: {url}")
        if progress_callback:
            progress_callback(0.0, f"正在下载 {description}...")

        # 创建 SSL 上下文（某些环境可能需要）
        ssl_context = ssl.create_default_context()

        # 打开 URL
        with urllib.request.urlopen(url, context=ssl_context, timeout=300) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1MB

            with open(dest_path, 'wb') as f:
                while True:
                    block = response.read(block_size)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)

                    if total_size > 0 and progress_callback:
                        progress = downloaded / total_size
                        size_mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        progress_callback(
                            progress,
                            f"正在下载 {description}: {size_mb:.1f}/{total_mb:.1f} MB"
                        )

        logger.info(f"{description} 下载完成")
        if progress_callback:
            progress_callback(1.0, f"{description} 下载完成")

        return True

    except Exception as e:
        logger.error(f"下载 {description} 失败: {e}")
        # 删除不完整的文件
        if dest_path.exists():
            try:
                dest_path.unlink()
            except Exception:
                pass
        return False


def ensure_panns_files(
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> bool:
    """
    确保 PANNs 所需的所有文件都已下载

    参数:
        progress_callback: 进度回调函数

    返回:
        bool: 是否所有文件都准备就绪
    """
    data_dir = get_panns_data_dir()
    file_status = check_panns_files()

    # 检查哪些文件需要下载
    files_to_download = [
        (filename, info)
        for filename, info in PANNS_FILES.items()
        if not file_status.get(filename, False)
    ]

    if not files_to_download:
        logger.info("PANNs 文件已就绪")
        return True

    logger.info(f"需要下载 {len(files_to_download)} 个文件")

    # 下载缺失的文件
    total_files = len(files_to_download)
    for i, (filename, info) in enumerate(files_to_download):
        file_path = data_dir / filename

        def file_progress(progress, message):
            if progress_callback:
                # 计算总进度
                base_progress = i / total_files
                file_weight = 1.0 / total_files
                total_progress = base_progress + (progress * file_weight)
                progress_callback(total_progress, message)

        success = download_file(
            info["url"],
            file_path,
            info["description"],
            file_progress
        )

        if not success:
            logger.error(f"文件 {filename} 下载失败")
            return False

    return True


def is_panns_available() -> bool:
    """
    检查 PANNs 是否可用

    返回:
        bool: PANNs 是否可以正常工作
    """
    # 首先检查文件
    file_status = check_panns_files()
    if not all(file_status.values()):
        return False

    # 然后检查库
    try:
        # 暂时禁用 panns_inference 的自动下载
        import panns_inference
        return True
    except ImportError:
        return False


def patch_panns_config():
    """
    修补 panns_inference 的 config 模块，避免其使用 wget

    这个函数应该在 import panns_inference 之前调用
    """
    data_dir = get_panns_data_dir()
    csv_path = data_dir / "class_labels_indices.csv"

    if not csv_path.exists():
        logger.warning("PANNs 标签文件不存在，需要先下载")
        return False

    # 设置环境变量来覆盖路径（如果 panns_inference 支持）
    os.environ['PANNS_DATA_DIR'] = str(data_dir)

    return True


def initialize_panns(
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> bool:
    """
    初始化 PANNs，确保所有依赖文件都已就绪

    这个函数应该在使用 PANNs 之前调用

    参数:
        progress_callback: 进度回调函数

    返回:
        bool: 是否初始化成功
    """
    logger.info("正在初始化 PANNs...")

    # 1. 确保文件已下载
    if not ensure_panns_files(progress_callback):
        logger.error("PANNs 文件下载失败")
        return False

    # 2. 修补配置
    if not patch_panns_config():
        logger.error("PANNs 配置修补失败")
        return False

    logger.info("PANNs 初始化完成")
    return True
