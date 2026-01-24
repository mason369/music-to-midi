"""
GPU检测和管理工具
"""
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """检查CUDA是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_count() -> int:
    """获取可用GPU数量"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    return 0


def get_gpu_info() -> List[dict]:
    """
    获取可用GPU信息

    返回:
        包含GPU信息（名称、内存等）的字典列表
    """
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "major": props.major,
                    "minor": props.minor,
                })
    except ImportError:
        logger.warning("未安装PyTorch，无法检测GPU")
    except Exception as e:
        logger.error(f"检测GPU时出错: {e}")

    return gpus


def get_device(prefer_gpu: bool = True, gpu_index: int = 0) -> str:
    """
    获取最佳可用设备

    参数:
        prefer_gpu: 是否优先使用GPU
        gpu_index: 多GPU时使用哪个GPU

    返回:
        设备字符串（'cuda:0', 'mps', 或 'cpu'）
    """
    try:
        import torch

        if prefer_gpu:
            if torch.cuda.is_available():
                device = f"cuda:{gpu_index}"
                logger.info(f"使用GPU: {torch.cuda.get_device_name(gpu_index)}")
                return device
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("使用Apple MPS")
                return "mps"

        logger.info("使用CPU")
        return "cpu"
    except ImportError:
        return "cpu"


def get_memory_info(device: str = "cuda:0") -> Optional[Tuple[float, float]]:
    """
    获取GPU内存使用情况

    参数:
        device: 设备字符串

    返回:
        (已用GB, 总共GB) 元组，如不可用则返回None
    """
    try:
        import torch
        if device.startswith("cuda") and torch.cuda.is_available():
            device_idx = int(device.split(":")[1]) if ":" in device else 0
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
            return (allocated, total)
    except Exception as e:
        logger.error(f"获取内存信息时出错: {e}")

    return None


def clear_gpu_memory() -> None:
    """清除GPU内存缓存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU内存缓存已清除")
    except Exception as e:
        logger.error(f"清除GPU内存时出错: {e}")
