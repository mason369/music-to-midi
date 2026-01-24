"""
GPU detection and management utilities.
"""
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_count() -> int:
    """Get number of available GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    return 0


def get_gpu_info() -> List[dict]:
    """
    Get information about available GPUs.

    Returns:
        List of dicts with GPU info (name, memory, etc.)
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
        logger.warning("PyTorch not installed, cannot detect GPUs")
    except Exception as e:
        logger.error(f"Error detecting GPUs: {e}")

    return gpus


def get_device(prefer_gpu: bool = True, gpu_index: int = 0) -> str:
    """
    Get the best available device.

    Args:
        prefer_gpu: Whether to prefer GPU if available
        gpu_index: Which GPU to use if multiple available

    Returns:
        Device string ('cuda:0', 'mps', or 'cpu')
    """
    try:
        import torch

        if prefer_gpu:
            if torch.cuda.is_available():
                device = f"cuda:{gpu_index}"
                logger.info(f"Using GPU: {torch.cuda.get_device_name(gpu_index)}")
                return device
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Using Apple MPS")
                return "mps"

        logger.info("Using CPU")
        return "cpu"
    except ImportError:
        return "cpu"


def get_memory_info(device: str = "cuda:0") -> Optional[Tuple[float, float]]:
    """
    Get GPU memory usage.

    Args:
        device: Device string

    Returns:
        Tuple of (used_gb, total_gb) or None if not available
    """
    try:
        import torch
        if device.startswith("cuda") and torch.cuda.is_available():
            device_idx = int(device.split(":")[1]) if ":" in device else 0
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
            return (allocated, total)
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")

    return None


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
    except Exception as e:
        logger.error(f"Error clearing GPU memory: {e}")
