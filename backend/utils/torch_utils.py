"""PyTorch utilities for Hallo2 backend."""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_device(device_id: Optional[int] = None) -> torch.device:
    """
    Get torch device with optional GPU selection.

    Args:
        device_id: GPU device ID (default: use first available GPU or CPU)

    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        if device_id is not None:
            if device_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{device_id}")
                logger.info(f"Using GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
            else:
                logger.warning(
                    f"GPU {device_id} not available (count: {torch.cuda.device_count()}), "
                    f"falling back to GPU 0"
                )
                device = torch.device("cuda:0")
        else:
            device = torch.device("cuda:0")
            logger.info(f"Using default GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU (inference will be slow)")

    return device


def get_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert dtype string to torch dtype.

    Args:
        dtype_str: Dtype string ('float16', 'float32', 'bfloat16')

    Returns:
        torch.dtype object
    """
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    dtype = dtype_map.get(dtype_str.lower(), torch.float32)

    if dtype == torch.float16:
        logger.info("Using float16 (lower memory, potential precision loss)")
    elif dtype == torch.bfloat16:
        logger.info("Using bfloat16")
    else:
        logger.info("Using float32 (full precision)")

    return dtype


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug(f"Set random seed to {seed}")


def empty_gpu_cache() -> None:
    """Empty GPU memory cache to free up space."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared GPU cache")


def get_gpu_memory() -> dict:
    """
    Get GPU memory usage statistics.

    Returns:
        Dictionary with memory info (MB)
    """
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "current_mb": torch.cuda.memory_allocated() / 1024 / 1024,
        "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """
    Enable gradient checkpointing to save memory during training.

    Args:
        model: PyTorch model
    """
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
        logger.debug("Enabled gradient checkpointing")
