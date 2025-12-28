"""Abstract base class for all AI models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all AI models."""

    def __init__(self, config: Dict[str, Any], device: str, dtype: torch.dtype):
        """
        Initialize model.

        Args:
            config: Model configuration dictionary
            device: Device to load model on (e.g., "cuda:0", "cpu")
            dtype: PyTorch data type (torch.float32, torch.float16, torch.bfloat16)
        """
        self.config = config
        self.device = device
        self.dtype = dtype
        self._initialized = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights and initialize."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory/GPU."""
        pass

    @property
    def initialized(self) -> bool:
        """Check if model is initialized."""
        return self._initialized

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model metadata.

        Returns:
            Dictionary with model info (name, version, params, etc.)
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup on exit."""
        try:
            self.unload()
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
