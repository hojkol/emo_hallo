"""Central registry for all available models."""

from typing import Dict, Type, Optional, Any
import torch
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for all available models.

    Implements the registry pattern for plugin-like model management.
    Models are lazily loaded on first use and can be unloaded to free memory.
    """

    _models: Dict[str, Type] = {}  # Model class registry
    _instances: Dict[str, Any] = {}  # Model instance cache
    _lock = None  # Thread lock for thread-safe access (set at runtime)

    @classmethod
    def register(cls, name: str, model_class: Type) -> None:
        """
        Register a model class.

        Args:
            name: Unique model identifier
            model_class: Model class (should inherit from BaseModel)
        """
        if name in cls._models:
            logger.warning(f"Overwriting existing model registration: {name}")
        cls._models[name] = model_class
        logger.info(f"Registered model: {name}")

    @classmethod
    def get_model(
        cls, name: str, config: Dict[str, Any], device: str, dtype: torch.dtype
    ) -> Any:
        """
        Get or create model instance (lazy loading).

        Args:
            name: Model name
            config: Model configuration
            device: Device to load on
            dtype: Data type

        Returns:
            Model instance

        Raises:
            ValueError: If model not registered
        """
        if name not in cls._models:
            raise ValueError(
                f"Model '{name}' not registered. Available: {list(cls._models.keys())}"
            )

        # Return cached instance if exists
        if name in cls._instances:
            logger.debug(f"Using cached model instance: {name}")
            return cls._instances[name]

        # Create and cache new instance
        logger.info(f"Loading model: {name}")
        model_class = cls._models[name]
        model = model_class(config, device, dtype)
        model.load()
        cls._instances[name] = model

        return model

    @classmethod
    def unload_model(cls, name: str) -> None:
        """
        Unload a specific model from memory.

        Args:
            name: Model name
        """
        if name in cls._instances:
            try:
                logger.info(f"Unloading model: {name}")
                cls._instances[name].unload()
                del cls._instances[name]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error unloading model {name}: {e}")

    @classmethod
    def unload_all_models(cls) -> None:
        """Unload all loaded models."""
        model_names = list(cls._instances.keys())
        for name in model_names:
            cls.unload_model(name)

    @classmethod
    def list_models(cls) -> list:
        """
        List all registered model names.

        Returns:
            List of registered model names
        """
        return list(cls._models.keys())

    @classmethod
    def list_loaded_models(cls) -> list:
        """
        List all currently loaded model instances.

        Returns:
            List of loaded model names
        """
        return list(cls._instances.keys())

    @classmethod
    def get_model_info(cls, name: str) -> Dict[str, Any]:
        """
        Get model information.

        Args:
            name: Model name

        Returns:
            Model info dictionary

        Raises:
            ValueError: If model not registered
        """
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not registered")

        if name in cls._instances:
            return cls._instances[name].get_model_info()
        else:
            return {"name": name, "initialized": False}

    @classmethod
    def get_all_models_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered models.

        Returns:
            Dictionary mapping model names to their info
        """
        info = {}
        for name in cls._models:
            info[name] = cls.get_model_info(name)
        return info
