"""Health check endpoints."""

import logging
import torch
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime

from models.schema import HealthResponse, ModelInfoResponse, ModelsListResponse
from services.registry import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def get_health(registry: ModelRegistry = Depends(lambda: None)):
    """
    Get system health status.

    Returns information about GPU availability and loaded models.
    """
    try:
        gpu_available = torch.cuda.is_available()
        gpu_name = None
        gpu_count = 0

        if gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)

        # Import here to avoid circular imports
        from app import get_model_registry
        registry = get_model_registry()

        # Get list of loaded models
        loaded_models = list(registry._instances.keys())

        return HealthResponse(
            status="healthy",
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            loaded_models=loaded_models,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    except Exception as e:
        logger.error(f"Error in health check: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """
    List all available and loaded models.

    Returns information about model availability and status.
    """
    try:
        from app import get_model_registry
        registry = get_model_registry()

        models_info = {}

        # Get info for all registered models
        for model_name in registry._models.keys():
            try:
                model_instance = registry.get_model(model_name, {}, "cpu", torch.float32)
                info = model_instance.get_model_info()

                models_info[model_name] = ModelInfoResponse(
                    name=info.get("name", model_name),
                    initialized=info.get("initialized", False),
                    version=info.get("version"),
                    device=info.get("device"),
                    dtype=info.get("dtype"),
                )

                # Unload to free memory
                if model_name in registry._instances:
                    try:
                        model_instance.unload()
                    except Exception:
                        pass

            except Exception as e:
                logger.warning(f"Error getting info for {model_name}: {str(e)}")
                models_info[model_name] = ModelInfoResponse(
                    name=model_name,
                    initialized=False,
                    version=None,
                    device=None,
                    dtype=None,
                )

        # Get loaded models
        loaded_models = list(registry._instances.keys())

        return ModelsListResponse(
            models=models_info,
            loaded_models=loaded_models,
        )

    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}", response_model=ModelInfoResponse)
async def get_model_info(model_name: str):
    """
    Get information about a specific model.

    Args:
        model_name: Name of the model (e.g., 'hallo2')
    """
    try:
        from app import get_model_registry
        registry = get_model_registry()

        # Check if model is registered
        if model_name not in registry._models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not registered")

        # Get model info
        model_instance = registry.get_model(model_name, {}, "cpu", torch.float32)
        info = model_instance.get_model_info()

        # Unload to free memory
        if model_name in registry._instances:
            try:
                model_instance.unload()
            except Exception:
                pass

        return ModelInfoResponse(
            name=info.get("name", model_name),
            initialized=info.get("initialized", False),
            version=info.get("version"),
            device=info.get("device"),
            dtype=info.get("dtype"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
