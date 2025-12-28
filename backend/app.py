"""FastAPI application for Hallo2 backend service."""

import logging
import sys
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add project root directory to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
emo_hallo_dir = os.path.dirname(backend_dir)
project_root = os.path.dirname(emo_hallo_dir)

# Add in reverse priority order (highest priority first)
for path in [backend_dir, emo_hallo_dir, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Use relative imports to avoid "attempted relative import beyond top-level package" error
from .config import get_settings
from .services.registry import ModelRegistry
from .services.task_manager import TaskManager

logger = logging.getLogger(__name__)


# Global instances
_model_registry: Optional[ModelRegistry] = None
_task_manager: Optional[TaskManager] = None
_settings = None


def get_model_registry() -> ModelRegistry:
    """Get global model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


def get_task_manager() -> TaskManager:
    """Get global task manager instance."""
    global _task_manager
    if _task_manager is None:
        settings = get_settings()
        max_concurrent = settings.get("max_concurrent_tasks", 1)
        _task_manager = TaskManager(max_concurrent=max_concurrent)
    return _task_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting Hallo2 backend service...")
    try:
        # Initialize settings
        global _settings
        _settings = get_settings()
        logger.info(f"Configuration: {_settings}")

        # Register Hallo2 model
        logger.info("Registering Hallo2 model...")
        from services.hallo2 import Hallo2Model
        registry = get_model_registry()
        registry.register("hallo2", Hallo2Model)
        logger.info("Successfully registered Hallo2 model")

        # Initialize task manager
        task_manager = get_task_manager()
        logger.info(f"Task manager initialized (max concurrent: {task_manager.max_concurrent})")

        logger.info("Hallo2 backend service started successfully")

    except Exception as e:
        logger.error(f"Startup error: {str(e)}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down Hallo2 backend service...")
    try:
        # Shutdown task manager
        task_manager = get_task_manager()
        task_manager.shutdown(timeout=30)

        # Unload all models
        registry = get_model_registry()
        for model_name in list(registry._instances.keys()):
            try:
                model = registry._instances[model_name]
                if hasattr(model, "unload"):
                    model.unload()
                    logger.info(f"Unloaded model: {model_name}")
            except Exception as e:
                logger.warning(f"Error unloading {model_name}: {str(e)}")

        logger.info("Hallo2 backend service shut down successfully")

    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}", exc_info=True)


# Create FastAPI application
app = FastAPI(
    title="Hallo2 Backend API",
    description="Backend service for Hallo2 talking head generation",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "error_code": "INTERNAL_ERROR"},
    )


# Dependency injection
def get_registry() -> ModelRegistry:
    """Dependency: Get model registry."""
    return get_model_registry()


def get_manager() -> TaskManager:
    """Dependency: Get task manager."""
    return get_task_manager()


def get_app_settings():
    """Dependency: Get settings."""
    return get_settings()


# Include routers
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Hallo2 Backend",
        "version": "1.0.0",
        "status": "running",
    }


# Import and include routers after app creation to avoid circular imports
async def include_routers():
    """Include API routers."""
    # Health router
    from controllers import health
    app.include_router(
        health.router,
        prefix="/api/v1",
        tags=["health"],
    )

    # Inference router
    from controllers import inference
    app.include_router(
        inference.router,
        prefix="/api/v1",
        tags=["inference"],
        dependencies=[],
    )

    # Tasks router
    from controllers import tasks
    app.include_router(
        tasks.router,
        prefix="/api/v1",
        tags=["tasks"],
    )


# Include routers on startup
@app.on_event("startup")
async def startup_event():
    """Include routers on startup."""
    await include_routers()


# Health check endpoint (simple, no dependencies)
@app.get("/health")
async def health_check():
    """Simple health check."""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

        return {
            "status": "healthy",
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "gpu_name": gpu_name,
        }
    except Exception as e:
        logger.warning(f"Error in health check: {str(e)}")
        return {"status": "healthy", "gpu_available": False}


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    return app


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get settings
    settings = get_settings()
    port = settings.get("backend_port", 8001)

    logger.info(f"Starting Hallo2 backend on port {port}")

    # Run uvicorn
    uvicorn.run(
        "emo_hallo.backend.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
