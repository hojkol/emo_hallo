"""Inference endpoints for Hallo2 model."""

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
import torch

from models.schema import Hallo2InferenceRequest, Hallo2InferenceResponse
from models.const import TASK_STATE_QUEUED

logger = logging.getLogger(__name__)

router = APIRouter()

# Upload directory - use project logs/uploads directory
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(os.path.dirname(backend_dir))
UPLOAD_DIR = os.path.join(project_root, "logs", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/inference/hallo2", response_model=Hallo2InferenceResponse)
async def create_hallo2_inference(
    image_file: UploadFile = File(..., description="Source image (JPG/PNG)"),
    audio_file: UploadFile = File(..., description="Driving audio (WAV/MP3)"),
    use_cache: bool = Form(True, description="Whether to use cached preprocessed data"),
):
    """
    Create a new Hallo2 inference task.

    Uploads image and audio files, validates them, and submits a task to the queue.

    Args:
        image_file: Source image file
        audio_file: Driving audio file
        use_cache: Whether to use cached preprocessed data

    Returns:
        Task ID and status
    """
    task_id = str(uuid.uuid4())

    try:
        logger.info(f"Creating inference task {task_id}")

        # Validate image file
        if not image_file.filename or not image_file.filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            raise HTTPException(status_code=400, detail="Invalid image file format")

        # Validate audio file
        if not audio_file.filename or not audio_file.filename.lower().endswith(
            (".wav", ".mp3", ".ogg", ".m4a")
        ):
            raise HTTPException(status_code=400, detail="Invalid audio file format")

        # Create task directory
        task_dir = os.path.join(UPLOAD_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)

        # Save image
        image_path = os.path.join(task_dir, f"image_{image_file.filename}")
        try:
            contents = await image_file.read()
            with open(image_path, "wb") as f:
                f.write(contents)
            logger.info(f"Saved image to {image_path}")
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            raise HTTPException(status_code=400, detail="Failed to save image file")

        # Save audio
        audio_path = os.path.join(task_dir, f"audio_{audio_file.filename}")
        try:
            contents = await audio_file.read()
            with open(audio_path, "wb") as f:
                f.write(contents)
            logger.info(f"Saved audio to {audio_path}")
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            raise HTTPException(status_code=400, detail="Failed to save audio file")

        # Submit task to task manager
        try:
            from app import get_task_manager, get_settings
            task_manager = get_task_manager()
            settings = get_settings()

            # Define task function
            def run_inference(task_id, image_path, audio_path, save_dir, use_cache):
                """Run Hallo2 inference."""
                try:
                    from services.registry import ModelRegistry
                    from services.hallo2 import Hallo2Model, Hallo2Pipeline

                    logger.info(f"[{task_id}] Starting inference...")

                    # Get model
                    registry = ModelRegistry()
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    dtype = torch.float16

                    model_config = settings.get_hallo2_config()
                    model = Hallo2Model(model_config, device=device, dtype=dtype)
                    model.load()

                    # Create pipeline
                    pipeline = Hallo2Pipeline(model)

                    # Prepare inputs
                    inputs = {
                        "image_path": image_path,
                        "audio_path": audio_path,
                        "save_dir": save_dir,
                    }

                    # Define progress callback
                    def progress_callback(message: str, progress: int):
                        task_manager.state_manager.update_task(
                            task_id,
                            progress=progress,
                            status_message=message,
                        )

                    # Run inference
                    preprocessed = pipeline.preprocess(inputs)
                    outputs = pipeline.inference(preprocessed, progress_callback)
                    results = pipeline.postprocess(outputs)

                    # Save results
                    results["video_path"] = os.path.join(save_dir, "output.mp4")
                    results["task_id"] = task_id

                    model.unload()

                    logger.info(f"[{task_id}] Inference complete")
                    return results

                except Exception as e:
                    logger.error(f"[{task_id}] Inference failed: {str(e)}", exc_info=True)
                    raise

            # Submit task
            task_manager.submit_task(
                task_id,
                run_inference,
                image_path=image_path,
                audio_path=audio_path,
                save_dir=task_dir,
                use_cache=use_cache,
            )

            logger.info(f"Task {task_id} submitted successfully")

            return Hallo2InferenceResponse(
                task_id=task_id,
                status="queued",
                message="Inference task queued successfully",
            )

        except Exception as e:
            logger.error(f"Error submitting task: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to submit task: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating inference task: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")


@router.post("/inference/hallo2/config")
async def configure_inference(config: dict):
    """
    Update inference configuration.

    Args:
        config: Configuration dictionary with parameters

    Returns:
        Confirmation message
    """
    try:
        logger.info(f"Updating inference configuration: {config}")

        # Update settings
        from app import get_settings
        settings = get_settings()

        # Apply config updates
        for key, value in config.items():
            if key in ["guidance_scale", "inference_steps", "enable_cache"]:
                settings.config_dict[key] = value

        logger.info("Configuration updated successfully")

        return {
            "status": "success",
            "message": "Configuration updated successfully",
        }

    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
