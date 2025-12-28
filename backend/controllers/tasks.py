"""Task management endpoints."""

import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse

from models.schema import TaskStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Upload directory - use project logs/uploads directory
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(os.path.dirname(backend_dir))
UPLOAD_DIR = os.path.join(project_root, "logs", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get status of an inference task.

    Args:
        task_id: Task identifier

    Returns:
        Task status with progress, result, and error information
    """
    try:
        logger.debug(f"Fetching status for task {task_id}")

        from app import get_task_manager
        task_manager = get_task_manager()

        # Get task status from state manager
        task_status = task_manager.get_task_status(task_id)

        if task_status is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        return TaskStatusResponse(
            task_id=task_status.get("task_id"),
            status=task_status.get("status"),
            progress=task_status.get("progress", 0),
            status_message=task_status.get("status_message", ""),
            result=task_status.get("result"),
            error=task_status.get("error"),
            created_at=task_status.get("created_at", ""),
            updated_at=task_status.get("updated_at", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}/video")
async def download_task_video(task_id: str):
    """
    Download generated video for a completed task.

    Args:
        task_id: Task identifier

    Returns:
        Video file download
    """
    try:
        logger.info(f"Downloading video for task {task_id}")

        from app import get_task_manager
        task_manager = get_task_manager()

        # Get task status
        task_status = task_manager.get_task_status(task_id)

        if task_status is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # Check task completion
        if task_status.get("status") != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Task not completed. Status: {task_status.get('status')}",
            )

        # Get video path from result
        result = task_status.get("result", {})
        video_path = result.get("video_path")

        if not video_path or not os.path.exists(video_path):
            # Try default location
            video_path = os.path.join(UPLOAD_DIR, task_id, "output.mp4")
            if not os.path.exists(video_path):
                raise HTTPException(status_code=404, detail="Video file not found")

        logger.info(f"Sending video: {video_path}")

        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"hallo2_{task_id}.mp4",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a queued task (cannot cancel running tasks).

    Args:
        task_id: Task identifier

    Returns:
        Cancellation status
    """
    try:
        logger.info(f"Cancelling task {task_id}")

        from app import get_task_manager
        task_manager = get_task_manager()

        # Attempt to cancel
        cancelled = task_manager.cancel_task(task_id)

        if not cancelled:
            raise HTTPException(
                status_code=400,
                detail="Task not found in queue or already running",
            )

        logger.info(f"Task {task_id} cancelled successfully")

        return {
            "status": "cancelled",
            "task_id": task_id,
            "message": "Task cancelled successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks")
async def list_tasks():
    """
    List all tasks and their status.

    Returns:
        Dictionary of all tasks
    """
    try:
        logger.debug("Listing all tasks")

        from app import get_task_manager
        task_manager = get_task_manager()

        # Get all tasks
        all_tasks = task_manager.state_manager.get_all_tasks()

        return {
            "tasks": all_tasks,
            "count": len(all_tasks),
        }

    except Exception as e:
        logger.error(f"Error listing tasks: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/logs")
async def get_task_logs(task_id: str):
    """
    Get logs for a specific task.

    Args:
        task_id: Task identifier

    Returns:
        Task logs (if available)
    """
    try:
        logger.info(f"Fetching logs for task {task_id}")

        # Try to read logs from task directory
        log_file = os.path.join(UPLOAD_DIR, task_id, "task.log")

        if not os.path.exists(log_file):
            return {
                "task_id": task_id,
                "logs": "No logs available",
            }

        with open(log_file, "r") as f:
            logs = f.read()

        return {
            "task_id": task_id,
            "logs": logs,
        }

    except Exception as e:
        logger.error(f"Error getting task logs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
