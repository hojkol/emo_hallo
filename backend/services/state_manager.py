"""Task state management."""

from typing import Dict, Any, Optional
from datetime import datetime
import logging
import threading

# Import with multiple fallback strategies
import sys
import os

# Add backend directory to path as fallback
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from ..models.const import (
    TASK_STATE_PENDING,
    TASK_STATE_PROCESSING,
    TASK_STATE_COMPLETED,
    TASK_STATE_FAILED,
)

logger = logging.getLogger(__name__)


class StateManager:
    """
    In-memory task state management.

    Tracks task status, progress, and results.
    Thread-safe using locks for concurrent access.
    """

    def __init__(self):
        """Initialize state manager."""
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create_task(self, task_id: str) -> None:
        """
        Create a new task entry.

        Args:
            task_id: Unique task identifier
        """
        with self._lock:
            if task_id in self._tasks:
                logger.warning(f"Task {task_id} already exists")
                return

            self._tasks[task_id] = {
                "task_id": task_id,
                "status": TASK_STATE_PENDING,
                "progress": 0,
                "status_message": "Task created",
                "result": None,
                "error": None,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            logger.info(f"Created task {task_id}")

    def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        status_message: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Update task state.

        Args:
            task_id: Task identifier
            status: New status
            progress: Progress percentage (0-100)
            status_message: Status message
            **kwargs: Additional fields to update
        """
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"Task {task_id} not found, creating...")
                self.create_task(task_id)

            if status:
                self._tasks[task_id]["status"] = status
            if progress is not None:
                self._tasks[task_id]["progress"] = min(100, max(0, progress))
            if status_message:
                self._tasks[task_id]["status_message"] = status_message

            # Update other fields
            for key, value in kwargs.items():
                if key not in ["task_id", "created_at"]:
                    self._tasks[task_id][key] = value

            self._tasks[task_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"

    def complete_task(
        self, task_id: str, result: Dict[str, Any], **kwargs
    ) -> None:
        """
        Mark task as completed.

        Args:
            task_id: Task identifier
            result: Task result dictionary
            **kwargs: Additional fields
        """
        with self._lock:
            self.update_task(
                task_id,
                status=TASK_STATE_COMPLETED,
                progress=100,
                status_message="Task completed",
                result=result,
                **kwargs,
            )
            logger.info(f"Task {task_id} completed")

    def fail_task(self, task_id: str, error: str, **kwargs) -> None:
        """
        Mark task as failed.

        Args:
            task_id: Task identifier
            error: Error message
            **kwargs: Additional fields
        """
        with self._lock:
            self.update_task(
                task_id,
                status=TASK_STATE_FAILED,
                status_message=f"Task failed: {error}",
                error=error,
                **kwargs,
            )
            logger.error(f"Task {task_id} failed: {error}")

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task state.

        Args:
            task_id: Task identifier

        Returns:
            Task state dictionary or None if not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                # Return a copy to avoid external modifications
                return dict(task)
            return None

    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all tasks.

        Returns:
            Dictionary of all tasks
        """
        with self._lock:
            return {task_id: dict(task) for task_id, task in self._tasks.items()}

    def delete_task(self, task_id: str) -> None:
        """
        Delete task from state.

        Args:
            task_id: Task identifier
        """
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                logger.info(f"Deleted task {task_id}")

    def clear_all_tasks(self) -> None:
        """Clear all tasks (use with caution)."""
        with self._lock:
            self._tasks.clear()
            logger.warning("Cleared all tasks")

    def task_exists(self, task_id: str) -> bool:
        """
        Check if task exists.

        Args:
            task_id: Task identifier

        Returns:
            True if task exists
        """
        with self._lock:
            return task_id in self._tasks
