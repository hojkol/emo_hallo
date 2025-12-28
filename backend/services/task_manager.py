"""Async task execution and queue management."""

import threading
import queue
from typing import Callable, Any, Dict, Optional
import logging

# Import with multiple fallback strategies
import sys
import os

# Add backend directory to path as fallback
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from .state_manager import StateManager
from ..models.const import TASK_STATE_PROCESSING

logger = logging.getLogger(__name__)


class TaskManager:
    """
    Manages asynchronous task execution.

    Implements a task queue with configurable concurrency limit.
    Executes tasks in background threads and tracks state via StateManager.
    """

    def __init__(self, max_concurrent: int = 1):
        """
        Initialize task manager.

        Args:
            max_concurrent: Maximum concurrent tasks (default: 1 for GPU safety)
        """
        self.max_concurrent = max_concurrent
        self.current_tasks = 0
        self.lock = threading.RLock()
        self.queue: queue.Queue = queue.Queue()
        self.state_manager = StateManager()
        self.stop_event = threading.Event()

    def submit_task(
        self,
        task_id: str,
        task_func: Callable,
        *args,
        **kwargs,
    ) -> None:
        """
        Submit a task for execution.

        If max concurrent tasks reached, task is queued.

        Args:
            task_id: Unique task identifier
            task_func: Function to execute
            *args: Positional arguments for task_func
            **kwargs: Keyword arguments for task_func
        """
        # Create task entry
        self.state_manager.create_task(task_id)

        with self.lock:
            if self.current_tasks < self.max_concurrent:
                self._execute_task_async(task_id, task_func, args, kwargs)
            else:
                logger.info(
                    f"Task {task_id} queued (max concurrent {self.max_concurrent} reached)"
                )
                self.queue.put((task_id, task_func, args, kwargs))

    def _execute_task_async(
        self,
        task_id: str,
        task_func: Callable,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """
        Execute task in background thread.

        Args:
            task_id: Task identifier
            task_func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
        """
        with self.lock:
            self.current_tasks += 1

        thread = threading.Thread(
            target=self._run_task,
            args=(task_id, task_func, args, kwargs),
            daemon=True,
        )
        thread.start()
        logger.info(f"Task {task_id} started in background thread")

    def _run_task(
        self,
        task_id: str,
        task_func: Callable,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """
        Run task with error handling and state management.

        Args:
            task_id: Task identifier
            task_func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
        """
        try:
            # Update status to processing
            self.state_manager.update_task(
                task_id,
                status=TASK_STATE_PROCESSING,
                status_message="Task processing",
            )

            logger.info(f"Executing task {task_id}")

            # Execute task function
            result = task_func(task_id, *args, **kwargs)

            # Mark as complete
            self.state_manager.complete_task(task_id, result or {})
            logger.info(f"Task {task_id} completed successfully")

        except Exception as e:
            # Mark as failed
            error_msg = str(e)
            self.state_manager.fail_task(task_id, error_msg)
            logger.error(f"Task {task_id} failed: {error_msg}", exc_info=True)

        finally:
            with self.lock:
                self.current_tasks -= 1

            # Check queue for pending tasks
            self._process_queue()

    def _process_queue(self) -> None:
        """Process next queued task if capacity available."""
        with self.lock:
            if self.current_tasks < self.max_concurrent and not self.queue.empty():
                try:
                    task_id, task_func, args, kwargs = self.queue.get_nowait()
                    self._execute_task_async(task_id, task_func, args, kwargs)
                except queue.Empty:
                    pass

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status.

        Args:
            task_id: Task identifier

        Returns:
            Task status dictionary
        """
        return self.state_manager.get_task(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a queued task (cannot cancel running tasks).

        Args:
            task_id: Task identifier

        Returns:
            True if cancelled, False if not found or running
        """
        # Convert queue to list, remove task, convert back
        temp_queue = []
        found = False

        while not self.queue.empty():
            try:
                queued_task = self.queue.get_nowait()
                if queued_task[0] != task_id:
                    temp_queue.append(queued_task)
                else:
                    found = True
                    logger.info(f"Cancelled queued task {task_id}")
            except queue.Empty:
                break

        # Restore queue
        for task in temp_queue:
            self.queue.put(task)

        return found

    def shutdown(self, timeout: float = 30.0) -> None:
        """
        Shutdown task manager and wait for running tasks.

        Args:
            timeout: Max wait time for tasks to complete
        """
        logger.info("Shutting down task manager...")
        self.stop_event.set()

        # Wait for tasks to complete
        start = threading.current_thread()
        for _ in range(int(timeout * 10)):
            with self.lock:
                if self.current_tasks == 0:
                    logger.info("All tasks completed")
                    return
            threading.Event().wait(0.1)

        logger.warning(
            f"Timeout waiting for {self.current_tasks} tasks to complete"
        )
