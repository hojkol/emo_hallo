"""HTTP client for Hallo2 backend service."""

import requests
import logging
import time
import os
from typing import Optional, Dict, Any, Generator
from pathlib import Path

logger = logging.getLogger(__name__)


class Hallo2Client:
    """
    HTTP client for communicating with Hallo2 backend.

    Handles:
    - Health checks
    - Task submission
    - Status polling
    - Video download
    - Error handling and retries
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize Hallo2 client.

        Args:
            base_url: Base URL of backend service (e.g., http://localhost:8001)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_prefix = f"{self.base_url}/api/v1"

    def health_check(self, retries: int = 0) -> Dict[str, Any]:
        """
        Check if backend service is healthy.

        Args:
            retries: Number of retries attempted

        Returns:
            Health status dictionary

        Raises:
            ConnectionError: If backend is not reachable
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError as e:
            if retries < self.max_retries:
                logger.warning(
                    f"Connection failed, retrying... ({retries + 1}/{self.max_retries})"
                )
                time.sleep(1)
                return self.health_check(retries=retries + 1)

            logger.error(f"Backend health check failed: {str(e)}")
            raise ConnectionError(f"Cannot connect to backend at {self.base_url}")

        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            raise

    def create_inference(
        self,
        image_path: str,
        audio_path: str,
        use_cache: bool = True,
    ) -> str:
        """
        Submit an inference task to the backend.

        Args:
            image_path: Path to source image
            audio_path: Path to driving audio
            use_cache: Whether to use cached preprocessed data

        Returns:
            Task ID string

        Raises:
            FileNotFoundError: If image or audio files don't exist
            ValueError: If file formats are invalid
        """
        # Validate files
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        # Validate formats
        valid_image_exts = (".jpg", ".jpeg", ".png")
        valid_audio_exts = (".wav", ".mp3", ".ogg", ".m4a")

        if not image_path.lower().endswith(valid_image_exts):
            raise ValueError(f"Invalid image format: {image_path}")

        if not audio_path.lower().endswith(valid_audio_exts):
            raise ValueError(f"Invalid audio format: {audio_path}")

        try:
            logger.info(f"Submitting inference task: image={image_path}, audio={audio_path}")

            # Prepare multipart form data
            with open(image_path, "rb") as img_f, open(audio_path, "rb") as aud_f:
                files = {
                    "image_file": (Path(image_path).name, img_f, "image/jpeg"),
                    "audio_file": (Path(audio_path).name, aud_f, "audio/wav"),
                }
                data = {"use_cache": use_cache}

                response = requests.post(
                    f"{self.api_prefix}/inference/hallo2",
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )

            response.raise_for_status()
            result = response.json()

            task_id = result.get("task_id")
            logger.info(f"Task created successfully: {task_id}")

            return task_id

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to submit task: {str(e)}")
            raise ValueError(f"Backend request failed: {str(e)}")

        except Exception as e:
            logger.error(f"Error submitting task: {str(e)}")
            raise

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get current status of a task.

        Args:
            task_id: Task identifier

        Returns:
            Task status dictionary with fields:
            - task_id: Task identifier
            - status: Current status (pending, processing, completed, failed)
            - progress: Progress percentage (0-100)
            - status_message: Detailed status message
            - result: Result data (when completed)
            - error: Error message (if failed)

        Raises:
            ValueError: If task not found
        """
        try:
            response = requests.get(
                f"{self.api_prefix}/tasks/{task_id}",
                timeout=self.timeout,
            )

            if response.status_code == 404:
                raise ValueError(f"Task not found: {task_id}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get task status: {str(e)}")
            raise ValueError(f"Backend request failed: {str(e)}")

        except Exception as e:
            logger.error(f"Error getting task status: {str(e)}")
            raise

    def wait_for_completion(
        self,
        task_id: str,
        poll_interval: int = 2,
        timeout: int = 600,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Poll task status until completion.

        Yields status updates as the task progresses.

        Args:
            task_id: Task identifier
            poll_interval: Seconds between status checks
            timeout: Maximum time to wait (seconds)

        Yields:
            Task status dictionaries

        Raises:
            TimeoutError: If task doesn't complete within timeout
            ValueError: If task fails
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

            try:
                status = self.get_task_status(task_id)
                yield status

                # Check completion
                task_status = status.get("status")

                if task_status == "completed":
                    logger.info(f"Task {task_id} completed successfully")
                    return

                if task_status == "failed":
                    error = status.get("error", "Unknown error")
                    raise ValueError(f"Task failed: {error}")

                # Wait before next poll
                time.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Error polling task: {str(e)}")
                raise

    def download_video(
        self,
        task_id: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Download generated video for a completed task.

        Args:
            task_id: Task identifier
            output_path: Path to save video (default: {task_id}.mp4)

        Returns:
            Path to downloaded video file

        Raises:
            ValueError: If task not completed
            IOError: If download fails
        """
        if output_path is None:
            output_path = f"{task_id}.mp4"

        try:
            logger.info(f"Downloading video for task {task_id}")

            response = requests.get(
                f"{self.api_prefix}/tasks/{task_id}/video",
                timeout=self.timeout,
                stream=True,
            )

            if response.status_code == 400:
                raise ValueError("Task not completed or video not ready")

            if response.status_code == 404:
                raise ValueError(f"Task or video not found: {task_id}")

            response.raise_for_status()

            # Save video
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Video downloaded to {output_path}")
            return output_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download video: {str(e)}")
            raise IOError(f"Video download failed: {str(e)}")

        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            raise

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a queued task.

        Args:
            task_id: Task identifier

        Returns:
            True if cancelled, False if task was running or not found

        Raises:
            ValueError: If cancellation fails
        """
        try:
            logger.info(f"Cancelling task {task_id}")

            response = requests.delete(
                f"{self.api_prefix}/tasks/{task_id}",
                timeout=self.timeout,
            )

            if response.status_code == 400:
                logger.warning(f"Cannot cancel task {task_id}: already running or not found")
                return False

            response.raise_for_status()
            logger.info(f"Task {task_id} cancelled successfully")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to cancel task: {str(e)}")
            raise ValueError(f"Cancellation failed: {str(e)}")

        except Exception as e:
            logger.error(f"Error cancelling task: {str(e)}")
            raise

    def generate_video(
        self,
        image_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
        poll_interval: int = 2,
        timeout: int = 600,
    ) -> str:
        """
        Complete end-to-end video generation workflow.

        Submits task, polls for completion, and downloads result.

        Args:
            image_path: Path to source image
            audio_path: Path to driving audio
            output_path: Path to save video (default: {task_id}.mp4)
            poll_interval: Seconds between status checks
            timeout: Maximum time to wait

        Returns:
            Path to generated video

        Raises:
            Various exceptions if any step fails
        """
        try:
            # Submit task
            task_id = self.create_inference(image_path, audio_path)
            logger.info(f"Created task {task_id}")

            if output_path is None:
                output_path = f"{task_id}.mp4"

            # Poll for completion
            for status in self.wait_for_completion(
                task_id,
                poll_interval=poll_interval,
                timeout=timeout,
            ):
                progress = status.get("progress", 0)
                message = status.get("status_message", "")
                logger.info(f"Task {task_id}: {progress}% - {message}")

            # Download video
            video_path = self.download_video(task_id, output_path)
            logger.info(f"Video generation complete: {video_path}")

            return video_path

        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            raise
