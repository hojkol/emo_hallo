"""Abstract base class for inference pipelines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """Abstract base class for inference pipelines."""

    def __init__(self, model: "BaseModel"):
        """
        Initialize pipeline.

        Args:
            model: Model instance to use for inference
        """
        self.model = model

    @abstractmethod
    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess inputs before inference.

        Args:
            inputs: Raw input data

        Returns:
            Preprocessed data ready for inference
        """
        pass

    @abstractmethod
    def inference(
        self,
        preprocessed: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run inference with progress updates.

        Args:
            preprocessed: Preprocessed input data
            progress_callback: Optional callback for progress updates (progress: 0-100, message: str)

        Returns:
            Raw inference outputs
        """
        pass

    @abstractmethod
    def postprocess(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess outputs after inference.

        Args:
            outputs: Raw inference outputs

        Returns:
            Final output data
        """
        pass

    def run(
        self,
        inputs: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full pipeline: preprocess -> inference -> postprocess.

        Args:
            inputs: Raw input data
            progress_callback: Optional callback for progress updates

        Returns:
            Final output data
        """
        try:
            logger.info("Starting pipeline preprocessing")
            preprocessed = self.preprocess(inputs)

            if progress_callback:
                progress_callback(10, "Preprocessing complete")

            logger.info("Starting inference")
            outputs = self.inference(preprocessed, progress_callback)

            if progress_callback:
                progress_callback(90, "Postprocessing results")

            logger.info("Starting postprocessing")
            results = self.postprocess(outputs)

            if progress_callback:
                progress_callback(100, "Pipeline complete")

            return results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise
