"""Image processor for Hallo2 model."""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Image preprocessing for Hallo2 inference.

    Wraps Hallo2's ImageProcessor to handle:
    - Image loading and resizing
    - Face detection and localization
    - Face embedding extraction
    - Multi-scale mask generation
    - Caching of preprocessed data
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        face_analysis_model_path: str,
        enable_cache: bool = True,
    ):
        """
        Initialize image processor.

        Args:
            image_size: Target image size (width, height)
            face_analysis_model_path: Path to face analysis model
            enable_cache: Whether to cache preprocessed images
        """
        self.image_size = image_size
        self.face_analysis_model_path = face_analysis_model_path
        self.enable_cache = enable_cache

        # Lazy import to avoid dependency issues
        self._processor = None

    def _get_processor(self):
        """Lazily initialize the Hallo2 ImageProcessor."""
        if self._processor is None:
            from hallo.datasets.image_processor import ImageProcessor as Hallo2ImageProcessor

            self._processor = Hallo2ImageProcessor(
                self.image_size, self.face_analysis_model_path
            )
        return self._processor

    def preprocess(
        self,
        image_path: str,
        save_dir: str,
        face_expand_ratio: float = 1.2,
    ) -> Tuple:
        """
        Preprocess image for Hallo2 inference.

        Performs face detection, embedding extraction, and mask generation.

        Args:
            image_path: Path to input image
            save_dir: Directory to save preprocessed masks and embeddings
            face_expand_ratio: Expansion ratio for face region

        Returns:
            Tuple of (
                source_image_pixels: torch.Tensor,
                source_image_face_region: torch.Tensor,
                source_image_face_emb: np.ndarray,
                source_image_full_mask: List[torch.Tensor],
                source_image_face_mask: List[torch.Tensor],
                source_image_lip_mask: List[torch.Tensor],
            )
        """
        logger.info(f"Preprocessing image: {image_path}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Check cache
        file_name = Path(image_path).stem
        cache_files = [
            os.path.join(save_dir, f"{file_name}_sep_face.png"),
            os.path.join(save_dir, f"{file_name}_sep_lip.png"),
            os.path.join(save_dir, f"{file_name}_sep_background.png"),
            os.path.join(save_dir, f"{file_name}_face_mask.png"),
            os.path.join(save_dir, f"{file_name}_face_emb.pt"),
        ]

        if self.enable_cache and all(os.path.exists(f) for f in cache_files):
            logger.info(f"Found cached preprocessing for {file_name}, loading from cache")
            return self._load_from_cache(save_dir, file_name)

        # Preprocess using Hallo2 processor
        try:
            processor = self._get_processor()
            result = processor.preprocess(image_path, save_dir, face_expand_ratio)
            logger.info(f"Successfully preprocessed image: {image_path}")
            return result
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}", exc_info=True)
            raise

    def _load_from_cache(self, save_dir: str, file_name: str) -> Tuple:
        """Load preprocessed data from cache."""
        logger.debug(f"Loading cached preprocessing for {file_name}")

        from hallo.datasets.image_processor import ImageProcessor as Hallo2ImageProcessor

        # Create dummy processor just to use its load capabilities
        processor = Hallo2ImageProcessor(self.image_size, self.face_analysis_model_path)

        # Load the image
        image_path = os.path.join(save_dir, f"{file_name}_sep_face.png")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Cached image not found: {image_path}")

        # Load cached embedding
        emb_path = os.path.join(save_dir, f"{file_name}_face_emb.pt")
        if os.path.exists(emb_path):
            face_emb = torch.load(emb_path, map_location="cpu").numpy()
        else:
            logger.warning(f"Cached embedding not found: {emb_path}, returning zeros")
            face_emb = __import__("numpy").zeros((1, 512))

        # For simplicity, rerun preprocessing to get all outputs
        # In a production system, you might want to save more complete cache
        image = Image.open(image_path)

        try:
            # Try to get complete preprocessing from cache/processor
            # This is a fallback; ideally we cache everything
            from hallo.datasets.image_processor import ImageProcessor as Hallo2ImageProcessor
            processor = Hallo2ImageProcessor(
                self.image_size, self.face_analysis_model_path
            )
            # Fall through to full preprocessing
            return processor.preprocess(
                os.path.join(os.path.dirname(save_dir), f"{file_name}.jpg"),
                save_dir,
                1.2,
            )
        except Exception as e:
            logger.error(f"Cache loading failed: {str(e)}")
            raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
