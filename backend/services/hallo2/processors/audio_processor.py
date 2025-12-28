"""Audio processor for Hallo2 model."""

import os
import logging
from typing import Tuple, Optional
import torch

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Audio preprocessing for Hallo2 inference.

    Wraps Hallo2's AudioProcessor to handle:
    - Audio loading and resampling
    - Vocal separation using audio separator
    - WAV2Vec2 feature extraction
    - Audio embedding generation
    - Temporal windowing
    - Caching of preprocessed audio
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        fps: int = 25,
        wav2vec_model_path: str = None,
        wav2vec_only_last_features: bool = True,
        audio_separator_model_dir: str = None,
        audio_separator_model_file: str = None,
        cache_dir: str = None,
        enable_cache: bool = True,
    ):
        """
        Initialize audio processor.

        Args:
            sample_rate: Audio sample rate (must be 16000 for WAV2Vec)
            fps: Video frames per second for temporal alignment
            wav2vec_model_path: Path to WAV2Vec2 model
            wav2vec_only_last_features: Whether to use only last layer features
            audio_separator_model_dir: Directory containing audio separator model
            audio_separator_model_file: Audio separator model filename
            cache_dir: Directory for caching preprocessed audio
            enable_cache: Whether to cache preprocessed audio
        """
        assert (
            sample_rate == 16000
        ), f"sample_rate must be 16000, got {sample_rate}"

        self.sample_rate = sample_rate
        self.fps = fps
        self.wav2vec_model_path = wav2vec_model_path
        self.wav2vec_only_last_features = wav2vec_only_last_features
        self.audio_separator_model_dir = audio_separator_model_dir
        self.audio_separator_model_file = audio_separator_model_file
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache

        # Lazy import to avoid dependency issues
        self._processor = None

    def _get_processor(self):
        """Lazily initialize the Hallo2 AudioProcessor."""
        if self._processor is None:
            from hallo.datasets.audio_processor import AudioProcessor as Hallo2AudioProcessor

            self._processor = Hallo2AudioProcessor(
                sample_rate=self.sample_rate,
                fps=self.fps,
                wav2vec_model_path=self.wav2vec_model_path,
                wav2vec_only_last_features=self.wav2vec_only_last_features,
                audio_separator_model_dir=self.audio_separator_model_dir,
                audio_separator_model_file=self.audio_separator_model_file,
                output_dir=self.cache_dir,
            )
        return self._processor

    def preprocess(
        self,
        audio_path: str,
        clip_length: int = 16,
        padding: bool = False,
        processed_length: int = 0,
    ) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio for Hallo2 inference.

        Performs vocal separation, WAV2Vec2 feature extraction, and temporal windowing.

        Args:
            audio_path: Path to input audio file
            clip_length: Frame length for each clip (typically 16)
            padding: Whether to apply padding for the last segment
            processed_length: Length of previously processed audio

        Returns:
            Tuple of (audio_emb, audio_length) where:
            - audio_emb: torch.Tensor of shape (frames, 768) or (frames, 12, 768)
            - audio_length: Integer number of audio frames
        """
        logger.info(f"Preprocessing audio: {audio_path}")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            processor = self._get_processor()
            audio_emb, audio_length = processor.preprocess(
                audio_path, clip_length, padding=padding, processed_length=processed_length
            )
            logger.info(
                f"Successfully preprocessed audio: {audio_path} "
                f"(shape: {audio_emb.shape}, length: {audio_length})"
            )
            return audio_emb, audio_length
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}", exc_info=True)
            raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
