"""Mask processor for Hallo2 model."""

import logging
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class MaskProcessor:
    """
    Mask prediction and processing for Hallo2 inference.

    Handles:
    - Face/lip mask generation from static source
    - Autoregressive mask prediction using MaskPredictUNet
    - Multi-scale mask smoothing and refinement
    - Mask blending for natural transitions
    """

    def __init__(
        self,
        mask_predict_model=None,
        clip_length: int = 16,
        mask_height: int = 64,
        mask_width: int = 64,
        smooth_kernel: int = 3,
        smooth_tau: float = 0.05,
        smooth_eps: float = 0.15,
    ):
        """
        Initialize mask processor.

        Args:
            mask_predict_model: Loaded MaskPredictUNet model (optional)
            clip_length: Number of frames per clip
            mask_height: Height of predicted masks
            mask_width: Width of predicted masks
            smooth_kernel: Kernel size for smoothing
            smooth_tau: Threshold for soft thresholding
            smooth_eps: Transition width for smooth thresholding
        """
        self.mask_predict_model = mask_predict_model
        self.clip_length = clip_length
        self.mask_height = mask_height
        self.mask_width = mask_width
        self.smooth_kernel = smooth_kernel
        self.smooth_tau = smooth_tau
        self.smooth_eps = smooth_eps

        self.transform_64 = transforms.Compose([
            transforms.Resize((mask_height, mask_width)),
            transforms.ToTensor(),
        ])

    def load_static_masks(
        self, face_mask_path: str, lip_mask_path: str, full_mask_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load and resize static masks to clip length.

        Args:
            face_mask_path: Path to face mask image
            lip_mask_path: Path to lip mask image
            full_mask_path: Path to full/background mask image

        Returns:
            Tuple of (face_seq, lip_seq, full_seq) each with shape (clip_length, C, H, W)
        """
        logger.debug(f"Loading static masks from {face_mask_path}, {lip_mask_path}, {full_mask_path}")

        try:
            # Load as grayscale (single channel)
            face = self.transform_64(Image.open(face_mask_path).convert("L"))
            lip = self.transform_64(Image.open(lip_mask_path).convert("L"))
            full = self.transform_64(Image.open(full_mask_path).convert("L"))

            # Expand to clip_length
            face_seq = face.unsqueeze(0).repeat(self.clip_length, 1, 1, 1)
            lip_seq = lip.unsqueeze(0).repeat(self.clip_length, 1, 1, 1)
            full_seq = full.unsqueeze(0).repeat(self.clip_length, 1, 1, 1)

            logger.debug(
                f"Loaded static masks: face={face_seq.shape}, "
                f"lip={lip_seq.shape}, full={full_seq.shape}"
            )
            return face_seq, lip_seq, full_seq

        except Exception as e:
            logger.error(f"Error loading static masks: {str(e)}", exc_info=True)
            raise

    def predict_masks(
        self,
        mask_input: torch.Tensor,
        audio_features: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict face/lip masks using MaskPredictUNet.

        Args:
            mask_input: Input mask tensor with shape (1, F, C, H, W)
            audio_features: Audio features with shape (1, F, window, blocks, dim)
            device: Device to run prediction on
            dtype: Data type for tensors

        Returns:
            Tuple of (face_masks, lip_masks, full_masks) each with shape (F, C, H, W)
        """
        if self.mask_predict_model is None:
            logger.warning("MaskPredictUNet not loaded, using fallback logic")
            return None, None, None

        logger.debug(f"Predicting masks with input shape {mask_input.shape}")

        try:
            with torch.no_grad():
                # Move to device and cast dtype
                mask_input = mask_input.to(device=device, dtype=dtype)
                audio_features = audio_features.to(device=device, dtype=dtype)

                # Forward pass
                pred_masks = self.mask_predict_model(mask_input, audio_features)
                pred_masks = torch.clamp(pred_masks.squeeze(0).cpu(), 0.0, 1.0)

                logger.debug(f"Predicted masks shape: {pred_masks.shape}")

                # Split into face/lip
                num_channels = 1  # Assuming single channel per mask
                face_masks = pred_masks[:, 0:num_channels, :, :]
                lip_masks = pred_masks[:, num_channels:, :, :]

                # Smooth masks
                face_masks = self._smooth_masks(face_masks)
                lip_masks = self._smooth_masks(lip_masks)

                # Compute full mask as complement
                full_masks = torch.clamp(1 - face_masks - lip_masks, 0.0, 1.0)

                logger.debug(
                    f"Face masks: {face_masks.shape}, "
                    f"Lip masks: {lip_masks.shape}, "
                    f"Full masks: {full_masks.shape}"
                )
                return face_masks, lip_masks, full_masks

        except Exception as e:
            logger.error(f"Error predicting masks: {str(e)}", exc_info=True)
            raise

    def _smooth_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Smooth masks using pooling and soft thresholding.

        Args:
            masks: Mask tensor with shape (F, C, H, W)

        Returns:
            Smoothed mask tensor with same shape
        """
        # Average pooling for smoothness
        masks = self._smooth_pool(masks, kernel_size=self.smooth_kernel)

        # Soft thresholding
        masks = self._smooth_shrink(masks, tau=self.smooth_tau, eps=self.smooth_eps)

        # Final pooling for refinement
        masks = self._smooth_pool(masks, kernel_size=5)

        return masks

    def _smooth_pool(
        self, x: torch.Tensor, kernel_size: int = 3
    ) -> torch.Tensor:
        """Average pooling with same padding."""
        orig_dtype = x.dtype
        pad = kernel_size // 2
        return F.avg_pool2d(
            x.float(), kernel_size=kernel_size, stride=1, padding=pad
        ).to(orig_dtype)

    def _smooth_shrink(
        self, x: torch.Tensor, tau: float, eps: float
    ) -> torch.Tensor:
        """
        Soft thresholding with smooth transitions.

        x <= tau           -> 0
        x >= tau + eps     -> x (mostly unchanged)
        Middle region uses smoothstep for smooth transitions
        """
        t = (x - tau) / eps
        t = torch.clamp(t, 0.0, 1.0)
        s = t * t * (3 - 2 * t)  # smoothstep function
        return s * x

    def blend_masks(
        self,
        prev_masks: Optional[torch.Tensor],
        curr_masks: torch.Tensor,
        blend_ratio: float = 0.1,
    ) -> torch.Tensor:
        """
        Blend previous and current masks for smooth transitions.

        Args:
            prev_masks: Previous frame masks or None for first frame
            curr_masks: Current frame masks
            blend_ratio: Blending weight (0 = use current, 1 = use previous)

        Returns:
            Blended mask tensor
        """
        if prev_masks is None:
            return curr_masks

        return (1 - blend_ratio) * curr_masks + blend_ratio * prev_masks
