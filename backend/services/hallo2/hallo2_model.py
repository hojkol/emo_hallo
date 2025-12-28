"""Hallo2 talking head generation model wrapper."""

import torch
import logging
from typing import Any, Dict, Optional
from pathlib import Path
import os
from omegaconf import OmegaConf

# Try relative import first, then absolute if it fails
try:
    from ..base.model_interface import BaseModel
except (ImportError, ValueError):
    try:
        from services.base.model_interface import BaseModel
    except ImportError:
        from base.model_interface import BaseModel

logger = logging.getLogger(__name__)


class Hallo2Model(BaseModel):
    """
    Hallo2 talking head generation model.

    Loads and manages all sub-models required for inference:
    - VAE (video autoencoder)
    - Reference UNet2D (image conditioning)
    - Denoising UNet3D (temporal video generation)
    - FaceLocator (face detection and localization)
    - ImageProj (image projection)
    - AudioProj (audio projection)
    - MaskPredictUNet (face/lip mask prediction)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize Hallo2Model.

        Args:
            config: Configuration dictionary with model paths and parameters
            device: Device to load models on (cuda/cpu)
            dtype: Data type for model weights (float16/float32/bfloat16)
        """
        super().__init__(config, device, dtype)
        self.config = config

        # Sub-models will be loaded on demand
        self.vae = None
        self.reference_unet = None
        self.denoising_unet = None
        self.face_locator = None
        self.image_proj = None
        self.audio_proj = None
        self.mask_predict = None
        self.net = None
        self.scheduler = None

        self._initialized = False

    def load(self) -> None:
        """Load all Hallo2 sub-models."""
        if self._initialized:
            logger.warning("Hallo2Model already loaded, skipping re-initialization")
            return

        logger.info("Loading Hallo2 model and sub-models...")

        try:
            self._load_vae()
            self._load_reference_unet()
            self._load_denoising_unet()
            self._load_face_locator()
            self._load_projectors()
            self._load_net()
            self._load_scheduler()
            self._load_mask_predictor()

            self._initialized = True
            logger.info("Successfully loaded all Hallo2 sub-models")

        except Exception as e:
            logger.error(f"Failed to load Hallo2 models: {str(e)}", exc_info=True)
            self.unload()
            raise

    def _load_vae(self) -> None:
        """Load VAE model."""
        logger.debug("Loading VAE...")
        from diffusers import AutoencoderKL

        vae_path = self.config.get("vae_model_path")
        if not vae_path:
            raise ValueError("vae_model_path not found in config")

        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.vae = self.vae.to(self.device, dtype=self.dtype)
        self.vae.requires_grad_(False)
        logger.debug(f"Loaded VAE from {vae_path}")

    def _load_reference_unet(self) -> None:
        """Load reference UNet2D model."""
        logger.debug("Loading Reference UNet2D...")
        from hallo.models.unet_2d_condition import UNet2DConditionModel

        base_model_path = self.config.get("base_model_path")
        if not base_model_path:
            raise ValueError("base_model_path not found in config")

        self.reference_unet = UNet2DConditionModel.from_pretrained(
            base_model_path, subfolder="unet"
        )
        self.reference_unet = self.reference_unet.to(self.device, dtype=self.dtype)
        self.reference_unet.requires_grad_(False)
        logger.debug(f"Loaded Reference UNet2D from {base_model_path}")

    def _load_denoising_unet(self) -> None:
        """Load denoising UNet3D model with motion module."""
        logger.debug("Loading Denoising UNet3D...")
        from hallo.models.unet_3d import UNet3DConditionModel

        base_model_path = self.config.get("base_model_path")
        motion_module_path = self.config.get("motion_module_path")

        if not base_model_path or not motion_module_path:
            raise ValueError(
                "base_model_path and motion_module_path required in config"
            )

        unet_additional_kwargs = self.config.get("unet_additional_kwargs", {})

        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            base_model_path,
            motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=unet_additional_kwargs,
            use_landmark=False,
        )
        self.denoising_unet = self.denoising_unet.to(self.device, dtype=self.dtype)
        self.denoising_unet.requires_grad_(False)
        logger.debug(f"Loaded Denoising UNet3D from {base_model_path}")

    def _load_face_locator(self) -> None:
        """Load FaceLocator model."""
        logger.debug("Loading FaceLocator...")
        from hallo.models.face_locator import FaceLocator

        self.face_locator = FaceLocator(conditioning_embedding_channels=320)
        self.face_locator = self.face_locator.to(self.device, dtype=self.dtype)
        self.face_locator.requires_grad_(False)
        logger.debug("Loaded FaceLocator")

    def _load_projectors(self) -> None:
        """Load ImageProj and AudioProj models."""
        logger.debug("Loading Projectors...")
        from hallo.models.image_proj import ImageProjModel
        from hallo.models.audio_proj import AudioProjModel

        # ImageProj
        self.image_proj = ImageProjModel(
            cross_attention_dim=self.denoising_unet.config.cross_attention_dim,
            clip_embeddings_dim=512,
            clip_extra_context_tokens=4,
        )
        self.image_proj = self.image_proj.to(self.device, dtype=self.dtype)
        self.image_proj.requires_grad_(False)

        # AudioProj
        self.audio_proj = AudioProjModel(
            seq_len=5,
            blocks=12,  # 12 layers' hidden states from wav2vec
            channels=768,  # audio embedding dimension
            intermediate_dim=512,
            output_dim=768,
            context_tokens=32,
        )
        self.audio_proj = self.audio_proj.to(self.device, dtype=self.dtype)
        self.audio_proj.requires_grad_(False)

        logger.debug("Loaded ImageProj and AudioProj")

    def _load_net(self) -> None:
        """Load combined Net module with all sub-models."""
        logger.debug("Loading Net module...")
        from torch import nn

        class Net(nn.Module):
            """Combined module for all Hallo2 models."""
            def __init__(self, reference_unet, denoising_unet, face_locator,
                         imageproj, audioproj):
                super().__init__()
                self.reference_unet = reference_unet
                self.denoising_unet = denoising_unet
                self.face_locator = face_locator
                self.imageproj = imageproj
                self.audioproj = audioproj

            def forward(self):
                """Empty forward for compatibility."""
                pass

            def get_modules(self):
                """Return all sub-modules."""
                return {
                    "reference_unet": self.reference_unet,
                    "denoising_unet": self.denoising_unet,
                    "face_locator": self.face_locator,
                    "imageproj": self.imageproj,
                    "audioproj": self.audioproj,
                }

        self.net = Net(
            self.reference_unet,
            self.denoising_unet,
            self.face_locator,
            self.image_proj,
            self.audio_proj,
        )

        # Load combined checkpoint
        audio_ckpt_dir = self.config.get("audio_ckpt_dir")
        if not audio_ckpt_dir:
            raise ValueError("audio_ckpt_dir not found in config")

        net_checkpoint = os.path.join(audio_ckpt_dir, "net.pth")
        if not os.path.exists(net_checkpoint):
            raise FileNotFoundError(f"Net checkpoint not found: {net_checkpoint}")

        checkpoint = torch.load(net_checkpoint, map_location="cpu")
        missing_keys, unexpected_keys = self.net.load_state_dict(checkpoint, strict=False)

        if missing_keys:
            logger.warning(
                f"Missing keys when loading Net: {len(missing_keys)} keys "
                f"(showing first 10): {missing_keys[:10]}"
            )
        if unexpected_keys:
            logger.warning(
                f"Unexpected keys when loading Net: {len(unexpected_keys)} keys "
                f"(showing first 10): {unexpected_keys[:10]}"
            )

        logger.debug(f"Loaded Net from {net_checkpoint}")

    def _load_scheduler(self) -> None:
        """Load DDIM scheduler."""
        logger.debug("Loading DDIM Scheduler...")
        from diffusers import DDIMScheduler

        sched_kwargs = self.config.get("noise_scheduler_kwargs", {})

        # Make a copy to avoid modifying original config
        sched_kwargs = dict(sched_kwargs)

        if self.config.get("enable_zero_snr"):
            sched_kwargs.update({
                "rescale_betas_zero_snr": True,
                "timestep_spacing": "trailing",
                "prediction_type": "v_prediction",
            })

        self.scheduler = DDIMScheduler(**sched_kwargs)
        logger.debug("Loaded DDIM Scheduler")

    def _load_mask_predictor(self) -> None:
        """Load MaskPredictUNet for mask prediction."""
        logger.debug("Loading MaskPredictUNet...")
        from hallo.models_adapter.mask_predict_unet3 import MaskPredictUNet

        mask_config_path = self.config.get("mask_config_path")
        mask_checkpoint_path = self.config.get("mask_checkpoint_path")

        if not mask_config_path or not mask_checkpoint_path:
            logger.warning(
                "mask_config_path or mask_checkpoint_path not found in config. "
                "MaskPredictUNet will not be loaded. This may limit functionality."
            )
            return

        if not os.path.exists(mask_config_path):
            logger.warning(f"Mask config not found: {mask_config_path}")
            return

        if not os.path.exists(mask_checkpoint_path):
            logger.warning(f"Mask checkpoint not found: {mask_checkpoint_path}")
            return

        try:
            mask_cfg = OmegaConf.load(mask_config_path)

            clip_length = self.config.get("n_sample_frames", 16)

            self.mask_predict = MaskPredictUNet(
                in_channels=mask_cfg.mask_predict.in_channel,
                out_channels=mask_cfg.mask_predict.in_channel,
                base_channels=getattr(mask_cfg.mask_predict, "base_channels", 64),
                channel_multipliers=tuple(
                    getattr(mask_cfg.mask_predict, "channel_multipliers", [1, 2, 4, 8])
                ),
                num_res_blocks=getattr(mask_cfg.mask_predict, "num_res_blocks", 2),
                use_temporal_at_levels=tuple(
                    getattr(mask_cfg.mask_predict, "use_temporal_at_levels", [1, 2, 3])
                ),
                temporal_type=getattr(mask_cfg.mask_predict, "temporal_type", "conv"),
                temporal_kernel_size=getattr(
                    mask_cfg.mask_predict, "temporal_kernel_size", 3
                ),
                spatial_kernel_size=getattr(
                    mask_cfg.mask_predict, "spatial_kernel_size", 3
                ),
                temporal_heads=getattr(mask_cfg.mask_predict, "temporal_heads", 8),
                cross_attention_dim=mask_cfg.mask_predict.cross_attention_dim,
                cross_attention_at_levels=tuple(
                    getattr(mask_cfg.mask_predict, "cross_attention_at_levels", [2, 3])
                ),
                audio_dim=mask_cfg.mask_predict.audio_dim,
                num_frames=clip_length,
                audio_tokens=getattr(mask_cfg.mask_predict, "audio_tokens", 32),
                patch_size=getattr(mask_cfg.mask_predict, "patch_size", 2),
                dropout=getattr(mask_cfg.mask_predict, "dropout", 0.05),
                use_positional_embeddings=getattr(
                    mask_cfg.mask_predict, "use_positional_embeddings", False
                ),
                use_emotion=getattr(mask_cfg.mask_predict, "use_emotion_emb", False),
                emotion_dim=getattr(mask_cfg.mask_predict, "emotion_dim", 768),
            )

            self.mask_predict = self.mask_predict.to(self.device, dtype=self.dtype)
            self.mask_predict.requires_grad_(False)

            # Load checkpoint
            ckpt = torch.load(mask_checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict):
                if "module" in ckpt:
                    state_dict = ckpt["module"]
                elif "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt

            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_k] = v

            missing_keys, unexpected_keys = self.mask_predict.load_state_dict(
                new_state_dict, strict=False
            )

            if missing_keys:
                logger.warning(
                    f"Missing keys in MaskPredictUNet: {len(missing_keys)} keys"
                )
            if unexpected_keys:
                logger.warning(
                    f"Unexpected keys in MaskPredictUNet: {len(unexpected_keys)} keys"
                )

            self.mask_predict.eval()
            logger.debug(f"Loaded MaskPredictUNet from {mask_checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to load MaskPredictUNet: {str(e)}")
            self.mask_predict = None

    def unload(self) -> None:
        """Unload all models from GPU/memory."""
        logger.info("Unloading Hallo2 models...")

        models_to_unload = [
            "vae", "reference_unet", "denoising_unet", "face_locator",
            "image_proj", "audio_proj", "mask_predict", "net", "scheduler"
        ]

        for model_name in models_to_unload:
            model = getattr(self, model_name, None)
            if model is not None:
                try:
                    if hasattr(model, "to"):
                        model.to("cpu")
                    del model
                    setattr(self, model_name, None)
                except Exception as e:
                    logger.warning(f"Error unloading {model_name}: {str(e)}")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._initialized = False
        logger.info("Successfully unloaded all Hallo2 models")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "name": "Hallo2",
            "version": "2.0",
            "device": self.device,
            "dtype": str(self.dtype),
            "models_loaded": {
                "vae": self.vae is not None,
                "reference_unet": self.reference_unet is not None,
                "denoising_unet": self.denoising_unet is not None,
                "face_locator": self.face_locator is not None,
                "image_proj": self.image_proj is not None,
                "audio_proj": self.audio_proj is not None,
                "mask_predict": self.mask_predict is not None,
                "net": self.net is not None,
                "scheduler": self.scheduler is not None,
            },
            "initialized": self._initialized,
            "config": {
                "base_model": self.config.get("base_model_path", "unknown"),
                "motion_module": self.config.get("motion_module_path", "unknown"),
                "audio_checkpoint_dir": self.config.get("audio_ckpt_dir", "unknown"),
                "sample_frames": self.config.get("n_sample_frames", 16),
                "image_height": self.config.get("image_height", 512),
                "image_width": self.config.get("image_width", 512),
            },
        }
