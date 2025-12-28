"""Configuration loader for Hallo2 backend."""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import toml

logger = logging.getLogger(__name__)


class Settings:
    """Configuration settings for Hallo2 backend service."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings from config file or environment.

        Args:
            config_path: Path to config.toml file. If not provided,
                        looks in standard locations.
        """
        self.config_path = config_path or self._find_config_file()
        self.config_dict = {}

        if self.config_path and os.path.exists(self.config_path):
            self._load_config_file()
        else:
            logger.warning(f"Config file not found at {self.config_path}")

        self._load_from_environment()
        self._set_defaults()

    def _find_config_file(self) -> Optional[str]:
        """Find config.toml in standard locations."""
        candidates = [
            "/remote-home/JJHe/MoneyPrinterTurbo/config.toml",
            os.path.expanduser("~/config.toml"),
            "./config.toml",
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.toml"),
        ]

        for path in candidates:
            if os.path.exists(path):
                logger.info(f"Found config file at {path}")
                return path

        logger.warning(f"No config file found in standard locations")
        return None

    def _load_config_file(self):
        """Load configuration from TOML file."""
        try:
            with open(self.config_path, "r") as f:
                config = toml.load(f)
                self.config_dict = config.get("emo_hallo", {})
                logger.info(f"Loaded config from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")

    def _load_from_environment(self):
        """Override config with environment variables."""
        env_mappings = {
            "EMO_HALLO_BACKEND_PORT": ("backend_port", int),
            "EMO_HALLO_GPU_IDS": ("gpu_ids", lambda x: list(map(int, x.split(",")))),
            "EMO_HALLO_DTYPE": ("dtype", str),
            "EMO_HALLO_MAX_CONCURRENT_TASKS": ("max_concurrent_tasks", int),
            "EMO_HALLO_CHECKPOINT_DIR": ("hallo2_checkpoint_dir", str),
        }

        for env_var, (key, converter) in env_mappings.items():
            if env_var in os.environ:
                try:
                    self.config_dict[key] = converter(os.environ[env_var])
                    logger.info(f"Loaded {key} from {env_var}")
                except Exception as e:
                    logger.warning(f"Error parsing {env_var}: {str(e)}")

    def _set_defaults(self):
        """Set default values for missing configuration."""
        defaults = {
            "backend_enabled": True,
            "backend_port": 8001,
            "gpu_ids": [0],
            "dtype": "float16",
            "hallo2_checkpoint_dir": "/remote-home/JJHe/hallo2/pretrained_models",
            "max_concurrent_tasks": 1,
            "task_timeout": 600,
            "enable_caching": True,
            "hallo2": {
                "clip_length": 16,
                "fps": 25,
                "output_width": 512,
                "output_height": 512,
            },
        }

        for key, value in defaults.items():
            if key not in self.config_dict:
                self.config_dict[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config_dict.get(key, default)

    def get_hallo2_config(self) -> Dict[str, Any]:
        """Get Hallo2-specific configuration."""
        base_config = {
            "base_model_path": os.path.join(
                self.get("hallo2_checkpoint_dir"),
                "stable-diffusion-v1-5",
            ),
            "vae_model_path": os.path.join(
                self.get("hallo2_checkpoint_dir"),
                "sd-vae-ft-mse",
            ),
            "motion_module_path": os.path.join(
                self.get("hallo2_checkpoint_dir"),
                "motion_module/mm_sd_v15_v2.ckpt",
            ),
            "face_analysis_model_path": os.path.join(
                self.get("hallo2_checkpoint_dir"),
                "face_analysis",
            ),
            "wav2vec_model_path": os.path.join(
                self.get("hallo2_checkpoint_dir"),
                "wav2vec/wav2vec2-base-960h",
            ),
            "audio_separator_model_dir": os.path.join(
                self.get("hallo2_checkpoint_dir"),
                "audio_separator",
            ),
            "audio_separator_model_file": "Kim_Vocal_2.onnx",
            "audio_ckpt_dir": os.path.join(
                self.get("hallo2_checkpoint_dir"),
                "hallo2",
            ),
            "mask_config_path": "/remote-home/JJHe/hallo2/configs/stage_2mask_unet.yaml",
            "mask_checkpoint_path": os.path.join(
                self.get("hallo2_checkpoint_dir"),
                "hallo2/net_g.pth",
            ),
            "image_width": self.get("hallo2", {}).get("output_width", 512),
            "image_height": self.get("hallo2", {}).get("output_height", 512),
            "n_sample_frames": self.get("hallo2", {}).get("clip_length", 16),
            "fps": self.get("hallo2", {}).get("fps", 25),
            "n_motion_frames": 2,
            "face_expand_ratio": 1.2,
            "guidance_scale": 3.5,
            "inference_steps": 40,
            "audio_sample_rate": 16000,
            "wav2vec_only_last_features": False,
            "enable_zero_snr": True,
            "enable_cache": self.get("enable_caching", True),
            "noise_scheduler_kwargs": {
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "beta_schedule": "linear",
                "clip_sample": False,
                "steps_offset": 1,
                "prediction_type": "v_prediction",
                "rescale_betas_zero_snr": True,
                "timestep_spacing": "trailing",
            },
            "unet_additional_kwargs": {
                "use_inflated_groupnorm": True,
                "unet_use_cross_frame_attention": False,
                "unet_use_temporal_attention": False,
                "use_motion_module": True,
                "use_audio_module": True,
                "motion_module_resolutions": [1, 2, 4, 8],
                "motion_module_mid_block": True,
                "motion_module_decoder_only": False,
                "motion_module_type": "Vanilla",
                "motion_module_kwargs": {
                    "num_attention_heads": 8,
                    "num_transformer_block": 1,
                    "attention_block_types": ["Temporal_Self", "Temporal_Self"],
                    "temporal_position_encoding": True,
                    "temporal_position_encoding_max_len": 32,
                    "temporal_attention_dim_div": 1,
                },
                "audio_attention_dim": 768,
                "stack_enable_blocks_name": ["up", "down", "mid"],
                "stack_enable_blocks_depth": [0, 1, 2, 3],
            },
        }

        return base_config

    def __repr__(self) -> str:
        """String representation."""
        return f"Settings(config_path={self.config_path}, backend_port={self.get('backend_port')})"


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings():
    """Reset global settings (for testing)."""
    global _settings
    _settings = None
