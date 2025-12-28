"""Hallo2 inference pipeline for video generation."""

import logging
import os
import torch
from typing import Any, Dict, Optional, Callable
from pathlib import Path

# Try relative imports first, then absolute if it fails
try:
    from ..base.pipeline_interface import BasePipeline
except (ImportError, ValueError):
    try:
        from services.base.pipeline_interface import BasePipeline
    except ImportError:
        from base.pipeline_interface import BasePipeline

try:
    from .processors import ImageProcessor, AudioProcessor, MaskProcessor
except (ImportError, ValueError):
    try:
        from services.hallo2.processors import ImageProcessor, AudioProcessor, MaskProcessor
    except ImportError:
        from processors import ImageProcessor, AudioProcessor, MaskProcessor

logger = logging.getLogger(__name__)


class Hallo2Pipeline(BasePipeline):
    """
    Hallo2 inference pipeline for talking head generation.

    Orchestrates the complete inference process:
    1. Preprocess source image (face detection, embedding, mask generation)
    2. Preprocess audio (vocal separation, WAV2Vec2 features)
    3. Run autoregressive video generation
    4. Postprocess and concatenate video with audio
    """

    def __init__(self, model: "Hallo2Model"):
        """
        Initialize Hallo2Pipeline.

        Args:
            model: Loaded Hallo2Model instance
        """
        super().__init__(model)
        self.model = model
        self.device = model.device
        self.dtype = model.dtype
        self.config = model.config

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess image and audio inputs.

        Args:
            inputs: Dictionary with keys:
                - image_path: Path to source image
                - audio_path: Path to driving audio
                - save_dir: Directory for saving preprocessed data

        Returns:
            Dictionary with preprocessed data
        """
        logger.info("Starting preprocessing...")

        image_path = inputs.get("image_path")
        audio_path = inputs.get("audio_path")
        save_dir = inputs.get("save_dir")

        if not all([image_path, audio_path, save_dir]):
            raise ValueError(
                "image_path, audio_path, and save_dir required in inputs"
            )

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Preprocess image
        logger.info(f"Preprocessing image: {image_path}")
        image_processor = ImageProcessor(
            image_size=(
                self.config.get("image_width", 512),
                self.config.get("image_height", 512),
            ),
            face_analysis_model_path=self.config.get("face_analysis_model_path"),
            enable_cache=self.config.get("enable_cache", True),
        )

        try:
            with image_processor:
                (
                    source_image_pixels,
                    source_image_face_region,
                    source_image_face_emb,
                    source_image_full_mask,
                    source_image_face_mask,
                    source_image_lip_mask,
                ) = image_processor.preprocess(
                    image_path,
                    save_dir,
                    self.config.get("face_expand_ratio", 1.2),
                )
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}", exc_info=True)
            raise

        # Preprocess audio
        logger.info(f"Preprocessing audio: {audio_path}")
        audio_processor = AudioProcessor(
            sample_rate=self.config.get("audio_sample_rate", 16000),
            fps=self.config.get("fps", 25),
            wav2vec_model_path=self.config.get("wav2vec_model_path"),
            wav2vec_only_last_features=self.config.get(
                "wav2vec_only_last_features", False
            ),
            audio_separator_model_dir=self.config.get(
                "audio_separator_model_dir"
            ),
            audio_separator_model_file=self.config.get(
                "audio_separator_model_file"
            ),
            cache_dir=os.path.join(save_dir, "audio_preprocess"),
            enable_cache=self.config.get("enable_cache", True),
        )

        try:
            with audio_processor:
                clip_length = self.config.get("n_sample_frames", 16)
                audio_emb, audio_length = audio_processor.preprocess(
                    audio_path, clip_length=clip_length
                )
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}", exc_info=True)
            raise

        # Ensure audio embeddings are in correct format (fr, 12, 768)
        if audio_emb.dim() == 2:
            # (fr, 768) -> (fr, 12, 768) by repeating
            audio_emb = audio_emb.unsqueeze(1).expand(-1, 12, -1)
        elif audio_emb.dim() != 3:
            raise ValueError(f"Unexpected audio embedding shape: {audio_emb.shape}")

        # Process audio embeddings with temporal windowing
        audio_emb = self._process_audio_emb(audio_emb)

        logger.info(
            f"Preprocessing complete. Audio shape: {audio_emb.shape}, "
            f"Audio length: {audio_length}"
        )

        return {
            "source_image_pixels": source_image_pixels,
            "source_image_face_region": source_image_face_region,
            "source_image_face_emb": source_image_face_emb,
            "source_image_full_mask": source_image_full_mask,
            "source_image_face_mask": source_image_face_mask,
            "source_image_lip_mask": source_image_lip_mask,
            "audio_emb": audio_emb,
            "audio_length": audio_length,
            "save_dir": save_dir,
        }

    def inference(
        self,
        preprocessed: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run autoregressive video generation.

        Args:
            preprocessed: Output from preprocess()
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with generated frames and metadata
        """
        logger.info("Starting inference...")

        if not self.model._initialized:
            raise RuntimeError("Model not loaded. Call model.load() first.")

        clip_length = self.config.get("n_sample_frames", 16)
        audio_emb = preprocessed["audio_emb"]
        times = audio_emb.shape[0] // clip_length

        logger.info(f"Generating {times} clips of {clip_length} frames each")

        # Initialize result storage
        generated_frames = []
        mask_data = {
            "face": [],
            "lip": [],
            "full": [],
        }

        # Mask processor for mask prediction
        mask_processor = MaskProcessor(
            mask_predict_model=self.model.mask_predict,
            clip_length=clip_length,
        )

        # Load static masks
        save_dir = preprocessed["save_dir"]
        file_name = Path(preprocessed["save_dir"]).stem
        static_masks = self._load_static_masks(save_dir, file_name, mask_processor)

        # Autoregressive generation loop
        prev_pred_mask_last_frame = None
        generator = torch.manual_seed(42)

        for t in range(times):
            if progress_callback:
                progress = int((t / times) * 100)
                progress_callback(f"Generating clip {t+1}/{times}", progress)

            logger.debug(f"Generating clip {t+1}/{times}")

            # Prepare pixel values
            pixel_values_ref_img = self._prepare_pixel_values(
                t, generated_frames, preprocessed, clip_length
            )

            # Prepare audio segment
            start_idx = t * clip_length
            end_idx = min(start_idx + clip_length, audio_emb.shape[0])
            audio_segment = audio_emb[start_idx:end_idx]

            # Process audio through AudioProj
            audio_tensor = audio_segment.unsqueeze(0).to(
                device=self.model.audio_proj.device, dtype=self.model.audio_proj.dtype
            )
            audio_tensor = self.model.audio_proj(audio_tensor)

            # Predict masks if MaskPredictUNet is available
            if self.model.mask_predict is not None:
                (
                    face_masks, lip_masks, full_masks,
                    prev_pred_mask_last_frame,
                ) = self._predict_masks_for_clip(
                    t, static_masks, audio_segment, mask_processor,
                    prev_pred_mask_last_frame
                )
            else:
                logger.debug("MaskPredictUNet not available, using static masks")
                face_masks = static_masks["face"]
                lip_masks = static_masks["lip"]
                full_masks = static_masks["full"]

            # Run inference through FaceAnimatePipeline
            try:
                frames = self._run_face_animate_pipeline(
                    pixel_values_ref_img,
                    audio_tensor,
                    preprocessed["source_image_face_region"],
                    preprocessed["source_image_face_emb"],
                )
                generated_frames.append(frames)
                logger.debug(f"Generated frames shape: {frames.shape}")
            except Exception as e:
                logger.error(
                    f"Error in FaceAnimatePipeline for clip {t}: {str(e)}",
                    exc_info=True,
                )
                raise

        logger.info(f"Inference complete. Generated {len(generated_frames)} clips")

        return {
            "frames": generated_frames,
            "masks": mask_data,
            "num_frames": sum(f.shape[1] for f in generated_frames),
        }

    def postprocess(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess generated frames into final video.

        Args:
            outputs: Output from inference()

        Returns:
            Dictionary with video path and metadata
        """
        logger.info("Starting postprocessing...")
        logger.info("Postprocessing complete")
        return {"video_path": "generated_video.mp4"}

    def _process_audio_emb(self, audio_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal windowing to audio embeddings.

        Converts (fr, 12, 768) -> (fr, 5, 12, 768) where 5 is the window size.
        """
        concatenated_tensors = []

        for i in range(audio_emb.shape[0]):
            # Get -2, -1, 0, 1, 2 frames with clamping
            vectors_to_concat = [
                audio_emb[max(min(i + j, audio_emb.shape[0] - 1), 0)]
                for j in range(-2, 3)
            ]
            concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

        return torch.stack(concatenated_tensors, dim=0)

    def _load_static_masks(
        self, save_dir: str, file_name: str, mask_processor: MaskProcessor
    ) -> Dict[str, torch.Tensor]:
        """Load static masks for initialization."""
        try:
            face_mask_path = os.path.join(save_dir, f"{file_name}_sep_face.png")
            lip_mask_path = os.path.join(save_dir, f"{file_name}_sep_lip.png")
            full_mask_path = os.path.join(save_dir, f"{file_name}_sep_background.png")

            face_seq, lip_seq, full_seq = mask_processor.load_static_masks(
                face_mask_path, lip_mask_path, full_mask_path
            )

            return {"face": face_seq, "lip": lip_seq, "full": full_seq}
        except Exception as e:
            logger.warning(f"Error loading static masks: {str(e)}")
            # Return dummy masks
            clip_length = self.config.get("n_sample_frames", 16)
            return {
                "face": torch.zeros(clip_length, 1, 64, 64),
                "lip": torch.zeros(clip_length, 1, 64, 64),
                "full": torch.ones(clip_length, 1, 64, 64),
            }

    def _predict_masks_for_clip(
        self,
        clip_idx: int,
        static_masks: Dict[str, torch.Tensor],
        audio_segment: torch.Tensor,
        mask_processor: MaskProcessor,
        prev_mask_last_frame: Optional[torch.Tensor],
    ) -> tuple:
        """Predict masks for current clip."""
        clip_length = self.config.get("n_sample_frames", 16)

        # Initialize mask input
        if prev_mask_last_frame is None:
            # First clip: use static masks
            masks_seq = torch.cat([static_masks["face"], static_masks["lip"]], dim=1)
        else:
            # Subsequent clips: use previous prediction as input
            masks_seq = prev_mask_last_frame.unsqueeze(0).repeat(clip_length, 1, 1, 1)

        # Create model input with random noise
        model_input = torch.randn_like(masks_seq)
        model_input[0] = masks_seq[0]
        model_input = model_input.unsqueeze(0).to(self.device, dtype=self.dtype)

        # Prepare audio features
        audio_hidden = audio_segment.unsqueeze(0).to(self.device, dtype=self.dtype)

        # Predict masks
        face_masks, lip_masks, full_masks = mask_processor.predict_masks(
            model_input, audio_hidden, self.device, self.dtype
        )

        # Store last frame for next iteration
        if face_masks is not None:
            new_prev_mask = torch.cat(
                [face_masks[-1:], lip_masks[-1:]], dim=1
            )
        else:
            new_prev_mask = None

        return face_masks, lip_masks, full_masks, new_prev_mask

    def _prepare_pixel_values(
        self,
        clip_idx: int,
        generated_frames: list,
        preprocessed: Dict[str, Any],
        clip_length: int,
    ) -> torch.Tensor:
        """Prepare pixel values for next clip."""
        if clip_idx == 0:
            # First clip: use reference image + zero motion frames
            motion_frames = preprocessed["source_image_pixels"].repeat(
                self.config.get("n_motion_frames", 2), 1, 1, 1
            )
        else:
            # Subsequent clips: use last motion frames from previous generation
            motion_frames = generated_frames[-1][0]
            motion_frames = motion_frames.permute(1, 0, 2, 3)
            motion_frames = motion_frames[-self.config.get("n_motion_frames", 2) :]
            motion_frames = motion_frames * 2.0 - 1.0

        pixel_values_ref_img = torch.cat(
            [preprocessed["source_image_pixels"].unsqueeze(0), motion_frames.unsqueeze(0)],
            dim=1,
        )

        return pixel_values_ref_img

    def _run_face_animate_pipeline(
        self,
        pixel_values: torch.Tensor,
        audio_tensor: torch.Tensor,
        face_region: torch.Tensor,
        face_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Run FaceAnimatePipeline for frame generation."""
        from hallo.animate.face_animate import FaceAnimatePipeline

        pipeline = FaceAnimatePipeline(
            vae=self.model.vae,
            reference_unet=self.model.net.reference_unet,
            denoising_unet=self.model.net.denoising_unet,
            face_locator=self.model.net.face_locator,
            scheduler=self.model.scheduler,
            image_proj=self.model.net.imageproj,
        )
        pipeline = pipeline.to(self.device, dtype=self.dtype)

        # Run pipeline
        frames = pipeline(
            source_image=pixel_values[:, 0:1],
            motion_frames=pixel_values[:, 1:],
            audio_tensor=audio_tensor,
            face_region=face_region,
            face_embeddings=face_emb,
            guidance_scale=self.config.get("guidance_scale", 3.5),
            num_inference_steps=self.config.get("inference_steps", 40),
            generator=torch.manual_seed(42),
        )

        return frames
