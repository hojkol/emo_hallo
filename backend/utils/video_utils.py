"""Video utilities for Hallo2 backend."""

import logging
import os
from typing import Optional, List

logger = logging.getLogger(__name__)


def get_video_duration(video_path: str) -> Optional[float]:
    """
    Get video duration in seconds.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds, or None if error
    """
    try:
        from moviepy.editor import VideoFileClip

        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return None

        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()

        logger.debug(f"Video duration: {duration:.2f}s")
        return duration

    except Exception as e:
        logger.warning(f"Error getting video duration: {str(e)}")
        return None


def concatenate_videos(video_paths: List[str], output_path: str, fps: int = 25) -> bool:
    """
    Concatenate multiple video files.

    Args:
        video_paths: List of video file paths
        output_path: Path to save concatenated video
        fps: Output frames per second

    Returns:
        True if successful
    """
    try:
        from moviepy.editor import concatenate_videoclips, VideoFileClip

        logger.info(f"Concatenating {len(video_paths)} videos...")

        clips = []
        for video_path in video_paths:
            if not os.path.exists(video_path):
                logger.warning(f"Video file not found, skipping: {video_path}")
                continue

            try:
                clip = VideoFileClip(video_path)
                clips.append(clip)
            except Exception as e:
                logger.warning(f"Error loading video {video_path}: {str(e)}")

        if not clips:
            logger.error("No valid video clips to concatenate")
            return False

        # Concatenate
        final_clip = concatenate_videoclips(clips)

        # Write output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        final_clip.write_videofile(
            output_path,
            fps=fps,
            verbose=False,
            logger=None,
        )

        # Clean up
        final_clip.close()
        for clip in clips:
            clip.close()

        logger.info(f"Concatenated video saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error concatenating videos: {str(e)}")
        return False


def add_audio_to_video(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> bool:
    """
    Add audio track to video.

    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        output_path: Path to save output video

    Returns:
        True if successful
    """
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip

        logger.info(f"Adding audio to video...")

        # Load video and audio
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        # Adjust audio length to match video
        if audio_clip.duration < video_clip.duration:
            logger.warning(
                f"Audio duration ({audio_clip.duration:.2f}s) is shorter than "
                f"video ({video_clip.duration:.2f}s), audio will be cut"
            )
            audio_clip = audio_clip.subclipped(0, video_clip.duration)
        elif audio_clip.duration > video_clip.duration:
            logger.warning(
                f"Audio duration ({audio_clip.duration:.2f}s) is longer than "
                f"video ({video_clip.duration:.2f}s), audio will be truncated"
            )
            audio_clip = audio_clip.subclipped(0, video_clip.duration)

        # Set audio
        final_clip = video_clip.set_audio(audio_clip)

        # Write output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        final_clip.write_videofile(
            output_path,
            verbose=False,
            logger=None,
        )

        # Clean up
        final_clip.close()
        video_clip.close()
        audio_clip.close()

        logger.info(f"Video with audio saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error adding audio to video: {str(e)}")
        return False


def get_video_info(video_path: str) -> Optional[dict]:
    """
    Get video information (duration, fps, resolution).

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video info, or None if error
    """
    try:
        from moviepy.editor import VideoFileClip

        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return None

        clip = VideoFileClip(video_path)
        info = {
            "duration": clip.duration,
            "fps": clip.fps,
            "width": clip.w,
            "height": clip.h,
            "size_mb": os.path.getsize(video_path) / 1024 / 1024,
        }
        clip.close()

        return info

    except Exception as e:
        logger.warning(f"Error getting video info: {str(e)}")
        return None
