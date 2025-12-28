"""File utilities for Hallo2 backend."""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, create if needed.

    Args:
        path: Directory path

    Returns:
        Absolute path to directory
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    logger.debug(f"Ensured directory exists: {abs_path}")
    return abs_path


def cleanup_dir(path: str, remove_self: bool = False) -> None:
    """
    Clean up a directory by removing all contents.

    Args:
        path: Directory path to clean
        remove_self: Whether to remove the directory itself
    """
    if not os.path.exists(path):
        logger.debug(f"Directory does not exist: {path}")
        return

    try:
        if remove_self:
            shutil.rmtree(path)
            logger.info(f"Removed directory: {path}")
        else:
            # Remove contents only
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            logger.info(f"Cleaned directory: {path}")

    except Exception as e:
        logger.warning(f"Error cleaning directory {path}: {str(e)}")


def get_file_size(path: str) -> int:
    """
    Get file size in bytes.

    Args:
        path: File path

    Returns:
        File size in bytes
    """
    if not os.path.exists(path):
        return 0

    return os.path.getsize(path)


def get_dir_size(path: str) -> int:
    """
    Get total directory size in bytes.

    Args:
        path: Directory path

    Returns:
        Total size in bytes
    """
    total = 0

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total += os.path.getsize(filepath)

    return total


def format_size(size_bytes: int) -> str:
    """
    Format byte size to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., '1.5 GB')
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024

    return f"{size_bytes:.1f} TB"


def safe_rename(src: str, dst: str) -> bool:
    """
    Safely rename file or directory.

    Args:
        src: Source path
        dst: Destination path

    Returns:
        True if successful
    """
    try:
        os.rename(src, dst)
        logger.debug(f"Renamed {src} to {dst}")
        return True

    except Exception as e:
        logger.warning(f"Error renaming {src} to {dst}: {str(e)}")
        return False


def safe_copy(src: str, dst: str) -> bool:
    """
    Safely copy file or directory.

    Args:
        src: Source path
        dst: Destination path

    Returns:
        True if successful
    """
    try:
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

        logger.debug(f"Copied {src} to {dst}")
        return True

    except Exception as e:
        logger.warning(f"Error copying {src} to {dst}: {str(e)}")
        return False


def is_valid_file(path: str, required_extensions: Optional[list] = None) -> bool:
    """
    Check if file is valid and has correct extension.

    Args:
        path: File path
        required_extensions: List of valid extensions (e.g., ['.jpg', '.png'])

    Returns:
        True if file is valid
    """
    if not os.path.isfile(path):
        return False

    if required_extensions:
        ext = Path(path).suffix.lower()
        if ext not in required_extensions:
            return False

    return True
