"""Image loading, resizing, and padding utilities."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from os import path, walk
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".heic", ".cr2"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg", ".webm"}


def get_files(directory: str | Path, file_type: str = "image") -> list[str]:
    """Recursively find image/video files in a directory."""
    if file_type == "image":
        exts = IMAGE_EXTENSIONS
    elif file_type == "video":
        exts = VIDEO_EXTENSIONS
    elif file_type == "both":
        exts = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    else:
        exts = IMAGE_EXTENSIONS

    return [
        path.join(dp, f)
        for dp, _dn, filenames in walk(str(directory))
        for f in filenames
        if path.splitext(f)[1].lower() in exts and not f.startswith("._")
    ]


def imread_safe(file_path: str) -> np.ndarray | None:
    """Read an image file, handling standard formats."""
    try:
        ext = path.splitext(file_path)[1].lower()
        if ext == ".heic":
            import pyheif
            from PIL import Image as PILImage

            heif_file = pyheif.read(file_path)
            image = PILImage.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            image = np.array(image)[:, :, ::-1]  # RGB to BGR
        elif ext == ".cr2":
            import rawpy

            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(use_camera_wb=True)
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            image = cv2.imread(file_path)
        return image
    except Exception:
        log.exception("Failed to read image: %s", file_path)
        return None


def resize_maintain_aspect(
    image: np.ndarray | str, height: int | None = 640, width: int | None = None
) -> np.ndarray | None:
    """Resize image while maintaining aspect ratio."""
    try:
        if isinstance(image, str):
            image = imread_safe(image)
            if image is None:
                return None

        h, w = image.shape[:2]
        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    except Exception:
        log.exception("Failed to resize image")
        return None


def pad_to_square(image: np.ndarray, size: int = 640) -> np.ndarray:
    """Resize and pad image to a square of given size."""
    aspect_ratio = size / max(image.shape[0], image.shape[1])
    resized = cv2.resize(
        image,
        (int(image.shape[1] * aspect_ratio), int(image.shape[0] * aspect_ratio)),
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return canvas


def prepare_batch(
    image_paths: list[str], resize_height: int = 640, max_workers: int = 20
) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Load, resize, and pad a batch of images. Returns (valid_paths, resized, padded)."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        resized_images = list(
            executor.map(resize_maintain_aspect, image_paths, repeat(resize_height))
        )

    # Filter out failures
    valid = [
        (p, img) for p, img in zip(image_paths, resized_images, strict=True) if img is not None
    ]
    if not valid:
        return [], [], []

    paths, resized = zip(*valid, strict=True)
    paths = list(paths)
    resized = list(resized)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        padded = list(executor.map(pad_to_square, resized, repeat(resize_height)))

    return paths, resized, padded
