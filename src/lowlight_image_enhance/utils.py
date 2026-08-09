import os
import numpy
import cv2

import logging
logger = logging.getLogger(__name__)

# === Utility functions for low-light image enhancement ===

# imread: Read an image from a file path.
# imwrite: Write an image to a file path in PNG format.
# normalize: Normalize an image from uint8 [0, 255] to float32 [0, 1].
# denormalize: De-normalize an image from float32 [0, 1] to uint8 [0, 255].

def imread(image_path: str) -> numpy.ndarray:
    """
    Read an image from a file path and convert it to BGR format.
    Args:
        image_path (str): Path to the image file.
    Returns:
        numpy.ndarray: Image read from the file, expected BGR, uint8 [0-255].
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    logger.info(f"Image read: {os.path.basename(image_path)}.")
    logger.debug(f" - type: {type(img_bgr)}, shape: {img_bgr.shape}, dtype = {img_bgr.dtype}")
    return img_bgr

def imwrite(image_path: str, img: numpy.ndarray) -> None:
    """
    Write an image to a file path in PNG format.
    Args:
        image_path (str): Path to save the image file.
        img (numpy.ndarray): Image to write, expected BGR, uint8 [0-255] or float32 [0-1].
    """
    if img.dtype == numpy.uint8:
        pass  # No conversion needed
    elif img.dtype == numpy.float32:
        img = denormalize(img)  # Convert float32 [0, 1] to uint8 [0, 255]
    else:
        raise ValueError(f"Unsupported image dtype: {img.dtype}. Expected uint8 or float32.")

    if not cv2.imwrite(image_path, img):
        raise IOError(f"Failed to write image to path: {image_path}")

    logger.info(f"Image written: {(image_path)}.")
    return None

def normalize(img_bgr: numpy.ndarray) -> numpy.ndarray:
    """
    Normalize an image to float32 [0, 1].
    Args:
        img_bgr (numpy.ndarray): Image to normalize, expected BGR, uint8 [0-255].
    Returns:
        numpy.ndarray: Normalized image, float32 [0, 1].
    """
    img_bgr_norm = img_bgr.astype(numpy.float32) / 255.0
    return img_bgr_norm

def denormalize(img_bgr_norm: numpy.ndarray) -> numpy.ndarray:
    """
    De-normalize an image from float32 [0, 1] to uint8 [0, 255].
    Args:
        img_bgr_norm (numpy.ndarray): Image to de-normalize, expected float32 [0, 1].
    Returns:
        numpy.ndarray: De-normalized image, uint8 [0, 255].
    """
    img_bgr = (img_bgr_norm * 255).clip(0, 255).astype(numpy.uint8)
    return img_bgr