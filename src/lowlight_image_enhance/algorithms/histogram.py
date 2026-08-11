import numpy
import cv2
from .. import utils

# equalize: Perform histogram equalization (HE) on a channel float32 [0, 1].
# equalize_cla: Perform CLAHE on a channel float32 [0, 1].

def equalize(channel: numpy.ndarray) -> numpy.ndarray:
    """
    Perform histogram equalization on a channel.
    Args:
        channel (numpy.ndarray): Input channel, expected float32 [0, 1].
    Returns:
        numpy.ndarray: Equalized channel, expected float32 [0, 1].
    """
    channel_u8 = utils.denormalize(channel)  # Convert float32 [0, 1] to uint8 [0, 255]
    channel_eq_u8 = cv2.equalizeHist(channel_u8)
    channel_eq = utils.normalize(channel_eq_u8)  # Convert back to float32 [0, 1]
    return channel_eq

def equalize_cla(channel: numpy.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 6) -> numpy.ndarray:
    """
    Perform CLAHE (Contrast Limited Adaptive Histogram Equalization) on a channel.
    Args:
        channel (numpy.ndarray): Input channel, expected float32 [0, 1].
        clip_limit (float): Threshold for contrast limiting, defaults to 2.0.
        tile_grid_size (int): Size of grid, defaults to 6.
    Returns:
        numpy.ndarray: CLAHE equalized channel, expected float32 [0, 1].
    """
    channel_u8 = utils.denormalize(channel)  # Convert float32 [0, 1] to uint8 [0, 255]
    clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = (tile_grid_size, tile_grid_size))
    channel_cla_u8 = clahe.apply(channel_u8)
    channel_cla = utils.normalize(channel_cla_u8)   # Convert back to float32 [0, 1]
    return channel_cla
