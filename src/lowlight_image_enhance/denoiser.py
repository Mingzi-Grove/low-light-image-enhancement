import numpy
import cv2

# 降噪

def gaussian_blur(channel: numpy.ndarray, kernel_size: int = 11, sigma: float = 0.0) -> numpy.ndarray:
    """
    Apply Gaussian blur to a single channel.
    Args:
        channel (numpy.ndarray): Input channel to be blurred.
        kernel_size (int): Size of the Gaussian kernel. Must be a positive odd integer. Defaults to 11.
        sigma (float): Standard deviation for the Gaussian kernel. Defaults to 0.0.
    Returns:
        numpy.ndarray: Blurred channel.
    """
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("Kernel size must be a positive odd integer.")
    blurred_channel = cv2.GaussianBlur(channel, (kernel_size, kernel_size), sigma)
    return blurred_channel

def bilateral_filter(channel: numpy.ndarray, diameter: int = 11, sigma_color: float = 0.2, sigma_space: float = 1.5) -> numpy.ndarray:
    """
    Apply bilateral filter to a single channel.
    Args:
        channel (numpy.ndarray): Input channel to be filtered.
        diameter (int): Diameter of each pixel neighborhood. Defaults to 11.
        sigma_color (float): Filter sigma in color space. Defaults to 0.2.
        sigma_space (float): Filter sigma in coordinate space. Defaults to 1.5.
    Returns:
        numpy.ndarray: Filtered channel.
    """
    filtered_channel = cv2.bilateralFilter(channel, diameter, sigma_color, sigma_space)
    return filtered_channel

# TODO: guided_filter()

def non_local_means(channel: numpy.ndarray, h: float = 10.0, template_window_size: int = 7, search_window_size: int = 21) -> numpy.ndarray:
    """
    Apply Non-Local Means denoising to a single channel.
    Args:
        channel (numpy.ndarray): Input channel to be denoised, expected float32 [0, 1].
        h (float): Parameter regulating filter strength. Defaults to 10.0.
        template_window_size (int): Size of the template patch. Defaults to 7.
        search_window_size (int): Size of the window used to compute weighted average. Defaults to 21.
    Returns:
        numpy.ndarray: Denoised channel.
    """
    # Convert the channel to uint8 [0, 255] for OpenCV processing
    channel_uint8 = (channel * 255).clip(0, 255).astype(numpy.uint8)
    # Apply Non-Local Means denoising
    channel_uint8_denoised = cv2.fastNlMeansDenoising(channel_uint8, None, h, template_window_size, search_window_size)
    # Convert back to float32 [0, 1]
    channel_denoised = channel_uint8_denoised.astype(numpy.float32) / 255.0
    return channel_denoised

def non_local_means_color(img_bgr: numpy.ndarray, h: float = 10.0, template_window_size: int = 7, search_window_size: int = 21) -> numpy.ndarray:
    """
    Apply Non-Local Means denoising to a color image.
    Args:
        img_bgr (numpy.ndarray): Input BGR image to be denoised, expected uint8 [0, 255].
        h (float): Parameter regulating filter strength. Defaults to 10.0.
        template_window_size (int): Size of the template patch. Defaults to 7.
        search_window_size (int): Size of the window used to compute weighted average. Defaults to 21.
    Returns:
        numpy.ndarray: Denoised BGR image, expected uint8 [0, 255].
    """
    img_bgr_denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, template_window_size, search_window_size)
    return img_bgr_denoised