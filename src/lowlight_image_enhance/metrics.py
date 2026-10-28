import numpy
import cv2

# 计算各种图像质量指标的函数

def psnr(reference: numpy.ndarray, target: numpy.ndarray, max_value: float = 1.0) -> float:
    """
    Compute PSNR for gray images.
    Args:
        reference (numpy.ndarray): Reference channel.
        target (numpy.ndarray): Target channel to compare.
        max_value (float): The maximum possible pixel value in the images. Defaults to 1.0.
    Returns:
        float: PSNR value in decibels (dB).
    """
    if (reference.shape != target.shape) or (reference.ndim != 2):
        raise ValueError("Input images must be gray images (1 channel).")
    if (reference.dtype != target.dtype) or (reference.dtype != numpy.float32):
        raise ValueError("Input images must be of type numpy.float32.")

    mse = numpy.mean((reference - target) ** 2)
    if mse == 0:
        return float('inf')  # No noise
    psnr_value = 10 * numpy.log10(max_value ** 2 / mse)
    return psnr_value

def psnr_bgr(reference: numpy.ndarray, target: numpy.ndarray, max_value: float = 1.0) -> float:
    """
    Compute PSNR for BGR images.
    Args:
        reference (numpy.ndarray): Reference BGR image.
        target (numpy.ndarray): Target BGR image to compare.
        max_value (float): The maximum possible pixel value in the images. Defaults to 1.0.
    Returns:
        float: PSNR value in decibels (dB).
    """
    if reference.shape != target.shape or reference.shape[2:] != (3,):
        raise ValueError("Input images must be BGR (3 channels).")
    if (reference.dtype != target.dtype) or (reference.dtype != numpy.float32):
        raise ValueError("Input images must be of type numpy.float32.")
    
    # Caculate mean MSE for each channel
    mse_b = numpy.mean((reference[:, :, 0] - target[:, :, 0]) ** 2)
    mse_g = numpy.mean((reference[:, :, 1] - target[:, :, 1]) ** 2)
    mse_r = numpy.mean((reference[:, :, 2] - target[:, :, 2]) ** 2)
    mse = (mse_b + mse_g + mse_r) / 3.0

    # Caculate PSNR
    if mse == 0:
        return float('inf')  # No noise
    psnr_value = 10 * numpy.log10(max_value ** 2 / mse)

    return psnr_value

def ssim(reference: numpy.ndarray, target: numpy.ndarray, window_size: int = 11, sigma: float = 1.5, max_value: float = 1.0) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two channels.
    Args:
        reference (numpy.ndarray): Reference channel.
        target (numpy.ndarray): Target channel to compare.
        window_size (int): Size of the window used for SSIM calculation. Defaults to 11.
        sigma (float): Standard deviation for the Gaussian kernel. Defaults to 1.5.
        max_value (float): The maximum possible pixel value in the images. Defaults to 1.0.
    Returns:
        float: SSIM value.
    """
    if (window_size <= 0) or (window_size % 2 == 0):
        raise ValueError("window_size must be a positive odd integer.")
    if (reference.shape != target.shape) or (reference.ndim != 2):
        raise ValueError("Input images must be gray images (1 channel).")
    if (reference.dtype != target.dtype) or (reference.dtype != numpy.float32):
        raise ValueError("Input images must be of type numpy.float32.")

    # 1. 计算稳定常数 C，防止除零（K1 = 0.01， K2 = 0.03）
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2

    # 2. 高斯模糊 与 均值、方差、协方差
    # 高斯模糊
    def gaussian_blur(img: numpy.ndarray) -> numpy.ndarray:
        return cv2.GaussianBlur(img, (window_size, window_size), sigma, borderType=cv2.BORDER_REFLECT)
    # 计算亮度均值 μ[x], μ[y]
    mu_x = gaussian_blur(reference)
    mu_y = gaussian_blur(target)
    # 计算方差 σ²[x] = E[x²] - μ²[x]
    sigma_x_sq = gaussian_blur(reference ** 2) - mu_x ** 2
    sigma_y_sq = gaussian_blur(target ** 2) - mu_y ** 2
    # 计算协方差 σ[x,y] = E[xy] - μ[x]μ[y]
    sigma_xy = gaussian_blur(reference * target) - mu_x * mu_y

    # 3. 计算亮度项 l 和 对比度*结构项 cs
    # 亮度项 l
    l = (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1)
    # 对比度*结构项 cs
    cs = (2 * sigma_xy + c2) / (sigma_x_sq + sigma_y_sq + c2)

    # 4. 计算 SSIM map 并取均值
    ssim_map = l * cs

    pad = (window_size - 1) // 2 
    if ssim_map.shape[0] > 2 * pad and ssim_map.shape[1] > 2 * pad:
        ssim_map = ssim_map[pad:-pad, pad:-pad]

    ssim_value = float(numpy.mean(ssim_map))

    return ssim_value

def ssim_bgr(reference: numpy.ndarray, target: numpy.ndarray, window_size: int = 11, sigma: float = 1.5, max_value: float = 1.0) -> float:
    """
    Compute SSIM for BGR images.
    Args:
        reference (numpy.ndarray): Reference BGR image.
        target (numpy.ndarray): Target BGR image to compare.
        window_size (int): Size of the window used for SSIM calculation. Defaults to 11.
        sigma (float): Standard deviation for the Gaussian kernel. Defaults to 1.5.
        max_value (float): The maximum possible pixel value in the images. Defaults to 1.0.
    Returns:
        float: SSIM value.
    """
    if reference.shape != target.shape or reference.shape[2:] != (3,):
        raise ValueError("Input images must be BGR (3 channels).")
    if (reference.dtype != target.dtype) or (reference.dtype != numpy.float32):
        raise ValueError("Input images must be of type numpy.float32.")

    # Calculate SSIM for each channel and average
    ssim_b = ssim(reference[:, :, 0], target[:, :, 0], window_size, sigma, max_value)
    ssim_g = ssim(reference[:, :, 1], target[:, :, 1], window_size, sigma, max_value)
    ssim_r = ssim(reference[:, :, 2], target[:, :, 2], window_size, sigma, max_value)

    return (ssim_b + ssim_g + ssim_r) / 3.0