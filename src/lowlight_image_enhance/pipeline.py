import numpy
import cv2
import logging
logger = logging.getLogger(__name__)
from . import utils, colorspace, lightness, chroma, denoiser


# Pipeline for low-light image enhancement
# TODO：以后要做可选参数，透传给 lightness.boost 和 chroma.compensate

def enhance(img_bgr_u8: numpy.ndarray) -> numpy.ndarray:
    """
    Enhance a low-light image using the defined pipeline.
    Args:
        img_bgr_u8 (numpy.ndarray): Input low-light image, expected BGR, uint8 [0-255].
    Returns:
        numpy.ndarray: Enhanced image, BGR, uint8 [0-255].
    """
    # 1. NLM denoising
    logger.debug("Denoising... method: Non-Local Means")
    img_bgr_u8_denoised = denoiser.non_local_means_color(img_bgr_u8)

    # 2. Normalize to float32 [0, 1]
    logger.debug("Normalizing to float32 [0, 1]...")
    img_bgr_f32 = utils.normalize(img_bgr_u8_denoised)

    # 3. Convert to LCH and split channels
    logger.debug("Converting to LCH space...")
    img_lch = colorspace.BGR2LCH(img_bgr_f32)
    L_lch, C_lch, H_lch = cv2.split(img_lch)
    
    # 4. Boost lightness
    logger.debug("Boosting lightness...")
    L_lch_boosted, delta_L = lightness.boost(L_lch)

    # 5. Compensate chroma with delta_L
    logger.debug("Compensating chroma...")
    C_lch_compensated, delta_C = chroma.compensate(C_lch, delta_L)

    # 6. Compose back and convert to BGR
    logger.debug("Composing back to BGR space...")
    img_lch_enhanced = cv2.merge((L_lch_boosted, C_lch_compensated, H_lch))
    img_bgr_f32_enhanced = colorspace.LCH2BGR(img_lch_enhanced)
    img_bgr_u8_enhanced = utils.denormalize(img_bgr_f32_enhanced)

    return img_bgr_u8_enhanced
