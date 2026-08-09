import numpy

# Gamma: img_gray float32 [0, 1] -> img_gray float32 [0, 1], Nonlinear mapping.

def gamma(img_gray: numpy.ndarray, gamma: float) -> numpy.ndarray:
    """
    Args:
        img_gray (numpy.ndarray): Expected img_gray, float32, [0, 1].
        gamma (float): Gamma value.
    Returns:
        numpy.ndarray: img_gray, float32, [0, 1].
    """
    gamma_img = numpy.power(img_gray, gamma)
    return gamma_img