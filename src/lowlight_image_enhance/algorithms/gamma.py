import numpy

# Gamma: img_gray float32 [0, 1] -> img_gray float32 [0, 1], Nonlinear mapping.

def gamma(channel: numpy.ndarray, exponent: float = 0.5) -> numpy.ndarray:
    """
    Args:
        channel (numpy.ndarray): Expected a channel of an image, float32, [0, 1].
        exponent (float): Gamma value, defaults to 0.5.
    Returns:
        numpy.ndarray: img_gray, float32, [0, 1].
    """
    gamma_img = numpy.power(channel, exponent)
    return gamma_img
    