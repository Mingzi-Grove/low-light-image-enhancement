import numpy

# gamma: img_gray float32 [0, 1] -> img_gray float32 [0, 1], Nonlinear mapping curve.
# kaga: img_gray float32 [0, 1] -> img_gray float32 [0, 1], Nonlinear gain curve.

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
    
def kaga(channel: numpy.ndarray, gamma: float = 0.35, k: float = 1.45, upper_limit: int = 50) -> numpy.ndarray:
    """
    Args:
        channel (numpy.ndarray): Expected a channel of an image, float32, [0, 1].
    Returns:
        numpy.ndarray: img_gray, float32, [0, 1]
    """
    gain_map = k * numpy.power(channel, gamma - 1).clip(0, upper_limit)
    channel_gained = (channel * gain_map).clip(0, 1)
    return channel_gained
        