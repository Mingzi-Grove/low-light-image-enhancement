import numpy

# gamma: img_gray float32 [0, 1] -> img_gray float32 [0, 1], Nonlinear mapping curve.
# kaga: img_gray float32 [0, 1] -> img_gray float32 [0, 1], Nonlinear gain curve.
# kaga_autok: img_gray float32 [0, 1] -> img_gray float32 [0, 1], Nonlinear gain curve with auto arg k.

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
    with numpy.errstate(divide="ignore"):
        gain_map = (k * numpy.power(channel, gamma - 1)).clip(0, upper_limit)
    channel_gained = (channel * gain_map).clip(0, 1)
    return channel_gained

def kaga_autok(channel: numpy.ndarray, gamma: float = 0.35, percentile: float = 99.9, upper_limit: int = 50) -> numpy.ndarray:
    """
    Args:
        channel (numpy.ndarray): Expected a channel of an image, float32, [0, 1].
        gamma (float): Gamma value, defaults to 0.35.
        percentile (float): Reference percentile of the channel, used to compute k, defaults to 99.9.
        upper_limit (int): Gain upper limit, defaults to 50.
    Returns:
        numpy.ndarray: img_gray, float32, [0, 1]
    """
    ref = numpy.percentile(channel, percentile)
    k = 1 / ref**gamma
    with numpy.errstate(divide="ignore"):
        gain_map = (k * numpy.power(channel, gamma - 1)).clip(0, upper_limit)
    channel_gained = (channel * gain_map)

    print(f"channel ref(p{percentile:g}): {ref:.2f}, k: {k:.2f}, channel_gained max: {channel_gained.max():.2f}")

    return channel_gained.clip(0, 1)
