from typing import NamedTuple
import numpy
from . import algorithms

# 亮度增益：处理 LCH 空间中的 Lightness 分量

class BoostResult(NamedTuple):
    """
    A named tuple to hold the result of the boost operation.
    Attributes:
        L_boosted (numpy.ndarray): L layer after boosting, float32, L[0, 1].
        delta_L (numpy.ndarray): Delta map (boosted - original), float32, L[0, 1].
    """
    L_boosted: numpy.ndarray
    delta_L: numpy.ndarray

def boost(L_layer: numpy.ndarray) -> BoostResult:
    """
    Args:
        L_layer (numpy.ndarray): Expected L layer in LCH space, float32, L[0, 1].
    Returns:
        BoostResult: L layer after boosting and the delta map (boosted - original).
    """
    L_boosted = algorithms.gamma(L_layer)
    delta_L = L_boosted - L_layer
    return BoostResult(L_boosted, delta_L)
    