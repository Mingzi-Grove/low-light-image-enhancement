from typing import NamedTuple
import numpy

# 饱和度补偿，处理 LCH 空间中的 Chroma 分量，由 ΔL 驱动的饱和度补偿

class CompResult(NamedTuple):
    """
    A named tuple to hold the result of the chroma compensation operation.
    Attributes:
        C_compensated (numpy.ndarray): C layer after applying compensation, float32, C[0, 1].
        delta_C (numpy.ndarray):  算法的理论增益量, float32, C[0, 1].
    """
    C_compensated: numpy.ndarray
    delta_C: numpy.ndarray

# TODO：线性补偿是占位符，后续要换成正经增益算法
def compensate(C_layer: numpy.ndarray, delta_L: numpy.ndarray, k: float = 0.3) -> CompResult:
    """
    Args:
        C_layer (numpy.ndarray): Expected C layer in LCH space, float32, C[0, 1].
        delta_L (numpy.ndarray): Delta map from lightness boost, float32, L[0, 1].
        k (float, optional): Compensation factor. Defaults to 0.3.
    Returns:
        CompResult: C layer after compensation and the delta map.
    """
    # 简单的线性补偿公式：ΔC = k * ΔL
    delta_C = k * delta_L
    C_compensated = C_layer + delta_C
    # 钳位到 [0, 1]
    C_compensated = numpy.clip(C_compensated, 0.0, 1.0)
    return CompResult(C_compensated, delta_C)

