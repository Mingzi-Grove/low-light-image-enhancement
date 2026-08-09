import numpy
import cv2

# BGR2LCH: BGR float32 [0, 1] -> LCH float32 L[0, 1], C[0, 1], H[-π, π].
# LCH2BGR: LCH float32 L[0, 1], C[0, 1], H[-π, π] -> BGR float32 [0, 1].

# BGR2LAB: BGR float32 [0, 1] -> LAB float32 L[0, 100], A[-128, 127], B[-128, 127].
# LAB2BGR: LAB float32 L[0, 100], A[-128, 127], B[-128, 127] -> BGR float32 [0, 1].
# LAB2LCH: LAB float32 L[0, 100], A[-128, 127], B[-128, 127] -> LCH float32 L[0, 1], C[0, 1], H[-π, π].
# LCH2LAB: LCH float32 L[0, 1], C[0, 1], H[-π, π] -> LAB float32 L[0, 100], A[-128, 127], B[-128, 127].

def BGR2LCH(bgr_img: numpy.ndarray) -> numpy.ndarray:
    """
    Args:
        bgr_img (numpy.ndarray): Expected bgr_img, float32, [0, 1].
    Returns:
        numpy.ndarray: lch_img, float32, L[0, 1], C[0, 1], H[-π, π].
    """
    lab_img = BGR2LAB(bgr_img)
    lch_img = LAB2LCH(lab_img)
    return lch_img

def LCH2BGR(lch_img: numpy.ndarray) -> numpy.ndarray:
    """
    Args:
        lch_img (numpy.ndarray): Expected lch_img, float32, L[0, 1], C[0, 1], H[-π, π].
    Returns:
        numpy.ndarray: bgr_img, float32, [0, 1].
    """
    lab_img = LCH2LAB(lch_img)
    bgr_img = LAB2BGR(lab_img)
    return bgr_img

def BGR2LAB(bgr_img: numpy.ndarray) -> numpy.ndarray:
    """
    Args:
        bgr_img (numpy.ndarray): Expected bgr_img, float32, [0, 1].
    Returns:
        numpy.ndarray: lab_img, float32, L[0, 100], A[-128, 127], B[-128, 127].
    """
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    return lab_img

def LAB2BGR(lab_img: numpy.ndarray) -> numpy.ndarray:
    """
    Args:
        lab_img (numpy.ndarray): Expected lab_img, float32, L[0, 100], A[-128, 127], B[-128, 127].
    Returns:
        numpy.ndarray: bgr_img, float32, [0, 1].
    """
    bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    return bgr_img

def LAB2LCH(lab_img: numpy.ndarray) -> numpy.ndarray:
    """
    Args:
        lab_img (numpy.ndarray): Expected lab_img, float32, L[0, 100], A[-128, 127], B[-128, 127].
    Returns:
        numpy.ndarray: lch_img, float32, L[0, 1], C[0, 1], H[-π, π].
    """
    L = lab_img[:, :, 0] / 100.0    # 归一化 [0, 1]
    A = lab_img[:, :, 1]
    B = lab_img[:, :, 2]

    C = numpy.sqrt(A**2 + B**2) / 181.0   # 归一化 [0, 1]，181 ≈ sqrt(128²+128²)
    H = numpy.arctan2(B, A)               # 保留原始弧度，范围 [-pi, pi]
    
    lch_img = numpy.dstack((L, C, H)).astype(numpy.float32)
    return lch_img

def LCH2LAB(lch_img: numpy.ndarray) -> numpy.ndarray:
    """
    Args:
        lch_img (numpy.ndarray): Expected lch_img, float32, L[0, 1], C[0, 1], H[-π, π].
    Returns:
        numpy.ndarray: lab_img, float32, L[0, 100], A[-128, 127], B[-128, 127].
    """
    L = lch_img[:, :, 0] * 100.0    # 反归一化 [0, 100]
    C = lch_img[:, :, 1] * 181.0    # 反归一化 [0, 181]
    H = lch_img[:, :, 2]

    A = C * numpy.cos(H)             # A = C * cos(H)
    B = C * numpy.sin(H)             # B = C * sin(H)
    
    lab_img = cv2.merge([L, A, B])  # compose back to LAB image
    return lab_img