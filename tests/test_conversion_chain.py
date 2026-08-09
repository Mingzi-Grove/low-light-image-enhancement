import numpy
import cv2
import pytest
from lowlight_image_enhance import colorspace, utils

# === 测试数据转换 ===

# 构造测试图像：随机 BGR 图，uint8，像素值在 [0, 255]
H, W = 64, 64
rng = numpy.random.default_rng(42)  # 固定随机种子
bgr_u8 = rng.integers(0, 256, size=(H, W, 3)).astype(numpy.uint8)

# 测试 roundtrip，包括每个环节的 dtype 和 range，还有最终的往返误差
def test_cvt_roundtrip():

    # 1. BGR uint8 [0, 255] -> BGR float32 [0, 1]
    bgr_f32 = utils.normalize(bgr_u8)
    assert bgr_f32.dtype == numpy.float32, f"dtype error: expected float32, got {bgr_f32.dtype}"
    assert bgr_f32.min() >= 0.0 and bgr_f32.max() <= 1.0, f"range error: expected [0, 1], got [{bgr_f32.min()}, {bgr_f32.max()}]"

    # 2. BGR float32 [0, 1] -> LAB float32 L[0, 100], A[-128, 127], B[-128, 127]
    lab_f32 = colorspace.BGR2LAB(bgr_f32)
    assert lab_f32.dtype == numpy.float32, f"dtype error: expected float32, got {lab_f32.dtype}"
    assert lab_f32[:, :, 0].min() >= 0.0 and lab_f32[:, :, 0].max() <= 100.0, f"range error: expected L [0, 100], got [{lab_f32[:, :, 0].min():.3f}, {lab_f32[:, :, 0].max():.3f}]"
    assert lab_f32[:, :, 1].min() >= -128.0 and lab_f32[:, :, 1].max() <= 127.0, f"range error: expected A [-128, 127], got [{lab_f32[:, :, 1].min():.3f}, {lab_f32[:, :, 1].max():.3f}]"
    assert lab_f32[:, :, 2].min() >= -128.0 and lab_f32[:, :, 2].max() <= 127.0, f"range error: expected B [-128, 127], got [{lab_f32[:, :, 2].min():.3f}, {lab_f32[:, :, 2].max():.3f}]"

    # 3.LAB -> LCH float32 L[0, 1], C[0, 1], H[-π, π]
    lch_f32 = colorspace.LAB2LCH(lab_f32)
    assert lch_f32.dtype == numpy.float32, f"dtype error: expected float32, got {lch_f32.dtype}"
    assert lch_f32[:, :, 0].min() >= 0.0 and lch_f32[:, :, 0].max() <= 1.0, f"range error: expected L [0, 1], got [{lch_f32[:, :, 0].min():.3f}, {lch_f32[:, :, 0].max():.3f}]"
    assert lch_f32[:, :, 1].min() >= 0.0 and lch_f32[:, :, 1].max() <= 1.01, f"range error: expected C [0, 1], got [{lch_f32[:, :, 1].min():.3f}, {lch_f32[:, :, 1].max():.3f}]"
    assert lch_f32[:, :, 2].min() >= -numpy.pi and lch_f32[:, :, 2].max() <= numpy.pi, f"range error: expected H [-π, π], got [{lch_f32[:, :, 2].min():.3f}, {lch_f32[:, :, 2].max():.3f}]"

    # 4. LCH -> LAB float32 L[0, 100], A[-128, 127], B[-128, 127]
    lab_f32_round = colorspace.LCH2LAB(lch_f32)
    assert lab_f32_round.dtype == numpy.float32, f"dtype error: expected float32, got {lab_f32_round.dtype}"
    assert lab_f32_round[:, :, 0].min() >= 0.0 and lab_f32_round[:, :, 0].max() <= 100.0, f"range error: expected L [0, 100], got [{lab_f32_round[:, :, 0].min():.3f}, {lab_f32_round[:, :, 0].max():.3f}]"
    assert lab_f32_round[:, :, 1].min() >= -128.0 and lab_f32_round[:, :, 1].max() <= 127.0, f"range error: expected A [-128, 127], got [{lab_f32_round[:, :, 1].min():.3f}, {lab_f32_round[:, :, 1].max():.3f}]"
    assert lab_f32_round[:, :, 2].min() >= -128.0 and lab_f32_round[:, :, 2].max() <= 127.0, f"range error: expected B [-128, 127], got [{lab_f32_round[:, :, 2].min():.3f}, {lab_f32_round[:, :, 2].max():.3f}]"

    # 5. LAB -> BGR float32 [0, 1]
    bgr_f32_round = colorspace.LAB2BGR(lab_f32_round)
    assert bgr_f32_round.dtype == numpy.float32, f"dtype error: expected float32, got {bgr_f32_round.dtype}"
    assert bgr_f32_round.min() >= 0.0 and bgr_f32_round.max() <= 1.0, f"range error: expected [0, 1], got [{bgr_f32_round.min():.3f}, {bgr_f32_round.max():.3f}]"

    # 6. BGR float32 [0, 1] -> BGR uint8 [0, 255]
    bgr_u8_round = utils.denormalize(bgr_f32_round)
    assert bgr_u8_round.dtype == numpy.uint8, f"dtype error: expected uint8, got {bgr_u8_round.dtype}" 
    assert bgr_u8_round.min() >= 0 and bgr_u8_round.max() <= 255, f"range error: expected [0, 255], got [{bgr_u8_round.min()}, {bgr_u8_round.max()}]"

    # 7. 计算往返误差
    f32_error = numpy.abs(bgr_f32_round - bgr_f32)
    f32_max_error = f32_error.max()
    assert f32_max_error <= 0.005, f"Maximum roundtrip error {f32_max_error:.4f}, BGR f32"
    u8_error = numpy.abs(bgr_u8_round.astype(numpy.int16) - bgr_u8.astype(numpy.int16))
    u8_max_error = u8_error.max()
    assert u8_max_error <= 1, f"Maximum roundtrip error {u8_max_error}, BGR u8"

# 分层测试误差来源
def test_cvt_error_sources():
    
    # 1. BGR uint8 [0, 255] <-> BGR float32 [0, 1]
    bgr_f32 = utils.normalize(bgr_u8)
    bgr_u8_r1 = utils.denormalize(bgr_f32)
    error_r1 = numpy.abs(bgr_u8_r1.astype(numpy.int16) - bgr_u8.astype(numpy.int16))
    print("BGR u8 <-> BGR f32")
    print(f" - maximum error: {error_r1.max()}")
    print(f" - mean error: {error_r1.mean():.4f}")
    assert error_r1.max() <= 1, f"roundtrip error {error_r1.max()}, r1"

    # 2. BGR <-> LAB
    lab_f32 = colorspace.BGR2LAB(bgr_f32)
    bgr_f32_r2 = colorspace.LAB2BGR(lab_f32)
    error_r2 = numpy.abs(bgr_f32_r2 - bgr_f32)
    print("BGR f32 <-> LAB f32")
    print(f" - maximum error: {error_r2.max():.4f}")
    print(f" - mean error: {error_r2.mean():.4f}")
    assert error_r2.max() <= 0.005, f"roundtrip error {error_r2.max():.4f}, r2"

    # 3. LAB <-> LCH
    lch_f32 = colorspace.LAB2LCH(lab_f32)
    lab_f32_r3 = colorspace.LCH2LAB(lch_f32)
    error_r3 = numpy.abs(lab_f32_r3 - lab_f32)
    print("LAB f32 <-> LCH f32")
    print(f" - maximum error: {error_r3.max():.4f}")
    print(f" - mean error: {error_r3.mean():.4f}")
    assert error_r3.max() <= 0.005, f"roundtrip error {error_r3.max():.4f}, r3"
