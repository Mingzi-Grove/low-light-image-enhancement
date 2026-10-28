"""metrics.py 的单元测试。

分两类：
1. 行为与异常：相同图像、形状不符、通道数不符、dtype 不符时函数应有明确的表现
2. 精度验证：与 skimage 的参考实现交叉比对，确认计算口径与 Wang et al. 2004 一致

测试数据：
- 合成数据：固定随机种子生成，保证每次运行结果一致
- 真实数据：examples/fortest 下裁自 RENOIR 数据集的 512×512 图像对
"""

from pathlib import Path

import numpy
import pytest

from lowlight_image_enhance import metrics, utils

# scikit-image 属于 dev 可选依赖，缺失时只跳过交叉验证部分，其余测试照常运行
try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

requires_skimage = pytest.mark.skipif(not HAS_SKIMAGE, reason="未安装 scikit-image")

# 真实图像对裁自 RENOIR 数据集：IMG_0055Noisy / IMG_0053Reference
# 取原图几何中心 512×512 区域，存为 PNG 保证无损
FORTEST_DIR = Path(__file__).resolve().parents[1] / "examples" / "fortest"
REAL_REF_PATH = FORTEST_DIR / "renoir_IMG_0053_ref.png"
REAL_NOISY_PATH = FORTEST_DIR / "renoir_IMG_0055_noisy.png"

requires_real_images = pytest.mark.skipif(
    not (REAL_REF_PATH.exists() and REAL_NOISY_PATH.exists()),
    reason="examples/fortest 下缺少真实图像对",
)

# SSIM 与 skimage 对齐后，实测差值在 1e-7 量级（float32 舍入），此处留 10 倍余量
SSIM_TOL = 1e-6
PSNR_TOL = 1e-5


# ---------- 生成测试数据 ----------

def make_gray(seed=0, size=32):
    """生成一张 float32 [0,1] 的灰度图像（固定种子，保证可复现）"""
    rng = numpy.random.default_rng(seed)
    return rng.random((size, size), dtype=numpy.float32)


def make_bgr(seed=0, size=32):
    """生成一张 float32 [0,1] 的 BGR 图像（固定种子，保证可复现）"""
    rng = numpy.random.default_rng(seed)
    return rng.random((size, size, 3), dtype=numpy.float32)


def make_noisy_pair(seed=0, size=128, sigma=0.05):
    """生成一对「原图 + 轻微加噪图」，SSIM 处于中间区间，便于观察分数变化"""
    base = make_gray(seed, size)
    noise = numpy.random.default_rng(seed + 1).normal(0, sigma, base.shape)
    noisy = numpy.clip(base + noise, 0, 1).astype(numpy.float32)
    return base, noisy


def load_real_pair():
    """读取 examples/fortest 下的真实图像对，归一化为 float32 [0,1]"""
    ref = utils.imread(str(REAL_REF_PATH))
    noisy = utils.imread(str(REAL_NOISY_PATH))
    return utils.normalize(ref), utils.normalize(noisy)


# ---------- psnr ----------

def test_psnr_identical_images():
    """完全相同的两张图，MSE 为 0，PSNR 应返回 inf 而不是报错"""
    img = make_gray()
    assert metrics.psnr(img, img) == float("inf")


def test_psnr_closed_form():
    """用解析解验证数值：每个像素都差 0.125，PSNR = 20*log10(1/0.125)"""
    ref = numpy.zeros((8, 8), dtype=numpy.float32)
    tgt = numpy.full((8, 8), 0.125, dtype=numpy.float32)   # 0.125 = 2^-3，float32 可精确表示
    expected = 20 * numpy.log10(1.0 / 0.125)
    assert metrics.psnr(ref, tgt) == pytest.approx(expected, abs=PSNR_TOL)


def test_psnr_shape_mismatch():
    """两张图尺寸不同应抛出 ValueError，避免静默算出无意义的数"""
    with pytest.raises(ValueError):
        metrics.psnr(make_gray(), make_gray(size=16))


def test_psnr_rejects_3d_input():
    """三通道图像应抛出 ValueError，灰度指标不能直接吃彩色图"""
    with pytest.raises(ValueError):
        metrics.psnr(make_bgr(), make_bgr())


def test_psnr_rejects_uint8():
    """uint8 输入应抛出 ValueError：uint8 做减法会回绕、做平方会溢出，结果不可信"""
    img = (make_gray() * 255).astype(numpy.uint8)
    with pytest.raises(ValueError):
        metrics.psnr(img, img)


# ---------- psnr_bgr ----------

def test_psnr_bgr_identical_images():
    """完全相同的两张 BGR 图，PSNR 应为 inf"""
    img = make_bgr()
    assert metrics.psnr_bgr(img, img) == float("inf")


def test_psnr_bgr_averages_mse_over_channels():
    """只有 B 通道差 0.3，三通道 MSE 平均 = 0.09/3，验证聚合方式为先平均 MSE 再取 dB"""
    ref = numpy.zeros((4, 4, 3), dtype=numpy.float32)
    tgt = ref.copy()
    tgt[:, :, 0] = 0.3
    mse = (0.3 ** 2) / 3
    expected = 10 * numpy.log10(1.0 / mse)
    assert metrics.psnr_bgr(ref, tgt) == pytest.approx(expected, abs=PSNR_TOL)


def test_psnr_bgr_rejects_gray_input():
    """灰度图应被拒绝，并抛出 ValueError 而不是 IndexError"""
    with pytest.raises(ValueError):
        metrics.psnr_bgr(make_gray(), make_gray())


def test_psnr_bgr_rejects_4_channel():
    """四通道（BGRA）应被拒绝，避免 alpha 通道被静默忽略"""
    img = numpy.random.default_rng(0).random((8, 8, 4), dtype=numpy.float32)
    with pytest.raises(ValueError):
        metrics.psnr_bgr(img, img)


def test_psnr_bgr_rejects_uint8():
    """uint8 输入应抛出 ValueError，理由同 psnr"""
    img = (make_bgr() * 255).astype(numpy.uint8)
    with pytest.raises(ValueError):
        metrics.psnr_bgr(img, img)


# ---------- ssim ----------

def test_ssim_identical_images():
    """完全相同的两张图，SSIM 应为 1"""
    img = make_gray()
    assert metrics.ssim(img, img) == pytest.approx(1.0, abs=1e-6)


def test_ssim_in_valid_range():
    """SSIM 的取值范围应落在 [-1, 1]，超出即说明实现有问题"""
    value = metrics.ssim(make_gray(1), make_gray(2))
    assert -1.0 <= value <= 1.0


def test_ssim_closer_image_scores_higher():
    """对同一张参考图，加少量噪声的版本应比无关图得分更高（单调性检查）"""
    base, _ = make_noisy_pair(seed=0, size=64, sigma=0.01)
    _, near = make_noisy_pair(seed=0, size=64, sigma=0.01)
    far = make_gray(2, size=64)
    assert metrics.ssim(base, near) > metrics.ssim(base, far)


def test_ssim_rejects_uint8():
    """uint8 输入应抛出 ValueError：内部有 reference**2，uint8 会溢出成荒谬的分数"""
    img = (make_gray() * 255).astype(numpy.uint8)
    with pytest.raises(ValueError):
        metrics.ssim(img, img)


def test_ssim_rejects_3d_input():
    """三通道图像应抛出 ValueError，单通道 SSIM 不接受彩色图"""
    with pytest.raises(ValueError):
        metrics.ssim(make_bgr(), make_bgr())


@pytest.mark.parametrize("window_size", [0, -1, 10])
def test_ssim_rejects_invalid_window_size(window_size):
    """窗口必须是正奇数：偶数或非正数应抛 ValueError，而不是让 OpenCV 抛出内部错误"""
    img = make_gray()
    with pytest.raises(ValueError):
        metrics.ssim(img, img, window_size=window_size)


# ---------- ssim_bgr ----------

def test_ssim_bgr_identical_images():
    """完全相同的两张 BGR 图，SSIM 应为 1"""
    img = make_bgr()
    assert metrics.ssim_bgr(img, img) == pytest.approx(1.0, abs=1e-6)


def test_ssim_bgr_equals_mean_of_channels():
    """彩色 SSIM 取三通道 SSIM 的算术平均，验证聚合方式"""
    ref = make_bgr(1)
    tgt = make_bgr(2)
    expected = numpy.mean([metrics.ssim(ref[:, :, c], tgt[:, :, c]) for c in range(3)])
    assert metrics.ssim_bgr(ref, tgt) == pytest.approx(expected, abs=1e-9)


def test_ssim_bgr_rejects_gray_input():
    """灰度图应被拒绝，并抛出 ValueError"""
    with pytest.raises(ValueError):
        metrics.ssim_bgr(make_gray(), make_gray())


def test_ssim_bgr_rejects_4_channel():
    """四通道（BGRA）应被拒绝"""
    img = numpy.random.default_rng(0).random((8, 8, 4), dtype=numpy.float32)
    with pytest.raises(ValueError):
        metrics.ssim_bgr(img, img)


# ---------- 与 skimage 交叉验证：合成数据 ----------

@requires_skimage
def test_psnr_matches_skimage():
    """PSNR 与 skimage 的参考实现一致（float32 舍入范围内）"""
    ref, tgt = make_noisy_pair(seed=7, size=128)
    theirs = sk_psnr(ref, tgt, data_range=1.0)
    assert metrics.psnr(ref, tgt) == pytest.approx(theirs, abs=PSNR_TOL)


@requires_skimage
def test_ssim_matches_skimage():
    """SSIM 与 skimage 的参考实现一致：11×11 高斯 σ=1.5、总体协方差、排除边界"""
    ref, tgt = make_noisy_pair(seed=7, size=128)
    theirs = sk_ssim(
        ref, tgt, data_range=1.0,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )
    assert metrics.ssim(ref, tgt) == pytest.approx(theirs, abs=SSIM_TOL)


@requires_skimage
def test_ssim_border_cropping_is_applied():
    """小图上验证「排除边界」：32×32 图边界占 52%，若不裁剪会与标准差 1e-2 量级"""
    ref, tgt = make_noisy_pair(seed=7, size=32)
    theirs = sk_ssim(
        ref, tgt, data_range=1.0,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )
    assert metrics.ssim(ref, tgt) == pytest.approx(theirs, abs=SSIM_TOL)


# ---------- 与 skimage 交叉验证：真实图像 ----------

@requires_skimage
@requires_real_images
def test_psnr_matches_skimage_on_real_image():
    """真实图像对上 PSNR 与 skimage 一致，确认实现不依赖合成数据的特殊性质"""
    ref, noisy = load_real_pair()
    gray_ref = ref[:, :, 1]
    gray_noisy = noisy[:, :, 1]
    theirs = sk_psnr(gray_ref, gray_noisy, data_range=1.0)
    assert metrics.psnr(gray_ref, gray_noisy) == pytest.approx(theirs, abs=PSNR_TOL)


@requires_skimage
@requires_real_images
def test_psnr_bgr_matches_skimage_on_real_image():
    """BGR 的 PSNR 与 skimage 一致：skimage 对整幅图求 MSE，与「三通道 MSE 平均」同口径"""
    ref, noisy = load_real_pair()
    theirs = sk_psnr(ref, noisy, data_range=1.0)
    assert metrics.psnr_bgr(ref, noisy) == pytest.approx(theirs, abs=PSNR_TOL)


@requires_skimage
@requires_real_images
def test_psnr_bgr_is_not_mean_of_channel_db():
    """钉住口径：psnr_bgr 是「先平均 MSE 再取 dB」，不是「三个 dB 值平均」（两者差 0.37 dB）"""
    ref, noisy = load_real_pair()
    merged = metrics.psnr_bgr(ref, noisy)
    mean_db = numpy.mean([
        sk_psnr(ref[:, :, c], noisy[:, :, c], data_range=1.0) for c in range(3)
    ])
    assert abs(merged - mean_db) > 0.1


@requires_skimage
@requires_real_images
def test_ssim_bgr_matches_skimage_on_real_image():
    """彩色 SSIM 与 skimage 一致：逐通道计算后取平均，与 channel_axis=2 的行为相同"""
    ref, noisy = load_real_pair()
    theirs = sk_ssim(
        ref, noisy, data_range=1.0, channel_axis=2,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )
    assert metrics.ssim_bgr(ref, noisy) == pytest.approx(theirs, abs=SSIM_TOL)
