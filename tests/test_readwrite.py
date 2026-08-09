"""测试 utils 模块：normalize / denormalize / imread / imwrite"""

import numpy
from lowlight_image_enhance import utils

def test_imread_missing_file_raises():
    """读取不存在的文件应抛出 FileNotFoundError"""
    import pytest

    with pytest.raises(FileNotFoundError):
        utils.imread("definitely_not_exist.png")


def test_imwrite_imread_roundtrip(tmp_path):
    """imwrite 后再 imread，应读回原图（uint8 无损）"""
    rng = numpy.random.default_rng(1)
    img = rng.integers(0, 256, size=(8, 8, 3)).astype(numpy.uint8)
    out_path = tmp_path / "test.png"

    utils.imwrite(str(out_path), img)
    loaded = utils.imread(str(out_path))

    assert numpy.array_equal(loaded, img)
