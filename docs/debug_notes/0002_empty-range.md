# colorspace.py 的输出有空值域

## 已知问题
- `colorspace.py` 中 LAB 与 LCH 空间的部分通道有“空值域”(empty range) 问题，因为 sRGB 的色彩空间经转换后无法达到所设定的值域
