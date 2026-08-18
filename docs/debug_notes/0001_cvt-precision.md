# colorspace.py 与 utils.py 的转换精度

## 已知问题
- `utils.normalize` 和 `utils.denormalize` 在转换 dtype 时用的 astype 截断方法而不是四舍五入，会引入稍大一点的误差
- `cv2.cvt` 做 BGR2LAB 转换时会引入较大的误差，是往返误差的主要来源
