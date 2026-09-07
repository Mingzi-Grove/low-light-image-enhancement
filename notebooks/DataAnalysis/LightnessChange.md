# Analysis - LightnessChange

## 目的
**分析数据集中亮暗样本的 Lightness 差异，寻找亮度增益的规律**

## 分析工具
- [analysis.ipynb](../../data/lol/analysis.ipynb) - Jupyter 脚本，调试单张分析图
- [ana_light.py](../../src/lowlight_image_enhance/analysis/ana_light.py) - 绘制分析图的工具

## 收获

### 分析图
![分析图](img_lc_1.png)


### 增益算法设计思路
#### histogram stretch
观察到 low_l 与 high_l 的直方图形状有关联，将蓝色直方图向右拉伸似乎能得到与橙色直方图相近的形状
#### power
从 mapping curve 观察到 high_l 与 low_l 有类似幂函数的关系，尝试摸索一种改进的增益算法：[Kaga.md](../Algorithms/Kaga/Kaga.md)