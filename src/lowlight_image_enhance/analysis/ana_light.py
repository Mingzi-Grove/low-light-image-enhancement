"""亮度分析/对比图工具。

提供面向亮度（L 层）的分析与可视化函数，用于对比"参照亮度图"与"各算法增益亮度图"。
"""
import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

GAIN_VMAX = 100.0   # gain 上限（clip），恒定


def plot_compare(low_l, **sources):
    """绘制"亮度来源 vs 低光参照"的对比图。

    每个亮度来源（参照 / 各算法）一行，共 5 列：
        亮度图 | 直方图 | gain_l | mapping curve | gain curve

    Args:
        low_l: 低光照亮度图，float32 [0,1]，2D。
        **sources: 每个关键字是一个亮度来源（如 ref_l / kaga_l），
                   值为 float32 [0,1] 的 2D 灰度图；关键字名作行标题。

    Returns:
        matplotlib.figure.Figure: n 行 × 5 列的对比图。
    """
    n = len(sources)
    fig, axes = plt.subplots(n, 5, figsize=(32, 5 * n))
    axes = numpy.atleast_2d(axes)   # 只有一行时 subplots 返回 1D，统一转成 2D

    for i, (name, L) in enumerate(sources.items()):
        # ---- 第 0 列：该来源的亮度图 ----
        axes[i, 0].set_title(name)
        axes[i, 0].imshow(L, cmap='gray', vmin=0.0, vmax=1.0)
        axes[i, 0].axis('off')

        # ---- 第 1 列：直方图（该来源 L vs low_l）----
        axes[i, 1].hist(L.ravel(), bins=100, range=(0, 1), alpha=0.6, label=name)
        axes[i, 1].hist(low_l.ravel(), bins=100, range=(0, 1), alpha=0.6, label='low_l')
        axes[i, 1].set_xlabel('L (brightness)')
        axes[i, 1].set_ylabel('pixel count')
        axes[i, 1].legend()

        # ---- 第 2 列：gain_l（L / low_l，clip 上限 GAIN_VMAX）----
        gain = numpy.minimum(L / numpy.maximum(low_l, 1e-6), GAIN_VMAX)
        im = axes[i, 2].imshow(gain, cmap='viridis', vmin=0, vmax=GAIN_VMAX)
        axes[i, 2].set_title('gain_map')
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2], orientation='horizontal',
                     fraction=0.04, pad=0.12, aspect=30)

        # ---- 第 3 列：mapping curve（low_l vs L 密度图）----
        hb = axes[i, 3].hist2d(low_l.ravel(), L.ravel(),
                               bins=200, range=[[0, 1], [0, 1]],
                               cmap='viridis_r', norm=mcolors.LogNorm())
        axes[i, 3].plot([0, 1], [0, 1], 'r--', lw=1)  # y=x 参考线
        axes[i, 3].set_xlabel('low_l')
        axes[i, 3].set_ylabel(name)
        axes[i, 3].set_title('Mapping Curve (density)')
        axes[i, 3].set_aspect('equal')
        plt.colorbar(hb[3], ax=axes[i, 3])

        # ---- 第 4 列：gain curve（low_l vs gain 密度图）----
        hb2 = axes[i, 4].hist2d(low_l.ravel(), gain.ravel(),
                                bins=200, range=[[0, 1], [0, GAIN_VMAX]],
                                cmap='viridis_r', norm=mcolors.LogNorm())
        axes[i, 4].axhline(1.0, color='r', linestyle='--', lw=1)  # gain=1 参考线
        axes[i, 4].set_xlabel('low_l')
        axes[i, 4].set_ylabel('gain')
        axes[i, 4].set_title('Gain Curve (density)')
        plt.colorbar(hb2[3], ax=axes[i, 4])

    fig.tight_layout()
    return fig


def plot_pair(low_l, normal_l, gain_vmax=GAIN_VMAX, low_min=1e-6):
    """绘制 low/normal 一对图像的 2x3 详细分析图。

    不截断 low_l，保留过暗像素的真实增益（仅对 gain clip 上限）。

    row 0:  low_l | normal_l | histograms (low_l 与 normal_l)
    row 1:  gain_l | Mapping Curve (low_l vs normal_l) | Gain Curve (low_l vs gain)

    Args:
        low_l: 低光照亮度图，float32 [0,1]，2D。
        normal_l: 正常光亮度图，float32 [0,1]，2D。
        gain_vmax: gain 上限（clip），默认 GAIN_VMAX=100。
        low_min: 防除零的最小分母。

    Returns:
        matplotlib.figure.Figure: 2x3 分析图。
    """
    gain_raw = normal_l / numpy.maximum(low_l, low_min)  # 真实增益（可能很大）
    gain = numpy.minimum(gain_raw, gain_vmax)            # 仅截断增益上限

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # row 0: low_l, normal_l, histograms
    axes[0, 0].imshow(low_l, cmap='gray', vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("low_l")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(normal_l, cmap='gray', vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("normal_l")
    axes[0, 1].axis('off')

    axes[0, 2].hist(low_l.ravel(), bins=100, range=(0, 1), alpha=0.6, label='low_l')
    axes[0, 2].hist(normal_l.ravel(), bins=100, range=(0, 1), alpha=0.6, label='normal_l')
    axes[0, 2].set_xlabel('L (brightness)')
    axes[0, 2].set_ylabel('pixel count')
    axes[0, 2].set_title('histograms')
    axes[0, 2].legend()

    # row 1: gain_l, Mapping Curve, Gain Curve
    im = axes[1, 0].imshow(gain, cmap='viridis', vmin=0, vmax=gain_vmax)
    axes[1, 0].set_title("gain_l")
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], orientation='horizontal',
                 fraction=0.04, pad=0.12, aspect=30)

    hb = axes[1, 1].hist2d(low_l.ravel(), normal_l.ravel(),
                           bins=200, range=[[0, 1], [0, 1]],
                           cmap='viridis_r', norm=mcolors.LogNorm())
    axes[1, 1].plot([0, 1], [0, 1], 'r--', lw=1)  # y=x 参考线
    axes[1, 1].set_xlabel('low_l')
    axes[1, 1].set_ylabel('normal_l')
    axes[1, 1].set_title('Mapping Curve (density)')
    axes[1, 1].set_aspect('equal')
    plt.colorbar(hb[3], ax=axes[1, 1])

    hb2 = axes[1, 2].hist2d(low_l.ravel(), gain.ravel(),
                            bins=200, range=[[0, 1], [0, gain_vmax]],
                            cmap='viridis_r', norm=mcolors.LogNorm())
    axes[1, 2].axhline(1.0, color='r', linestyle='--', lw=1)  # gain=1 参考线
    axes[1, 2].set_xlabel('low_l')
    axes[1, 2].set_ylabel('gain')
    axes[1, 2].set_title('Gain Curve (density)')
    plt.colorbar(hb2[3], ax=axes[1, 2])

    fig.tight_layout()
    return fig
