"""批量生成 LOL 数据集的亮度分析图，并汇总每张图的亮度/增益统计表。

对 low/ 与 high/ 目录中的每一对同名图像，生成 2x3 分析大图：
    row 0: low_l,  normal_l, histograms
    row 1: gain_l, Mapping Curve (density), Gain Curve (density)

- normal 指"正常光图"（物理目录仍为 high/，LOL 数据集原始命名）
- 低位截断：low_l 低于 LOW_CLIP 的像素视为无信息，替换为阈值，
  gain_l 上限统一为 1/LOW_CLIP，保证各样本可比
- 输出：保存到 ana-out/<日期>/ 目录，文件名与输入同名（.png）
- 统计：每张图的亮度/增益统计写入同目录 stats.csv
"""
import os
import csv
import datetime
import numpy
import cv2
import matplotlib
matplotlib.use('Agg')   # 无界面后端：批量保存时不弹窗口、不依赖 GUI
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from lowlight_image_enhance import utils, colorspace

LOW_DIR = os.path.join(os.path.dirname(__file__), "low")
NORMAL_DIR = os.path.join(os.path.dirname(__file__), "high")  # 物理目录仍为 high/
OUT_DIR = os.path.join(os.path.dirname(__file__), "ana-out")

# 参数（与 analysis.ipynb 保持一致）
LOW_CLIP = 0.01               # low_l 低位截断阈值：低于它的像素无信息，替换为阈值
GAIN_VMAX = 1.0 / LOW_CLIP    # gain_l 统一上限 = 100

# 统计表列名
STATS_COLUMNS = [
    "filename",
    "low_l_min", "low_l_max", "low_l_mean",
    "normal_l_min", "normal_l_max", "normal_l_mean",
    "gain_l_min", "gain_l_max", "gain_l_mean",
    "percent_clipped",
]


def analyze_pair(low_path: str, normal_path: str, out_path: str) -> dict:
    """读取一对 low/normal 图像，生成分析图保存为 PNG，返回该图的统计信息。"""
    # 1-2. read & normalize
    low_bgr_u8 = utils.imread(low_path)
    normal_bgr_u8 = utils.imread(normal_path)
    low_bgr_f32 = utils.normalize(low_bgr_u8)
    normal_bgr_f32 = utils.normalize(normal_bgr_u8)

    # 3-4. convert to LCH, get L channel + 统一低位截断
    low_lch = colorspace.BGR2LCH(low_bgr_f32)
    normal_lch = colorspace.BGR2LCH(normal_bgr_f32)
    low_l = low_lch[:, :, 0]
    normal_l = normal_lch[:, :, 0]
    low_l_clip = numpy.maximum(low_l, LOW_CLIP)  # 无信息像素替换为阈值
    gain_l = normal_l / low_l_clip               # 相对增益，上限统一 = 1/LOW_CLIP
    percent_clipped = 100.0 * (low_l < LOW_CLIP).sum() / low_l.size

    # 统计（写入 stats.csv）
    stats = {
        "filename": os.path.basename(low_path),
        "low_l_min": round(float(low_l.min()), 4),
        "low_l_max": round(float(low_l.max()), 4),
        "low_l_mean": round(float(low_l.mean()), 4),
        "normal_l_min": round(float(normal_l.min()), 4),
        "normal_l_max": round(float(normal_l.max()), 4),
        "normal_l_mean": round(float(normal_l.mean()), 4),
        "gain_l_min": round(float(gain_l.min()), 4),
        "gain_l_max": round(float(gain_l.max()), 4),
        "gain_l_mean": round(float(gain_l.mean()), 4),
        "percent_clipped": round(percent_clipped, 4),
    }

    # 5. plot 2x3 大图
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

    # row 1: gain_l, Mapping Curve (density), Gain Curve (density)
    im = axes[1, 0].imshow(gain_l, cmap='viridis', vmin=0, vmax=GAIN_VMAX)
    axes[1, 0].set_title("gain_l")
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], orientation='horizontal',
                 fraction=0.04, pad=0.12, aspect=30)  # 色标横放在增益图下方

    hb = axes[1, 1].hist2d(low_l.ravel(), normal_l.ravel(),
                           bins=200, range=[[0, 1], [0, 1]],
                           cmap='viridis_r', norm=mcolors.LogNorm())
    axes[1, 1].plot([0, 1], [0, 1], 'r--', lw=1)
    axes[1, 1].set_xlabel('low_l')
    axes[1, 1].set_ylabel('normal_l')
    axes[1, 1].set_title('Mapping Curve (density)')
    axes[1, 1].set_aspect('equal')
    plt.colorbar(hb[3], ax=axes[1, 1])

    # Gain Curve (density): x=截断后 low_l, y=增益(线性 0~100), 色标 LogNorm
    hb2 = axes[1, 2].hist2d(low_l_clip.ravel(), gain_l.ravel(),
                            bins=200, range=[[LOW_CLIP, 1.0], [0, GAIN_VMAX]],
                            cmap='viridis_r', norm=mcolors.LogNorm())
    axes[1, 2].axhline(1.0, color='r', linestyle='--', lw=1)  # gain=1 参考线（无变化）
    axes[1, 2].set_xticks([LOW_CLIP, 0.2, 0.4, 0.6, 0.8, 1.0])  # x 轴起点标出截断阈值
    axes[1, 2].set_xlabel('low_l (clipped)')
    axes[1, 2].set_ylabel('gain')
    axes[1, 2].set_title('Gain Curve (density)')
    plt.colorbar(hb2[3], ax=axes[1, 2])

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)  # 释放内存，防止 488 张图堆积

    return stats


def main() -> None:
    today = datetime.date.today()
    out_dir = os.path.join(OUT_DIR, f"{today.year}-{today.month}-{today.day}")
    os.makedirs(out_dir, exist_ok=True)
    stats_path = os.path.join(out_dir, "stats.csv")

    low_names = sorted(os.listdir(LOW_DIR))
    all_stats = []
    total = 0
    skipped = []
    for name in low_names:
        low_path = os.path.join(LOW_DIR, name)
        normal_path = os.path.join(NORMAL_DIR, name)
        if not os.path.isfile(low_path) or not os.path.isfile(normal_path):
            skipped.append(name)
            continue
        out_path = os.path.join(out_dir, name)
        all_stats.append(analyze_pair(low_path, normal_path, out_path))
        total += 1
        if total % 50 == 0:
            print(f"  processed {total} pairs...")

    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATS_COLUMNS)
        writer.writeheader()
        writer.writerows(all_stats)

    print(f"Done. {total} analysis images saved to: {out_dir}")
    print(f"Stats table saved to: {stats_path}")
    if skipped:
        print(f"Skipped (missing pair): {skipped}")


if __name__ == "__main__":
    main()
