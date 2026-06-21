"""
Crop the embedded Ultralytics title from PR curve PNGs and regenerate pr_comparison.png.
Ultralytics matplotlib figures include a top-level title inside the pixel data;
this script removes that strip before compositing the side-by-side comparison.
"""

import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_DIR   = r"D:\microcrystals\revised\overleafpaper\img"
PR11_SRC  = r"D:\microcrystals\notebooks881616-test\yolo_grid_search_results_old2\bs16_lr0.001_augmented\run\PR_curve.png"
PR26_SRC  = r"D:\microcrystals\results\yolo26_comparison\yolo26m_augmented\run\BoxPR_curve.png"

DPI      = 300
FS_TITLE = 13
CROP_TOP = 90   # pixels to remove from top of each Ultralytics figure (title area)


def load_and_crop(path, crop_top=CROP_TOP):
    img = np.array(Image.open(path).convert("RGB"))
    return img[crop_top:, :, :]


if __name__ == "__main__":
    img11 = load_and_crop(PR11_SRC)
    img26 = load_and_crop(PR26_SRC)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=DPI)
    for ax, img, subtitle in [
        (axes[0], img11, "(a)  YOLO11m  —  Precision-Recall Curves (augmented, mAP@0.5 = 0.720)"),
        (axes[1], img26, "(b)  YOLO26m  —  Precision-Recall Curves (augmented, mAP@0.5 = 0.744)"),
    ]:
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(subtitle, fontsize=FS_TITLE, fontweight="bold", pad=8)

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "pr_comparison.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pr_comparison.png  ({os.path.getsize(out)//1024} KB)")
