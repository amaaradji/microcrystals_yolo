"""
Regenerate confusion matrix figures with correct normalization.

YOLO's confusion matrix is (nc+1 x nc+1):
  rows = predicted class  (0..nc-1 = classes, nc = background/missed)
  cols = true class       (0..nc-1 = classes, nc = background/FP)

Correct normalization: divide each column by its sum BEFORE dropping the
background column, so each true-class column sums to 1.0 and the bottom
"Background (Missed)" row shows the false-negative rate.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Paths are set via argparse in __main__; see defaults for the reference values used in the paper.

DPI      = 300
FS_TITLE = 12
FS_TICK  = 11
FS_CELL  = 22   # large readable numbers inside cells


def run_val(pt_path, device=0):
    model = YOLO(pt_path)
    res   = model.val(data=DATA_YAML, device=device, verbose=False, split="val")
    mat   = res.confusion_matrix.matrix           # (nc+1, nc+1)
    names = [res.names[k] for k in sorted(res.names)]
    return mat, names


def draw_cm(ax, mat, class_names, title):
    """
    Correctly normalized confusion matrix including the background/missed row.
    mat shape: (nc+1, nc+1)
    Displayed: (nc+1) rows x nc cols  (background col dropped).
    """
    nc = len(class_names)
    # Normalize full matrix by column sum first (so each col sums to 1)
    col_sums = mat.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    norm_full = mat / col_sums            # (nc+1, nc+1)

    # Drop background column (col nc) — it represents unmatched predictions,
    # less informative for the per-true-class view
    disp = norm_full[:nc+1, :nc]          # shape (nc+1, nc)

    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=1, interpolation="nearest",
                   aspect="auto")
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=10)

    ax.set_xticks(range(nc))
    ax.set_xticklabels(class_names, fontsize=FS_TICK, fontweight="bold")
    ax.set_yticks(range(nc + 1))
    y_labels = class_names + ["Background\n(Missed)"]
    ax.set_yticklabels(y_labels, fontsize=FS_TICK, fontweight="bold")
    ax.set_xlabel("True Class",      fontsize=FS_TICK + 1, fontweight="bold")
    ax.set_ylabel("Predicted Class", fontsize=FS_TICK + 1, fontweight="bold")

    for i in range(nc + 1):
        for j in range(nc):
            v = disp[i, j]
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=FS_CELL, fontweight="bold", color=color)
    return im


if __name__ == "__main__":
    import argparse
    _D = r"D:\microcrystals"  # root used in the paper — override all paths below as needed
    p = argparse.ArgumentParser(description="Generate confusion matrix figures for the microcrystal paper.")
    p.add_argument("--yolo11", default=rf"{_D}\notebooks881616-test\yolo_grid_search_results_old2\bs16_lr0.001_augmented\run\weights\best.pt",
                   help="YOLO11m best.pt weights")
    p.add_argument("--yolo26", default=rf"{_D}\results\yolo26_comparison\yolo26m_augmented\run\weights\best.pt",
                   help="YOLO26m best.pt weights")
    p.add_argument("--data",   default=rf"{_D}\notebooks881616-test\Nanocrystals2-12\data.yaml",
                   help="data.yaml for the augmented dataset")
    p.add_argument("--img-dir", default=rf"{_D}\revised\overleafpaper\img",
                   help="Output directory for figures")
    p.add_argument("--device11", default=1, type=int, help="GPU index for YOLO11m val()")
    p.add_argument("--device26", default=2, type=int, help="GPU index for YOLO26m val()")
    args = p.parse_args()

    IMG_DIR   = args.img_dir
    YOLO11_PT = args.yolo11
    YOLO26_PT = args.yolo26
    DATA_YAML = args.data

    os.makedirs(IMG_DIR, exist_ok=True)

    print("Running YOLO11m val() on GPU 1 ...")
    mat11, names = run_val(YOLO11_PT, device=args.device11)
    print("Running YOLO26m val() on GPU 2 ...")
    mat26, _     = run_val(YOLO26_PT, device=args.device26)

    # Print diagonal values for verification
    nc = len(names)
    col_sums11 = mat11.sum(axis=0)
    col_sums26 = mat26.sum(axis=0)
    print("\nYOLO11m per-class recall (confusion matrix diagonal):")
    for j, name in enumerate(names):
        val = mat11[j, j] / col_sums11[j] if col_sums11[j] > 0 else 0
        print(f"  {name}: {val:.3f}")
    print("YOLO26m per-class recall (confusion matrix diagonal):")
    for j, name in enumerate(names):
        val = mat26[j, j] / col_sums26[j] if col_sums26[j] > 0 else 0
        print(f"  {name}: {val:.3f}")

    # ── 1. Individual YOLO11m figure (single column = smaller) ───────────────
    fig, ax = plt.subplots(figsize=(4.5, 4.2), dpi=DPI)
    im = draw_cm(ax, mat11, names,
                 "YOLO11m Normalized Confusion Matrix\n(augmented dataset, mAP@0.5 = 0.720)")
    plt.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    plt.tight_layout()
    out = os.path.join(IMG_DIR, "cm_yolo11m.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved cm_yolo11m.png  ({os.path.getsize(out)//1024} KB)")

    # ── 2. Individual YOLO26m figure ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 4.2), dpi=DPI)
    im = draw_cm(ax, mat26, names,
                 "YOLO26m Normalized Confusion Matrix\n(augmented dataset, mAP@0.5 = 0.744)")
    plt.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    plt.tight_layout()
    out = os.path.join(IMG_DIR, "cm_yolo26m.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cm_yolo26m.png  ({os.path.getsize(out)//1024} KB)")

    # ── 3. Side-by-side comparison ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=DPI)
    im0 = draw_cm(axes[0], mat11, names,
                  "(a) YOLO11m  (mAP@0.5 = 0.720)")
    im1 = draw_cm(axes[1], mat26, names,
                  "(b) YOLO26m  (mAP@0.5 = 0.744)")
    for ax, im in zip(axes, [im0, im1]):
        plt.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    plt.suptitle(
        "Normalized Confusion Matrices: YOLO11m vs. YOLO26m (augmented dataset)",
        fontsize=FS_TITLE + 1, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(IMG_DIR, "cm_comparison.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cm_comparison.png  ({os.path.getsize(out)//1024} KB)")

    print("\nDone.")
