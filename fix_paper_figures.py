"""
fix_paper_figures.py — Fix three paper figures:
1. metrics_yolo11m.png  — regenerate from first 100 epochs only
                          (current file shows 200 epochs, concatenated artifact)
2. cm_comparison.png    — crop embedded suptitle from top
3. pr_comparison.png    — remove overlapping panel titles, increase crop
"""

import os
import csv
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

_D  = r"D:\microcrystals"
IMG = r"D:\microcrystals\revised\overleafpaper\img"

CSV_PATH = (rf"{_D}\notebooks881616-test"
            r"\yolo_grid_search_results_old2\bs16_lr0.001_augmented\run\results.csv")
PR11_SRC = (rf"{_D}\notebooks881616-test"
            r"\yolo_grid_search_results_old2\bs16_lr0.001_augmented\run\PR_curve.png")
PR26_SRC = (rf"{_D}\results\yolo26_comparison\yolo26m_augmented\run\BoxPR_curve.png")


# ── 1. metrics_yolo11m.png (first 100 epochs only, EMA-smoothed) ─────────────
def ema(values, alpha=0.1):
    """Exponential moving average — good for loss curves (heavy smoothing)."""
    out = []
    v = values[0]
    for x in values:
        v = alpha * x + (1 - alpha) * v
        out.append(v)
    return out

def rolling_mean(values, window=10):
    """Rolling mean — recovers faster from transient spikes than EMA."""
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start:i + 1])))
    return out

def regenerate_metrics():
    epochs, tbox, tcls, vbox, vcls, prec, rec, map50 = [], [], [], [], [], [], [], []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 100:
                break
            epochs.append(int(row["epoch"]))
            tbox.append(float(row["train/box_loss"]))
            tcls.append(float(row["train/cls_loss"]))
            vbox.append(float(row["val/box_loss"]))
            vcls.append(float(row["val/cls_loss"]))
            prec.append(float(row["metrics/precision(B)"]))
            rec.append(float(row["metrics/recall(B)"]))
            map50.append(float(row["metrics/mAP50(B)"]))

    # Apply EMA smoothing (alpha=0.1, same as Ultralytics default for plots).
    # Val cls loss has an extreme spike at epoch 5 (>3000) from a transient
    # training instability; including it distorts the y-axis for 50+ epochs.
    # We show the three stable loss curves instead.
    # Panel (a): causal EMA for loss curves
    # Panel (b): non-causal Gaussian filter (sigma=3) — same approach as
    #            Ultralytics' own plot_results(); symmetric averaging means
    #            a transient crash at epoch 5 is balanced by recovery epochs
    #            on both sides, avoiding the long EMA tail.
    tbox_s  = ema(tbox,  0.1)
    tcls_s  = ema(tcls,  0.1)
    vbox_s  = ema(vbox,  0.1)
    prec_s  = gaussian_filter1d(np.array(prec,  dtype=float), sigma=3)
    rec_s   = gaussian_filter1d(np.array(rec,   dtype=float), sigma=3)
    map50_s = gaussian_filter1d(np.array(map50, dtype=float), sigma=3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    ax = axes[0]
    ax.plot(epochs, tbox_s, label="Train box loss")
    ax.plot(epochs, tcls_s, label="Train cls loss")
    ax.plot(epochs, vbox_s, label="Val box loss", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    peak = max(max(tbox_s), max(tcls_s), max(vbox_s))
    ax.set_ylim(bottom=0, top=min(3.0, peak * 1.15))
    ax.set_title("(a)  Training & Validation Loss", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, prec_s,  label="Precision")
    ax.plot(epochs, rec_s,   label="Recall")
    ax.plot(epochs, map50_s, label="mAP@0.5")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("(b)  Validation Metrics", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(IMG, "metrics_yolo11m.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[1] metrics_yolo11m.png saved — {len(epochs)} epochs, "
          f"{os.path.getsize(out)//1024} KB")


# ── 2. cm_comparison.png — crop suptitle ─────────────────────────────────────
def crop_cm():
    path = os.path.join(IMG, "cm_comparison.png")
    img  = Image.open(path)
    w, h = img.size
    # Crop top ~5 % of height (where suptitle lives); adjust if needed
    crop_px = int(h * 0.055)
    cropped = img.crop((0, crop_px, w, h))
    cropped.save(path)
    print(f"[2] cm_comparison.png cropped {crop_px}px from top  "
          f"(was {w}x{h}, now {w}x{h - crop_px})")


# ── 3. pr_comparison.png — larger crop, no overflowing titles ────────────────
def regenerate_pr():
    crop_top = 130  # was 90 — increased to clear Ultralytics title fully

    def load_crop(path):
        img = np.array(Image.open(path).convert("RGB"))
        return img[crop_top:, :, :]

    img11 = load_crop(PR11_SRC)
    img26 = load_crop(PR26_SRC)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
    for ax, img, label in [
        (axes[0], img11, "(a)  YOLO11m"),
        (axes[1], img26, "(b)  YOLO26m"),
    ]:
        ax.imshow(img)
        ax.axis("off")
        # Short panel label below each image — avoids title overflow
        ax.text(0.5, -0.03, label, transform=ax.transAxes,
                ha="center", va="top", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(IMG, "pr_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[3] pr_comparison.png saved — {os.path.getsize(out)//1024} KB")


if __name__ == "__main__":
    regenerate_metrics()
    crop_cm()
    regenerate_pr()
    print("All figures updated.")
