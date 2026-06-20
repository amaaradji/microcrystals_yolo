"""
Generate the YOLO26m training dynamics figure (metrics_yolo26m.png) for the paper.

Usage:
    python gen_yolo26_metrics.py
    python gen_yolo26_metrics.py --csv path/to/results.csv --img-dir path/to/img/
"""

import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI, FS_TITLE, FS_TICK = 300, 13, 11


def read_csv(path):
    with open(path) as f:
        lines = f.read().strip().split("\n")
    header = [h.strip() for h in lines[0].split(",")]
    data = {k: [] for k in header}
    for line in lines[1:]:
        for k, v in zip(header, [x.strip() for x in line.split(",")]):
            try:    data[k].append(float(v))
            except: data[k].append(0.0)
    return data


if __name__ == "__main__":
    _D = r"D:\microcrystals"  # root used in the paper — override with --csv / --img-dir
    p = argparse.ArgumentParser(description="Generate YOLO26m training metrics figure.")
    p.add_argument("--csv", default=rf"{_D}\results\yolo26_comparison\yolo26m_augmented\run\results.csv",
                   help="Path to YOLO26m results.csv")
    p.add_argument("--img-dir", default=rf"{_D}\revised\overleafpaper\img",
                   help="Output directory for figures")
    args = p.parse_args()

    IMG_DIR    = args.img_dir
    YOLO26_CSV = args.csv

    os.makedirs(IMG_DIR, exist_ok=True)
    d = read_csv(YOLO26_CSV)
    epochs = list(range(1, len(d["epoch"]) + 1))
    mkey = "metrics/mAP50(B)"
    print(f"Epochs: {len(epochs)}, final mAP50: {d[mkey][-1]:.4f}, max mAP50: {max(d[mkey]):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=DPI)

    ax = axes[0]
    for col, label, color in [
        ("train/box_loss", "Train box loss", "#1A4E8A"),
        ("train/cls_loss", "Train cls loss", "#4C90CA"),
        ("val/box_loss",   "Val box loss",   "#C04020"),
        ("val/cls_loss",   "Val cls loss",   "#E07B39"),
    ]:
        if col in d:
            ax.plot(epochs, d[col], label=label, color=color, lw=1.6)
    ax.set_xlabel("Epoch", fontsize=FS_TICK + 1); ax.set_ylabel("Loss", fontsize=FS_TICK + 1)
    ax.set_title("(a)  Training & Validation Loss", fontsize=FS_TITLE, fontweight="bold")
    ax.legend(fontsize=FS_TICK - 1); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for col, label, color in [
        ("metrics/precision(B)", "Precision", "#1A4E8A"),
        ("metrics/recall(B)",    "Recall",    "#E07B39"),
        ("metrics/mAP50(B)",     "mAP@0.5",  "#2A9E4F"),
    ]:
        if col in d:
            ax.plot(epochs, d[col], label=label, color=color, lw=1.6)
    ax.set_xlabel("Epoch", fontsize=FS_TICK + 1); ax.set_ylabel("Score", fontsize=FS_TICK + 1)
    ax.set_title("(b)  Validation Metrics", fontsize=FS_TITLE, fontweight="bold")
    ax.legend(fontsize=FS_TICK - 1); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "metrics_yolo26m.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}  ({os.path.getsize(out)//1024} KB)")
