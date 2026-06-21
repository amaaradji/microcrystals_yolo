"""
K-fold cross-validation for YOLO11m on the microcrystal SEM dataset.

Splits the augmented training set (253 images) into k stratified folds.
For each fold: trains YOLO11m from pretrained weights using the best
hyperparameters found in the grid search (batch=16, lr=0.001, 100 epochs),
then evaluates on the held-out fold.  Reports mean ± std across folds.

The fixed validation and test splits from Roboflow are NOT used during CV;
they remain as an independent held-out benchmark.

Usage:
    python cross_validate.py \
        --data-dir    D:/microcrystals/notebooks881616-test/Nanocrystals2-12 \
        --results-dir D:/microcrystals/results/cross_validation \
        --folds       5 \
        --epochs      100 \
        --gpu         0

Outputs:
    <results-dir>/
        fold_k/run/          -- YOLO training artefacts for fold k
        cv_results.csv       -- per-fold P / R / mAP@0.5 / mAP@0.5-0.95
        cv_summary.txt       -- mean ± std summary printed to stdout and saved
"""

import os
import csv
import yaml
import shutil
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO

BEST_BATCH = 16
BEST_LR    = 0.001
IMG_SIZE   = 640
MODEL_PT   = "yolo11m.pt"


def get_image_paths(images_dir: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return sorted(
        str(p) for p in Path(images_dir).iterdir()
        if p.suffix.lower() in exts
    )


def write_path_list(paths: list[str], out_file: str):
    with open(out_file, "w") as f:
        f.write("\n".join(paths))


def make_fold_yaml(train_txt: str, val_txt: str, nc: int,
                   names: list[str], out_yaml: str):
    data = {
        "train": train_txt.replace("\\", "/"),
        "val":   val_txt.replace("\\", "/"),
        "nc":    nc,
        "names": names,
    }
    with open(out_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def train_fold(fold_idx: int, train_paths: list[str], val_paths: list[str],
               nc: int, names: list[str], results_dir: str,
               epochs: int, gpu: int) -> dict:

    fold_dir = os.path.join(results_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    train_txt = os.path.join(fold_dir, "train.txt")
    val_txt   = os.path.join(fold_dir, "val.txt")
    fold_yaml = os.path.join(fold_dir, "data.yaml")

    write_path_list(train_paths, train_txt)
    write_path_list(val_paths,   val_txt)
    make_fold_yaml(train_txt, val_txt, nc, names, fold_yaml)

    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx}  |  train={len(train_paths)}  val={len(val_paths)}")
    print(f"{'='*60}")

    model = YOLO(MODEL_PT)
    model.train(
        data=fold_yaml,
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=BEST_BATCH,
        lr0=BEST_LR,
        device=gpu,
        project=fold_dir,
        name="run",
        exist_ok=True,
        workers=2,
        amp=False,
        verbose=False,
    )

    # Evaluate the best checkpoint on the fold's validation set
    best_pt = os.path.join(fold_dir, "run", "weights", "best.pt")
    val_model = YOLO(best_pt)
    val_res = val_model.val(
        data=fold_yaml,
        split="val",
        device=gpu,
        verbose=False,
    )

    m = val_res.results_dict
    result = {
        "fold":       fold_idx,
        "train_imgs": len(train_paths),
        "val_imgs":   len(val_paths),
        "precision":  round(float(m.get("metrics/precision(B)", 0)), 4),
        "recall":     round(float(m.get("metrics/recall(B)",    0)), 4),
        "mAP50":      round(float(m.get("metrics/mAP50(B)",     0)), 4),
        "mAP50_95":   round(float(m.get("metrics/mAP50-95(B)",  0)), 4),
    }
    print(f"  Fold {fold_idx} result: P={result['precision']}  "
          f"R={result['recall']}  mAP@0.5={result['mAP50']}  "
          f"mAP@0.5-0.95={result['mAP50_95']}")
    return result


def main():
    _D = r"D:\microcrystals"
    parser = argparse.ArgumentParser(
        description="K-fold cross-validation for YOLO11m microcrystal detection"
    )
    parser.add_argument("--data-dir",    default=rf"{_D}\notebooks881616-test\Nanocrystals2-12",
                        help="Root of Roboflow dataset (contains train/valid/test folders)")
    parser.add_argument("--results-dir", default=rf"{_D}\results\cross_validation",
                        help="Output directory for fold results and CSV summary")
    parser.add_argument("--folds",   type=int, default=5,  help="Number of CV folds")
    parser.add_argument("--epochs",  type=int, default=100, help="Training epochs per fold")
    parser.add_argument("--gpu",     type=int, default=0,   help="GPU index")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load dataset info ────────────────────────────────────────────────────
    yaml_path = os.path.join(args.data_dir, "data.yaml")
    with open(yaml_path) as f:
        ds = yaml.safe_load(f)
    nc    = int(ds["nc"])
    names = list(ds["names"])

    train_img_dir = os.path.join(args.data_dir, "train", "images")
    all_images = get_image_paths(train_img_dir)
    print(f"\nDataset: {len(all_images)} training images  |  {args.folds}-fold CV")
    print(f"Classes ({nc}): {names}")

    # ── K-fold split ─────────────────────────────────────────────────────────
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    all_images = np.array(all_images)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_images), start=1):
        train_paths = list(all_images[train_idx])
        val_paths   = list(all_images[val_idx])
        result = train_fold(
            fold_idx, train_paths, val_paths,
            nc, names, args.results_dir, args.epochs, args.gpu
        )
        fold_results.append(result)

    # ── Save per-fold CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(args.results_dir, "cv_results.csv")
    fieldnames = ["fold", "train_imgs", "val_imgs",
                  "precision", "recall", "mAP50", "mAP50_95"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fold_results)
    print(f"\nPer-fold results saved to: {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    metrics = ["precision", "recall", "mAP50", "mAP50_95"]
    lines = [
        "",
        "=" * 50,
        f"  {args.folds}-Fold Cross-Validation Summary",
        f"  Model: YOLO11m  |  batch={BEST_BATCH}  lr={BEST_LR}  epochs={args.epochs}",
        "=" * 50,
    ]
    for m in metrics:
        vals = [r[m] for r in fold_results]
        lines.append(f"  {m:<12}:  {np.mean(vals):.4f}  ±  {np.std(vals):.4f}"
                     f"  (min={min(vals):.4f}  max={max(vals):.4f})")
    lines += ["=" * 50, ""]

    summary = "\n".join(lines)
    print(summary)

    summary_path = os.path.join(args.results_dir, "cv_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
