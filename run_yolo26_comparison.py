"""
Train and compare YOLO11m, YOLOv8m, and YOLO26m on the microcrystal SEM dataset.

Extends the original compare_models.py to include YOLO26m (ultralytics >= 8.3.x).
Uses the best hyperparameters found in the grid search: batch=16, lr=0.001, 100 epochs.

Usage:
    python run_yolo26_comparison.py \
        --no-aug-yaml  D:/microcrystals/notebooks881616-test/Nanocrystals2-14/data.yaml \
        --aug-yaml     D:/microcrystals/notebooks881616-test/Nanocrystals2-12/data.yaml \
        --results-dir  D:/microcrystals/results/yolo26_comparison \
        --gpu          0

Outputs:
    <results-dir>/comparison_results.csv
    <results-dir>/yolo26m_no_aug/run/weights/best.pt
    <results-dir>/yolo26m_augmented/run/weights/best.pt
"""

import os
import csv
import time
import argparse
import torch
from ultralytics import YOLO

BEST_BATCH = 16
BEST_LR    = 0.001
EPOCHS     = 100
IMG_SIZE   = 640

# Known YOLO11m and YOLOv8m reference results (from earlier experiments)
KNOWN_RESULTS = [
    {"model": "YOLOv8m",  "dataset": "Non-aug.",  "params": "25.9M", "gflops": "79.1",
     "precision": 0.677,  "recall": 0.599, "mAP50": 0.652, "inference_ms": 10.1},
    {"model": "YOLO11m",  "dataset": "Non-aug.",  "params": "20.0M", "gflops": "67.7",
     "precision": 0.707,  "recall": 0.647, "mAP50": 0.690, "inference_ms": 10.9},
    {"model": "YOLOv8m",  "dataset": "Augmented", "params": "25.9M", "gflops": "79.1",
     "precision": 0.831,  "recall": 0.677, "mAP50": 0.773, "inference_ms":  9.6},
    {"model": "YOLO11m",  "dataset": "Augmented", "params": "20.0M", "gflops": "67.7",
     "precision": 0.760,  "recall": 0.702, "mAP50": 0.720, "inference_ms": 10.9},
]


def get_model_info(model):
    """Return (params_M, gflops) strings for the model."""
    try:
        info = model.info(verbose=False)
        # info returns (parameters, gflops)
        params = f"{info[0] / 1e6:.1f}M"
        gflops = f"{info[1]:.1f}"
        return params, gflops
    except Exception:
        return "--", "--"


def train_and_eval(model_name, dataset_yaml, tag, results_dir, gpu):
    save_path = os.path.join(results_dir, tag)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n[GPU {gpu}] Training {tag} ...")
    start = time.time()

    model = YOLO(model_name)
    params_str, gflops_str = get_model_info(model)

    train_results = model.train(
        data=dataset_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BEST_BATCH,
        lr0=BEST_LR,
        device=gpu,
        project=save_path,
        name="run",
        exist_ok=True,
        workers=2,
        amp=False,
    )

    duration = round((time.time() - start) / 60, 2)
    metrics   = train_results.results_dict
    precision = round(float(metrics.get("metrics/precision(B)", 0.0)), 4)
    recall    = round(float(metrics.get("metrics/recall(B)",    0.0)), 4)
    map50     = round(float(metrics.get("metrics/mAP50(B)",     0.0)), 4)

    best_weights = os.path.join(save_path, "run", "weights", "best.pt")
    val_model    = YOLO(best_weights)
    val_results  = val_model.val(data=dataset_yaml, device=gpu, verbose=False)
    inference_ms = round(val_results.speed.get("inference", 0.0), 2)

    torch.cuda.empty_cache()
    label = tag.split("_")[0].upper()  # e.g. "yolo26m" → "YOLO26M"
    dataset_label = "Augmented" if "aug" in tag.lower() and "no_aug" not in tag.lower() else "Non-aug."
    print(f"[GPU {gpu}] Done {tag} | P={precision} R={recall} mAP50={map50} | {inference_ms}ms/img | {duration}min")
    return {
        "model":        label,
        "dataset":      dataset_label,
        "params":       params_str,
        "gflops":       gflops_str,
        "precision":    precision,
        "recall":       recall,
        "mAP50":        map50,
        "inference_ms": inference_ms,
        "duration_min": duration,
        "weights":      best_weights,
    }


def print_table(rows):
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY  (batch=16, lr=0.001, 100 epochs, val set)")
    print("=" * 80)
    print(f"{'Model':<10} {'Dataset':<12} {'Params':>7} {'GFLOPs':>8} "
          f"{'P':>7} {'R':>7} {'mAP50':>8} {'ms/img':>9}")
    print("-" * 80)
    for r in rows:
        print(f"{str(r['model']):<10} {str(r['dataset']):<12} "
              f"{str(r.get('params', '--')):>7} {str(r.get('gflops', '--')):>8} "
              f"{str(r['precision']):>7} {str(r['recall']):>7} "
              f"{str(r['mAP50']):>8} {str(r['inference_ms']):>9}")


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO26m and compare with YOLO11m and YOLOv8m"
    )
    parser.add_argument("--no-aug-yaml", required=True,
                        help="data.yaml for the non-augmented dataset")
    parser.add_argument("--aug-yaml",    required=True,
                        help="data.yaml for the augmented dataset")
    parser.add_argument("--results-dir", default="D:/microcrystals/results/yolo26_comparison")
    parser.add_argument("--model",       default="yolo26m.pt",
                        help="YOLO26m weights (auto-downloaded if not present)")
    parser.add_argument("--gpu",         type=int, default=0,
                        help="GPU index for no_aug; aug uses --gpu+1")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    r_no_aug = train_and_eval(
        args.model, args.no_aug_yaml,
        "yolo26m_no_aug", args.results_dir, args.gpu
    )
    r_no_aug["model"] = "YOLO26m"
    r_no_aug["dataset"] = "Non-aug."

    r_aug = train_and_eval(
        args.model, args.aug_yaml,
        "yolo26m_augmented", args.results_dir, args.gpu + 1
    )
    r_aug["model"] = "YOLO26m"
    r_aug["dataset"] = "Augmented"

    all_rows = KNOWN_RESULTS + [r_no_aug, r_aug]

    csv_path = os.path.join(args.results_dir, "comparison_results_with_yolo26.csv")
    fieldnames = ["model", "dataset", "params", "gflops", "precision", "recall", "mAP50", "inference_ms"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print_table(all_rows)
    print(f"\nFull results saved to: {csv_path}")
    print("\n>>> Copy the YOLO26m rows into Table 2 (tab:model_comparison) in the LaTeX file.")
    print(">>> Also copy best.pt paths for generating updated figures if needed.")


if __name__ == "__main__":
    main()
