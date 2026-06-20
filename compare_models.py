"""
Compare YOLO11m vs YOLOv8m on the microcrystal SEM detection dataset.

Trains YOLOv8m with the best hyperparameters found in the grid search
(batch=16, lr=0.001) and produces a side-by-side comparison table.

Usage:
    python compare_models.py \
        --no-aug-yaml path/to/non_augmented/data.yaml \
        --aug-yaml    path/to/augmented/data.yaml \
        --gpu         1

Outputs:
    model_comparison_results/comparison_results.csv
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

# Known YOLO11m reference results from grid search (batch=16, lr=0.001)
YOLO11M_NO_AUG  = {"model": "yolo11m", "dataset": "no_aug",    "precision": 0.7074, "recall": 0.6466, "mAP50": 0.6903}
YOLO11M_AUGMENT = {"model": "yolo11m", "dataset": "augmented", "precision": 0.76,   "recall": 0.70,   "mAP50": 0.72}


def train_and_eval(model_name, dataset_yaml, tag, results_dir, gpu):
    save_path = os.path.join(results_dir, tag)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n[GPU {gpu}] Training {tag} ...")
    start = time.time()

    model = YOLO(model_name)
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
    recall    = round(float(metrics.get("metrics/recall(B)", 0.0)), 4)
    map50     = round(float(metrics.get("metrics/mAP50(B)", 0.0)), 4)

    # Measure inference speed via val()
    best_weights = os.path.join(save_path, "run", "weights", "best.pt")
    val_model    = YOLO(best_weights)
    val_results  = val_model.val(data=dataset_yaml, device=gpu, verbose=False)
    inference_ms = round(val_results.speed.get("inference", 0.0), 2)

    torch.cuda.empty_cache()
    print(f"[GPU {gpu}] Done {tag} | P={precision} R={recall} mAP50={map50} | {inference_ms}ms/img | {duration}min")
    return {
        "model":        model_name.replace(".pt", ""),
        "dataset":      tag.split("_", 1)[1],
        "precision":    precision,
        "recall":       recall,
        "mAP50":        map50,
        "inference_ms": inference_ms,
        "duration_min": duration,
        "weights":      best_weights,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare YOLO11m vs YOLOv8m for SEM microcrystal detection"
    )
    parser.add_argument("--no-aug-yaml", required=True)
    parser.add_argument("--aug-yaml",    required=True)
    parser.add_argument("--results-dir", default="model_comparison_results")
    parser.add_argument("--model",       default="yolov8m.pt",
                        help="Path to YOLOv8m weights (downloaded automatically if bare filename)")
    parser.add_argument("--gpu",         type=int, default=1,
                        help="Starting GPU index; uses --gpu for no_aug and --gpu+1 for augmented")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    r_no_aug = train_and_eval(
        args.model, args.no_aug_yaml,
        "yolov8m_no_aug", args.results_dir, args.gpu
    )
    r_aug = train_and_eval(
        args.model, args.aug_yaml,
        "yolov8m_augmented", args.results_dir, args.gpu + 1
    )

    all_rows = [
        YOLO11M_NO_AUG | {"inference_ms": "see benchmark"},
        YOLO11M_AUGMENT | {"inference_ms": "see benchmark"},
        r_no_aug,
        r_aug,
    ]

    csv_path = os.path.join(args.results_dir, "comparison_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "dataset", "precision", "recall", "mAP50", "inference_ms"])
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n" + "=" * 65)
    print("MODEL COMPARISON SUMMARY (batch=16, lr=0.001, 100 epochs)")
    print("=" * 65)
    print(f"{'Model':<12} {'Dataset':<12} {'P':>7} {'R':>7} {'mAP50':>8} {'ms/img':>9}")
    print("-" * 65)
    for r in all_rows:
        print(f"{str(r['model']):<12} {str(r['dataset']):<12} "
              f"{str(r['precision']):>7} {str(r['recall']):>7} "
              f"{str(r['mAP50']):>8} {str(r.get('inference_ms', '-')):>9}")
    print(f"\nFull results saved to: {csv_path}")


if __name__ == "__main__":
    main()
