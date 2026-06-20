"""
YOLO11 Hyperparameter Grid Search and Augmented Training for Microcrystal SEM Detection.

Paper: "Learning-Based Detection of Microcrystals in Scanning Electron Microscopy Images"
Dataset: https://universe.roboflow.com/team-ifitikhar/nanocrystals2/dataset/14
Code repository: https://github.com/amaaradji/microcrystals_yolo

Usage:
    python yolo_grid_search_and_compare.py \
        --no-aug-yaml path/to/non_augmented/data.yaml \
        --aug-yaml    path/to/augmented/data.yaml \
        --model       yolo11m.pt \
        --epochs      100 \
        --results-dir yolo_grid_search_results

The script:
  1. Runs a 4x4 grid search (batch sizes x learning rates) on the non-augmented dataset.
  2. Identifies the best hyperparameter combination by mAP@0.5.
  3. Re-trains with the best combination on the augmented dataset.
"""

import os
import time
import csv
import argparse
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO

LEARNING_RATES = [0.01, 0.001, 0.0001, 0.00001]
BATCH_SIZES = [2, 4, 8, 16]


def train_model(batch, lr, gpu, dataset_yaml, results_dir, epochs, img_size, model_path, tag_suffix="no_aug"):
    tag = f"bs{batch}_lr{lr}_{tag_suffix}"
    save_path = os.path.join(results_dir, tag)
    os.makedirs(save_path, exist_ok=True)

    print(f"[GPU {gpu}] Starting {tag}")
    start = time.time()

    model = YOLO(model_path)
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        lr0=lr,
        device=gpu,
        project=save_path,
        name="run",
        exist_ok=True,
        workers=2,
        amp=False,
    )

    duration = round((time.time() - start) / 60, 2)
    try:
        metrics = results.results_dict
        precision = round(float(metrics.get("metrics/precision(B)", 0.0)), 4)
        recall    = round(float(metrics.get("metrics/recall(B)", 0.0)), 4)
        map50     = round(float(metrics.get("metrics/mAP50(B)", 0.0)), 4)
    except Exception as e:
        print(f"Warning: could not extract metrics for {tag}: {e}")
        precision = recall = map50 = "N/A"

    torch.cuda.empty_cache()
    del model

    print(f"[GPU {gpu}] Done {tag} | mAP50={map50}  {duration} min")
    return [batch, lr, precision, recall, map50, duration, save_path, tag_suffix]


def run_grid_search(no_aug_yaml, results_dir, epochs, img_size, model_path):
    csv_path = os.path.join(results_dir, "grid_search_results.csv")
    combos = [(bs, lr) for bs in BATCH_SIZES for lr in LEARNING_RATES]
    results = []

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {}
        for i, (bs, lr) in enumerate(combos):
            f = executor.submit(
                train_model, bs, lr, i % 1,
                no_aug_yaml, results_dir, epochs, img_size, model_path
            )
            futures[f] = (bs, lr)
            time.sleep(1)

        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Training failed: {e}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "learning_rate", "precision", "recall",
                         "mAP50", "duration_min", "results_dir", "dataset"])
        writer.writerows(results)

    print(f"\nGrid search results saved to: {csv_path}")
    return csv_path


def find_best_combo(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                float(row["mAP50"]); float(row["duration_min"])
                rows.append(row)
            except ValueError:
                continue
    if not rows:
        return None
    return sorted(rows, key=lambda r: (-float(r["mAP50"]), float(r["duration_min"])))[0]


def run_best_on_augmented(csv_path, aug_yaml, results_dir, epochs, img_size, model_path):
    best = find_best_combo(csv_path)
    if not best:
        print("No valid rows in CSV.")
        return
    bs  = int(best["batch_size"])
    lr  = float(best["learning_rate"])
    print(f"\nBest combo: batch={bs}, lr={lr}, mAP50={best['mAP50']}")
    result = train_model(bs, lr, 0, aug_yaml, results_dir, epochs, img_size, model_path,
                         tag_suffix="augmented")
    print(f"Augmented training done | mAP50={result[4]}, time={result[5]} min")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO11 grid search + augmented training for SEM microcrystal detection"
    )
    parser.add_argument("--no-aug-yaml", required=True,
                        help="Path to non-augmented dataset data.yaml")
    parser.add_argument("--aug-yaml", required=True,
                        help="Path to augmented dataset data.yaml")
    parser.add_argument("--model", default="yolo11m.pt",
                        help="YOLO model weights (downloaded automatically if not found)")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--img-size",   type=int, default=640)
    parser.add_argument("--results-dir", default="yolo_grid_search_results")
    parser.add_argument("--skip-grid",  action="store_true",
                        help="Skip grid search and use existing CSV to find best combo")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "grid_search_results.csv")

    if not args.skip_grid:
        csv_path = run_grid_search(
            args.no_aug_yaml, args.results_dir,
            args.epochs, args.img_size, args.model
        )

    run_best_on_augmented(
        csv_path, args.aug_yaml, args.results_dir,
        args.epochs, args.img_size, args.model
    )


if __name__ == "__main__":
    main()
