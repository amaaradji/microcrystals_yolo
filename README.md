# Microcrystal Detection in SEM Images — Code Repository

Code for the paper:

> **Learning-Based Detection of Microcrystals in Scanning Electron Microscopy Images**  
> A. Maaradji, A. Mehadjbia, A. Jaouadi  
> *Open Computer Science*, De Gruyter (2026)

---

## Overview

This repository contains all scripts to reproduce the experiments in the paper: hyperparameter search, model comparison (YOLO11m / YOLOv8m / YOLO26m), inference benchmarking, and publication-quality figure generation.

The method trains a YOLO-based object detector to simultaneously localize two classes in scanning electron microscopy (SEM) images:
- **Useful microcrystal** — the primary target for material quality assessment
- **Bacterial structure** — a co-occurring contaminant that must be distinguished

---

## Dataset

The annotated SEM dataset (120 images, 2,747 bounding-box instances in COCO format) is **available upon request** from the corresponding author:

**Abderrahmane Maaradji** — abderrahmane.maaradji@udst.edu.qa  
University of Doha for Science and Technology, Doha, Qatar

Both the non-augmented split (88 train / 16 val / 16 test) and the augmented split (253 train / 16 val / 16 test) are provided in COCO format alongside their `data.yaml` files.

---

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.10+
- ultralytics 8.4.72 (required for YOLO26m support)
- PyTorch 2.0+ with CUDA
- matplotlib, Pillow (for figure generation)

---

## Scripts

### 1. Hyperparameter grid search + augmented training
`yolo_grid_search_and_compare.py`

Runs a 4 × 4 grid (batch sizes × learning rates, 100 epochs each) on the non-augmented dataset, identifies the best combination by mAP@0.5, then retrains with that configuration on the augmented dataset.

```bash
python yolo_grid_search_and_compare.py \
    --no-aug-yaml path/to/non_augmented/data.yaml \
    --aug-yaml    path/to/augmented/data.yaml \
    --model       yolo11m.pt \
    --epochs      100 \
    --results-dir yolo_grid_search_results
```

Best configuration found: **batch=16, lr=0.001** → mAP@0.5 = 0.690 (non-aug), 0.720 (augmented).

---

### 2. Model comparison — YOLO11m vs YOLOv8m
`compare_models.py`

Trains YOLOv8m under identical conditions (batch=16, lr=0.001, 100 epochs) and outputs a side-by-side comparison CSV.

```bash
python compare_models.py \
    --no-aug-yaml path/to/non_augmented/data.yaml \
    --aug-yaml    path/to/augmented/data.yaml \
    --gpu         0
```

Output: `model_comparison_results/comparison_results.csv`

---

### 3. Three-model comparison — YOLO11m / YOLOv8m / YOLO26m
`run_yolo26_comparison.py`

Trains and evaluates all three YOLO generations under identical conditions and produces a combined CSV for Table 2 in the paper.

```bash
python run_yolo26_comparison.py \
    --no-aug-yaml path/to/non_augmented/data.yaml \
    --aug-yaml    path/to/augmented/data.yaml \
    --results-dir path/to/output \
    --model       yolo26m.pt \
    --gpu         0
```

Helper: `_run_yolo26.ps1` (PowerShell launcher — update the three path variables at the top).

---

### 4. Inference benchmark
`benchmark_inference.py`

Measures per-image latency (preprocess / GPU forward / postprocess) and saves annotated prediction images for qualitative inspection.

```bash
python benchmark_inference.py \
    --model  path/to/best.pt \
    --data   path/to/data.yaml \
    --gpu    0 \
    --output inference_benchmark_results
```

Paper result (NVIDIA L4 GPU): 0.2 ms preprocess + 10.9 ms inference + 8.2 ms postprocess = **19.3 ms/image (~52 FPS)**.

---

### 5. Figure generation

**Confusion matrices** (correct normalization with background/missed row):
```bash
python gen_cm_correct.py \
    --yolo11  path/to/yolo11m/best.pt \
    --yolo26  path/to/yolo26m/best.pt \
    --data    path/to/augmented/data.yaml \
    --img-dir path/to/output/img/ \
    --device11 0 --device26 1
```

Outputs: `cm_yolo11m.png`, `cm_yolo26m.png`, `cm_comparison.png`

**Training dynamics figure** (YOLO26m, 100 epochs):
```bash
python gen_yolo26_metrics.py \
    --csv     path/to/yolo26m_augmented/run/results.csv \
    --img-dir path/to/output/img/
```

Output: `metrics_yolo26m.png`

---

## Key Results

| Model    | Dataset   | Params | GFLOPs | P     | R     | mAP@0.5 | ms/img |
|----------|-----------|--------|--------|-------|-------|---------|--------|
| YOLOv8m  | Non-aug   | 25.9M  | 79.1   | 0.677 | 0.599 | 0.652   | 10.1   |
| YOLO11m  | Non-aug   | 20.0M  | 67.7   | 0.707 | 0.647 | 0.690   | 10.9   |
| YOLOv8m  | Augmented | 25.9M  | 79.1   | 0.831 | 0.677 | 0.773   | 9.6    |
| YOLO11m  | Augmented | 20.0M  | 67.7   | 0.760 | 0.702 | 0.720   | 10.9   |
| YOLO26m  | Non-aug   | 20.4M  | 67.9   | 0.788 | 0.594 | 0.705   | 10.9   |
| YOLO26m  | Augmented | 20.4M  | 67.9   | 0.768 | 0.667 | 0.744   | 10.9   |

All metrics on the held-out validation set (16 images, 96 annotated instances).  
Hardware: NVIDIA L4 GPU, CUDA 12.8, Python 3.11.3, PyTorch 2.5.1.
