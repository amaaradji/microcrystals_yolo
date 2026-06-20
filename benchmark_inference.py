"""
Benchmark YOLO11m inference speed on the SEM microcrystal test set.

Measures per-image latency (preprocess / inference / postprocess) and
runs validation to report per-class metrics. Prediction images are saved
for qualitative failure-case inspection.

Usage:
    python benchmark_inference.py \
        --model  path/to/best.pt \
        --data   path/to/data.yaml \
        --gpu    0 \
        --output inference_benchmark_results

Outputs:
    <output>/val_run/   -- annotated prediction images
    <output>/timing_stats.json
"""

import os
import json
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO11m inference and generate prediction images"
    )
    parser.add_argument("--model",  required=True, help="Path to best.pt weights")
    parser.add_argument("--data",   required=True, help="Path to dataset data.yaml")
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--output", default="inference_benchmark_results")
    parser.add_argument("--conf",   type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--split",  default="test",
                        help="Dataset split to evaluate: train / val / test")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model = YOLO(args.model)

    print(f"Running inference on '{args.split}' split  (GPU {args.gpu}, conf>={args.conf}) ...")
    val_results = model.val(
        data=args.data,
        device=args.gpu,
        split=args.split,
        conf=args.conf,
        save=True,
        save_json=True,
        project=args.output,
        name="val_run",
        exist_ok=True,
        verbose=True,
    )

    speed = val_results.speed
    total_ms = sum(speed.values())

    print("\n" + "=" * 50)
    print("INFERENCE TIMING  (ms per image)")
    print("=" * 50)
    print(f"  Preprocessing:  {speed.get('preprocess',  0):.2f} ms")
    print(f"  Inference:      {speed.get('inference',   0):.2f} ms  <-- GPU forward pass")
    print(f"  Postprocessing: {speed.get('postprocess', 0):.2f} ms")
    print(f"  TOTAL:          {total_ms:.2f} ms   ({1000/total_ms:.1f} FPS)")

    print("\nDETECTION METRICS (on test set):")
    print(f"  mAP@0.5:   {val_results.box.map50:.4f}")
    print(f"  Precision: {val_results.box.mp:.4f}")
    print(f"  Recall:    {val_results.box.mr:.4f}")

    timing = {
        "preprocess_ms":  round(speed.get("preprocess",  0), 2),
        "inference_ms":   round(speed.get("inference",   0), 2),
        "postprocess_ms": round(speed.get("postprocess", 0), 2),
        "total_ms":       round(total_ms, 2),
        "fps":            round(1000 / total_ms, 1),
        "map50":          round(val_results.box.map50, 4),
        "precision":      round(val_results.box.mp,   4),
        "recall":         round(val_results.box.mr,   4),
    }

    out_json = os.path.join(args.output, "timing_stats.json")
    with open(out_json, "w") as f:
        json.dump(timing, f, indent=2)

    print(f"\nTiming stats saved to : {out_json}")
    print(f"Prediction images in  : {args.output}/val_run/")
    print("\n>>> For the paper: use 'inference_ms' value from timing_stats.json")


if __name__ == "__main__":
    main()
