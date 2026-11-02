"""Fine-tune YOLOv8 for accident detection with severity classes (moderate, severe).

Hyperparameters match the notebook run reported in the README.

    python scripts/train.py --data data/Accident-Detection-1/data.yaml
"""

from __future__ import annotations

import argparse

import torch
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="path to the dataset's data.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="n | s | m | l checkpoint")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="reduce to 8 on out-of-memory")
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="accident_detection")
    args = ap.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=15,  # early stopping
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3,
        # augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        copy_paste=0.1,
        device=0 if torch.cuda.is_available() else "cpu",
        plots=True,
        save=True,
    )
    print(f"Training complete. Best weights: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
