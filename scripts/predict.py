"""Detect accidents and their severity in images, folders, videos, streams or a webcam.

    python scripts/predict.py --weights best.pt --source path/to/image_or_folder_or_video
    python scripts/predict.py --weights best.pt --source 0            # webcam
    python scripts/predict.py --weights best.onnx --source rtsp://...  # ONNX works too

Annotated outputs are saved under runs/predict/; a per-frame summary is printed.
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--project", default="runs/predict")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.weights)
    totals: Counter = Counter()
    for i, result in enumerate(
        model.predict(
            source=source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=0 if torch.cuda.is_available() else "cpu",
            stream=True,
            save=True,
            project=args.project,
            verbose=False,
        )
    ):
        boxes = result.boxes
        labels = [model.names[int(c)] for c in boxes.cls.tolist()] if boxes is not None else []
        confs = [round(float(c), 3) for c in boxes.conf.tolist()] if boxes is not None else []
        totals.update(labels)
        summary = ", ".join(f"{lbl} ({c})" for lbl, c in zip(labels, confs, strict=True)) or "no accident detected"
        print(f"[{i}] {result.path}: {summary}")
    print(f"Detections by class: {dict(totals)}")


if __name__ == "__main__":
    main()
