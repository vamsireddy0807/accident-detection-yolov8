"""Evaluate trained weights and write overall and per-class metrics to JSON.

    python scripts/evaluate.py --weights runs/detect/accident_detection/weights/best.pt \
        --data data/Accident-Detection-1/data.yaml --split val --out metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--out", type=Path, default=Path("metrics.json"))
    args = ap.parse_args()

    model = YOLO(args.weights)
    res = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=0 if torch.cuda.is_available() else "cpu",
        plots=True,
    )
    names = model.names
    per_class = {}
    for i, cls_idx in enumerate(res.box.ap_class_index):
        p, r, ap50, ap = res.box.class_result(i)
        per_class[names[int(cls_idx)]] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "mAP50": round(float(ap50), 4),
            "mAP50_95": round(float(ap), 4),
        }
    metrics = {
        "split": args.split,
        "mAP50": round(float(res.box.map50), 4),
        "mAP50_95": round(float(res.box.map), 4),
        "precision": round(float(res.box.mp), 4),
        "recall": round(float(res.box.mr), 4),
        "per_class": per_class,
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in res.speed.items()},
    }
    args.out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
