"""Export trained weights to ONNX for deployment (ONNX Runtime, TensorRT, OpenVINO).

    python scripts/export_onnx.py --weights runs/detect/accident_detection/weights/best.pt
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    path = YOLO(args.weights).export(format="onnx", imgsz=args.imgsz, simplify=True)
    print(f"ONNX model written to {path}")


if __name__ == "__main__":
    main()
