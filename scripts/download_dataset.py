"""Download the accident-severity dataset from Roboflow in YOLOv8 format.

    export ROBOFLOW_API_KEY=...        # never commit this key
    python scripts/download_dataset.py --out data/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from roboflow import Roboflow

WORKSPACE = "accident-detection-ffdrf"
PROJECT = "accident-detection-8dvh5"
VERSION = 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workspace", default=WORKSPACE)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--version", type=int, default=VERSION)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set the ROBOFLOW_API_KEY environment variable first.")

    args.out.mkdir(parents=True, exist_ok=True)
    os.chdir(args.out)
    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace(args.workspace).project(args.project).version(args.version).download("yolov8")

    data_yaml = Path(dataset.location) / "data.yaml"
    cfg = yaml.safe_load(data_yaml.read_text())
    print(f"Dataset   : {dataset.location}")
    print(f"data.yaml : {data_yaml}")
    print(f"Classes   : {cfg['names']}")


if __name__ == "__main__":
    main()
