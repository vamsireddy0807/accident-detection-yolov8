# Accident Detection and Severity Grading with YOLOv8

A YOLOv8-small detector that finds road accidents in images and video and grades each one as **moderate** or **severe**. It reaches **98.6% mAP@50 and 91.6% mAP@50-95** on 1,347 validation images, at **8.7 ms of inference per image on a Tesla T4**, and exports to ONNX for deployment.

## Results

Validation split: 1,347 images, 1,406 labelled accidents. Thresholds are `conf=0.25` and `iou=0.45`. All numbers come from the training run saved in [`notebooks/accident_detection.ipynb`](notebooks/accident_detection.ipynb).

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| **all** | 1,347 | 1,406 | **0.984** | **0.973** | **0.986** | **0.916** |
| moderate | 327 | 329 | 0.969 | 0.957 | 0.977 | 0.900 |
| severe | 1,024 | 1,077 | 0.998 | 0.989 | 0.994 | 0.932 |

| Deployment detail | Value |
|---|---|
| Model | YOLOv8s, 11.1M parameters, 28.4 GFLOPs (fused) |
| Speed on Tesla T4 | 1.2 ms pre-process + 8.7 ms inference + 0.8 ms post-process per image |
| Weights | `best.pt` 21.5 MB; `best.onnx` 42.7 MB (opset 20, simplified) |

> The model was selected and evaluated on the same validation split, so treat these as optimistic estimates. Run `scripts/evaluate.py --split test` for an untouched estimate.

## How it works

1. **Data.** A Roboflow accident dataset in YOLO format with two classes, `moderate` and `severe`. Training runs 610 batches of 16 per epoch.
2. **Model.** COCO-pretrained YOLOv8s with its 80-class head replaced by a 2-class head (349 of 355 weight tensors transferred).
3. **Training.** 50 epochs at 640 px, batch 16, AdamW (`lr0=1e-3`, `lrf=0.01`, weight decay 5e-4), 3 warm-up epochs, early stopping with patience 15, and mixed precision.
4. **Augmentation.** Mosaic, mixup 0.1, copy-paste 0.1, HSV jitter, horizontal flip, ±5° rotation, translate, scale, shear and perspective, so the detector is robust to camera angle and lighting.
5. **Export.** ONNX for ONNX Runtime, TensorRT or OpenVINO.

## Quick start

```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY=...                     # your own key; never commit it
python scripts/download_dataset.py --out data/
python scripts/train.py --data data/Accident-Detection-1/data.yaml
python scripts/evaluate.py --weights runs/detect/accident_detection/weights/best.pt \
    --data data/Accident-Detection-1/data.yaml --out metrics.json
python scripts/predict.py --weights runs/detect/accident_detection/weights/best.pt --source path/to/video.mp4
python scripts/export_onnx.py --weights runs/detect/accident_detection/weights/best.pt
```

`predict.py` accepts an image, a folder, a video file, an RTSP or HTTP stream, or a webcam index (`--source 0`). It saves annotated frames and prints the detected classes and confidences for each frame.

Prefer Colab? Open the notebook, set the runtime to GPU, add `ROBOFLOW_API_KEY` to Colab secrets, and run all cells. The notebook also plots class balance, training curves, the PR curve, the confusion matrix, confidence histograms and box-size distributions.

## Repository layout

```
notebooks/accident_detection.ipynb   end-to-end run with saved outputs (the run of record)
scripts/download_dataset.py          Roboflow download (key from the environment)
scripts/train.py                     training with the notebook's hyperparameters
scripts/evaluate.py                  overall and per-class metrics -> JSON
scripts/predict.py                   image, folder, video, stream and webcam inference
scripts/export_onnx.py               ONNX export
```

## Limitations

* There are two severity levels only, and "no accident" is handled as "no detection". Add negative-heavy footage before using this for alerting.
* The dataset comes from a single public Roboflow source; performance on other cameras, weather conditions and night footage has not been measured.
* The reported metrics come from the validation split that was also used for checkpoint selection.

## License

MIT for this code. The dataset and the Ultralytics YOLOv8 weights keep their own licenses (Ultralytics is AGPL-3.0); check them before commercial use.
