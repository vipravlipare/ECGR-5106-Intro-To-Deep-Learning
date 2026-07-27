# Real-Time Road Object Detection Using YOLO and Faster R-CNN

Final project for ECGR 5106 Intro to Deep Learning, expanded from the earlier
`Assignment_2_Implement_Detection_Networks_RCNN_and_YOLO` traffic-light detector.

This version follows the new proposal: train and evaluate road-scene object detectors on
BDD100K for cars, buses, trucks, pedestrians, traffic lights, and traffic signs. YOLO is the
primary real-time detector, while Faster R-CNN is included as the stronger two-stage
comparison model, matching the structure of the previous project.

## Project Goals

- Convert BDD100K detection annotations into YOLO format.
- Fine-tune a pretrained YOLO detector with transfer learning, heavy but realistic
  augmentation, cosine learning-rate scheduling, AdamW, AMP, and multi-scale training.
- Train CPU-friendly MobileNet-FPN and final ResNet50-FPN Faster R-CNN comparisons.
- Evaluate precision, recall, F1, mAP50, and mAP50:95 for both detectors.
- Run realtime webcam/video detection and benchmark inference FPS.
- Export YOLO weights to deployment formats such as ONNX, TensorRT, OpenVINO, or TFLite.

## Why This Is Bigger Than The Previous Project

The earlier assignment used YOLO and Faster R-CNN for traffic-light detection. This project
keeps that detection-network core but expands it to a multi-class driving-scene detector:

- Dataset: LISA traffic-light data becomes BDD100K road-scene data.
- Classes: one class becomes six proposal-aligned classes.
- Training: notebook-only experiments become reusable scripts that can run locally,
  on Colab, or on a GPU workstation.
- Evaluation: qualitative detection becomes metrics, plots, validation reports, and
  realtime FPS benchmarking.
- Deployment: live camera detection is paired with export/benchmark tools inspired by
  the ML-for-IoT project.

## Setup

```powershell
cd C:\Users\vipra\OneDrive\Documents\GitHub\ECGR-5106-Intro-To-Deep-Learning\Final_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
jupyter lab
```

Open `notebooks/01_YOLO_BDD100K_Training.ipynb` or
`notebooks/02_Faster_RCNN_BDD100K_Training.ipynb`. Both notebooks default to a
CPU-safe profile and show data checks, training curves, validation/test metrics,
per-class metrics, prediction examples, low-light failures, inference FPS, and the
saved best checkpoint.

## Expected BDD100K Layout

Download BDD100K images and detection labels, then place or unzip them like this:

```text
bdd100k/
  images/
    100k/
      train/
      val/
      test/
  labels/
    det_20/
      det_train.json
      det_val.json
```

The converter also tries older label locations such as
`labels/bdd100k_labels_images_train.json`. Public BDD100K test labels are normally
withheld, so the converter reproducibly reserves 20% of labeled validation data as a
local test split.

## Convert BDD100K To YOLO Format

Quick smoke-test subset:

```powershell
.\scripts\prepare_bdd100k.ps1 -BddRoot C:\path\to\bdd100k -MaxImagesPerSplit 500
```

Full conversion:

```powershell
python -m road_detection.bdd100k_to_yolo `
  --bdd-root C:\path\to\bdd100k `
  --output data\bdd100k_yolo `
  --splits train val `
  --copy-mode hardlink `
  --val-test-fraction 0.20 `
  --seed 42
```

This creates:

```text
data/bdd100k_yolo/
  data.yaml
  conversion_summary.json
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

## Notebook Training Profiles

- `cpu_quick`: small end-to-end run for the current CPU.
- `cpu_practical`: larger CPU experiment using more data.
- `accuracy`: full-data, high-resolution final run; a CUDA GPU is strongly recommended.

The requested 80% validation/test target is checked as F1 at IoU 0.50. Object detection
does not have one classification-style accuracy, so the notebooks also report the
proposal's precision, recall, mAP50, and mAP50:95. The acceptance tables report whether
the trained model actually reaches 0.80 rather than claiming it in advance.

## Train YOLO From The Command Line

CPU baseline:

```powershell
python -m road_detection.yolo_train `
  --data data\bdd100k_yolo\data.yaml `
  --model yolo11n.pt `
  --epochs 20 `
  --imgsz 640 `
  --batch 4 `
  --device cpu `
  --workers 0 `
  --fraction 0.05 `
  --freeze 10
```

Final higher-accuracy run:

```powershell
python -m road_detection.yolo_train `
  --data data\bdd100k_yolo\data.yaml `
  --model yolo11m.pt `
  --epochs 120 `
  --imgsz 960 `
  --batch -1 `
  --device 0 `
  --cache disk `
  --multi-scale
```

If your installed Ultralytics release provides a newer YOLO family, pass it directly, for
example `--model yolo26m.pt`.

## Validate And Export YOLO

```powershell
python -m road_detection.yolo_eval_export `
  --weights runs\yolo\bdd100k_road_objects\weights\best.pt `
  --data data\bdd100k_yolo\data.yaml `
  --imgsz 960 `
  --device 0 `
  --split test
```

Export for fast inference:

```powershell
python -m road_detection.yolo_eval_export `
  --weights runs\yolo\bdd100k_road_objects\weights\best.pt `
  --data data\bdd100k_yolo\data.yaml `
  --export onnx `
  --half
```

For NVIDIA GPUs, use `--export engine --half` after validating the PyTorch model.

## Train Faster R-CNN

```powershell
python -m road_detection.rcnn_train `
  --dataset data\bdd100k_yolo `
  --variant mobilenet `
  --epochs 8 `
  --batch 1 `
  --min-size 480 `
  --max-size 640 `
  --workers 0 `
  --device cpu `
  --max-train-images 1000 `
  --max-val-images 250 `
  --max-test-images 250 `
  --output models\fasterrcnn_bdd100k.pth
```

Final Faster R-CNN ResNet50 comparison:

```powershell
python -m road_detection.rcnn_train `
  --dataset data\bdd100k_yolo `
  --variant resnet50 `
  --epochs 24 `
  --batch 2 `
  --min-size 800 `
  --max-size 1024 `
  --trainable-backbone-layers 5 `
  --device cuda
```

The R-CNN initializer reuses compatible COCO prediction-head weights, tunes confidence
on validation data, and applies that fixed threshold to test data.

## Realtime Detection

YOLO webcam:

```powershell
python YOLO_Live_Capture.py `
  --backend yolo `
  --weights runs\yolo\bdd100k_road_objects\weights\best.pt `
  --source 0 `
  --conf 0.35 `
  --device 0
```

Faster R-CNN webcam:

```powershell
python RCNN_Live_Capture.py `
  --backend rcnn `
  --weights models\fasterrcnn_bdd100k.pth `
  --source 0 `
  --conf 0.45
```

Video demo:

```powershell
python -m road_detection.realtime_detect `
  --backend yolo `
  --weights runs\yolo\bdd100k_road_objects\weights\best.pt `
  --source C:\path\to\driving_clip.mp4 `
  --save outputs\road_detection_demo.mp4
```

## Benchmark Speed

```powershell
python -m road_detection.benchmark_inference `
  --weights runs\yolo\bdd100k_road_objects\weights\best.pt `
  --source data\bdd100k_yolo\images\val `
  --imgsz 960 `
  --device 0
```

## Accuracy And Speed Strategy

- Use pretrained COCO YOLO weights for transfer learning because the proposal classes
  overlap heavily with COCO road objects.
- Start with `yolo11n.pt` for CPU iteration, then move to `yolo11m.pt` or a newer
  supported YOLO model for the final run.
- Keep image size at `960` or `1024` so small traffic lights and signs are not crushed.
- Use multi-scale training, mosaic, mixup, color jitter, and cosine learning
  rate decay for robustness to BDD100K weather/time-of-day variation.
- Use frozen-backbone transfer learning first, then unfreeze at a lower learning rate.
- Tune confidence on validation only and evaluate once on the local held-out test split.
- Use `--cache disk`, AMP, and automatic batch sizing for speed without changing model
  behavior.
- Export the final YOLO model to ONNX or TensorRT for realtime demos.
- Use Faster R-CNN as a higher-cost two-stage baseline, especially useful for analyzing
  missed small or overlapping objects.

## Sources

- BDD100K dataset: https://bdd-data.berkeley.edu/
- BDD100K paper: https://arxiv.org/abs/1805.04687
- Ultralytics object detection docs: https://docs.ultralytics.com/tasks/detect/
- Ultralytics training docs: https://docs.ultralytics.com/modes/train/
- COCO dataset: https://cocodataset.org/
