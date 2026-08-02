# Real-Time BDD100K Road Object Detection

ECGR 5106 final project comparing a one-stage YOLO11s detector with a two-stage
Faster R-CNN ResNet50-FPN V2 detector. Both models recognize six BDD100K road
classes: **car, bus, truck, pedestrian, traffic light, and traffic sign**. The
project extends the earlier ECGR 5116 traffic-light detector to full road scenes
with class-aware sampling, small-object training, calibrated confidence
thresholds, and webcam/video inference.

## Final Deliverables

- [`notebooks/01_YOLO11s_Training.ipynb`](notebooks/01_YOLO11s_Training.ipynb):
  final YOLO11s training, refinement, test evaluation, threshold calibration,
  and deployment benchmark. All 9 code cells are executed with no saved errors.
- [`notebooks/02_RCNN_Training.ipynb`](notebooks/02_RCNN_Training.ipynb): final
  Faster R-CNN training, staged backbone fine-tuning, test evaluation, and
  deployment benchmark. All 10 code cells are executed with no saved errors.
- [`src/road_detection/realtime_detect.py`](src/road_detection/realtime_detect.py):
  shared real-time inference application for both models.

Object detection does not have one classification-style accuracy value.
**Precision** measures how many displayed detections are correct, **recall**
measures how many labeled objects were found, and **mAP** summarizes the
precision-recall curve. A displayed confidence of 0.80 is a filtering score,
not 80% accuracy.

## Final Results

These are the saved held-out **test** results from the executed notebooks. YOLO
01G is the final model now stored as `01_YOLO11s_Training.ipynb`.

| Model | Test images | Precision | Recall | F1 | mAP50 | mAP50:95 |
|---|---:|---:|---:|---:|---:|---:|
| Initial YOLO11s | 1,000 | 63.76% | 49.22% | 55.55% | 52.82% | 28.56% |
| **Final YOLO11s** | 2,000 | 67.11% | **55.06%** | **60.49%** | **59.67%** | **33.41%** |
| Faster R-CNN, balanced (`conf=0.40`) | 1,500 | **73.46%** | 51.19% | 60.34% | 46.77% | 26.18% |
| Faster R-CNN, strict (`conf=0.55`) | 1,500 | **81.75%** | 46.40% | 59.20% | 46.77% | 26.18% |

The final YOLO has the best recall, F1, mAP50, and mAP50:95. Faster R-CNN has
the highest precision when false positives are more costly. The final test
subsets have different sizes, so the strongest YOLO comparison is the
notebook's same-subset candidate evaluation: final YOLO improved over the
initial YOLO by **4.66 precision points, 1.84 recall points, 3.04 F1 points,
5.76 mAP50 points, and 4.07 mAP50:95 points**.

### YOLO Confidence Calibration

The balanced thresholds maximize per-class validation F1. The strict profile
requires at least 0.70 displayed confidence and met the 80% validation precision
target for every class, with the expected reduction in recall.

| Class | Balanced threshold | Balanced F1 | Strict threshold | Strict precision | Strict recall |
|---|---:|---:|---:|---:|---:|
| Car | 0.327 | 71.76% | 0.701 | 98.07% | 37.30% |
| Bus | 0.327 | 52.59% | 0.703 | 80.59% | 32.71% |
| Truck | 0.394 | 54.23% | 0.701 | 82.48% | 30.67% |
| Pedestrian | 0.249 | 57.24% | 0.701 | 96.95% | 6.05% |
| Traffic light | 0.323 | 61.26% | 0.701 | 96.09% | 7.15% |
| Traffic sign | 0.280 | 60.20% | 0.701 | 93.85% | 18.58% |

## Complexity And Runtime

Measurements are from the executed notebooks on an NVIDIA GeForce RTX 3050
Laptop GPU with 4 GB VRAM.

| Measurement | Final YOLO11s | Faster R-CNN ResNet50-FPN V2 |
|---|---:|---:|
| Parameters | 9,430,114 | 43,281,778 |
| FLOPs | 21.6 GFLOPs | Dynamic; 280.37 GFLOPs at Torchvision's 800x800 reference input |
| Final checkpoint size | 18.27 MiB | 165.48 MiB |
| Training completed | 56 main + 12 refinement epochs | 32 epochs |
| Total measured training time | 16.90 hours | 10.21 hours |
| Network inference measurement | 41.5 ms/image at 704 px | 150.4 ms/image |
| Saved end-to-end benchmark | 0.51 images/s | 6.65 images/s |

The YOLO validation timer isolates model inference and is the more useful guide
for live camera frames. Its separate 100-file end-to-end benchmark included
abnormally slow host image I/O and measured 1,958.1 ms/image; rerun that cell on
an idle machine before using it as a deployment FPS claim. Faster R-CNN uses
dynamic 360-640 px resizing, so its FLOPs and speed are not directly comparable
to fixed 704 px YOLO inference.

### Loss And Checkpoint Selection

| Run | First-epoch loss | Final executed loss | Validation protection |
|---|---|---|---|
| YOLO main | box 1.491, cls 1.362, DFL 0.998 | box 1.254, cls 0.906, DFL 0.911 | Best checkpoint selected by held-out F1/mAP quality |
| YOLO refinement | box 1.462, cls 1.210, DFL 0.982 | box 1.366, cls 1.010, DFL 0.946 | Stopped at epoch 12; best result was epoch 5 |
| Faster R-CNN | total 1.359 | total 0.913 | Best checkpoint was epoch 27; later deep unfreezing did not replace it |

Training loss decreased for both architectures. Early stopping and held-out
checkpoint selection prevented later refinement from overwriting a better
validation model. Complete loss curves and per-epoch metrics remain visible in
the notebooks.

## Training Configuration

### Final YOLO11s

- COCO-pretrained YOLO11s, initialized from the completed initial YOLO best
  checkpoint for the recorded run.
- Class-aware subsets: 4,800 main images, 3,200 refinement images, 600
  validation images per epoch, and 2,000 images for each final split.
- Main stage: 56 epochs, 576 px, batch 16, AdamW, `lr0=0.0008`, cosine decay,
  weight decay 0.0005, AMP, rectangular batches, and patience 8.
- Refinement: maximum 14 epochs, 704 px, batch 12, AdamW, `lr0=0.0003`, and
  patience 7. It stopped after 12 epochs.
- Mild HSV, translation, scale, rotation, and horizontal-flip augmentation.
  Mosaic and mixup were disabled to preserve small-object geometry.

### Final Faster R-CNN

- COCO-pretrained Faster R-CNN ResNet50-FPN V2 with compatible COCO classifier
  and box-regression rows transferred to the six project classes.
- 32,000-image weighted candidate pool, 640 unique images per epoch, 128 proxy
  validation images per epoch, and 1,500 images for each final split.
- 32 epochs, batch 4, AMP, minimum size 360, maximum size 640.
- Epochs 1-3 trained the detection head/FPN; epochs 4-27 added ResNet layer 4;
  epochs 28-32 added layer 3. Each stage used cosine learning-rate decay.
- Bounded RPN/ROI proposal counts reduce two-stage computation while retaining
  the best validation checkpoint.

Set `RUN_MODE = "rtx3050_more_data"` in both notebooks for the recorded
configuration. For a clean run set `RESUME_TRAINING = False`; use `True` only
when the matching `last.pt` or `last.pth` checkpoint still exists.

## Environment Setup

The tested environment is `C:\tf214_hw2` with 64-bit **Python 3.11.9**,
PyTorch 2.10.0+cu126, Torchvision 0.25.0, and Ultralytics 8.4.112.

```powershell
cd C:\Users\vipra\OneDrive\Documents\GitHub\ECGR-5106-Intro-To-Deep-Learning\Final_Project
& C:\tf214_hw2\Scripts\Activate.ps1
python -m pip install -e .
python -m ipykernel install --user --name tf214_hw2 --display-name "Python (tf214_hw2)"
jupyter lab
```

Select **Python (tf214_hw2)** as the Jupyter kernel. Verify CUDA with:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

For another computer, Python 3.11 is recommended:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the correct CUDA-enabled PyTorch build before selecting a GPU training
profile. CPU inference is supported, but both final training profiles require
CUDA.

## Download And Convert BDD100K

Download **100K Images** and **Detection 2020 Labels** from the
[official BDD100K site](https://bdd-data.berkeley.edu/). The site may require an
authenticated browser session. Extract them so the root resembles:

```text
C:\datasets\bdd100k\
  images\100k\train\
  images\100k\val\
  labels\det_20\det_train.json
  labels\det_20\det_val.json
```

Convert the complete dataset from the project directory:

```powershell
.\scripts\prepare_bdd100k.ps1 `
  -BddRoot C:\datasets\bdd100k `
  -Output .\data\bdd100k_yolo
```

For a quick pipeline check, append `-MaxImagesPerSplit 500`. Do not use the
literal placeholder `C:\path\to\bdd100k`. Public BDD100K test labels are
withheld, so the converter reproducibly reserves 20% of labeled validation
images as the local test split using seed 42.

The notebooks stage their selected subsets outside OneDrive under
`%LOCALAPPDATA%\bdd100k_road_detection_cache`. Override that location with:

```powershell
$env:BDD100K_FAST_CACHE = "C:\tf214_hw2\bdd_fast_cache"
```

## Real-Time Detection

The final model files must exist at the paths shown below. They are ignored by
Git because of their size, so a teammate must receive them separately or
produce them by running the notebooks.

### YOLO Webcam

Balanced profile, recommended for general detection:

```powershell
python -m road_detection.realtime_detect `
  --backend yolo `
  --weights outputs\yolo_more_data\bdd100k_yolo11s_more_data_best.pt `
  --config outputs\yolo_more_data\deployment_config.json `
  --threshold-profile balanced `
  --source 0 `
  --device 0
```

High-precision profile with at least 0.70 displayed confidence:

```powershell
python -m road_detection.realtime_detect `
  --backend yolo `
  --weights outputs\yolo_more_data\bdd100k_yolo11s_more_data_best.pt `
  --config outputs\yolo_more_data\deployment_config.json `
  --threshold-profile high_precision_80 `
  --source 0 `
  --device 0
```

### Faster R-CNN Webcam

Balanced threshold `0.40` is stored in the checkpoint:

```powershell
python -m road_detection.realtime_detect `
  --backend rcnn `
  --weights models\fasterrcnn_bdd100k_resnet50_best.pth `
  --source 0 `
  --device cuda
```

For the measured high-precision operating point, add `--conf 0.55`.

### Video Or Saved Output

Replace `--source 0` with a video path and optionally save the annotated video:

```powershell
python -m road_detection.realtime_detect `
  --backend yolo `
  --weights outputs\yolo_more_data\bdd100k_yolo11s_more_data_best.pt `
  --config outputs\yolo_more_data\deployment_config.json `
  --threshold-profile balanced `
  --source .\input.mp4 `
  --save .\annotated_output.mp4 `
  --device 0
```

Press `q` to close the live window. Use `--device cpu --precision fp32` when
CUDA is unavailable. Pass `--no-show` when processing a video without a display.

## Tests

```powershell
C:\tf214_hw2\Scripts\python.exe -m pytest -q
```

## Repository Notes

`data/`, `runs/`, `tmp/`, `models/`, `outputs/`, and model checkpoint formats
are intentionally ignored by Git. The executed notebook outputs and tables in
this README preserve the experiment evidence, while datasets and large model
files must be distributed separately.

Technical references: [BDD100K](https://arxiv.org/abs/1805.04687),
[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/),
[Torchvision Faster R-CNN](https://docs.pytorch.org/vision/master/models/faster_rcnn.html),
and [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144).
