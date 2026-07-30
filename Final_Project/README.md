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

## Setup

### Use The Existing Project Environment

This project was developed with the virtual environment at `C:\tf214_hw2`. It uses
64-bit **Python 3.11.9**.

```powershell
cd C:\Users\vipra\OneDrive\Documents\GitHub\ECGR-5106-Intro-To-Deep-Learning\Final_Project
& C:\tf214_hw2\Scripts\Activate.ps1
python --version
python -m pip install -e .
jupyter lab
```

After activation, `python --version` should print `Python 3.11.9`, and the PowerShell
prompt will normally begin with `(tf214_hw2)`. The `pip install -e .` command installs
the project package and dependencies, including JupyterLab. Run `deactivate` when
finished.

If PowerShell prevents activation, the environment can still be used directly:

```powershell
C:\tf214_hw2\Scripts\python.exe -m pip install -e .
C:\tf214_hw2\Scripts\python.exe -m jupyter lab
```

The laptop's RTX 3050 requires CUDA-enabled PyTorch wheels. The tested environment
uses PyTorch 2.10.0 with CUDA 12.6:

```powershell
C:\tf214_hw2\Scripts\python.exe -m pip install --force-reinstall `
  torch==2.10.0 torchvision==0.25.0 `
  --index-url https://download.pytorch.org/whl/cu126
C:\tf214_hw2\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Restart Jupyter after changing PyTorch. The verification command should print `True`
and `NVIDIA GeForce RTX 3050 Laptop GPU`.
The improved notebook requires `ultralytics>=8.4.112`; `python -m pip install -e .`
installs or upgrades it.

### Create A General Project-Local Environment

For another computer or user, create a virtual environment inside the project. The
project supports Python 3.10 or newer; Python 3.11 is recommended to match the tested
environment.

```powershell
cd C:\path\to\ECGR-5106-Intro-To-Deep-Learning\Final_Project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -e .
jupyter lab
```

Open `notebooks/01_YOLO_BDD100K_Training.ipynb`,
`notebooks/01B_YOLO_BDD100K_Overnight_Accuracy.ipynb`,
`notebooks/01C_YOLO_BDD100K_Fast_High_Accuracy.ipynb`,
`notebooks/01D_YOLO_One_Minute_Epoch_High_Precision.ipynb`,
`notebooks/01E_YOLO_BDD100K_Two_Minute_High_Accuracy.ipynb`, or
`notebooks/02_Faster_RCNN_BDD100K_Training.ipynb`. The original YOLO notebook
preserves the completed CPU baseline, and `01B` preserves the first accuracy run.
Use `01E` for the recommended two-minute accuracy continuation. Use `01D` only when
every training epoch must finish in approximately one minute.

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
`labels/bdd100k_labels_images_train.json`, plus the per-image JSON layout
`labels/train/*.json` and `labels/val/*.json`. Public BDD100K test labels are normally
withheld, so the converter reproducibly reserves 20% of labeled validation data as a
local test split.

## Convert BDD100K To YOLO Format

Quick smoke-test subset:

```powershell
.\scripts\prepare_bdd100k.ps1 `
  -BddRoot .\data\bdd100k `
  -Output .\data\bdd100k_yolo_smoke `
  -MaxImagesPerSplit 500
```

Full conversion:

```powershell
python -m road_detection.bdd100k_to_yolo `
  --bdd-root .\data\bdd100k `
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
  index/train.jsonl
  index/val.jsonl
  index/test.jsonl
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

The compact index is created during conversion while annotations are already in
memory. The `01C` notebook uses it to build class-aware samples without reopening
70,000 individual label files.

## Notebook Training Profiles

The recommended `01E_YOLO_BDD100K_Two_Minute_High_Accuracy.ipynb` profile:

- Starts explicitly from the strongest `01C` checkpoint and excludes the weaker
  executed `01D` checkpoint.
- Trains YOLO11n at 576 pixels with batch 16 and `freeze=10`.
- Draws 2,200 new class-balanced scenes without replacement every epoch from a
  12,000-image pool while preserving optimizer and cosine-schedule state.
- Uses inverse-frequency classification loss weights for the rare bus and truck
  classes and moderate BDD100K augmentation.
- Measured 139.0 seconds for the warm-up epoch and 104.7 seconds for the steady epoch
  with a 200-image validation sample on the RTX 3050. The notebook's 400-image
  validation sample adds about four seconds.
- Runs 240 bounded epochs by default, then compares the starting and trained
  checkpoints on all 8,000 validation images and evaluates the winner on all 2,000
  held-out test images.
- Saves balanced and strict 0.80-minimum-confidence deployment profiles. Confidence
  filtering is reported separately from mAP/F1 and is never described as accuracy.

The measured `01D_YOLO_One_Minute_Epoch_High_Precision.ipynb` profile:

- Automatically starts from the newest BDD100K checkpoint.
- Uses 300 rare-class-balanced images at 320 pixels, batch 32, and freezes the first
  16 model modules.
- Measured 29.9 seconds training, 4.2 seconds epoch validation, and 46.9 seconds total
  first-epoch wall time on the RTX 3050.
- Always compares the micro-tuned model with the untouched starting checkpoint on all
  8,000 validation images before deployment.
- Provides a strict deployment mode targeting 90% validation precision with a minimum
  displayed confidence of 0.70.

The current checkpoint measured 90.5% aggregate precision at confidence 0.70 on the
1,200-image validation sample, but recall was only 7.5%. This is a strict operating
precision result, not 90% F1, mAP, or overall accuracy. A micro-epoch processes 300
selected images rather than the full 70,000-image training set. The one-time full
validation and test evaluation therefore take longer than one minute.

The longer `01C_YOLO_BDD100K_Fast_High_Accuracy.ipynb` provides:

- `fast_smoke`: a one-epoch end-to-end check.
- `fast_overnight`: the measured RTX 3050 continuation. It automatically starts from
  the newest prior accuracy checkpoint, retains the 4,000 class-aware scenes and adds
  4,000 fresh scenes at batch 10/576 pixels, then uses batch 8/640 pixels for
  full-network refinement.
- `larger_gpu`: a longer high-resolution run for a larger GPU.
- `cpu_fallback`: a bounded CPU continuation.

The measured `v2` run took about 56-65 minutes per 4,000-image epoch at batch 6 and
704 pixels. A local `01C` fit check processed batch 10 at 576 pixels in 3.0 seconds per
step instead of about 5.2 seconds per step, or roughly 2.9 times more images per second.
It also replaces full 8,000-image validation during every epoch with a fixed 1,200-image
validation subset. Finalist selection still uses all 8,000 validation images, and the
final report still uses all 2,000 held-out test images.

The earlier `01B_YOLO_BDD100K_Overnight_Accuracy.ipynb` provides:

- `gpu_smoke`: one-epoch pipeline validation.
- `overnight_gpu`: an 11-hour continuation/refinement run for the RTX 3050 that
  preserves the completed YOLO11n baseline training.
- `nextgen_gpu`: a fresh YOLO26s experiment with a 34-hour budget on this laptop.
- `full_accuracy`: a longer YOLO26m run for a larger GPU.
- `cpu_fallback`: a bounded CPU run that is not expected to match the GPU profile.

The unchanged baseline notebook provides:

- `cpu_quick`: small end-to-end run for the current CPU.
- `cpu_practical`: larger CPU experiment using more data.
- `accuracy`: full-data, high-resolution final run; a CUDA GPU is strongly recommended.

The requested 70-80% validation/test range is checked as F1 at IoU 0.50. Object detection
does not have one classification-style accuracy, so the notebooks also report the
proposal's precision, recall, mAP50, and mAP50:95. The acceptance tables report whether
the trained model actually reaches the target rather than claiming it in advance.

For the current continuation, open `01E`, leave `RUN_TAG = "v5_rotating_2min"`, and
use **Run All Cells**. It automatically starts from:

```text
runs/notebooks/yolo_accuracy/bdd100k_fast_overnight_v3_fast_main/weights/best.pt
```

If training is interrupted, keep the same tag and set `RESUME_TRAINING = True`. On a
fresh clone with no local checkpoint, `01E` falls back to COCO-pretrained
`yolo11n.pt` and starts transfer learning from it.

For the one-minute workflow, open `01D`, leave `RUN_TAG = "v4_one_minute"`, and use
**Run All Cells**. The default 60 micro-epochs take approximately 45-60 minutes in
total, followed by the slower one-time full validation and test evaluation. If
interrupted, keep the tag and set `RESUME_TRAINING = True`.

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

Calibrated YOLO webcam after running the improved notebook:

```powershell
python -m road_detection.realtime_detect `
  --backend yolo `
  --weights outputs\yolo_accuracy_fast\bdd100k_yolo_fast_best.pt `
  --config outputs\yolo_accuracy_fast\deployment_config.json `
  --threshold-profile high_precision `
  --source 0 `
  --device 0
```

The deployment JSON contains `balanced` and `high_precision` per-class thresholds
selected on validation data. The default high-precision profile displays only scores
of 0.70 or greater and targets 0.75 validation precision where the measured curve
supports it. This score floor is not the same as 70% mAP and can reduce recall.
Passing `--conf` explicitly overrides all calibrated thresholds.

Strict 90%-precision-mode webcam after running `01D`:

```powershell
python -m road_detection.realtime_detect `
  --backend yolo `
  --weights outputs\yolo_one_minute\bdd100k_yolo_selected.pt `
  --config outputs\yolo_one_minute\deployment_config.json `
  --threshold-profile high_precision_90 `
  --source 0 `
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
- Continue the strongest adapted YOLO11n checkpoint rather than restarting from COCO.
- Use 8,000 varied images in the measured batch-10 576-pixel main stage, then unfreeze
  the full network for a batch-8 640-pixel refinement stage. Final model selection
  protects against regression.
- Keep YOLO26s as the longer next-generation experiment; measured local throughput
  makes it a multi-day run rather than an overnight run.
- Use class-aware sampling and partial inverse-frequency class weighting for rare buses
  and trucks.
- Use mosaic, mild mixup, brightness/color augmentation, and cosine learning-rate decay
  for BDD100K weather and time-of-day variation.
- Compare the main and rare-class-refined checkpoints on validation data before choosing
  the deployment model.
- Tune confidence on validation only and evaluate once on the local held-out test split.
- Use AMP, channels-last tensors, refinement-set RAM caching, measured batch sizes, and
  nondeterministic CUDA kernels for speed.
- Export the final YOLO model to ONNX or TensorRT for realtime demos.
- Use Faster R-CNN as a higher-cost two-stage baseline, especially useful for analyzing
  missed small or overlapping objects.

## Sources

- BDD100K dataset: https://bdd-data.berkeley.edu/
- BDD100K paper: https://arxiv.org/abs/1805.04687
- Ultralytics object detection docs: https://docs.ultralytics.com/tasks/detect/
- Ultralytics training docs: https://docs.ultralytics.com/modes/train/
- COCO dataset: https://cocodataset.org/
