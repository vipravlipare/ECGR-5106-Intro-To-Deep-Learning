# Real-Time BDD100K Road Object Detection

ECGR 5106 final project using YOLO11 and Faster R-CNN to detect cars, buses,
trucks, pedestrians, traffic lights, and traffic signs. The project expands the
earlier ECGR 5116 traffic-light assignment to diverse BDD100K road scenes while
keeping training practical on an NVIDIA RTX 3050 Laptop GPU.

## Supported Training Notebooks

Only two training notebooks are maintained:

- `notebooks/01_YOLO_BDD100K_Training.ipynb`: recommended realtime detector.
  YOLO11s at 704 px, small-object-aware sampling, AMP, local data staging,
  low-learning-rate 768 px refinement, no-regression model selection, full
  validation/test reporting, and per-class live threshold calibration.
- `notebooks/02B_Faster_RCNN_Fast_High_Accuracy.ipynb`: accuracy-oriented
  two-stage comparison. ResNet50-FPN V2, transferred COCO head rows, bounded
  proposal counts, staged backbone fine-tuning, preloaded micro-epochs, and
  held-out checkpoint selection.

Each notebook contains a `smoke`, `rtx3050_balanced`, and
`overnight_accuracy` profile. Run `smoke` once to verify the installation, then
switch back to `rtx3050_balanced` for the real experiment.

## Accuracy Terminology

Object detection does not have one classification-style accuracy value. The
notebooks report:

- **Precision**: fraction of displayed detections that are correct.
- **Recall**: fraction of labeled objects found.
- **F1**: harmonic mean of precision and recall at a selected confidence.
- **mAP50**: area under the per-class precision/recall curves at IoU 0.50.
- **mAP50:95**: stricter COCO-style localization score.
- **Confidence**: model score used to filter a prediction, not accuracy.

The old traffic-light assignment's roughly 90% "accuracy" did not penalize
false positives and could omit missed images from its denominator. Its
ResNet50-FPN, high-resolution, balanced-data, and staged-training ideas were
reused, but its metric was not. Neither notebook claims an 80% or 90% result
before measuring it. Both protect the best starting checkpoint if fine-tuning
regresses.

## Existing Virtual Environment

This computer uses `C:\tf214_hw2` with 64-bit **Python 3.11.9**, PyTorch 2.10.0
with CUDA 12.6, Torchvision 0.25.0, and Ultralytics 8.4.112 or newer.

```powershell
cd C:\Users\vipra\OneDrive\Documents\GitHub\ECGR-5106-Intro-To-Deep-Learning\Final_Project
& C:\tf214_hw2\Scripts\Activate.ps1
python --version
python -m pip install -e .
python -m ipykernel install --user --name tf214_hw2 --display-name "Python (tf214_hw2)"
jupyter lab
```

Select **Python (tf214_hw2)** as the notebook kernel. The notebooks stop before a
real run if this computer accidentally starts them with CPU-only PyTorch.

Verify CUDA:

```powershell
C:\tf214_hw2\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The expected result includes `True` and `NVIDIA GeForce RTX 3050 Laptop GPU`.
If PowerShell blocks activation, use the environment directly:

```powershell
C:\tf214_hw2\Scripts\python.exe -m pip install -e .
C:\tf214_hw2\Scripts\jupyter-lab.exe
```

## General Virtual Environment

Python 3.10 or newer is supported. Python 3.11 is recommended.

```powershell
cd C:\path\to\Final_Project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m ipykernel install --user --name road-detection --display-name "Python (road-detection)"
jupyter lab
```

Install a CUDA-enabled PyTorch build appropriate for the other computer before
running a GPU profile.

## Download BDD100K

Register and download the **100K Images** and **Detection 2020 Labels** archives
from the [official BDD100K download site](https://bdd-data.berkeley.edu/). A
browser error on a guessed direct archive URL does not mean the conversion code
is broken; the official downloads may require an authenticated browser session.

Extract to a real folder, for example:

```text
C:\datasets\bdd100k\
  images\
    100k\
      train\
      val\
      test\
  labels\
    det_20\
      det_train.json
      det_val.json
```

The public BDD test labels are withheld. The converter reproducibly reserves
20% of labeled validation scenes as the local held-out test split.

## Convert The Dataset

Do not type the literal placeholder `C:\path\to\bdd100k`. Use the folder where
the archives were actually extracted:

```powershell
.\scripts\prepare_bdd100k.ps1 `
  -BddRoot C:\datasets\bdd100k `
  -Output .\data\bdd100k_yolo `
  -MaxImagesPerSplit 500
```

That command creates a quick conversion for a pipeline check. For the real
training set, omit `-MaxImagesPerSplit`:

```powershell
.\scripts\prepare_bdd100k.ps1 `
  -BddRoot C:\datasets\bdd100k `
  -Output .\data\bdd100k_yolo
```

Equivalent Python command:

```powershell
python -m road_detection.bdd100k_to_yolo `
  --bdd-root C:\datasets\bdd100k `
  --output .\data\bdd100k_yolo `
  --splits train val `
  --copy-mode hardlink `
  --val-test-fraction 0.20 `
  --seed 42
```

Converted layout:

```text
data\bdd100k_yolo\
  data.yaml
  conversion_summary.json
  index\
    train.jsonl
    val.jsonl
    test.jsonl
  images\
    train\
    val\
    test\
  labels\
    train\
    val\
    test\
```

## Fast Local Data Cache

The repository is inside OneDrive. On this computer a YOLO smoke run spent
about 12.9 minutes hydrating cloud-synced images, then its next 128-image epoch
took only 9.8 seconds. The YOLO notebook now copies its compact selected subset
in parallel to `%LOCALAPPDATA%\bdd100k_road_detection_cache` before training.
The cache is reused on later runs.

To keep the cache beside the virtual environment instead:

```powershell
$env:BDD100K_FAST_CACHE = "C:\tf214_hw2\bdd_fast_cache"
jupyter lab
```

The first staging pass can still take time when OneDrive files are online-only.
It is a one-time I/O cost shown separately from epoch training. Selecting
**Always keep on this device** for `data\bdd100k_yolo` also avoids hydration
delays.

## YOLO Training

Open `notebooks/01_YOLO_BDD100K_Training.ipynb`, leave:

```python
RUN_MODE = "rtx3050_balanced"
RESUME_TRAINING = True
RUN_REFINEMENT = True
```

The default profile uses:

- COCO-pretrained YOLO11s rather than the lower-capacity YOLO11n baseline.
- 1,200 fixed class- and small-object-aware training scenes per epoch.
- 300 fixed validation scenes per epoch.
- Fixed 704 px shapes, batch 8, AMP, and zero Windows DataLoader workers with
  RAM-cached images. Fixed shapes avoid the severe first-epoch cuDNN autotuning
  cost measured with dynamic multi-scale training on this GPU.
- 60 continuous main epochs and up to 12 low-learning-rate 768 px refinement
  epochs.
- Mild road-scene geometry and color augmentation, then closed mosaic.
- Final selection among main, refinement, and the old retained checkpoint on
  exactly the same validation scenes.
- Up to 1,000 final validation and 1,000 local test scenes.

The measured 128-image steady smoke epoch was 9.8 seconds after local hydration.
An earlier direct 704 px benchmark projected the 1,200-image profile near
90-120 seconds per steady epoch. Actual time depends on power mode and cooling.

## Faster R-CNN Training

Open `notebooks/02B_Faster_RCNN_Fast_High_Accuracy.ipynb` and leave:

```python
RUN_MODE = "rtx3050_balanced"
RESUME_TRAINING = True
```

The default profile uses:

- COCO-pretrained Faster R-CNN ResNet50-FPN V2.
- Compatible COCO classification and box-regression rows for road classes.
- A fixed 8,000-scene candidate pool with rare-class and tiny-object weights.
- 320 unique preloaded scenes per epoch, batch 4, AMP, and bounded proposals.
- Three head/FPN warm-up epochs, then low-learning-rate ResNet layer 3/4
  fine-tuning.
- A protected zero-step baseline and combined F1/mAP checkpoint selection.
- 150 proxy-validation scenes per epoch and up to 1,000 final validation/test
  scenes.

The untouched transferred ResNet model locally measured `0.518 F1` and
`0.405 mAP50` on the 150-image proxy before BDD fine-tuning, already far above
the retained MobileNet result (`0.306 F1`, `0.180 mAP50`). A smoke run completed
the full training/evaluation/deployment path and reached 83.0% test precision
at its strict operating threshold; smoke metrics are not final-model claims.

## Realtime Detection

After YOLO training:

```powershell
python -m road_detection.realtime_detect `
  --backend yolo `
  --weights outputs\yolo_final\bdd100k_yolo11s_best.pt `
  --config outputs\yolo_final\deployment_config.json `
  --threshold-profile high_precision_80 `
  --source 0 `
  --device 0
```

Use `--threshold-profile balanced` for better recall. The strict profile uses
per-class thresholds with a 0.70 confidence floor and targets at least 80%
validation precision where the data supports it.

After Faster R-CNN training:

```powershell
python -m road_detection.realtime_detect `
  --backend rcnn `
  --weights models\fasterrcnn_bdd100k_resnet50_best.pth `
  --source 0 `
  --device cuda
```

YOLO should normally be used for realtime deployment. Faster R-CNN is the
two-stage accuracy and error-analysis comparison.

## Tests

```powershell
C:\tf214_hw2\Scripts\python.exe -m pytest -q
```

## Technical References

- [BDD100K paper](https://arxiv.org/abs/1805.04687)
- [Ultralytics YOLO11 models](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics training settings](https://docs.ultralytics.com/modes/train/)
- [Ultralytics road segmentation discussion](https://github.com/orgs/ultralytics/discussions/20550)
- [Road semantic segmentation reference](https://github.com/NikolasEnt/Road-Semantic-Segmentation)
- [Torchvision Faster R-CNN](https://docs.pytorch.org/vision/master/models/faster_rcnn.html)
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)
- [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [SAHI sliced inference for small objects](https://arxiv.org/abs/2202.06934)

SAHI-style tiled inference can improve tiny-object recall, but it is not the
default realtime path because multiple crops increase latency.
