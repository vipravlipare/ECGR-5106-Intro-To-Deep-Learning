# Training Notebooks

Use the `Python (tf214_hw2)` Jupyter kernel and run one smoke profile before the
real experiment.

## 01_YOLO_BDD100K_Training.ipynb

Recommended realtime workflow:

- YOLO11s at 704 px on the RTX 3050.
- Fixed rare-class and tiny-object-aware training manifests.
- Parallel staging from OneDrive to a reusable local cache.
- AMP, fixed shapes, realistic augmentation, and closed mosaic.
- Short 768 px small-object refinement.
- Main/refinement/old-checkpoint comparison to prevent regression.
- Final validation/test metrics and per-class live thresholds.

Use `RUN_MODE = "rtx3050_balanced"` for the real run. `RESUME_TRAINING = True`
continues an interrupted run with the same profile-specific tag.

## 02B_Faster_RCNN_Fast_High_Accuracy.ipynb

Accuracy-oriented two-stage comparison:

- Faster R-CNN ResNet50-FPN V2 with compatible COCO head transfer.
- Rare-class and tiny-object-aware 320-image epochs.
- Parallel per-epoch image preload to avoid OneDrive stalls.
- Bounded RPN/ROI proposal work, batch 4, and AMP.
- Head/FPN warm-up followed by low-learning-rate layer 3/4 fine-tuning.
- Protected COCO-transfer baseline and proper F1/mAP checkpoint selection.
- Final balanced and 80%-precision operating points.

Use `RUN_MODE = "rtx3050_balanced"` for the real run.

## Interpreting Results

Confidence is not accuracy. Report precision, recall, F1, mAP50, and mAP50:95
from the final held-out tables. A strict confidence threshold can raise
precision while sharply reducing recall.
