# Training Notebooks

Use the `Python (tf214_hw2)` Jupyter kernel and run one smoke profile before the
real experiment.

The completed YOLO notebook, its larger-data successor, and the current Faster
R-CNN notebook are kept as separate files. Previous committed iterations and
recovered executed results are stored under `recovered/`; do not run them over
the current experiments.

## 01_YOLO_BDD100K_Training.ipynb

Completed realtime baseline. Its exact executed copy is archived under
`recovered/`; keep this notebook unchanged while comparing later experiments.

## 01G_YOLO11s_More_Data_Class_Balanced.ipynb

Recommended larger-data experiment:

- Continues from the completed YOLO11s checkpoint rather than restarting.
- 4,800-image main and 3,200-image refinement manifests.
- Guaranteed per-class image minimums plus small-object-aware weighting.
- Rectangular 576 px bulk training and 704 px refinement for efficient batches.
- Moderate inverse-frequency classification weights for rare classes.
- Parallel staging from OneDrive to a reusable local cache.
- Main/refinement/completed-checkpoint comparison to prevent regression.
- Final validation/test metrics and per-class live thresholds.

Use `RUN_MODE = "rtx3050_more_data"` for the real run. `RESUME_TRAINING = True`
continues an interrupted run with the same profile-specific tag.

## 02B_Faster_RCNN_Fast_High_Accuracy.ipynb

Accuracy-oriented staged comparison:

- Faster R-CNN ResNet50-FPN V2 with compatible COCO head transfer.
- Rare-class and tiny-object-aware 640-image coverage epochs.
- Parallel per-epoch image preload to avoid OneDrive stalls.
- Bounded RPN/ROI proposal work, batch 4, and AMP.
- Head/FPN warm-up, layer-4 fine-tuning, and a short layer-3/4 finish.
- Protected COCO-transfer baseline and proper F1/mAP checkpoint selection.
- Final balanced and 80%-precision operating points.

Use `RUN_MODE = "rtx3050_more_data"` for the real run.

## Preservation Policy

Future changes must be made in a newly named notebook first. Fully executed
notebooks are archived under `recovered/` before any source cells are changed.
See `recovered/README.md` for which files are exact recoveries and which YOLO
result notebook was reconstructed from surviving run artifacts.

## Interpreting Results

Confidence is not accuracy. Report precision, recall, F1, mAP50, and mAP50:95
from the final held-out tables. A strict confidence threshold can raise
precision while sharply reducing recall.
