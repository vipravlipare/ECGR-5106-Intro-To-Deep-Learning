# Recovered Notebook Archive

The current improved notebooks remain unchanged in `notebooks/`. This folder
preserves earlier work separately so future improvements never overwrite saved
results again.

## Fully Executed Results

- `2026-08-01_125201_02B_Faster_RCNN_Completed_Executed.ipynb` is an exact,
  byte-for-byte archive of the completed 32-epoch ResNet50-FPN V2 coverage run.
  It contains the 20,480-image training history, 1,500-image validation and test
  evaluation, per-class metrics, confidence operating points, visual predictions,
  and throughput benchmark. Its balanced test result is 0.735 precision, 0.512
  recall, 0.603 F1, 0.468 mAP50, and 0.262 mAP50:95.
- `2026-08-01_021051_01_YOLO_BDD100K_Completed_Executed.ipynb` is an exact,
  byte-for-byte archive of the completed YOLO11s 704/768-pixel run. It contains
  the main and refinement training outputs, candidate comparison, 1,000-image
  held-out validation and test evaluation, confidence calibration, and speed
  benchmark. The final test result is 0.638 precision, 0.492 recall, 0.556 F1,
  0.528 mAP50, and 0.286 mAP50:95.
- `02B_Faster_RCNN_Fully_Executed_Recovered.ipynb` is an **exact byte-for-byte
  recovery** of the fully executed RCNN notebook. It contains 12 executed code
  cells, 137 output blocks, training history through epoch 60, plots, final
  validation/test results, and the saved best epoch.
- `01_YOLO_BDD100K_Previous_Results_Reconstructed.ipynb` combines the original
  committed YOLO source with the exact surviving run CSV/JSON, curves, confusion
  matrix, and prediction image. Its numeric results and artifacts are exact, but
  its output-cell ordering is reconstructed because the fully executed working
  notebook itself was not committed before it was overwritten.
- `01_YOLO_BDD100K_Training_Older_Executed_Backup.ipynb` is an exact older smoke
  execution from July 27. It is not the later full run.

## Original Code And Prior Iterations

- `01_YOLO_BDD100K_Training_Original_Committed.ipynb`
- `01B_YOLO_BDD100K_Overnight_Accuracy.ipynb`
- `01C_YOLO_BDD100K_Fast_High_Accuracy.ipynb`
- `01D_YOLO_One_Minute_Epoch_High_Precision.ipynb`
- `01E_YOLO_BDD100K_Two_Minute_High_Accuracy.ipynb`
- `02_Faster_RCNN_BDD100K_Training.ipynb`

These are exact files recovered from earlier Git commits. Most were committed
without execution outputs.

## OneDrive Version History

The only remaining route to a byte-for-byte copy of the later fully executed
YOLO notebook is OneDrive version history. In OneDrive, locate
`notebooks/01_YOLO_BDD100K_Training.ipynb`, choose **Version history**, and
download a version from before approximately **July 31, 2026 at 4:05 AM ET**.
Look for the roughly 2.8 MB version. Download it under a new filename; do not
restore it over the currently running improved notebook.
