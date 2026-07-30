# Notebooks

The model-training workflows are Jupyter notebooks so data checks, curves, metrics,
predictions, and failure cases remain visible with each experiment.

- `01_YOLO_BDD100K_Training.ipynb`: primary proposal model with staged transfer
  learning, validation/test mAP, qualitative results, and a speed benchmark.
- `01B_YOLO_BDD100K_Overnight_Accuracy.ipynb`: preserved first accuracy-focused
  continuation and its visible executed results.
- `01C_YOLO_BDD100K_Fast_High_Accuracy.ipynb`: recommended RTX 3050 continuation
  with measured batch sizes, broader training data, refinement-set RAM caching, fast
  epoch validation, full finalist validation/test evaluation, and 0.70+
  high-precision live thresholds.
- `01D_YOLO_One_Minute_Epoch_High_Precision.ipynb`: measured sub-minute
  micro-epochs for the RTX 3050, full no-regression model selection, and a strict
  90%-precision deployment profile. This means operating precision, not 90% F1 or mAP.
- `02_Faster_RCNN_BDD100K_Training.ipynb`: MobileNet or ResNet50 Faster R-CNN with
  COCO-head transfer, proper AP metrics, threshold tuning, and a speed benchmark.

Run `pip install -e .`, launch `jupyter lab` from the project root, choose a `RUN_MODE`
near the top, and run all cells. For the current laptop run, open `01C`, leave
`RUN_MODE = "fast_overnight"`, and run all cells. It automatically starts from the
newest prior accuracy checkpoint without modifying either older notebook.

When the hard requirement is one minute or less per training epoch, open `01D` and run
all cells. Its measured first epoch took 46.9 seconds including startup; the one-time
full validation and test evaluation after training take longer.
