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
- `01E_YOLO_BDD100K_Two_Minute_High_Accuracy.ipynb`: recommended successor to
  `01D`. It trains at 576 pixels on 2,200 newly sampled scenes per epoch, rotates
  rare-class-aware samples through a 12,000-image pool, preserves optimizer state,
  and performs full validation/test selection plus 0.80-confidence calibration.
- `02_Faster_RCNN_BDD100K_Training.ipynb`: MobileNet or ResNet50 Faster R-CNN with
  COCO-head transfer, proper AP metrics, threshold tuning, and a speed benchmark.

Run `pip install -e .`, launch `jupyter lab` from the project root, choose a `RUN_MODE`
near the top, and run all cells. For the current laptop run, open `01E`, keep its
default `v5_rotating_2min` settings, and run all cells. It starts from the strongest
`01C` checkpoint without modifying any executed notebook.

When the hard requirement is one minute or less per training epoch, open `01D` and run
all cells. For the better speed/quality tradeoff, use `01E`; its measured warm-up and
steady benchmark epochs took 139.0 and 104.7 seconds.
