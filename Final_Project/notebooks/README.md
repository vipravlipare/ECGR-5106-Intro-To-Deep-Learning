# Notebooks

The model-training workflows are Jupyter notebooks so data checks, curves, metrics,
predictions, and failure cases remain visible with each experiment.

- `01_YOLO_BDD100K_Training.ipynb`: primary proposal model with staged transfer
  learning, validation/test mAP, qualitative results, and a speed benchmark.
- `02_Faster_RCNN_BDD100K_Training.ipynb`: MobileNet or ResNet50 Faster R-CNN with
  COCO-head transfer, proper AP metrics, threshold tuning, and a speed benchmark.

Run `pip install -e .`, launch `jupyter lab` from the project root, choose a `RUN_MODE`
near the top, and run all cells. Start with `cpu_quick`; use `accuracy` for the final
full-data result on a GPU.
