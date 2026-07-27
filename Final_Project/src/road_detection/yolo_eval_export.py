from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and optionally export a YOLO detector.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/bdd100k_yolo/data.yaml"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.001, help="Low validation confidence is standard for mAP sweeps.")
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--export", choices=["none", "onnx", "engine", "openvino", "tflite"], default="none")
    parser.add_argument("--half", action="store_true", help="Use FP16 export where supported.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        split=args.split,
        batch=args.batch,
        workers=args.workers,
        plots=True,
        save_json=True,
    )
    print(metrics)
    if args.export != "none":
        exported = model.export(format=args.export, imgsz=args.imgsz, half=args.half, simplify=True)
        print(f"Exported model: {exported}")


if __name__ == "__main__":
    main()
