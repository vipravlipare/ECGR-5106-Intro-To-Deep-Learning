from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an Ultralytics YOLO detector on BDD100K road objects.")
    parser.add_argument("--data", type=Path, default=Path("data/bdd100k_yolo/data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt", help="Use yolo11n.pt on CPU, yolo11s.pt/yolo11m.pt for final accuracy.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4, help="Use a small fixed batch on CPU; -1 enables GPU auto-batch.")
    parser.add_argument("--device", default=None, help="Example: 0, 0,1, cpu, or mps.")
    parser.add_argument("--project", default="runs/yolo")
    parser.add_argument("--name", default="bdd100k_road_objects")
    parser.add_argument("--workers", type=int, default=0, help="Zero is the safest setting for Windows/Jupyter.")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", choices=["ram", "disk", "False"], default="False")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training images to use.")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze the first N model layers for faster transfer learning.")
    parser.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    cache = False if args.cache == "False" else args.cache
    project = Path(args.project).resolve()
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        pretrained=True,
        optimizer="AdamW",
        cos_lr=True,
        close_mosaic=10,
        mosaic=1.0,
        mixup=0.10,
        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.40,
        degrees=2.0,
        translate=0.08,
        scale=0.50,
        fliplr=0.50,
        multi_scale=args.multi_scale,
        amp=args.amp,
        cache=cache,
        fraction=args.fraction,
        freeze=args.freeze,
        seed=args.seed,
        deterministic=True,
        resume=args.resume,
        plots=True,
        save=True,
        exist_ok=True,
    )
    save_dir = Path(results.save_dir)
    print(f"Training run: {save_dir}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
