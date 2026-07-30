from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from road_detection.budgeted_yolo_trainer import (
    make_budgeted_detection_trainer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark rotating, class-balanced YOLO micro-epochs."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("runs/yolo_epoch_bench"))
    parser.add_argument("--name", default="budgeted_probe")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = make_budgeted_detection_trainer(
        samples_per_epoch=args.samples,
        balance_power=0.35,
        sampler_seed=args.seed,
    )
    model = YOLO(str(args.model))
    results = model.train(
        trainer=trainer,
        data=str(args.data),
        epochs=args.epochs,
        patience=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.00035,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0006,
        warmup_epochs=0.25,
        freeze=args.freeze,
        amp=True,
        cos_lr=True,
        close_mosaic=1,
        deterministic=False,
        cache=False,
        plots=False,
        save=True,
        val=True,
        box=7.5,
        cls=0.65,
        cls_pw=0.35,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.35,
        translate=0.08,
        scale=0.45,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
        verbose=True,
        seed=args.seed,
    )
    print(f"Benchmark run: {results.save_dir}")


if __name__ == "__main__":
    main()
