from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from road_detection.constants import PROJECT_CLASSES
from road_detection.rcnn_dataset import YoloDetectionDataset, collate_fn
from road_detection.rcnn_metrics import collect_predictions, evaluate_predictions, tune_score_threshold
from road_detection.rcnn_model import (
    FAST_ACCURATE_RCNN_CONFIG,
    RCNN_VARIANTS,
    build_faster_rcnn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on the converted BDD100K YOLO dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("data/bdd100k_yolo"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--variant", choices=RCNN_VARIANTS, default="mobilenet")
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--max-size", type=int, default=768)
    parser.add_argument("--trainable-backbone-layers", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("models/fasterrcnn_bdd100k.pth"))
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-f1", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def maybe_subset(dataset, limit: int | None):
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    train_ds = maybe_subset(
        YoloDetectionDataset(args.dataset, "train", max_size=args.max_size, augment=args.augment),
        args.max_train_images,
    )
    val_ds = maybe_subset(YoloDetectionDataset(args.dataset, "val", max_size=args.max_size), args.max_val_images)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = build_faster_rcnn(
        num_classes=len(PROJECT_CLASSES) + 1,
        variant=args.variant,
        min_size=args.min_size,
        max_size=args.max_size,
        trainable_backbone_layers=args.trainable_backbone_layers,
        class_names=PROJECT_CLASSES,
        transfer_coco_head=True,
        performance_config=FAST_ACCURATE_RCNN_CONFIG,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Faster R-CNN epoch {epoch}/{args.epochs}")
        for images, targets in progress:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            progress.set_postfix(loss=f"{running_loss / max(1, progress.n):.4f}")
        scheduler.step()
        predictions, targets = collect_predictions(model, val_loader, device)
        metrics = evaluate_predictions(
            predictions,
            targets,
            score_threshold=args.score_threshold,
            class_names=PROJECT_CLASSES,
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            **asdict(metrics),
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": PROJECT_CLASSES,
                    "epoch": epoch,
                    "metrics": row,
                    "variant": args.variant,
                    "min_size": args.min_size,
                    "max_size": args.max_size,
                    "performance_config": FAST_ACCURATE_RCNN_CONFIG.to_dict(),
                },
                args.output,
            )
            print(f"Saved best Faster R-CNN checkpoint to {args.output}")
    (args.output.parent / "fasterrcnn_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    checkpoint = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_predictions, val_targets = collect_predictions(model, val_loader, device)
    tuned_val = tune_score_threshold(
        val_predictions,
        val_targets,
        class_names=PROJECT_CLASSES,
    )
    report = {
        "target_f1": args.target_f1,
        "target_met_on_validation": tuned_val.f1 >= args.target_f1,
        "validation": asdict(tuned_val),
    }
    test_image_dir = args.dataset / "images" / "test"
    if test_image_dir.exists() and any(test_image_dir.iterdir()):
        test_ds = maybe_subset(
            YoloDetectionDataset(args.dataset, "test", max_size=args.max_size),
            args.max_test_images,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
        )
        test_predictions, test_targets = collect_predictions(model, test_loader, device)
        test_metrics = evaluate_predictions(
            test_predictions,
            test_targets,
            score_threshold=tuned_val.score_threshold,
            class_names=PROJECT_CLASSES,
        )
        report["target_met_on_test"] = test_metrics.f1 >= args.target_f1
        report["test"] = asdict(test_metrics)
    checkpoint["score_threshold"] = tuned_val.score_threshold
    checkpoint["final_evaluation"] = report
    torch.save(checkpoint, args.output)
    (args.output.parent / "fasterrcnn_evaluation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
