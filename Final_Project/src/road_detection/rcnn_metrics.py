from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torchvision.ops import box_iou


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float
    score_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)


def collect_predictions(model, loader, device) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets_cpu: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            predictions.extend(
                {
                    "boxes": output["boxes"].detach().cpu(),
                    "labels": output["labels"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                }
                for output in outputs
            )
            targets_cpu.extend(
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                }
                for target in targets
            )
    return predictions, targets_cpu


def _match_counts(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    score_threshold: float,
    iou_threshold: float,
    class_id: int | None = None,
) -> tuple[int, int, int]:
    true_positives = false_positives = false_negatives = 0
    for output, target in zip(predictions, targets):
        pred_keep = output["scores"] >= score_threshold
        if class_id is not None:
            pred_keep &= output["labels"] == class_id
        pred_boxes = output["boxes"][pred_keep]
        pred_labels = output["labels"][pred_keep]
        pred_scores = output["scores"][pred_keep]

        gt_keep = torch.ones(len(target["labels"]), dtype=torch.bool)
        if class_id is not None:
            gt_keep &= target["labels"] == class_id
        gt_boxes = target["boxes"][gt_keep]
        gt_labels = target["labels"][gt_keep]
        matched_gt: set[int] = set()

        for pred_index in torch.argsort(pred_scores, descending=True).tolist():
            same_class = torch.where(gt_labels == pred_labels[pred_index])[0]
            candidates = [int(index) for index in same_class.tolist() if int(index) not in matched_gt]
            if not candidates:
                false_positives += 1
                continue
            ious = box_iou(pred_boxes[pred_index].unsqueeze(0), gt_boxes[candidates]).squeeze(0)
            best_iou, best_position = torch.max(ious, dim=0)
            if float(best_iou) >= iou_threshold:
                true_positives += 1
                matched_gt.add(candidates[int(best_position)])
            else:
                false_positives += 1
        false_negatives += len(gt_boxes) - len(matched_gt)
    return true_positives, false_positives, false_negatives


def _average_precision_for_class(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    class_id: int,
    iou_threshold: float,
) -> float | None:
    ground_truth: dict[int, torch.Tensor] = {}
    prediction_rows: list[tuple[float, int, torch.Tensor]] = []
    total_ground_truth = 0

    for image_index, (output, target) in enumerate(zip(predictions, targets)):
        gt_boxes = target["boxes"][target["labels"] == class_id]
        ground_truth[image_index] = gt_boxes
        total_ground_truth += len(gt_boxes)
        keep = output["labels"] == class_id
        for box, score in zip(output["boxes"][keep], output["scores"][keep]):
            prediction_rows.append((float(score), image_index, box))

    if total_ground_truth == 0:
        return None

    prediction_rows.sort(key=lambda row: row[0], reverse=True)
    matched: dict[int, set[int]] = {image_index: set() for image_index in ground_truth}
    tp = torch.zeros(len(prediction_rows), dtype=torch.float64)
    fp = torch.zeros(len(prediction_rows), dtype=torch.float64)

    for prediction_index, (_, image_index, pred_box) in enumerate(prediction_rows):
        gt_boxes = ground_truth[image_index]
        available = [index for index in range(len(gt_boxes)) if index not in matched[image_index]]
        if not available:
            fp[prediction_index] = 1.0
            continue
        ious = box_iou(pred_box.unsqueeze(0), gt_boxes[available]).squeeze(0)
        best_iou, best_position = torch.max(ious, dim=0)
        if float(best_iou) >= iou_threshold:
            tp[prediction_index] = 1.0
            matched[image_index].add(available[int(best_position)])
        else:
            fp[prediction_index] = 1.0

    cumulative_tp = torch.cumsum(tp, dim=0)
    cumulative_fp = torch.cumsum(fp, dim=0)
    recalls = cumulative_tp / max(1, total_ground_truth)
    precisions = cumulative_tp / torch.clamp(cumulative_tp + cumulative_fp, min=1e-12)
    recall_points = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    interpolated = [
        torch.max(precisions[recalls >= recall_point]).item() if torch.any(recalls >= recall_point) else 0.0
        for recall_point in recall_points
    ]
    return float(sum(interpolated) / len(interpolated))


def evaluate_predictions(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    score_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    class_names: Sequence[str] | None = None,
) -> DetectionMetrics:
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must contain the same number of images.")
    max_label = max(
        [int(labels.max()) for item in targets for labels in [item["labels"]] if len(labels)] + [0]
    )
    class_ids = list(range(1, len(class_names) + 1)) if class_names else list(range(1, max_label + 1))

    tp, fp, fn = _match_counts(predictions, targets, score_threshold, iou_threshold)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)

    per_class: dict[str, dict[str, float]] = {}
    ap50_values = []
    ap5095_values = []
    iou_sweep = [0.50 + 0.05 * index for index in range(10)]
    for class_id in class_ids:
        name = class_names[class_id - 1] if class_names else str(class_id)
        class_tp, class_fp, class_fn = _match_counts(
            predictions,
            targets,
            score_threshold,
            iou_threshold,
            class_id,
        )
        class_precision = class_tp / max(1, class_tp + class_fp)
        class_recall = class_tp / max(1, class_tp + class_fn)
        class_f1 = 2.0 * class_precision * class_recall / max(1e-12, class_precision + class_recall)
        class_aps = [
            _average_precision_for_class(predictions, targets, class_id, threshold)
            for threshold in iou_sweep
        ]
        valid_aps = [value for value in class_aps if value is not None]
        if valid_aps:
            ap50_values.append(valid_aps[0])
            ap5095_values.append(sum(valid_aps) / len(valid_aps))
        per_class[name] = {
            "precision": class_precision,
            "recall": class_recall,
            "f1": class_f1,
            "ap50": valid_aps[0] if valid_aps else 0.0,
            "map50_95": sum(valid_aps) / len(valid_aps) if valid_aps else 0.0,
            "ground_truth": float(class_tp + class_fn),
        }

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        map50=sum(ap50_values) / max(1, len(ap50_values)),
        map50_95=sum(ap5095_values) / max(1, len(ap5095_values)),
        score_threshold=score_threshold,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        per_class=per_class,
    )


def tune_score_threshold(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    thresholds: Sequence[float] | None = None,
    class_names: Sequence[str] | None = None,
) -> DetectionMetrics:
    thresholds = thresholds or [0.10 + 0.05 * index for index in range(13)]
    candidates = [
        evaluate_predictions(
            predictions,
            targets,
            score_threshold=threshold,
            class_names=class_names,
        )
        for threshold in thresholds
    ]
    return max(candidates, key=lambda metrics: (metrics.f1, metrics.precision))


def evaluate_simple(model, loader, device, score_threshold: float = 0.25, iou_threshold: float = 0.50) -> DetectionMetrics:
    predictions, targets = collect_predictions(model, loader, device)
    return evaluate_predictions(
        predictions,
        targets,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
