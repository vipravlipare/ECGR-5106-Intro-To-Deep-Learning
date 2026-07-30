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


@dataclass
class _MatchRecords:
    scores: torch.Tensor
    labels: torch.Tensor
    true_positives: torch.Tensor
    ground_truth_by_class: torch.Tensor


def collect_predictions(
    model,
    loader,
    device,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
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


def _build_match_records(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    iou_threshold: float,
    num_classes: int,
) -> _MatchRecords:
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    true_positives: list[torch.Tensor] = []
    ground_truth_by_class = torch.zeros(num_classes + 1, dtype=torch.int64)

    for output, target in zip(predictions, targets):
        pred_boxes = output["boxes"]
        pred_labels = output["labels"]
        pred_scores = output["scores"]
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        if len(gt_labels):
            ground_truth_by_class += torch.bincount(gt_labels, minlength=num_classes + 1)[
                : num_classes + 1
            ]
        matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
        matched_predictions = torch.zeros(len(pred_boxes), dtype=torch.bool)
        ious = box_iou(pred_boxes, gt_boxes)

        for pred_index in torch.argsort(pred_scores, descending=True).tolist():
            available = (gt_labels == pred_labels[pred_index]) & ~matched_gt
            if not torch.any(available):
                continue
            candidate_ious = ious[pred_index].masked_fill(~available, -1.0)
            best_iou, best_gt = torch.max(candidate_ious, dim=0)
            if float(best_iou) >= iou_threshold:
                matched_predictions[pred_index] = True
                matched_gt[int(best_gt)] = True

        scores.append(pred_scores)
        labels.append(pred_labels)
        true_positives.append(matched_predictions)

    return _MatchRecords(
        scores=torch.cat(scores) if scores else torch.empty(0, dtype=torch.float32),
        labels=torch.cat(labels) if labels else torch.empty(0, dtype=torch.int64),
        true_positives=(
            torch.cat(true_positives) if true_positives else torch.empty(0, dtype=torch.bool)
        ),
        ground_truth_by_class=ground_truth_by_class,
    )


def _counts_from_records(
    records: _MatchRecords,
    score_threshold: float,
    class_id: int | None = None,
) -> tuple[int, int, int]:
    keep = records.scores >= score_threshold
    if class_id is not None:
        keep &= records.labels == class_id
        ground_truth = int(records.ground_truth_by_class[class_id])
    else:
        ground_truth = int(records.ground_truth_by_class[1:].sum())
    true_positives = int((records.true_positives & keep).sum())
    false_positives = int((~records.true_positives & keep).sum())
    false_negatives = ground_truth - true_positives
    return true_positives, false_positives, false_negatives


def _match_counts(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    score_threshold: float,
    iou_threshold: float,
    class_id: int | None = None,
) -> tuple[int, int, int]:
    max_label = max(
        [int(labels.max()) for item in targets for labels in [item["labels"]] if len(labels)]
        + [class_id or 0]
    )
    records = _build_match_records(predictions, targets, iou_threshold, max_label)
    return _counts_from_records(records, score_threshold, class_id)


def _average_precision_for_class(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    class_id: int,
    iou_threshold: float,
) -> float | None:
    return _average_precision_sweep_for_class(
        predictions,
        targets,
        class_id,
        [iou_threshold],
    )[0]


def _average_precision_sweep_for_class(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    class_id: int,
    iou_thresholds: Sequence[float],
) -> list[float | None]:
    ground_truth_counts: dict[int, int] = {}
    iou_matrices: dict[int, torch.Tensor] = {}
    prediction_rows: list[tuple[float, int, int]] = []
    total_ground_truth = 0

    for image_index, (output, target) in enumerate(zip(predictions, targets)):
        gt_boxes = target["boxes"][target["labels"] == class_id]
        pred_keep = output["labels"] == class_id
        pred_boxes = output["boxes"][pred_keep]
        pred_scores = output["scores"][pred_keep]
        ground_truth_counts[image_index] = len(gt_boxes)
        iou_matrices[image_index] = box_iou(pred_boxes, gt_boxes)
        total_ground_truth += len(gt_boxes)
        for local_index, score in enumerate(pred_scores):
            prediction_rows.append((float(score), image_index, local_index))

    if total_ground_truth == 0:
        return [None] * len(iou_thresholds)

    prediction_rows.sort(key=lambda row: row[0], reverse=True)
    average_precisions: list[float | None] = []
    for iou_threshold in iou_thresholds:
        matched = {
            image_index: torch.zeros(count, dtype=torch.bool)
            for image_index, count in ground_truth_counts.items()
        }
        tp = torch.zeros(len(prediction_rows), dtype=torch.float64)
        fp = torch.zeros(len(prediction_rows), dtype=torch.float64)

        for prediction_index, (_, image_index, local_index) in enumerate(prediction_rows):
            available = ~matched[image_index]
            if not torch.any(available):
                fp[prediction_index] = 1.0
                continue
            ious = iou_matrices[image_index][local_index].masked_fill(~available, -1.0)
            best_iou, best_position = torch.max(ious, dim=0)
            if float(best_iou) >= iou_threshold:
                tp[prediction_index] = 1.0
                matched[image_index][int(best_position)] = True
            else:
                fp[prediction_index] = 1.0

        cumulative_tp = torch.cumsum(tp, dim=0)
        cumulative_fp = torch.cumsum(fp, dim=0)
        recalls = cumulative_tp / total_ground_truth
        precisions = cumulative_tp / torch.clamp(cumulative_tp + cumulative_fp, min=1e-12)
        recall_points = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
        interpolated = [
            torch.max(precisions[recalls >= recall_point]).item()
            if torch.any(recalls >= recall_point)
            else 0.0
            for recall_point in recall_points
        ]
        average_precisions.append(float(sum(interpolated) / len(interpolated)))
    return average_precisions


def evaluate_predictions(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    score_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    class_names: Sequence[str] | None = None,
    compute_map: bool = True,
    _records: _MatchRecords | None = None,
) -> DetectionMetrics:
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must contain the same number of images.")
    max_label = max(
        [int(labels.max()) for item in targets for labels in [item["labels"]] if len(labels)] + [0]
    )
    class_ids = (
        list(range(1, len(class_names) + 1))
        if class_names
        else list(range(1, max_label + 1))
    )
    num_classes = max(max_label, len(class_names) if class_names else 0)
    records = _records or _build_match_records(
        predictions,
        targets,
        iou_threshold,
        num_classes,
    )

    tp, fp, fn = _counts_from_records(records, score_threshold)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)

    per_class: dict[str, dict[str, float]] = {}
    ap50_values = []
    ap5095_values = []
    iou_sweep = [0.50 + 0.05 * index for index in range(10)]
    for class_id in class_ids:
        name = class_names[class_id - 1] if class_names else str(class_id)
        class_tp, class_fp, class_fn = _counts_from_records(records, score_threshold, class_id)
        class_precision = class_tp / max(1, class_tp + class_fp)
        class_recall = class_tp / max(1, class_tp + class_fn)
        class_f1 = 2.0 * class_precision * class_recall / max(1e-12, class_precision + class_recall)
        class_aps = (
            _average_precision_sweep_for_class(
                predictions,
                targets,
                class_id,
                iou_sweep,
            )
            if compute_map
            else []
        )
        valid_aps = [value for value in class_aps if value is not None]
        if valid_aps:
            ap50_values.append(valid_aps[0])
            ap5095_values.append(sum(valid_aps) / len(valid_aps))
        per_class[name] = {
            "precision": class_precision,
            "recall": class_recall,
            "f1": class_f1,
            "ap50": valid_aps[0] if valid_aps else float("nan"),
            "map50_95": sum(valid_aps) / len(valid_aps) if valid_aps else float("nan"),
            "ground_truth": float(class_tp + class_fn),
        }

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        map50=sum(ap50_values) / len(ap50_values) if ap50_values else float("nan"),
        map50_95=sum(ap5095_values) / len(ap5095_values) if ap5095_values else float("nan"),
        score_threshold=score_threshold,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        per_class=per_class,
    )


def evaluate_operating_points(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    thresholds: Sequence[float],
    iou_threshold: float = 0.50,
    class_names: Sequence[str] | None = None,
) -> list[DetectionMetrics]:
    max_label = max(
        [int(labels.max()) for item in targets for labels in [item["labels"]] if len(labels)] + [0]
    )
    num_classes = max(max_label, len(class_names) if class_names else 0)
    records = _build_match_records(predictions, targets, iou_threshold, num_classes)
    return [
        evaluate_predictions(
            predictions,
            targets,
            score_threshold=threshold,
            iou_threshold=iou_threshold,
            class_names=class_names,
            compute_map=False,
            _records=records,
        )
        for threshold in thresholds
    ]


def tune_score_threshold(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    thresholds: Sequence[float] | None = None,
    class_names: Sequence[str] | None = None,
    compute_map: bool = True,
) -> DetectionMetrics:
    thresholds = thresholds or [0.10 + 0.05 * index for index in range(13)]
    candidates = evaluate_operating_points(
        predictions,
        targets,
        thresholds,
        class_names=class_names,
    )
    best = max(candidates, key=lambda metrics: (metrics.f1, metrics.precision))
    if not compute_map:
        return best
    return evaluate_predictions(
        predictions,
        targets,
        score_threshold=best.score_threshold,
        class_names=class_names,
        compute_map=True,
    )


def evaluate_simple(
    model,
    loader,
    device,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.50,
) -> DetectionMetrics:
    predictions, targets = collect_predictions(model, loader, device)
    return evaluate_predictions(
        predictions,
        targets,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
