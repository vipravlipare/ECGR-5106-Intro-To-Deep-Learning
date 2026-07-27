from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from road_detection.constants import PROJECT_CLASSES


RCNN_VARIANTS = ("mobilenet", "resnet50")
COCO_CLASS_ALIASES = {
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "pedestrian": "person",
    "traffic light": "traffic light",
    "traffic sign": "stop sign",
}


def _transfer_coco_predictor_weights(model, new_predictor, categories: list[str], class_names: list[str]) -> None:
    old_predictor = model.roi_heads.box_predictor
    with torch.no_grad():
        new_predictor.cls_score.weight[0].copy_(old_predictor.cls_score.weight[0])
        new_predictor.cls_score.bias[0].copy_(old_predictor.cls_score.bias[0])
        new_predictor.bbox_pred.weight[:4].copy_(old_predictor.bbox_pred.weight[:4])
        new_predictor.bbox_pred.bias[:4].copy_(old_predictor.bbox_pred.bias[:4])
        for project_index, project_name in enumerate(class_names, start=1):
            coco_name = COCO_CLASS_ALIASES.get(project_name)
            if coco_name not in categories:
                continue
            coco_index = categories.index(coco_name)
            new_predictor.cls_score.weight[project_index].copy_(old_predictor.cls_score.weight[coco_index])
            new_predictor.cls_score.bias[project_index].copy_(old_predictor.cls_score.bias[coco_index])
            new_slice = slice(project_index * 4, project_index * 4 + 4)
            coco_slice = slice(coco_index * 4, coco_index * 4 + 4)
            new_predictor.bbox_pred.weight[new_slice].copy_(old_predictor.bbox_pred.weight[coco_slice])
            new_predictor.bbox_pred.bias[new_slice].copy_(old_predictor.bbox_pred.bias[coco_slice])


def build_faster_rcnn(
    num_classes: int,
    variant: str = "mobilenet",
    min_size: int = 512,
    max_size: int = 768,
    trainable_backbone_layers: int | None = None,
    pretrained: bool = True,
    class_names: list[str] | None = None,
    transfer_coco_head: bool = True,
):
    if variant == "mobilenet":
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights,
            weights_backbone=None,
            min_size=min_size,
            max_size=max_size,
            trainable_backbone_layers=trainable_backbone_layers,
            box_score_thresh=0.05,
        )
    elif variant == "resnet50":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            weights_backbone=None,
            min_size=min_size,
            max_size=max_size,
            trainable_backbone_layers=trainable_backbone_layers,
            box_score_thresh=0.05,
        )
    else:
        raise ValueError(f"Unknown Faster R-CNN variant '{variant}'. Choose from {RCNN_VARIANTS}.")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    new_predictor = FastRCNNPredictor(in_features, num_classes)
    selected_names = class_names or PROJECT_CLASSES
    if pretrained and transfer_coco_head and len(selected_names) + 1 == num_classes and weights is not None:
        _transfer_coco_predictor_weights(
            model,
            new_predictor,
            list(weights.meta["categories"]),
            selected_names,
        )
    model.roi_heads.box_predictor = new_predictor
    return model


def load_faster_rcnn_checkpoint(weights: Path, device: torch.device):
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    classes = metadata.get("classes", PROJECT_CLASSES)
    variant = metadata.get("variant", "resnet50")
    min_size = int(metadata.get("min_size", 512))
    max_size = int(metadata.get("max_size", 960))
    model = build_faster_rcnn(
        num_classes=len(classes) + 1,
        variant=variant,
        min_size=min_size,
        max_size=max_size,
        pretrained=False,
    )
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device).eval()
    return model, classes, metadata
