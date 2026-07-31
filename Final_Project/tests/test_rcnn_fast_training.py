from pathlib import Path
from types import SimpleNamespace

import math
from PIL import Image
import torch

from road_detection.rcnn_dataset import RareClassBalancedSampler, YoloDetectionDataset
from road_detection.rcnn_metrics import evaluate_predictions, tune_score_threshold
from road_detection.rcnn_model import FasterRCNNPerformanceConfig, configure_faster_rcnn_performance


def _fake_detector():
    sampler = lambda: SimpleNamespace(batch_size_per_image=0)
    return SimpleNamespace(
        rpn=SimpleNamespace(
            _pre_nms_top_n={},
            _post_nms_top_n={},
            fg_bg_sampler=sampler(),
        ),
        roi_heads=SimpleNamespace(
            fg_bg_sampler=sampler(),
            detections_per_img=0,
        ),
    )


def test_fast_config_sets_every_proposal_limit():
    model = _fake_detector()
    config = FasterRCNNPerformanceConfig()

    configure_faster_rcnn_performance(model, config)

    assert model.rpn._pre_nms_top_n == {"training": 1000, "testing": 600}
    assert model.rpn._post_nms_top_n == {"training": 512, "testing": 300}
    assert model.rpn.fg_bg_sampler.batch_size_per_image == 256
    assert model.roi_heads.fg_bg_sampler.batch_size_per_image == 384
    assert model.roi_heads.detections_per_img == 100


def test_rare_class_sampler_is_reproducible_and_rotates(tmp_path: Path):
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index in range(6):
        Image.new("RGB", (16, 16)).save(image_dir / f"{index}.jpg")
        class_id = 1 if index == 5 else 0
        (label_dir / f"{index}.txt").write_text(
            f"{class_id} 0.5 0.5 0.5 0.5\n",
            encoding="utf-8",
        )

    dataset = YoloDetectionDataset(tmp_path, "train")
    sampler = RareClassBalancedSampler(dataset, num_samples=3, num_classes=2, seed=7)
    first = sampler.sample_indices()
    sampler.set_epoch(1)
    second = sampler.sample_indices()
    sampler.set_epoch(0)

    assert sampler.sample_indices() == first
    assert second != first
    assert sampler.class_weights[1] > sampler.class_weights[0]
    assert len(set(first)) == len(first)


def test_sampler_boosts_small_objects_and_dataset_can_preload(tmp_path: Path):
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index in range(2):
        Image.new("RGB", (32, 16), color=(index * 20, 0, 0)).save(
            image_dir / f"{index}.jpg"
        )
    (label_dir / "0.txt").write_text(
        "4 0.5 0.5 0.01 0.01\n4 0.6 0.5 0.01 0.01\n",
        encoding="utf-8",
    )
    (label_dir / "1.txt").write_text(
        "4 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    dataset = YoloDetectionDataset(tmp_path, "train")
    sampler = RareClassBalancedSampler(
        dataset,
        num_samples=1,
        num_classes=6,
        small_object_boost=1.0,
    )
    dataset.cache_samples([0], workers=1)
    image, target = dataset[0]

    assert sampler.small_object_counts == [2, 0]
    assert sampler.weights[0] > sampler.weights[1]
    assert 0 in dataset._image_bytes
    assert image.shape == (3, 16, 32)
    assert target["boxes"].shape == (2, 4)


def test_threshold_tuning_reuses_matches_and_full_map_is_available():
    predictions = [
        {
            "boxes": torch.tensor(
                [[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([1, 1, 2]),
            "scores": torch.tensor([0.9, 0.8, 0.7]),
        },
        {
            "boxes": torch.tensor([[40, 40, 50, 50]], dtype=torch.float32),
            "labels": torch.tensor([2]),
            "scores": torch.tensor([0.6]),
        },
    ]
    targets = [
        {
            "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
            "labels": torch.tensor([1]),
        },
        {
            "boxes": torch.tensor([[40, 40, 50, 50]], dtype=torch.float32),
            "labels": torch.tensor([2]),
        },
    ]

    quick = tune_score_threshold(
        predictions,
        targets,
        thresholds=[0.55, 0.65, 0.75],
        class_names=["one", "two"],
        compute_map=False,
    )
    full = evaluate_predictions(
        predictions,
        targets,
        score_threshold=quick.score_threshold,
        class_names=["one", "two"],
    )

    assert quick.score_threshold == 0.55
    assert math.isnan(quick.map50)
    assert full.f1 == quick.f1
    assert full.map50 == 0.75
