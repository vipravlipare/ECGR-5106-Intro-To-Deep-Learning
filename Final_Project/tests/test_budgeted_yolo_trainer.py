from __future__ import annotations

import numpy as np

from road_detection.budgeted_yolo_trainer import (
    RotatingClassBalancedSampler,
)


class FakeDetectionDataset:
    data = {"nc": 3}
    labels = [
        {"cls": np.array([[0], [1]])},
        {"cls": np.array([[0]])},
        {"cls": np.array([[0]])},
        {"cls": np.array([[2]])},
    ]

    def __len__(self) -> int:
        return len(self.labels)


def test_sampler_rotates_without_replacement_and_can_resume() -> None:
    sampler = RotatingClassBalancedSampler(
        FakeDetectionDataset(),
        samples_per_epoch=3,
        balance_power=0.5,
        seed=7,
    )

    epoch_zero = list(sampler)
    epoch_one = list(sampler)
    sampler.set_epoch(0)

    assert len(epoch_zero) == len(set(epoch_zero)) == 3
    assert epoch_zero != epoch_one
    assert list(sampler) == epoch_zero
    assert sampler.class_image_counts.tolist() == [3, 1, 1]
    assert sampler.weights[0] > sampler.weights[1]
    assert sampler.weights[3] > sampler.weights[2]
