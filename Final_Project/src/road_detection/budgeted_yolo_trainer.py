from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import torch_distributed_zero_first


class RotatingClassBalancedSampler(Sampler[int]):
    """Sample a fresh, rare-class-aware image budget without replacement each epoch."""

    def __init__(
        self,
        dataset: Any,
        samples_per_epoch: int,
        balance_power: float = 0.35,
        seed: int = 42,
    ) -> None:
        dataset_size = len(dataset)
        if not 1 <= samples_per_epoch <= dataset_size:
            raise ValueError(
                f"samples_per_epoch must be between 1 and {dataset_size}, "
                f"received {samples_per_epoch}"
            )
        if not 0.0 <= balance_power <= 1.0:
            raise ValueError("balance_power must be between 0 and 1")

        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        self.weights, self.class_image_counts = self._image_weights(
            dataset, balance_power
        )

    @staticmethod
    def _image_weights(
        dataset: Any, balance_power: float
    ) -> tuple[torch.Tensor, np.ndarray]:
        class_count = int(dataset.data["nc"])
        class_image_counts = np.zeros(class_count, dtype=np.int64)
        image_classes: list[np.ndarray] = []

        for label in dataset.labels:
            classes = np.unique(label["cls"].astype(np.int64).reshape(-1))
            classes = classes[(classes >= 0) & (classes < class_count)]
            image_classes.append(classes)
            class_image_counts[classes] += 1

        largest_count = max(1, int(class_image_counts.max()))
        class_weights = np.ones(class_count, dtype=np.float64)
        present = class_image_counts > 0
        class_weights[present] = (
            largest_count / class_image_counts[present]
        ) ** balance_power

        image_weights = np.ones(len(image_classes), dtype=np.float64)
        for index, classes in enumerate(image_classes):
            if classes.size:
                image_weights[index] = float(class_weights[classes].max())
        return torch.as_tensor(image_weights, dtype=torch.double), class_image_counts

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.samples_per_epoch,
            replacement=False,
            generator=generator,
        )
        self.epoch += 1
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def make_budgeted_detection_trainer(
    samples_per_epoch: int,
    balance_power: float = 0.35,
    sampler_seed: int = 42,
) -> type[DetectionTrainer]:
    """Create a single-GPU trainer with short epochs over rotating full-data samples."""

    class BudgetedDetectionTrainer(DetectionTrainer):
        def get_dataloader(
            self,
            dataset_path: str,
            batch_size: int = 16,
            rank: int = 0,
            mode: str = "train",
        ) -> InfiniteDataLoader:
            if mode != "train":
                return super().get_dataloader(
                    dataset_path, batch_size=batch_size, rank=rank, mode=mode
                )
            if rank != -1:
                raise RuntimeError(
                    "BudgetedDetectionTrainer currently supports one GPU or CPU."
                )

            with torch_distributed_zero_first(rank):
                dataset = self.build_dataset(dataset_path, mode, batch_size)
            sampler = RotatingClassBalancedSampler(
                dataset,
                samples_per_epoch=min(samples_per_epoch, len(dataset)),
                balance_power=balance_power,
                seed=sampler_seed,
            )
            self.budget_sampler = sampler

            batch_size = min(batch_size, len(sampler))
            batches = math.ceil(len(sampler) / batch_size)
            workers = min(os.cpu_count() or 1, self.args.workers, batches)
            generator = torch.Generator()
            generator.manual_seed(6148914691236517205 + sampler_seed)

            return InfiniteDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=workers,
                prefetch_factor=4 if workers > 0 else None,
                pin_memory=self.device.type == "cuda",
                collate_fn=getattr(dataset, "collate_fn", None),
                worker_init_fn=seed_worker,
                generator=generator,
                drop_last=self.args.compile,
            )

        def _setup_train(self, *args, **kwargs) -> None:
            super()._setup_train(*args, **kwargs)
            if hasattr(self, "budget_sampler"):
                self.budget_sampler.set_epoch(self.start_epoch)
                counts = self.budget_sampler.class_image_counts.tolist()
                LOGGER.info(
                    colorstr("train: ")
                    + f"rotating {len(self.budget_sampler):,} of "
                    f"{len(self.train_loader.dataset):,} images per epoch; "
                    f"class image counts={counts}"
                )

    BudgetedDetectionTrainer.__name__ = "BudgetedDetectionTrainer"
    return BudgetedDetectionTrainer
