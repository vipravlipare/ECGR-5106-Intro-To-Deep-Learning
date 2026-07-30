from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RareClassBalancedSampler(Sampler[int]):
    """Draw a short, reproducible epoch without ignoring uncommon object classes."""

    def __init__(
        self,
        dataset: "YoloDetectionDataset",
        num_samples: int,
        num_classes: int,
        seed: int = 42,
        rarity_exponent: float = 0.5,
        max_weight: float = 4.0,
        candidate_indices: Sequence[int] | None = None,
        cache_path: Path | None = None,
        scan_workers: int = 8,
    ) -> None:
        self.candidate_indices = (
            list(range(len(dataset)))
            if candidate_indices is None
            else [int(index) for index in candidate_indices]
        )
        if not self.candidate_indices:
            raise ValueError("candidate_indices must not be empty.")
        if min(self.candidate_indices) < 0 or max(self.candidate_indices) >= len(dataset):
            raise ValueError("candidate_indices contains an index outside the dataset.")
        if len(set(self.candidate_indices)) != len(self.candidate_indices):
            raise ValueError("candidate_indices must be unique.")
        if not 0 < num_samples <= len(self.candidate_indices):
            raise ValueError("num_samples must be between 1 and the candidate pool length.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if not 0.0 <= rarity_exponent <= 1.0:
            raise ValueError("rarity_exponent must be between 0 and 1.")
        if scan_workers < 1:
            raise ValueError("scan_workers must be positive.")

        self.dataset = dataset
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.seed = seed
        self.epoch = 0

        def read_image_classes(dataset_index: int) -> set[int]:
            image_path = dataset.images[dataset_index]
            label_path = dataset.root / "labels" / dataset.split / f"{image_path.stem}.txt"
            class_ids: set[int] = set()
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if not fields:
                        continue
                    class_id = int(float(fields[0]))
                    if 0 <= class_id < num_classes:
                        class_ids.add(class_id)
            return class_ids

        cache_path = Path(cache_path) if cache_path is not None else None
        cache_key = {
            "root": str(dataset.root.resolve()),
            "split": dataset.split,
            "dataset_size": len(dataset),
            "num_classes": num_classes,
            "candidate_indices": self.candidate_indices,
        }
        cached_classes = None
        if cache_path is not None and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("key") == cache_key:
                cached_classes = [set(class_ids) for class_ids in cached["image_classes"]]

        if cached_classes is None:
            with ThreadPoolExecutor(max_workers=scan_workers) as executor:
                self.image_classes = list(executor.map(read_image_classes, self.candidate_indices))
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(
                        {
                            "key": cache_key,
                            "image_classes": [sorted(class_ids) for class_ids in self.image_classes],
                        }
                    ),
                    encoding="utf-8",
                )
        else:
            self.image_classes = cached_classes

        class_image_counts = torch.zeros(num_classes, dtype=torch.float64)
        for class_ids in self.image_classes:
            for class_id in class_ids:
                class_image_counts[class_id] += 1

        nonzero_counts = class_image_counts[class_image_counts > 0]
        reference_count = float(nonzero_counts.max()) if len(nonzero_counts) else 1.0
        class_weights = torch.ones(num_classes, dtype=torch.float64)
        present = class_image_counts > 0
        class_weights[present] = (reference_count / class_image_counts[present]).pow(rarity_exponent)
        class_weights.clamp_(max=max_weight)
        self.class_image_counts = class_image_counts
        self.class_weights = class_weights
        self.weights = torch.tensor(
            [
                max(
                    (float(class_weights[class_id]) for class_id in classes),
                    default=1.0,
                )
                for classes in self.image_classes
            ],
            dtype=torch.float64,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def sample_indices(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        positions = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=False,
            generator=generator,
        ).tolist()
        return [self.candidate_indices[position] for position in positions]

    def __iter__(self):
        return iter(self.sample_indices())

    def __len__(self) -> int:
        return self.num_samples


class YoloDetectionDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        max_size: int = 768,
        augment: bool = False,
        horizontal_flip_probability: float = 0.5,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.max_size = max_size
        self.augment = augment
        self.horizontal_flip_probability = horizontal_flip_probability
        self.color_jitter = ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.02)
        image_dir = self.root / "images" / split
        self.images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.images:
            raise FileNotFoundError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        label_path = self.root / "labels" / self.split / f"{image_path.stem}.txt"
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        scale = min(1.0, self.max_size / max(width, height))
        if scale < 1.0:
            new_size = (int(height * scale), int(width * scale))
            image = F.resize(image, new_size, antialias=True)

        new_width, new_height = image.size
        boxes = []
        labels = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id, xc, yc, bw, bh = map(float, line.split()[:5])
                x1 = (xc - bw / 2.0) * new_width
                y1 = (yc - bh / 2.0) * new_height
                x2 = (xc + bw / 2.0) * new_width
                y2 = (yc + bh / 2.0) * new_height
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(class_id) + 1)

        if self.augment:
            image = self.color_jitter(image)
            if torch.rand(1).item() < self.horizontal_flip_probability:
                image = F.hflip(image)
                boxes = [[new_width - x2, y1, new_width - x1, y2] for x1, y1, x2, y2 in boxes]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
            target["boxes"][:, 2] - target["boxes"][:, 0]
        )
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)
        return F.to_tensor(image), target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
