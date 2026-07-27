from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
