from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class IndexedImage:
    path: Path
    classes: frozenset[int]


@dataclass(frozen=True)
class FastDataFiles:
    main_manifest: Path
    refine_manifest: Path
    validation_manifest: Path
    main_yaml: Path
    refine_yaml: Path
    source: str


def load_dataset_config(data_yaml: Path) -> tuple[dict, Path, list[str]]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    names = config["names"]
    class_names = (
        [str(names[index]) for index in sorted(names)]
        if isinstance(names, dict)
        else [str(name) for name in names]
    )
    return config, root, class_names


def latest_best_checkpoint(
    run_root: Path,
    fallback: Path | None = None,
    exclude_run_names: Iterable[str] = (),
) -> Path:
    excluded = set(exclude_run_names)
    candidates = [
        path
        for path in run_root.glob("*/weights/best.pt")
        if path.parent.parent.name not in excluded
    ]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)
    if fallback is not None and fallback.exists():
        return fallback
    raise FileNotFoundError(f"No best.pt checkpoint found under {run_root}")


def read_image_manifest(path: Path) -> list[Path]:
    images = [
        Path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not images:
        raise ValueError(f"Image manifest is empty: {path}")
    return images


def write_image_manifest(path: Path, images: Iterable[Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = [image.resolve().as_posix() for image in images]
    path.write_text("\n".join(resolved) + "\n", encoding="utf-8")
    return path


def load_split_index(index_path: Path, image_dir: Path) -> list[IndexedImage]:
    records = []
    with index_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                image = image_dir / payload["image"]
                classes = frozenset(int(value) for value in payload["classes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid record at {index_path}:{line_number}"
                ) from exc
            records.append(IndexedImage(image, classes))
    if not records:
        raise ValueError(f"Dataset index is empty: {index_path}")
    return records


def weighted_sample(
    records: list[IndexedImage],
    count: int,
    class_count: int,
    exponent: float,
    seed: int,
) -> list[Path]:
    if count >= len(records):
        return [record.path for record in records]
    image_counts = Counter(
        class_id for record in records for class_id in record.classes
    )
    largest = max(image_counts.values(), default=1)
    class_weights = {
        class_id: (largest / max(1, image_counts[class_id])) ** exponent
        for class_id in range(class_count)
    }
    rng = random.Random(seed)
    weighted = []
    for record in records:
        weight = max(
            (class_weights[class_id] for class_id in record.classes),
            default=1.0,
        )
        weighted.append((rng.random() ** (1.0 / weight), record.path))
    weighted.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in weighted[:count]]


def uniform_sample(images: list[Path], count: int, seed: int) -> list[Path]:
    if count >= len(images):
        return list(images)
    return random.Random(seed).sample(images, count)


def extend_sample(
    preferred: list[Path],
    all_images: list[Path],
    count: int,
    seed: int,
) -> list[Path]:
    preferred = list(dict.fromkeys(preferred))
    if len(preferred) >= count:
        return uniform_sample(preferred, count, seed)
    preferred_set = set(preferred)
    available = [image for image in all_images if image not in preferred_set]
    needed = count - len(preferred)
    if len(available) < needed:
        raise ValueError(
            f"Requested {count} images, but only {len(preferred) + len(available)} are available"
        )
    selected = preferred + uniform_sample(available, needed, seed)
    random.Random(seed + 1000).shuffle(selected)
    return selected


def _resolve_split_path(dataset_root: Path, split_value: str) -> Path:
    path = Path(split_value)
    return path if path.is_absolute() else dataset_root / path


def _write_variant_yaml(
    path: Path,
    source_config: dict,
    dataset_root: Path,
    train_manifest: Path,
    validation_manifest: Path,
) -> Path:
    variant = dict(source_config)
    variant["path"] = dataset_root.resolve().as_posix()
    variant["train"] = train_manifest.resolve().as_posix()
    variant["val"] = validation_manifest.resolve().as_posix()
    path.write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
    return path


def prepare_fast_data_files(
    data_yaml: Path,
    output_dir: Path,
    run_root: Path,
    preferred_manifest_tag: str,
    main_count: int,
    refine_count: int,
    validation_count: int,
    seed: int = 42,
) -> FastDataFiles:
    config, dataset_root, class_names = load_dataset_config(data_yaml)
    train_dir = _resolve_split_path(dataset_root, config["train"])
    validation_dir = _resolve_split_path(dataset_root, config["val"])
    if not train_dir.is_dir() or not validation_dir.is_dir():
        raise FileNotFoundError(
            f"Expected image directories at {train_dir} and {validation_dir}"
        )

    source_dir = run_root / "manifests" / preferred_manifest_tag
    source_main = source_dir / "main.txt"
    source_refine = source_dir / "refine.txt"
    index_path = dataset_root / "index" / "train.jsonl"

    prior_main = read_image_manifest(source_main) if source_main.exists() else []
    prior_refine = (
        read_image_manifest(source_refine) if source_refine.exists() else []
    )
    if len(prior_main) >= main_count and len(prior_refine) >= refine_count:
        main_images = uniform_sample(prior_main, main_count, seed)
        refine_images = uniform_sample(prior_refine, refine_count, seed + 1)
        source = f"reused class-aware manifests from {source_dir}"
    elif index_path.exists():
        records = load_split_index(index_path, train_dir)
        main_images = weighted_sample(
            records, main_count, len(class_names), exponent=0.25, seed=seed
        )
        refine_images = weighted_sample(
            records,
            refine_count,
            len(class_names),
            exponent=0.65,
            seed=seed + 1,
        )
        source = f"class-aware dataset index {index_path}"
    elif prior_main or prior_refine:
        train_images = sorted(train_dir.glob("*.jpg"))
        if not train_images:
            raise FileNotFoundError(f"No training images found in {train_dir}")
        main_images = extend_sample(
            prior_main, train_images, main_count, seed
        )
        refine_images = extend_sample(
            prior_refine, train_images, refine_count, seed + 1
        )
        source = (
            f"expanded prior class-aware manifests from {source_dir} with "
            "deterministic fresh scenes"
        )
    else:
        train_images = sorted(train_dir.glob("*.jpg"))
        if not train_images:
            raise FileNotFoundError(f"No training images found in {train_dir}")
        main_images = uniform_sample(train_images, main_count, seed)
        refine_images = uniform_sample(train_images, refine_count, seed + 1)
        source = (
            "uniform image sampling because no prior manifest or conversion-time "
            "class index was available"
        )

    validation_images = sorted(validation_dir.glob("*.jpg"))
    if not validation_images:
        raise FileNotFoundError(f"No validation images found in {validation_dir}")
    validation_images = uniform_sample(
        validation_images, validation_count, seed + 2
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    main_manifest = write_image_manifest(output_dir / "main.txt", main_images)
    refine_manifest = write_image_manifest(
        output_dir / "refine.txt", refine_images
    )
    validation_manifest = write_image_manifest(
        output_dir / "epoch_val.txt", validation_images
    )
    main_yaml = _write_variant_yaml(
        output_dir / "main.yaml",
        config,
        dataset_root,
        main_manifest,
        validation_manifest,
    )
    refine_yaml = _write_variant_yaml(
        output_dir / "refine.yaml",
        config,
        dataset_root,
        refine_manifest,
        validation_manifest,
    )
    return FastDataFiles(
        main_manifest=main_manifest,
        refine_manifest=refine_manifest,
        validation_manifest=validation_manifest,
        main_yaml=main_yaml,
        refine_yaml=refine_yaml,
        source=source,
    )
