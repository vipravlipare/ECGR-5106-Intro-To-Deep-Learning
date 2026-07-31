from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml


@dataclass(frozen=True)
class IndexedImage:
    path: Path
    classes: frozenset[int]
    small_object_count: int = 0
    box_count: int = 0


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


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_positions = [
        index for index, value in enumerate(parts) if value.lower() == "images"
    ]
    if not image_positions:
        raise ValueError(f"Image path does not contain an 'images' directory: {image_path}")
    parts[image_positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def stage_images_and_labels(
    images: Iterable[Path],
    cache_root: Path,
    split: str,
    workers: int = 12,
) -> list[Path]:
    """Copy selected data to local storage in parallel and reuse completed files."""
    source_images = [Path(path) for path in images]
    image_dir = Path(cache_root) / "images" / split
    label_dir = Path(cache_root) / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    def copy_if_needed(source: Path, destination: Path) -> None:
        if destination.exists() and destination.stat().st_size == source.stat().st_size:
            return
        temporary = destination.with_suffix(destination.suffix + ".part")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)

    def stage_one(source_image: Path) -> Path:
        destination_image = image_dir / source_image.name
        source_label = label_path_for_image(source_image)
        destination_label = label_dir / f"{source_image.stem}.txt"
        copy_if_needed(source_image, destination_image)
        if source_label.exists():
            copy_if_needed(source_label, destination_label)
        elif not destination_label.exists():
            destination_label.write_text("", encoding="utf-8")
        return destination_image

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        return list(executor.map(stage_one, source_images))


def stage_fast_data_files(
    data_files: FastDataFiles,
    cache_root: Path,
    source_data_yaml: Path,
    workers: int = 12,
) -> FastDataFiles:
    """Stage compact YOLO manifests away from cloud-synced folders."""
    main_images = read_image_manifest(data_files.main_manifest)
    refine_images = read_image_manifest(data_files.refine_manifest)
    validation_images = read_image_manifest(data_files.validation_manifest)
    staged_main = stage_images_and_labels(main_images, cache_root, "train", workers)
    staged_refine = stage_images_and_labels(refine_images, cache_root, "train", workers)
    staged_validation = stage_images_and_labels(
        validation_images,
        cache_root,
        "val",
        workers,
    )

    manifest_dir = Path(cache_root) / "manifests"
    main_manifest = write_image_manifest(manifest_dir / "main.txt", staged_main)
    refine_manifest = write_image_manifest(manifest_dir / "refine.txt", staged_refine)
    validation_manifest = write_image_manifest(
        manifest_dir / "epoch_val.txt",
        staged_validation,
    )
    source_config = yaml.safe_load(source_data_yaml.read_text(encoding="utf-8"))
    main_yaml = _write_variant_yaml(
        manifest_dir / "main.yaml",
        source_config,
        Path(cache_root),
        main_manifest,
        validation_manifest,
    )
    refine_yaml = _write_variant_yaml(
        manifest_dir / "refine.yaml",
        source_config,
        Path(cache_root),
        refine_manifest,
        validation_manifest,
    )
    return FastDataFiles(
        main_manifest=main_manifest,
        refine_manifest=refine_manifest,
        validation_manifest=validation_manifest,
        main_yaml=main_yaml,
        refine_yaml=refine_yaml,
        source=f"{data_files.source}; staged in local cache {Path(cache_root)}",
    )


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
                small_object_count = int(payload.get("small_object_count", 0))
                box_count = int(payload.get("box_count", 0))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid record at {index_path}:{line_number}"
                ) from exc
            records.append(IndexedImage(image, classes, small_object_count, box_count))
    if not records:
        raise ValueError(f"Dataset index is empty: {index_path}")
    return records


def weighted_sample(
    records: list[IndexedImage],
    count: int,
    class_count: int,
    exponent: float,
    seed: int,
    small_object_boost: float = 0.0,
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
        bounded_small_count = min(9.0, float(record.small_object_count))
        weight *= 1.0 + small_object_boost * (bounded_small_count**0.5 / 3.0)
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


def _find_label_dir(dataset_root: Path, image_dir: Path) -> Path:
    if image_dir.parent.name == "images":
        return image_dir.parent.parent / "labels" / image_dir.name
    return dataset_root / "labels" / image_dir.name


def build_candidate_index(
    image_dir: Path,
    label_dir: Path,
    index_path: Path,
    candidate_count: int,
    seed: int,
    small_object_class_ids: Sequence[int] = (3, 4, 5),
    small_object_area_threshold: float = 0.0025,
    scan_workers: int = 12,
) -> list[IndexedImage]:
    """Build a reusable class and object-size index without scanning all 70k images."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in image_extensions
    )
    if not images:
        raise FileNotFoundError(f"No training images found in {image_dir}")
    selected = uniform_sample(images, min(candidate_count, len(images)), seed)
    selected.sort()
    small_object_class_ids = frozenset(int(value) for value in small_object_class_ids)

    def read_record(image_path: Path) -> IndexedImage:
        label_path = label_dir / f"{image_path.stem}.txt"
        classes: set[int] = set()
        small_object_count = 0
        box_count = 0
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) < 5:
                    continue
                class_id = int(float(fields[0]))
                classes.add(class_id)
                box_count += 1
                if (
                    class_id in small_object_class_ids
                    and float(fields[3]) * float(fields[4]) < small_object_area_threshold
                ):
                    small_object_count += 1
        return IndexedImage(
            image_path,
            frozenset(classes),
            small_object_count,
            box_count,
        )

    with ThreadPoolExecutor(max_workers=max(1, scan_workers)) as executor:
        records = list(executor.map(read_record, selected))

    index_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "image": record.path.name,
                "classes": sorted(record.classes),
                "small_object_count": record.small_object_count,
                "box_count": record.box_count,
            }
        )
        for record in records
    ]
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


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
    candidate_scan_count: int = 12000,
    scan_workers: int = 12,
) -> FastDataFiles:
    config, dataset_root, class_names = load_dataset_config(data_yaml)
    train_dir = _resolve_split_path(dataset_root, config["train"])
    validation_dir = _resolve_split_path(dataset_root, config["val"])
    if not train_dir.is_dir() or not validation_dir.is_dir():
        raise FileNotFoundError(
            f"Expected image directories at {train_dir} and {validation_dir}"
        )

    index_path = dataset_root / "index" / "train.jsonl"
    candidate_index_path = output_dir / "candidate_index.jsonl"

    if index_path.exists():
        records = load_split_index(index_path, train_dir)
        source = f"class-aware dataset index {index_path}"
    elif candidate_index_path.exists():
        records = load_split_index(candidate_index_path, train_dir)
        if len(records) < max(main_count, refine_count):
            records = build_candidate_index(
                train_dir,
                _find_label_dir(dataset_root, train_dir),
                candidate_index_path,
                candidate_count=max(candidate_scan_count, main_count, refine_count),
                seed=seed,
                scan_workers=scan_workers,
            )
            source = (
                f"expanded small-object-aware candidate index {candidate_index_path} "
                f"({len(records)} scenes)"
            )
        else:
            source = (
                f"reused small-object-aware candidate index {candidate_index_path}"
            )
    else:
        records = build_candidate_index(
            train_dir,
            _find_label_dir(dataset_root, train_dir),
            candidate_index_path,
            candidate_count=max(candidate_scan_count, main_count, refine_count),
            seed=seed,
            scan_workers=scan_workers,
        )
        source = (
            f"new small-object-aware candidate index {candidate_index_path} "
            f"({len(records)} scenes)"
        )

    main_images = weighted_sample(
        records,
        main_count,
        len(class_names),
        exponent=0.35,
        seed=seed,
        small_object_boost=0.60,
    )
    refine_images = weighted_sample(
        records,
        refine_count,
        len(class_names),
        exponent=0.60,
        seed=seed + 1,
        small_object_boost=1.00,
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
