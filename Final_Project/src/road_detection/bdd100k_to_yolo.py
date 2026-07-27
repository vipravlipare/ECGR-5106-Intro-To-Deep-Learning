from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm

from road_detection.constants import BDD_ALIASES, CLASS_TO_ID, PROJECT_CLASSES


@dataclass(frozen=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BDD100K detection labels to Ultralytics YOLO format."
    )
    parser.add_argument("--bdd-root", type=Path, required=True, help="Root folder containing BDD100K images and labels.")
    parser.add_argument("--output", type=Path, default=Path("data/bdd100k_yolo"), help="Converted YOLO dataset output folder.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="BDD splits to convert.")
    parser.add_argument("--copy-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--min-box-area", type=float, default=16.0, help="Drop boxes smaller than this area in pixels.")
    parser.add_argument("--max-images-per-split", type=int, default=None, help="Small debug subset per split.")
    parser.add_argument("--include-empty", action="store_true", help="Include images with no selected classes.")
    parser.add_argument(
        "--val-test-fraction",
        type=float,
        default=0.20,
        help="Move this fraction of labeled BDD validation frames into a reproducible local test split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for subset selection and the local test holdout.")
    return parser.parse_args()


def find_label_file(root: Path, split: str) -> Path:
    candidates = [
        root / "labels" / "det_20" / f"det_{split}.json",
        root / "labels" / "100k" / split / "labels_images.json",
        root / "labels" / f"bdd100k_labels_images_{split}.json",
        root / "labels" / f"bdd100k_labels_images_{split}_weather.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.glob(f"**/*{split}*.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find a BDD100K label JSON for split '{split}' under {root}")


def find_image(root: Path, split: str, name: str) -> Path | None:
    candidates = [
        root / "images" / "100k" / split / name,
        root / "images" / split / name,
        root / "images" / "det" / split / name,
    ]
    if not Path(name).suffix:
        candidates.extend([candidate.with_suffix(".jpg") for candidate in list(candidates)])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list((root / "images").glob(f"**/{split}/**/{name}")) if (root / "images").exists() else []
    return matches[0] if matches else None


def iter_objects(frame: dict) -> Iterable[dict]:
    if "labels" in frame:
        yield from frame.get("labels") or []
    for nested in frame.get("frames") or []:
        yield from nested.get("objects") or nested.get("labels") or []


def frame_name(frame: dict) -> str | None:
    name = frame.get("name")
    if name:
        return name if Path(name).suffix else f"{name}.jpg"
    frames = frame.get("frames") or []
    if frames and frames[0].get("name"):
        nested = frames[0]["name"]
        return nested if Path(nested).suffix else f"{nested}.jpg"
    return None


def yolo_boxes(frame: dict, image_size: tuple[int, int], min_box_area: float) -> list[Box]:
    width, height = image_size
    boxes: list[Box] = []
    for obj in iter_objects(frame):
        mapped = BDD_ALIASES.get(str(obj.get("category", "")).lower())
        box = obj.get("box2d")
        if mapped is None or not box:
            continue
        x1 = max(0.0, min(float(box["x1"]), width - 1))
        y1 = max(0.0, min(float(box["y1"]), height - 1))
        x2 = max(0.0, min(float(box["x2"]), width - 1))
        y2 = max(0.0, min(float(box["y2"]), height - 1))
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw * bh < min_box_area:
            continue
        boxes.append(
            Box(
                class_id=CLASS_TO_ID[mapped],
                x_center=((x1 + x2) / 2.0) / width,
                y_center=((y1 + y2) / 2.0) / height,
                width=bw / width,
                height=bh / height,
            )
        )
    return boxes


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def write_label(path: Path, boxes: list[Box]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
        for box in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_frames(label_file: Path) -> list[dict]:
    frames = json.loads(label_file.read_text(encoding="utf-8"))
    if isinstance(frames, dict):
        frames = frames.get("frames") or frames.get("images") or frames.get("data") or []
    if not isinstance(frames, list):
        raise ValueError(f"Expected a list of frames in {label_file}")
    return frames


def select_frames(frames: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or limit >= len(frames):
        return list(frames)
    rng = random.Random(seed)
    return rng.sample(frames, limit)


def convert_split(
    args: argparse.Namespace,
    source_split: str,
    output_split: str | None = None,
    frames: list[dict] | None = None,
    label_file: Path | None = None,
) -> dict:
    output_split = output_split or source_split
    label_file = label_file or find_label_file(args.bdd_root, source_split)
    frames = frames if frames is not None else load_frames(label_file)

    image_out = args.output / "images" / output_split
    label_out = args.output / "labels" / output_split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    class_counts = Counter()
    missing_images = 0

    for frame in tqdm(frames, desc=f"Converting {output_split}"):
        name = frame_name(frame)
        if not name:
            stats["missing_frame_name"] += 1
            continue
        image_path = find_image(args.bdd_root, source_split, name)
        if image_path is None:
            missing_images += 1
            continue
        with Image.open(image_path) as img:
            image_size = img.size
        boxes = yolo_boxes(frame, image_size, args.min_box_area)
        if not boxes and not args.include_empty:
            stats["skipped_empty"] += 1
            continue
        target_image = image_out / Path(name).name
        target_label = label_out / f"{Path(name).stem}.txt"
        link_or_copy(image_path, target_image, args.copy_mode)
        write_label(target_label, boxes)
        for box in boxes:
            class_counts[PROJECT_CLASSES[box.class_id]] += 1
        stats["images"] += 1
        stats["boxes"] += len(boxes)

    stats["missing_images"] = missing_images
    return {
        "split": output_split,
        "source_split": source_split,
        "label_file": str(label_file),
        "stats": dict(stats),
        "class_counts": dict(class_counts),
    }


def write_data_yaml(output: Path, splits: list[str]) -> None:
    lines = [f"path: {output.resolve().as_posix()}"]
    for split in splits:
        lines.append(f"{split}: images/{split}")
    if "test" not in splits and (output / "images" / "test").exists():
        lines.append("test: images/test")
    lines.append(f"nc: {len(PROJECT_CLASSES)}")
    lines.append("names:")
    for idx, name in enumerate(PROJECT_CLASSES):
        lines.append(f"  {idx}: {name}")
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.val_test_fraction < 1.0:
        raise ValueError("--val-test-fraction must be in [0, 1).")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split_index, split in enumerate(args.splits):
        label_file = find_label_file(args.bdd_root, split)
        frames = select_frames(
            load_frames(label_file),
            args.max_images_per_split,
            args.seed + split_index,
        )
        if split == "val" and args.val_test_fraction > 0.0:
            rng = random.Random(args.seed)
            shuffled = list(frames)
            rng.shuffle(shuffled)
            test_count = max(1, round(len(shuffled) * args.val_test_fraction))
            test_frames = shuffled[:test_count]
            val_frames = shuffled[test_count:]
            if not val_frames:
                raise ValueError("The validation/test split needs at least one frame in each split.")
            summaries.append(convert_split(args, "val", "val", val_frames, label_file))
            summaries.append(convert_split(args, "val", "test", test_frames, label_file))
        else:
            summaries.append(convert_split(args, split, split, frames, label_file))
    output_splits = [summary["split"] for summary in summaries]
    write_data_yaml(args.output, output_splits)
    merged_counts = defaultdict(int)
    for summary in summaries:
        for cls, count in summary["class_counts"].items():
            merged_counts[cls] += count
    report = {"splits": summaries, "classes": PROJECT_CLASSES, "total_class_counts": dict(merged_counts)}
    (args.output / "conversion_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote YOLO dataset to {args.output.resolve()}")
    print(f"Data config: {(args.output / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
