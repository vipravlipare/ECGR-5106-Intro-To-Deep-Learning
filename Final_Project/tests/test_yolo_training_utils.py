from pathlib import Path

from PIL import Image

from road_detection.yolo_training_utils import (
    IndexedImage,
    build_candidate_index,
    label_path_for_image,
    load_split_index,
    stage_images_and_labels,
    weighted_sample,
)


def test_small_object_weight_changes_deterministic_sample() -> None:
    records = [
        IndexedImage(Path(f"{index}.jpg"), frozenset({0}), 0, 1)
        for index in range(9)
    ]
    records.append(IndexedImage(Path("small.jpg"), frozenset({0}), 9, 9))

    plain_hits = sum(
        weighted_sample(records, 1, 1, exponent=0.0, seed=seed)
        == [Path("small.jpg")]
        for seed in range(100)
    )
    boosted_hits = sum(
        weighted_sample(
            records,
            1,
            1,
            exponent=0.0,
            seed=seed,
            small_object_boost=5.0,
        )
        == [Path("small.jpg")]
        for seed in range(100)
    )

    assert boosted_hits > plain_hits * 2


def test_candidate_index_records_class_and_small_object_stats(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(image_dir / "scene.jpg")
    (label_dir / "scene.txt").write_text(
        "4 0.5 0.5 0.01 0.01\n0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "candidate.jsonl"

    records = build_candidate_index(
        image_dir,
        label_dir,
        index_path,
        candidate_count=1,
        seed=42,
        scan_workers=1,
    )
    reloaded = load_split_index(index_path, image_dir)

    assert records == reloaded
    assert reloaded[0].classes == frozenset({0, 4})
    assert reloaded[0].small_object_count == 1
    assert reloaded[0].box_count == 2


def test_staging_copies_matching_image_and_label_layout(tmp_path: Path) -> None:
    source_image_dir = tmp_path / "source" / "images" / "train"
    source_label_dir = tmp_path / "source" / "labels" / "train"
    source_image_dir.mkdir(parents=True)
    source_label_dir.mkdir(parents=True)
    source_image = source_image_dir / "scene.jpg"
    Image.new("RGB", (16, 16)).save(source_image)
    (source_label_dir / "scene.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    staged = stage_images_and_labels(
        [source_image],
        tmp_path / "cache",
        "train",
        workers=1,
    )

    assert label_path_for_image(source_image) == source_label_dir / "scene.txt"
    assert staged == [tmp_path / "cache" / "images" / "train" / "scene.jpg"]
    assert staged[0].exists()
    assert (tmp_path / "cache" / "labels" / "train" / "scene.txt").read_text(
        encoding="utf-8"
    ).startswith("0 ")
