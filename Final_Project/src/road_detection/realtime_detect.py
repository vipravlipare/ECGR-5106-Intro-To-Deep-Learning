from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from road_detection.constants import PROJECT_CLASSES
from road_detection.rcnn_model import load_faster_rcnn_checkpoint


COLORS = np.array(
    [
        [28, 126, 214],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [255, 127, 14],
        [23, 190, 207],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime road-object detection with YOLO or Faster R-CNN.")
    parser.add_argument("--backend", choices=["yolo", "rcnn"], default="yolo")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", default="0", help="Webcam index, image, or video path.")
    parser.add_argument("--conf", type=float, default=None, help="Defaults to 0.35 for YOLO or the tuned checkpoint value for R-CNN.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YOLO deployment JSON with calibrated thresholds.")
    parser.add_argument(
        "--threshold-profile",
        default=None,
        help="Threshold profile from the deployment JSON, such as high_precision or balanced.",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Defaults to the deployment config value or 960.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--precision",
        choices=["auto", "fp16", "fp32"],
        default="auto",
        help="FP16 is faster on CUDA; auto uses FP16 on CUDA and FP32 on CPU.",
    )
    parser.add_argument("--max-det", type=int, default=None, help="Maximum detections per frame.")
    parser.add_argument("--camera-buffer", type=int, default=1, help="Small webcam buffers reduce display latency.")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def draw_box(frame, box, label: str, score: float, class_id: int) -> None:
    color = tuple(int(v) for v in COLORS[class_id % len(COLORS)].tolist())
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def source_value(source: str):
    return int(source) if source.isdigit() else source


def load_yolo_config(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Deployment config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected an object in deployment config: {path}")
    return config


def torch_device(device: str) -> torch.device:
    return torch.device(f"cuda:{device}" if device.isdigit() else device)


def yolo_threshold_settings(config: dict, requested_profile: str | None) -> tuple[str | None, float, dict[int, float]]:
    profiles = config.get("threshold_profiles") or {}
    profile_name = requested_profile or config.get("default_threshold_profile")
    profile = {}
    if profile_name:
        if profile_name not in profiles:
            available = ", ".join(sorted(profiles)) or "none"
            raise ValueError(
                f"Threshold profile '{profile_name}' is unavailable. Available profiles: {available}"
            )
        profile = profiles[profile_name]

    configured_thresholds = (
        profile.get("class_thresholds")
        or config.get("class_thresholds")
        or {}
    )
    class_thresholds = {
        int(class_id): float(value)
        for class_id, value in configured_thresholds.items()
    }
    confidence = float(
        profile.get(
            "inference_confidence",
            config.get("inference_confidence", 0.35),
        )
    )
    return profile_name, confidence, class_thresholds


def main() -> None:
    args = parse_args()
    device = torch_device(args.device)
    cap = cv2.VideoCapture(source_value(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, args.camera_buffer))
    writer = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    yolo_config = load_yolo_config(args.config) if args.backend == "yolo" else {}
    yolo = YOLO(str(args.weights)) if args.backend == "yolo" else None
    if args.backend == "rcnn":
        rcnn, rcnn_classes, rcnn_metadata = load_faster_rcnn_checkpoint(args.weights, device)
        confidence = args.conf if args.conf is not None else float(rcnn_metadata.get("score_threshold", 0.35))
        image_size = args.imgsz or 960
        class_thresholds = {}
        max_detections = args.max_det or 300
        yolo_quantize = None
    else:
        rcnn = None
        rcnn_classes = PROJECT_CLASSES
        profile_name, configured_confidence, class_thresholds = yolo_threshold_settings(
            yolo_config, args.threshold_profile
        )
        confidence = args.conf if args.conf is not None else configured_confidence
        image_size = args.imgsz or int(yolo_config.get("imgsz", 640))
        max_detections = args.max_det or int(yolo_config.get("max_det", 100))
        use_fp16 = (
            args.precision == "fp16"
            or (
                args.precision == "auto"
                and device.type == "cuda"
                and torch.cuda.is_available()
            )
        )
        if use_fp16 and device.type != "cuda":
            raise ValueError("--precision fp16 requires a CUDA device.")
        yolo_quantize = 16 if use_fp16 else None
        yolo.fuse()
        if profile_name:
            print(f"YOLO threshold profile: {profile_name}")
    last = time.perf_counter()
    fps_smooth = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if yolo is not None:
            result = yolo.predict(
                frame,
                imgsz=image_size,
                conf=confidence,
                device=args.device,
                quantize=yolo_quantize,
                max_det=max_detections,
                verbose=False,
            )[0]
            boxes = result.boxes
            coordinates = boxes.xyxy.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)
            scores = boxes.conf.detach().cpu().numpy()
            for box, class_id, score in zip(coordinates, class_ids, scores):
                threshold = confidence if args.conf is not None else class_thresholds.get(class_id, confidence)
                if class_id < len(PROJECT_CLASSES) and score >= threshold:
                    draw_box(frame, box, PROJECT_CLASSES[class_id], float(score), int(class_id))
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
            with torch.no_grad():
                output = rcnn([tensor])[0]
            for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
                if float(score) < confidence:
                    continue
                class_id = int(label) - 1
                if 0 <= class_id < len(rcnn_classes):
                    draw_box(frame, box.detach().cpu().numpy(), rcnn_classes[class_id], float(score), class_id)

        now = time.perf_counter()
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(1e-6, now - last)) if fps_smooth else 1.0 / max(1e-6, now - last)
        last = now
        cv2.putText(frame, f"{args.backend.upper()} {fps_smooth:.1f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 255, 30), 2)
        if writer:
            writer.write(frame)
        if args.show:
            cv2.imshow("Road Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
