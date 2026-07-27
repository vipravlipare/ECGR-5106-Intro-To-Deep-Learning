from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure YOLO inference FPS on a folder or video.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=300)
    return parser.parse_args()


def iter_frames(source: str, limit: int):
    path = Path(source)
    if path.is_dir():
        images = sorted(p for p in path.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        for image_path in images[:limit]:
            frame = cv2.imread(str(image_path))
            if frame is not None:
                yield frame
        return
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    seen = 0
    while seen < limit:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
        seen += 1
    cap.release()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    frames = list(iter_frames(args.source, args.limit))
    if not frames:
        raise RuntimeError(f"No readable frames found in {args.source}")
    for frame in frames[:10]:
        model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
    start = time.perf_counter()
    for frame in frames:
        model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
    elapsed = time.perf_counter() - start
    print(f"Frames: {len(frames)}")
    print(f"Elapsed: {elapsed:.3f} s")
    print(f"FPS: {len(frames) / max(elapsed, 1e-9):.2f}")


if __name__ == "__main__":
    main()

