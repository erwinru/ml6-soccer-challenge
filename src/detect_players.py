"""Run YOLO object detection on the soccer video and write an annotated video.

Processes the source video frame by frame with a YOLO26 model (via the
``ultralytics`` library), draws bounding boxes with class labels and confidence
scores, and writes the result to an output video.

Usage:
    python src/detect_players.py
    python src/detect_players.py --source data/soccer_match.mp4 --output data/soccer_match_detected.mp4
    python src/detect_players.py --all-classes --conf 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# Relevant COCO class ids for soccer analytics.
#   0  = person   (players, referees, keepers)
#   32 = sports ball
SOCCER_CLASSES = [0, 32]


def detect_on_video(
    model_path: Path,
    source_path: Path,
    output_path: Path,
    conf: float = 0.25,
    classes: list[int] | None = None,
) -> None:
    """Run detection frame by frame and write an annotated output video.

    Args:
        model_path: Path to the YOLO ``.pt`` weights.
        source_path: Path to the input video.
        output_path: Path where the annotated video is written.
        conf: Confidence threshold for detections.
        classes: Optional list of COCO class ids to keep (None = all classes).
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Video not found: {source_path}")

    model = YOLO(str(model_path))

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # mp4v is broadly available with OpenCV's bundled FFmpeg.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open video writer for: {output_path}")

    print(
        f"Model: {model_path.name} | source: {source_path.name} "
        f"({width}x{height}, {fps:.1f} fps, {total_frames} frames)\n"
        f"conf={conf} | classes={classes if classes is not None else 'all'}"
    )

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        # verbose=False keeps per-frame logging quiet; classes filters detections.
        results = model.predict(frame, conf=conf, classes=classes, verbose=False)
        result = results[0]

        # results.plot() returns a BGR image with boxes, labels and confidences.
        annotated = result.plot()
        writer.write(annotated)

        frame_index += 1
        if frame_index % 25 == 0 or frame_index == total_frames:
            n = len(result.boxes)
            print(f"  frame {frame_index}/{total_frames} | {n} detections")

    capture.release()
    writer.release()
    print(f"Done. Annotated video written to {output_path}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Run YOLO detection on a video and save an annotated video."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "models" / "yolo26l.pt",
        help="Path to YOLO weights (auto-downloaded by ultralytics if missing).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "data" / "soccer_match.mp4",
        help="Path to the input video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "soccer_match_detected.mp4",
        help="Path for the annotated output video.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Detect all COCO classes (default: only person + sports ball).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = None if args.all_classes else SOCCER_CLASSES
    detect_on_video(
        model_path=args.model,
        source_path=args.source,
        output_path=args.output,
        conf=args.conf,
        classes=classes,
    )


if __name__ == "__main__":
    main()
