"""Assign detected people to teams by jersey colour and count players per team.

Runs YOLO26 person detection on the video, classifies each detected person by
the dominant colour of their torso region against a set of known kit colours,
counts how many players of each team are on screen, and writes:

  * an annotated video (boxes coloured by team + a live per-team count panel)
  * a CSV of per-frame counts
  * a plot of the per-team counts over time

Team / role colour mapping (customer-provided):
    Team 1 : white players, blue goalkeeper
    Team 2 : red players,   neon-green goalkeeper
    Referee: yellow
    Anyone else (crowd, staff) -> "other" and excluded from team counts.

Usage:
    python src/team_assignment.py
    python src/team_assignment.py --limit 60          # quick test on first N frames
    python src/team_assignment.py --model models/yolo26l.pt --conf 0.3
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Kit colour definitions.
#
# Ranges are in OpenCV HSV space: H in [0, 179], S in [0, 255], V in [0, 255].
# Each category is a list of (lower, upper) HSV boxes (a list, so hues that
# wrap around 0 like red can use two boxes). Classification is by pixel-voting:
# for each category we count how many torso pixels fall inside its ranges and
# pick the category with the largest share.
# ---------------------------------------------------------------------------
COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    # White: near-zero saturation, high brightness.
    "team1_player": [((0, 0, 165), (179, 45, 255))],
    # Blue goalkeeper. This kit is a fairly DARK blue (measured torso median
    # ~H112 S140 V60-120), so the value floor is kept low while still requiring
    # enough saturation to avoid dark shadows / black clothing.
    "team1_keeper": [((98, 80, 40), (132, 255, 255))],
    # Red wraps around the hue circle -> two boxes.
    "team2_player": [((0, 110, 70), (10, 255, 255)), ((168, 110, 70), (179, 255, 255))],
    # Neon / fluorescent green: same hue band as grass but far more saturated
    # AND brighter, which is what separates the keeper from the pitch.
    "team2_keeper": [((36, 150, 150), (85, 255, 255))],
    # Referee yellow (between red and green hues).
    "referee": [((20, 90, 120), (34, 255, 255))],
}

# Which team each category counts toward.
CATEGORY_TEAM = {
    "team1_player": "team1",
    "team1_keeper": "team1",
    "team2_player": "team2",
    "team2_keeper": "team2",
    "referee": "referee",
    "other": "other",
}

# Short label + BGR draw colour per category.
CATEGORY_STYLE = {
    "team1_player": ("T1", (255, 255, 255)),
    "team1_keeper": ("T1-GK", (255, 160, 0)),   # blue-ish (BGR)
    "team2_player": ("T2", (0, 0, 255)),
    "team2_keeper": ("T2-GK", (0, 255, 0)),
    "referee": ("REF", (0, 255, 255)),
    "other": ("?", (150, 150, 150)),
}

# A pixel is "grass" if green-hued but not saturated/bright enough to be neon.
# Grass pixels are ignored so a white player on grass reads as white, not green.
GRASS_HSV = ((30, 40, 40), (90, 255, 255))

MIN_VOTE_FRACTION = 0.12  # min share of torso pixels to accept a colour class

# Location gate for the blue (team 1) goalkeeper. The keeper stays near his goal
# on the RIGHT side of the pitch at roughly middle height. Blue-jacketed
# spectators in the stands (left/centre) and staff on the near touchline
# (very bottom of the frame) otherwise get mis-read as the keeper, so a blue
# detection outside this normalised region is demoted to "other".
#   cx: fraction of frame width  (0 = left edge, 1 = right edge)
#   cy: fraction of frame height (0 = top edge,  1 = bottom edge)
KEEPER_MIN_CX = 0.50          # must be in the right half
KEEPER_CY_RANGE = (0.12, 0.85)  # exclude near-touchline foreground / far crowd top


def keeper_location_ok(cx_norm: float, cy_norm: float) -> bool:
    """True if a blue detection is where the team-1 keeper plausibly is."""
    lo, hi = KEEPER_CY_RANGE
    return cx_norm >= KEEPER_MIN_CX and lo <= cy_norm <= hi


def torso_region(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    """Return the jersey/torso sub-box of a person box.

    Takes the central-horizontal, upper-body band: below the head and above the
    shorts, trimmed on the sides to drop background and arms.
    """
    w = x2 - x1
    h = y2 - y1
    tx1 = x1 + int(0.25 * w)
    tx2 = x2 - int(0.25 * w)
    ty1 = y1 + int(0.12 * h)
    ty2 = y1 + int(0.55 * h)
    return tx1, ty1, tx2, ty2


def grass_mask(hsv: np.ndarray) -> np.ndarray:
    """Binary mask of grass pixels: green-hued but below the neon bar."""
    (glo, ghi) = GRASS_HSV
    grass = cv2.inRange(hsv, np.array(glo), np.array(ghi))
    neon = cv2.inRange(hsv, np.array((36, 150, 150)), np.array((85, 255, 255)))
    return cv2.bitwise_and(grass, cv2.bitwise_not(neon))


def filtered_torso(crop_bgr: np.ndarray) -> np.ndarray:
    """Copy of the torso crop with grass pixels blacked out (for inspection)."""
    out = crop_bgr.copy()
    if out.size:
        out[grass_mask(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)) > 0] = 0
    return out


def classify_torso(crop_bgr: np.ndarray) -> str:
    """Classify a torso crop into a kit category via HSV pixel-voting."""
    if crop_bgr.size == 0:
        return "other"

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    non_grass = cv2.bitwise_not(grass_mask(hsv))

    total = int(cv2.countNonZero(non_grass))
    if total == 0:
        return "other"

    best_cat, best_share = "other", 0.0
    for category, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        # Only count colour pixels that aren't grass.
        mask = cv2.bitwise_and(mask, non_grass)
        share = cv2.countNonZero(mask) / total
        if share > best_share:
            best_cat, best_share = category, share

    return best_cat if best_share >= MIN_VOTE_FRACTION else "other"


def save_torso_inspect(torso_dir: Path, category: str, frame_index: int,
                       det_index: int, clean_frame: np.ndarray,
                       person_box: tuple[int, int, int, int],
                       torso_box: tuple[int, int, int, int],
                       inspect_height: int = 160, context_pad: int = 40) -> None:
    """Save a 3-panel inspection image: context | raw torso | filtered torso.

    Panel 1 shows the person with the full detection box (blue) and the torso
    band (red) drawn, padded for context. Panels 2 and 3 are the raw torso crop
    and the grass-filtered crop (the exact pixels that vote in classification).
    Images are grouped into one subfolder per assigned category, so flipping
    through a folder makes misclassifications easy to spot. All panels are
    scaled to ``inspect_height`` so small (distant) crops stay viewable.
    """
    H, W = clean_frame.shape[:2]
    x1, y1, x2, y2 = person_box
    tx1, ty1, tx2, ty2 = torso_box

    crop = clean_frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return

    # Panel 1: padded person context with the boxes drawn on a local copy.
    px1, py1 = max(0, x1 - context_pad), max(0, y1 - context_pad)
    px2, py2 = min(W, x2 + context_pad), min(H, y2 + context_pad)
    context = clean_frame[py1:py2, px1:px2].copy()
    cv2.rectangle(context, (x1 - px1, y1 - py1), (x2 - px1, y2 - py1),
                  (255, 100, 0), 2)
    cv2.rectangle(context, (tx1 - px1, ty1 - py1), (tx2 - px1, ty2 - py1),
                  (0, 0, 255), 2)

    panels = [context, crop, filtered_torso(crop)]
    resized = []
    for p in panels:
        scale = inspect_height / p.shape[0]
        interp = cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA
        resized.append(cv2.resize(
            p, (max(1, int(p.shape[1] * scale)), inspect_height),
            interpolation=interp))

    gap = np.full((inspect_height, 6, 3), 255, dtype=np.uint8)
    composite = np.hstack([resized[0], gap, resized[1], gap, resized[2]])

    subdir = torso_dir / category
    subdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(subdir / f"f{frame_index:04d}_p{det_index:02d}.jpg"), composite)


def draw_count_panel(frame: np.ndarray, counts: dict[str, int]) -> None:
    """Draw a per-team count panel in the bottom-left corner."""
    lines = [
        (f"Team 1 (white/blue): {counts['team1']}", (255, 255, 255)),
        (f"Team 2 (red/neon):   {counts['team2']}", (0, 0, 255)),
        (f"Referees:            {counts['referee']}", (0, 255, 255)),
    ]
    h = frame.shape[0]
    x0, y0 = 20, h - 20 - 26 * len(lines)
    cv2.rectangle(frame, (x0 - 12, y0 - 28), (x0 + 360, y0 + 26 * len(lines) - 6),
                  (0, 0, 0), thickness=-1)
    for i, (text, color) in enumerate(lines):
        y = y0 + i * 26
        cv2.putText(frame, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    cv2.LINE_AA)


def process(
    model_path: Path,
    source_path: Path,
    output_path: Path,
    csv_path: Path,
    plot_path: Path,
    conf: float = 0.3,
    limit: int | None = None,
    torso_dir: Path | None = None,
) -> None:
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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open video writer for: {output_path}")

    print(
        f"Model: {model_path.name} | source: {source_path.name} "
        f"({width}x{height}, {fps:.1f} fps, {total_frames} frames) | conf={conf}"
    )

    per_frame_rows: list[dict] = []
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        results = model.predict(frame, conf=conf, classes=[0], verbose=False)
        boxes = results[0].boxes

        # Crops must come from an unannotated copy: drawing happens in place on
        # `frame` inside the loop, so cropping from `frame` would leak already-
        # drawn boxes of overlapping players into later torso crops and skew
        # their colour vote.
        clean = frame.copy()

        counts = {"team1": 0, "team2": 0, "referee": 0, "other": 0}

        for det_index, box in enumerate(boxes):
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            tx1, ty1, tx2, ty2 = torso_region(x1, y1, x2, y2)
            tx1, ty1 = max(0, tx1), max(0, ty1)
            tx2, ty2 = min(width, tx2), min(height, ty2)
            crop = clean[ty1:ty2, tx1:tx2]

            category = classify_torso(crop)

            # Location gate: reject blue-jacketed non-keepers (crowd / touchline).
            if category == "team1_keeper":
                cx_norm = ((x1 + x2) / 2) / width
                cy_norm = ((y1 + y2) / 2) / height
                if not keeper_location_ok(cx_norm, cy_norm):
                    category = "other"

            # Optional inspection dump: context|raw|filtered, sorted by category.
            if torso_dir is not None:
                save_torso_inspect(torso_dir, category, frame_index, det_index,
                                   clean, (x1, y1, x2, y2), (tx1, ty1, tx2, ty2))

            counts[CATEGORY_TEAM[category]] += 1

            label, color = CATEGORY_STYLE[category]
            det_conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {det_conf:.2f}"
            cv2.putText(frame, text, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        draw_count_panel(frame, counts)
        writer.write(frame)

        per_frame_rows.append({
            "frame": frame_index,
            "time_s": round(frame_index / fps, 2),
            "team1": counts["team1"],
            "team2": counts["team2"],
            "referee": counts["referee"],
            "other": counts["other"],
        })

        frame_index += 1
        if frame_index % 25 == 0 or frame_index == total_frames:
            print(f"  frame {frame_index}/{total_frames} | "
                  f"T1={counts['team1']} T2={counts['team2']} "
                  f"REF={counts['referee']} other={counts['other']}")

        if limit is not None and frame_index >= limit:
            break

    capture.release()
    writer.release()

    # Write per-frame counts CSV.
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "time_s", "team1", "team2",
                                          "referee", "other"])
        w.writeheader()
        w.writerows(per_frame_rows)

    _save_plot(per_frame_rows, plot_path)

    print(f"Done.\n  video: {output_path}\n  csv:   {csv_path}\n  plot:  {plot_path}")


def _save_plot(rows: list[dict], plot_path: Path) -> None:
    """Save a line plot of per-team counts over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r["time_s"] for r in rows]
    plt.figure(figsize=(12, 5))
    plt.plot(t, [r["team1"] for r in rows], label="Team 1 (white/blue)", color="black")
    plt.plot(t, [r["team2"] for r in rows], label="Team 2 (red/neon)", color="red")
    plt.plot(t, [r["referee"] for r in rows], label="Referees", color="goldenrod")
    plt.xlabel("time (s)")
    plt.ylabel("players on screen")
    plt.title("Players per team visible over time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Team assignment + per-team counting.")
    p.add_argument("--model", type=Path, default=root / "models" / "yolo26l.pt")
    p.add_argument("--source", type=Path, default=root / "data" / "soccer_match.mp4")
    p.add_argument("--output", type=Path,
                   default=root / "data" / "soccer_match_teams.mp4")
    p.add_argument("--csv", type=Path, default=root / "data" / "team_counts.csv")
    p.add_argument("--plot", type=Path, default=root / "data" / "team_counts.png")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N frames (for quick tuning).")
    p.add_argument("--torso-dir", type=Path, default=None,
                   help="If set, save every raw|filtered torso crop here, "
                        "grouped in one subfolder per assigned category "
                        "(for manual inspection).")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    process(a.model, a.source, a.output, a.csv, a.plot, conf=a.conf, limit=a.limit,
            torso_dir=a.torso_dir)


if __name__ == "__main__":
    main()
