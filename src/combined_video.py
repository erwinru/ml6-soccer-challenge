"""Combine the annotated match video with a synced players-per-team timeline.

Stacks the annotated video on top of the counts-over-time graph and sweeps a
vertical cursor across the graph as the video plays, so it is always visible
where the current frame sits on the timeline.

The graph is rendered once with matplotlib at the video's width; per frame only
the cursor line is drawn onto a copy, which keeps the loop fast. The mapping
from frame time to graph x-pixel uses matplotlib's own data-to-display
transform, so the cursor is exactly aligned with the plotted curves.

Usage:
    python src/combined_video.py
    python src/combined_video.py --video data/soccer_match_teams.mp4 \\
        --csv data/team_counts.csv --output data/soccer_match_combined.mp4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

GRAPH_HEIGHT = 320  # pixels of timeline strip under the video


def load_counts(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return [
            {k: float(v) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def render_graph(rows: list[dict], width: int, height: int):
    """Render the counts graph once; return (BGR image, x-pixel per frame,
    (y_top, y_bottom) pixel span of the plot area for the cursor line)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    t = [r["time_s"] for r in rows]
    ax.plot(t, [r["team1"] for r in rows], label="Team 1 (white/blue)",
            color="black", linewidth=1.6)
    ax.plot(t, [r["team2"] for r in rows], label="Team 2 (red/neon)",
            color="red", linewidth=1.6)
    ax.plot(t, [r["referee"] for r in rows], label="Referees",
            color="goldenrod", linewidth=1.6)

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, max(max(r["team1"], r["team2"]) for r in rows) + 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("players visible")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=1.2)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # RGBA -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Data -> pixel x for each frame's timestamp (y flips: display origin is
    # bottom-left, image origin is top-left).
    xs = ax.transData.transform([(r["time_s"], 0) for r in rows])[:, 0]
    bbox = ax.get_window_extent()
    y_top = int(height - bbox.y1)
    y_bottom = int(height - bbox.y0)
    plt.close(fig)

    return img, xs.astype(int), (y_top, y_bottom)


def combine(video_path: Path, csv_path: Path, output_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    rows = load_counts(csv_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    graph, xs, (gy0, gy1) = render_graph(rows, W, GRAPH_HEIGHT)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H + GRAPH_HEIGHT))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open writer: {output_path}")

    n = min(total, len(rows))
    print(f"Combining {video_path.name} + {csv_path.name} "
          f"({n} frames, {W}x{H + GRAPH_HEIGHT})")

    fi = 0
    while fi < n:
        ok, frame = cap.read()
        if not ok:
            break

        strip = graph.copy()
        x = int(xs[fi])
        cv2.line(strip, (x, gy0), (x, gy1), (0, 200, 0), 2)
        # small current-position marker triangle above the plot area
        cv2.drawMarker(strip, (x, gy0 + 8), (0, 200, 0),
                       cv2.MARKER_TRIANGLE_DOWN, 12, 2)

        writer.write(np.vstack([frame, strip]))
        fi += 1
        if fi % 100 == 0 or fi == n:
            print(f"  frame {fi}/{n}")

    cap.release()
    writer.release()
    print(f"Done. Combined video written to {output_path}")


def parse_args():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Video + synced count timeline.")
    p.add_argument("--video", type=Path,
                   default=root / "data" / "soccer_match_teams.mp4",
                   help="Annotated input video (frame count must match CSV).")
    p.add_argument("--csv", type=Path,
                   default=root / "data" / "team_counts.csv")
    p.add_argument("--output", type=Path,
                   default=root / "data" / "soccer_match_combined.mp4")
    return p.parse_args()


def main():
    a = parse_args()
    combine(a.video, a.csv, a.output)


if __name__ == "__main__":
    main()
