# Soccer Match Analytics

Analyze soccer match footage: detect players with YOLO26, assign them to teams
by jersey color, and count how many players per team are visible over time.

Input: `data/soccer_match.mp4` (1 min recording, 1920x1080 @ 10 fps).

## Setup

Requires Python >= 3.11.

With pip and requirements.txt:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively with poetry:

```bash
poetry install
poetry shell   # activate the environment
```

Model weights (`yolo26*.pt`) are downloaded automatically by ultralytics on
first run.

## Scripts

### `src/detect_players.py` — object detection

Runs YOLO26 frame by frame over the video and writes an annotated video with
bounding boxes, class labels and confidence scores. By default only the
`person` and `sports ball` classes are kept.

```bash
python src/detect_players.py \
    --model models/yolo26l.pt \
    --output data/soccer_match_detected_l.mp4
```

### `src/team_assignment.py` — team assignment + counting

Detects persons, crops each detection's torso region, removes grass pixels and
classifies the jersey color against predefined HSV ranges (pixel-share voting):
Team 1 white/blue GK, Team 2 red/neon GK, referee yellow, everything else
"other". Writes an annotated video with a live count panel, a per-frame counts
CSV and a counts-over-time plot. Optionally dumps every torso crop for manual
inspection with `--torso-dir`.

```bash
python src/team_assignment.py \
    --output data/soccer_match_teams.mp4 \
    --csv data/team_counts.csv \
    --plot data/team_counts.png
```

### `src/combined_video.py` — video + synced timeline

Stacks the annotated match video on top of the players-per-team graph and
sweeps a vertical cursor across the graph in sync with playback, so the current
position on the timeline is always visible.

```bash
python src/combined_video.py \
    --video data/soccer_match_teams.mp4 \
    --csv data/team_counts.csv \
    --output data/soccer_match_combined.mp4
```