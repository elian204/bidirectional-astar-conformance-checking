#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_SCRIPT="$SCRIPT_DIR/video_im_collapsed_sweep.py"
MODELS_DIR="$SCRIPT_DIR/models"
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$MODELS_DIR/video_im_overnight_${RUN_ID}"
LATEST_LINK="$MODELS_DIR/video_im_overnight_latest"

mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$LATEST_LINK"
printf '%s\n' "$RUN_DIR" > "$MODELS_DIR/video_im_overnight_latest_path.txt"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_job() {
  local job_name="$1"
  shift
  local out_dir="$RUN_DIR/$job_name"
  mkdir -p "$out_dir"
  log "Starting $job_name"
  python -u "$SWEEP_SCRIPT" "$@" --output-dir "$out_dir" | tee "$RUN_DIR/${job_name}.log"
  log "Finished $job_name"
}

COMMON_SAMPLING=(
  --sampling-fractions 0.5 0.6 0.7 0.8 0.9
  --sampling-seeds 1 2 3 4 5
)

log "Run directory: $RUN_DIR"

run_job \
  breakfast_global \
  --datasets breakfast \
  --partition-mode global \
  --metric token \
  "${COMMON_SAMPLING[@]}"

run_job \
  breakfast_recipe \
  --datasets breakfast \
  --partition-mode recipe \
  --metric token \
  "${COMMON_SAMPLING[@]}" \
  --variant-selection auto \
  --variant-max-subset-size 3

run_job \
  50salads_global \
  --datasets 50salads \
  --partition-mode global \
  --metric token \
  "${COMMON_SAMPLING[@]}" \
  --variant-selection auto \
  --variant-max-subset-size 4

run_job \
  gtea_global \
  --datasets gtea \
  --partition-mode global \
  --metric token \
  "${COMMON_SAMPLING[@]}" \
  --variant-selection auto \
  --variant-max-subset-size 4

python - "$RUN_DIR" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

run_dir = Path(sys.argv[1])
best_paths = sorted(run_dir.glob("*/video_im_collapsed_token_best.csv"))
frames = []
for path in best_paths:
    df = pd.read_csv(path)
    df.insert(0, "job_name", path.parent.name)
    frames.append(df)

if not frames:
    raise SystemExit("No best-model CSV files were produced.")

all_best = pd.concat(frames, ignore_index=True)
all_best.to_csv(run_dir / "all_best_models.csv", index=False)

global_best = (
    all_best[all_best["partition"] == "global"]
    .sort_values(["dataset", "score_min_f_p", "fitness", "precision"], ascending=[True, False, False, False])
    .groupby("dataset", sort=False)
    .head(1)
    .reset_index(drop=True)
)
global_best.to_csv(run_dir / "global_best_models.csv", index=False)

recommended = all_best[all_best["meets_target"] == True].copy()  # noqa: E712
recommended = recommended.sort_values(
    ["dataset", "partition", "score_min_f_p", "fitness", "precision"],
    ascending=[True, True, False, False, False],
).reset_index(drop=True)
recommended.to_csv(run_dir / "recommended_models.csv", index=False)

summary_lines = []
summary_lines.append("Global Best Models")
summary_lines.append(global_best.to_string(index=False))
summary_lines.append("")
summary_lines.append("Recommended Models (meets target)")
summary_lines.append(recommended.to_string(index=False) if not recommended.empty else "None")
(run_dir / "overnight_summary.txt").write_text("\n".join(summary_lines) + "\n")

status = {
    "run_dir": str(run_dir),
    "jobs": [path.parent.name for path in best_paths],
    "global_best_rows": len(global_best),
    "recommended_rows": len(recommended),
    "files": {
        "all_best_models": str(run_dir / "all_best_models.csv"),
        "global_best_models": str(run_dir / "global_best_models.csv"),
        "recommended_models": str(run_dir / "recommended_models.csv"),
        "overnight_summary": str(run_dir / "overnight_summary.txt"),
    },
}
(run_dir / "overnight_status.json").write_text(json.dumps(status, indent=2) + "\n")
PY

log "Wrote consolidated outputs to $RUN_DIR"
