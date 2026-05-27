#!/usr/bin/env python3
"""
Exact target search for a medium-strength GTEA process model.

The search space is the set of collapsed-trace variant subsets up to a chosen
size. Each discovered model is evaluated on the full collapsed GTEA log using
token-based fitness and precision, and ranked by closeness to a target point.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT.parent
SWEEP_SCRIPT = REPO_ROOT / "discovery/video_im_collapsed_sweep.py"


def load_sweep_module():
    spec = importlib.util.spec_from_file_location("video_im_collapsed_sweep", SWEEP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for progress, summary, and best-model artifacts.",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.7,
        help="Target value for both fitness and precision.",
    )
    parser.add_argument(
        "--band-low",
        type=float,
        default=0.65,
        help="Lower bound for the balanced target band.",
    )
    parser.add_argument(
        "--band-high",
        type=float,
        default=0.75,
        help="Upper bound for the balanced target band.",
    )
    parser.add_argument(
        "--min-subset-size",
        type=int,
        default=1,
        help="Smallest number of unique variants in a discovery subset.",
    )
    parser.add_argument(
        "--max-subset-size",
        type=int,
        default=4,
        help="Largest number of unique variants in a discovery subset.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write a progress checkpoint every N subset combinations.",
    )
    parser.add_argument(
        "--noise-grid",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help="Noise thresholds to evaluate.",
    )
    parser.add_argument(
        "--require-sound",
        action="store_true",
        help="Prefer and report only sound WF-net candidates as valid matches.",
    )
    parser.add_argument(
        "--require-1-safe",
        action="store_true",
        help="Prefer and report only 1-safe candidates as valid matches.",
    )
    return parser.parse_args()


def total_combinations(n_variants: int, min_size: int, max_size: int) -> int:
    return sum(math.comb(n_variants, size) for size in range(min_size, max_size + 1))


def row_key(row: dict[str, object]) -> tuple[bool, bool, float, float, float]:
    return (
        bool(row["passes_hard_constraints"]),
        bool(row["in_target_band"]),
        -float(row["target_distance"]),
        float(row["fitness"]),
        float(row["precision"]),
    )


def write_status(
    status_path: Path,
    combo_index: int,
    total_combos: int,
    row_count: int,
    start_time: float,
    best_row: dict[str, object] | None,
) -> None:
    payload: dict[str, object] = {
        "completed_combos": combo_index,
        "total_combos": total_combos,
        "completed_rows": row_count,
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    if best_row is not None:
        payload["best_row"] = best_row
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sweep = load_sweep_module()
    config = sweep.DATASETS["gtea"]
    raw_df = pd.read_csv(config.log_path)
    full_df = sweep.preprocess_log(raw_df, config.drop_activity_ids)
    variants, variant_to_cases = sweep.get_variant_case_groups(full_df)

    min_size = max(1, args.min_subset_size)
    max_size = min(args.max_subset_size, len(variants))
    if min_size > max_size:
        raise ValueError("Invalid subset-size range for the number of available variants.")

    n_traces = full_df["case:concept:name"].nunique()
    n_events = len(full_df)
    n_activities = full_df["concept:name"].nunique()
    total_combos = total_combinations(len(variants), min_size, max_size)

    summary_path = args.output_dir / "gtea_target07_summary.csv"
    best_path = args.output_dir / "gtea_target07_best.csv"
    status_path = args.output_dir / "gtea_target07_status.json"
    top_path = args.output_dir / "gtea_target07_top50.csv"

    fieldnames = [
        "dataset",
        "partition",
        "metric_family",
        "traces",
        "events",
        "activities",
        "variant_strategy",
        "variant_indices",
        "variant_subset_size",
        "discovery_traces",
        "discovery_mode",
        "noise_threshold",
        "fitness",
        "precision",
        "places",
        "transitions",
        "arcs",
        "invisible_transitions",
        "target",
        "band_low",
        "band_high",
        "target_distance",
        "in_target_band",
        "is_sound_wfnet",
        "is_one_safe",
        "passes_hard_constraints",
        "score_min_f_p",
    ]

    best_row: dict[str, object] | None = None
    top_rows: list[dict[str, object]] = []
    combo_index = 0
    row_count = 0
    start_time = time.time()

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for size in range(min_size, max_size + 1):
            for combo in __import__("itertools").combinations(range(len(variants)), size):
                combo_index += 1
                selected_cases: list[str] = []
                for idx in combo:
                    selected_cases.extend(variant_to_cases[variants[idx]])
                discovery_df = full_df[full_df["case:concept:name"].isin(selected_cases)].copy()
                discovery_traces = discovery_df["case:concept:name"].nunique()

                for noise in args.noise_grid:
                    net, initial_marking, final_marking = sweep.discover_inductive_model(
                        discovery_df, noise
                    )
                    metrics = sweep.score_model(
                        full_df, net, initial_marking, final_marking, metric="token"
                    )
                    constraints = sweep.analyze_model_constraints(
                        net, initial_marking, final_marking
                    )
                    passes_hard_constraints = True
                    if args.require_sound:
                        passes_hard_constraints = (
                            passes_hard_constraints and constraints["is_sound_wfnet"]
                        )
                    if args.require_1_safe:
                        passes_hard_constraints = (
                            passes_hard_constraints and constraints["is_one_safe"]
                        )
                    row = {
                        "dataset": "gtea",
                        "partition": "global",
                        "metric_family": "token",
                        "traces": n_traces,
                        "events": n_events,
                        "activities": n_activities,
                        "variant_strategy": "exhaustive_target07",
                        "variant_indices": ",".join(str(idx) for idx in combo),
                        "variant_subset_size": len(combo),
                        "discovery_traces": discovery_traces,
                        "discovery_mode": "variants_exhaustive_target07",
                        "noise_threshold": noise,
                        **metrics,
                        "target": args.target,
                        "band_low": args.band_low,
                        "band_high": args.band_high,
                        "target_distance": math.hypot(
                            float(metrics["fitness"]) - args.target,
                            float(metrics["precision"]) - args.target,
                        ),
                        "in_target_band": (
                            args.band_low <= float(metrics["fitness"]) <= args.band_high
                            and args.band_low <= float(metrics["precision"]) <= args.band_high
                        ),
                        **constraints,
                        "passes_hard_constraints": passes_hard_constraints,
                        "score_min_f_p": min(
                            float(metrics["fitness"]), float(metrics["precision"])
                        ),
                    }

                    writer.writerow(row)
                    row_count += 1

                    if best_row is None or row_key(row) > row_key(best_row):
                        best_row = row
                        pd.DataFrame([best_row]).to_csv(best_path, index=False)

                    top_rows.append(row)
                    top_rows.sort(key=row_key, reverse=True)
                    del top_rows[50:]

                handle.flush()

                if combo_index % args.checkpoint_every == 0:
                    write_status(
                        status_path=status_path,
                        combo_index=combo_index,
                        total_combos=total_combos,
                        row_count=row_count,
                        start_time=start_time,
                        best_row=best_row,
                    )
                    best_str = (
                        "none"
                        if best_row is None
                        else (
                            f"fit={best_row['fitness']:.4f} "
                            f"prec={best_row['precision']:.4f} "
                            f"dist={best_row['target_distance']:.4f} "
                            f"combo={best_row['variant_indices']} "
                            f"noise={best_row['noise_threshold']}"
                        )
                    )
                    elapsed = time.time() - start_time
                    print(
                        f"[{elapsed:8.1f}s] combos {combo_index}/{total_combos} "
                        f"rows {row_count} best {best_str}",
                        flush=True,
                    )

    pd.DataFrame(top_rows).to_csv(top_path, index=False)
    write_status(
        status_path=status_path,
        combo_index=combo_index,
        total_combos=total_combos,
        row_count=row_count,
        start_time=start_time,
        best_row=best_row,
    )

    if best_row is None:
        raise RuntimeError("Search completed without producing any rows.")

    elapsed = time.time() - start_time
    print(
        f"Completed exact target search in {elapsed:.1f}s. "
        f"Best fit={best_row['fitness']:.4f} prec={best_row['precision']:.4f} "
        f"dist={best_row['target_distance']:.4f} combo={best_row['variant_indices']} "
        f"noise={best_row['noise_threshold']}",
        flush=True,
    )
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {best_path}", flush=True)
    print(f"Wrote {top_path}", flush=True)
    print(f"Wrote {status_path}", flush=True)


if __name__ == "__main__":
    main()
