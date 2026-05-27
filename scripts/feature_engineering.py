#!/usr/bin/env python3
"""Build a one-row-per-instance Stage B feature table from Stage A targets."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from experiments.benchmark_loader import load_model
    from scripts.positional_structure import compute_positional_features
    from scripts.selection_semantics import deduplicate_latest_rows
except ImportError:  # pragma: no cover - fallback for direct script invocation
    from positional_structure import compute_positional_features  # type: ignore
    from selection_semantics import deduplicate_latest_rows  # type: ignore
    from experiments.benchmark_loader import load_model  # type: ignore


KEY_COLS = ["dataset_name", "model_id", "trace_id", "trace_hash"]
OUTPUT_CSV = "selection_features_full.csv"
OUTPUT_SUMMARY = "feature_extraction_summary.json"
OUTPUT_FAILURES = "feature_extraction_failures.csv"
FAILURE_COLS = [
    "_target_row_idx",
    "dataset_name",
    "model_id",
    "trace_id",
    "trace_hash",
    "model_path",
    "failure_reason",
]


@dataclass
class ModelCacheEntry:
    """Reusable model-level feature bundle."""

    model_path: str
    model_places: int
    model_transitions: int
    model_silent_transitions: int
    xor_splits: int
    xor_joins: int
    and_splits: int
    and_joins: int
    tau_ratio: float
    enabled_initial_count: int
    enabled_final_count: int
    visible_alphabet: set[str]
    visible_transition_count_by_label: Counter
    positional_features: Dict[str, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage B feature table")
    parser.add_argument("--driver-csv", type=Path, required=True)
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def normalize_key_frame(df: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    """Normalize join keys to a stable string representation."""

    work = df.copy()
    for column in key_cols:
        if column in work.columns:
            work[f"__key_{column}"] = work[column].astype("string")
    return work


def parse_trace_activities(value: object) -> List[str]:
    """Parse the canonical pipe-delimited trace representation."""

    if pd.isna(value):
        return []
    text = str(value)
    if not text:
        return []
    return text.split("|")


def visible_transition_label(label: object) -> Optional[str]:
    """Normalize transition labels and suppress silent placeholders."""

    if label is None:
        return None
    text = str(label).strip()
    if not text or text.lower() == "tau":
        return None
    return text


def build_model_cache_entry(model_path: str) -> ModelCacheEntry:
    """Load a model and compute reusable structural features."""

    wf, _, _, _ = load_model(model_path)
    net = wf.net

    place_input_degree: Dict[object, int] = {place: 0 for place in net.places}
    place_output_degree: Dict[object, int] = {place: 0 for place in net.places}
    visible_transition_count_by_label: Counter = Counter()

    xor_splits = 0
    xor_joins = 0
    and_splits = 0
    and_joins = 0
    silent_transitions = 0
    visible_alphabet: set[str] = set()

    for transition in net.transitions:
        preset = net.preset(transition)
        postset = net.postset(transition)
        if len(preset) > 1:
            and_joins += 1
        if len(postset) > 1:
            and_splits += 1

        label = visible_transition_label(getattr(transition, "label", None))
        if label is None:
            silent_transitions += 1
        else:
            visible_alphabet.add(label)
            visible_transition_count_by_label[label] += 1

        for place in preset:
            place_output_degree[place] += 1
        for place in postset:
            place_input_degree[place] += 1

    for place in net.places:
        if place_output_degree[place] > 1:
            xor_splits += 1
        if place_input_degree[place] > 1:
            xor_joins += 1

    transition_count = len(net.transitions)
    tau_ratio = (
        float(silent_transitions / transition_count)
        if transition_count > 0
        else 0.0
    )
    positional_features = compute_positional_features(wf)

    return ModelCacheEntry(
        model_path=model_path,
        model_places=len(net.places),
        model_transitions=transition_count,
        model_silent_transitions=silent_transitions,
        xor_splits=xor_splits,
        xor_joins=xor_joins,
        and_splits=and_splits,
        and_joins=and_joins,
        tau_ratio=tau_ratio,
        enabled_initial_count=len(net.enabled_transitions(wf.initial_marking)),
        enabled_final_count=len(net.enabled_transitions(wf.final_marking)),
        visible_alphabet=visible_alphabet,
        visible_transition_count_by_label=visible_transition_count_by_label,
        positional_features=positional_features,
    )


def shannon_entropy(counts: Iterable[int]) -> float:
    """Compute Shannon entropy in base 2."""

    total = sum(counts)
    if total <= 0:
        return 0.0

    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def compute_trace_features(trace_activities: List[str]) -> Dict[str, object]:
    """Compute per-trace features from parsed activities."""

    trace_counter = Counter(trace_activities)
    return {
        "trace_length": len(trace_activities),
        "distinct_activities": len(trace_counter),
        "trace_entropy": shannon_entropy(trace_counter.values()),
        "trace_max_repetitions": max(trace_counter.values()) if trace_counter else 0,
        "trace_counter": trace_counter,
        "trace_alphabet": set(trace_counter.keys()),
    }


def compute_interaction_features(
    trace_activities: List[str],
    trace_counter: Counter,
    trace_alphabet: set[str],
    model_entry: ModelCacheEntry,
) -> Dict[str, float]:
    """Compute fast interaction features between the model and the trace."""

    trace_length = len(trace_activities)
    visible_alphabet = model_entry.visible_alphabet

    log_only_count = sum(1 for activity in trace_activities if activity not in visible_alphabet)
    log_only_prop = float(log_only_count / trace_length) if trace_length > 0 else 0.0

    if visible_alphabet:
        model_only_count = sum(1 for label in visible_alphabet if label not in trace_alphabet)
        model_only_prop = float(model_only_count / len(visible_alphabet))
    else:
        model_only_prop = 0.0

    union = visible_alphabet | trace_alphabet
    if union:
        alphabet_overlap = float(len(visible_alphabet & trace_alphabet) / len(union))
    else:
        alphabet_overlap = 1.0

    branching_imbalance = float(
        model_entry.enabled_initial_count / (model_entry.enabled_final_count + 1)
    )

    sync_moves_count = sum(
        count * model_entry.visible_transition_count_by_label.get(activity, 0)
        for activity, count in trace_counter.items()
    )
    sp_places = model_entry.model_places + trace_length + 1
    sp_transitions = model_entry.model_transitions + trace_length + sync_moves_count
    sp_branching_factor = float(sp_transitions / sp_places) if sp_places > 0 else 0.0

    return {
        "log_only_prop": log_only_prop,
        "model_only_prop": model_only_prop,
        "alphabet_overlap": alphabet_overlap,
        "branching_imbalance": branching_imbalance,
        "sp_places": int(sp_places),
        "sp_transitions": int(sp_transitions),
        "sp_branching_factor": sp_branching_factor,
    }


def uses_me_method(method_name: object) -> bool:
    """Detect ME-based heuristic names."""

    return isinstance(method_name, str) and method_name.endswith("_me")


def build_failure_row(row: Dict[str, object], reason: str) -> Dict[str, object]:
    """Capture a failed feature extraction row."""

    return {
        "_target_row_idx": int(row["_target_row_idx"]),
        "dataset_name": row.get("dataset_name"),
        "model_id": row.get("model_id"),
        "trace_id": row.get("trace_id"),
        "trace_hash": row.get("trace_hash"),
        "model_path": row.get("model_path"),
        "failure_reason": reason,
    }


def rank_fallback_candidate(path: Path) -> tuple[int, int, str]:
    """Rank replacement model paths deterministically."""

    parts = path.parts
    if "data" in parts:
        priority = 0
    elif "tmp_smoke" in parts and "models_selected" in parts:
        priority = 1
    else:
        priority = 2
    return (priority, len(parts), str(path))


def resolve_model_path(
    model_path: str,
    basename_cache: Dict[str, Optional[str]],
) -> Optional[str]:
    """Resolve stale model paths by basename when the recorded path is missing."""

    if os.path.exists(model_path):
        return model_path

    basename = Path(model_path).name
    if not basename:
        return None
    if basename in basename_cache:
        return basename_cache[basename]

    candidates = sorted(REPO_ROOT.rglob(basename), key=rank_fallback_candidate)
    resolved = str(candidates[0]) if candidates else None
    basename_cache[basename] = resolved
    return resolved


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_wall_start = time.perf_counter()

    driver_df = pd.read_csv(args.driver_csv)
    driver_df.insert(0, "_target_row_idx", np.arange(len(driver_df), dtype=int))
    missing_key_cols = [column for column in KEY_COLS if column not in driver_df.columns]
    if missing_key_cols:
        raise ValueError(f"Driver CSV is missing required key columns: {missing_key_cols}")

    aggregate_df = pd.read_csv(args.aggregate_csv)
    deduped_aggregate = deduplicate_latest_rows(aggregate_df, KEY_COLS)

    aggregate_cols = [
        column
        for column in (
            *KEY_COLS,
            "model_path",
            "model_name",
            "trace_activities",
        )
        if column in deduped_aggregate.columns
    ]
    aggregate_base = deduped_aggregate.loc[:, aggregate_cols].copy()

    driver_norm = normalize_key_frame(driver_df, KEY_COLS)
    aggregate_norm = normalize_key_frame(aggregate_base, KEY_COLS)
    merge_keys = [f"__key_{column}" for column in KEY_COLS]
    driver_enriched = driver_norm.merge(
        aggregate_norm.drop(columns=[column for column in KEY_COLS if column in aggregate_norm.columns]),
        on=merge_keys,
        how="left",
        suffixes=("", "_aggregate"),
    )

    model_cache: Dict[str, ModelCacheEntry] = {}
    missing_model_path_cache: Dict[str, Optional[str]] = {}
    model_cache_hits = 0
    model_cache_misses = 0
    model_load_failures = 0
    missing_model_path_fallback_hits = 0
    missing_model_path_fallback_misses = 0
    extraction_times_ms: List[float] = []
    failures: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []

    for row_dict in driver_enriched.to_dict(orient="records"):
        row_start = time.perf_counter()

        feature_row: Dict[str, object] = {
            "target_forward_method": row_dict.get("forward_expansions_method"),
            "target_dibbs_method": row_dict.get("dibbs_expansions_method"),
            "target_mm_method": row_dict.get("bidir_mm_expansions_method"),
            "uses_potentially_inconsistent_dibbs_target": bool(
                uses_me_method(row_dict.get("dibbs_expansions_method"))
            ),
            "uses_potentially_inconsistent_mm_target": bool(
                uses_me_method(row_dict.get("bidir_mm_expansions_method"))
            ),
            "uses_potentially_inconsistent_bidirectional_target": bool(
                uses_me_method(row_dict.get("dibbs_expansions_method"))
                or uses_me_method(row_dict.get("bidir_mm_expansions_method"))
            ),
        }

        model_path = row_dict.get("model_path")
        if pd.isna(model_path) or not model_path:
            extraction_ms = (time.perf_counter() - row_start) * 1000.0
            feature_row["extraction_time_ms"] = extraction_ms
            extraction_times_ms.append(extraction_ms)
            feature_rows.append(feature_row)
            failures.append(build_failure_row(row_dict, "missing_aggregate_enrichment"))
            continue

        model_path = str(model_path)
        resolved_model_path = resolve_model_path(model_path, missing_model_path_cache)
        if resolved_model_path is None:
            missing_model_path_fallback_misses += 1
            extraction_ms = (time.perf_counter() - row_start) * 1000.0
            feature_row["extraction_time_ms"] = extraction_ms
            extraction_times_ms.append(extraction_ms)
            feature_rows.append(feature_row)
            failures.append(build_failure_row(row_dict, "model_path_not_found"))
            continue
        if resolved_model_path != model_path:
            missing_model_path_fallback_hits += 1
            model_path = resolved_model_path

        if model_path in model_cache:
            model_entry = model_cache[model_path]
            model_cache_hits += 1
        else:
            try:
                model_entry = build_model_cache_entry(model_path)
            except Exception as exc:  # pragma: no cover - exercised in smoke runs
                model_load_failures += 1
                extraction_ms = (time.perf_counter() - row_start) * 1000.0
                feature_row["extraction_time_ms"] = extraction_ms
                extraction_times_ms.append(extraction_ms)
                feature_rows.append(feature_row)
                failures.append(build_failure_row(row_dict, f"model_load_failed:{exc}"))
                continue

            model_cache[model_path] = model_entry
            model_cache_misses += 1

        trace_activities = parse_trace_activities(row_dict.get("trace_activities"))
        trace_features = compute_trace_features(trace_activities)
        interaction_features = compute_interaction_features(
            trace_activities=trace_activities,
            trace_counter=trace_features["trace_counter"],
            trace_alphabet=trace_features["trace_alphabet"],
            model_entry=model_entry,
        )

        feature_row.update(
            {
                "model_path": model_entry.model_path,
                "xor_splits": model_entry.xor_splits,
                "xor_joins": model_entry.xor_joins,
                "and_splits": model_entry.and_splits,
                "and_joins": model_entry.and_joins,
                "tau_ratio": model_entry.tau_ratio,
                "trace_length": trace_features["trace_length"],
                "distinct_activities": trace_features["distinct_activities"],
                "trace_entropy": trace_features["trace_entropy"],
                "trace_max_repetitions": trace_features["trace_max_repetitions"],
            }
        )
        feature_row.update(model_entry.positional_features)
        feature_row.update(interaction_features)

        extraction_ms = (time.perf_counter() - row_start) * 1000.0
        feature_row["extraction_time_ms"] = extraction_ms
        extraction_times_ms.append(extraction_ms)
        feature_rows.append(feature_row)

    features_df = pd.DataFrame(feature_rows)
    output_df = pd.concat(
        [driver_df.reset_index(drop=True), features_df.reset_index(drop=True)],
        axis=1,
    )
    output_df = output_df.sort_values("_target_row_idx", kind="stable").reset_index(drop=True)

    key_check = output_df.loc[:, ["_target_row_idx", *KEY_COLS]]
    if len(output_df) != len(driver_df):
        raise ValueError("Feature output row count does not match Stage A driver row count")
    if not key_check.equals(driver_df.loc[:, ["_target_row_idx", *KEY_COLS]]):
        raise ValueError("Feature output key order does not match Stage A driver order")
    if output_df.duplicated(subset=KEY_COLS, keep=False).any():
        raise ValueError("Feature output contains duplicate Stage A keys")

    failures_df = pd.DataFrame(failures, columns=FAILURE_COLS)
    output_df.to_csv(args.out_dir / OUTPUT_CSV, index=False)
    failures_df.to_csv(args.out_dir / OUTPUT_FAILURES, index=False)

    extraction_series = pd.Series(extraction_times_ms, dtype=float)
    summary = {
        "driver_csv": str(args.driver_csv.resolve()),
        "aggregate_csv": str(args.aggregate_csv.resolve()),
        "output_csv": str((args.out_dir / OUTPUT_CSV).resolve()),
        "lp_features_included": False,
        "deferred_lp_columns": ["h_f", "h_b", "heuristic_asymmetry"],
        "rows_in_driver": int(len(driver_df)),
        "rows_in_output": int(len(output_df)),
        "rows_with_failures": int(len(failures_df)),
        "unique_models_loaded": int(len(model_cache)),
        "model_cache_hits": int(model_cache_hits),
        "model_cache_misses": int(model_cache_misses),
        "model_load_failures": int(model_load_failures),
        "missing_model_path_fallback_hits": int(missing_model_path_fallback_hits),
        "missing_model_path_fallback_misses": int(missing_model_path_fallback_misses),
        "total_wall_time_ms": float((time.perf_counter() - total_wall_start) * 1000.0),
        "total_extraction_time_ms": float(extraction_series.sum()) if not extraction_series.empty else 0.0,
        "mean_extraction_time_ms": float(extraction_series.mean()) if not extraction_series.empty else 0.0,
        "median_extraction_time_ms": float(extraction_series.median()) if not extraction_series.empty else 0.0,
    }
    (args.out_dir / OUTPUT_SUMMARY).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {args.out_dir / OUTPUT_CSV}")
    print(f"Wrote: {args.out_dir / OUTPUT_FAILURES}")
    print(f"Wrote: {args.out_dir / OUTPUT_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
