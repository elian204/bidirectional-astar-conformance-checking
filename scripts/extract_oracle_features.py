#!/usr/bin/env python3
"""Extract oracle LP features using the existing marking-equation heuristic."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from core.synchronous_product import SynchronousProduct
    from core.trace_model import build_trace_net
    from experiments.benchmark_loader import load_model
    from heuristics.marking_equation import create_marking_equation_heuristic
    from scripts.feature_engineering import parse_trace_activities, resolve_model_path
except ImportError:  # pragma: no cover - fallback for direct script invocation
    from core.synchronous_product import SynchronousProduct  # type: ignore
    from core.trace_model import build_trace_net  # type: ignore
    from experiments.benchmark_loader import load_model  # type: ignore
    from heuristics.marking_equation import create_marking_equation_heuristic  # type: ignore
    from feature_engineering import parse_trace_activities, resolve_model_path  # type: ignore


KEY_COLS = ["dataset_name", "model_id", "trace_id", "trace_hash"]
OUTPUT_CSV = "selection_features_oracle.csv"
OUTPUT_FAILURES = "selection_features_oracle_failures.csv"
OUTPUT_SUMMARY = "selection_features_oracle_summary.json"
CHECKPOINT_CSV = "selection_features_oracle.checkpoint.csv"
FAILURE_COLS = [
    "_target_row_idx",
    "dataset_name",
    "model_id",
    "trace_id",
    "trace_hash",
    "model_path",
    "failure_reason",
]

_WORKER_MODEL_CACHE: Dict[str, object] = {}
_WORKER_LP_TIMEOUT_SECONDS: Optional[float] = None


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description="Extract oracle LP features")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("tmp_smoke/stage_b_features_current/selection_features_full.csv"),
    )
    parser.add_argument(
        "--trace-source-csv",
        type=Path,
        default=Path(
            "tmp_smoke/aggregated_current/analysis_ready/selection_trace_table_apriori.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp_smoke/stage_b_features_current"),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--lp-timeout-seconds",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke runs.",
    )
    return parser


def normalize_key_frame(df: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    """Normalize join keys to a stable string representation."""

    work = df.copy()
    for column in key_cols:
        work[f"__key_{column}"] = work[column].astype("string")
    return work


def build_failure_row(row: Dict[str, object], reason: str) -> Dict[str, object]:
    """Capture a failed oracle extraction row."""

    return {
        "_target_row_idx": int(row["_target_row_idx"]),
        "dataset_name": row.get("dataset_name"),
        "model_id": row.get("model_id"),
        "trace_id": row.get("trace_id"),
        "trace_hash": row.get("trace_hash"),
        "model_path": row.get("model_path"),
        "failure_reason": reason,
    }


def deduplicate_oracle_input(
    df: pd.DataFrame,
    subset_cols: Sequence[str],
    order_col: Optional[str] = None,
) -> tuple[pd.DataFrame, int]:
    """Keep the last stable row for each logical oracle input key."""

    work = df.copy()
    if order_col and order_col in work.columns:
        work = work.sort_values(order_col, kind="stable")
    else:
        work = work.reset_index(drop=False).rename(columns={"index": "__row_order"})
        work = work.sort_values("__row_order", kind="stable")

    before = len(work)
    work = work.drop_duplicates(subset=list(subset_cols), keep="last").copy()
    if "__row_order" in work.columns:
        work = work.drop(columns="__row_order")
    return work.reset_index(drop=True), int(before - len(work))


def prepare_driver(
    features_csv: Path,
    trace_source_csv: Path,
    limit: Optional[int] = None,
) -> tuple[pd.DataFrame, int, int, Dict[str, int]]:
    """Prepare the oracle extraction driver table."""

    driver_df = pd.read_csv(features_csv)
    if "_target_row_idx" not in driver_df.columns:
        driver_df.insert(0, "_target_row_idx", np.arange(len(driver_df), dtype=int))

    required_driver_cols = ["_target_row_idx", *KEY_COLS, "model_path", "trace_length"]
    missing_driver_cols = [
        column for column in required_driver_cols if column not in driver_df.columns
    ]
    if missing_driver_cols:
        raise ValueError(f"Feature CSV is missing required columns: {missing_driver_cols}")
    driver_df, driver_dup_drops = deduplicate_oracle_input(
        driver_df,
        KEY_COLS,
        order_col="_target_row_idx",
    )

    trace_df = pd.read_csv(trace_source_csv)
    required_trace_cols = [*KEY_COLS, "trace_activities"]
    missing_trace_cols = [
        column for column in required_trace_cols if column not in trace_df.columns
    ]
    if missing_trace_cols:
        raise ValueError(
            f"Trace source CSV is missing required columns: {missing_trace_cols}"
        )
    trace_df = trace_df.loc[:, required_trace_cols].copy()
    trace_df, trace_dup_drops = deduplicate_oracle_input(trace_df, KEY_COLS)

    driver_norm = normalize_key_frame(driver_df, KEY_COLS)
    trace_norm = normalize_key_frame(trace_df, KEY_COLS)
    merge_keys = [f"__key_{column}" for column in KEY_COLS]
    merged = driver_norm.merge(
        trace_norm.drop(columns=KEY_COLS),
        on=merge_keys,
        how="left",
    )
    merged = merged.sort_values("_target_row_idx", kind="stable").reset_index(drop=True)
    if limit is not None:
        merged = merged.head(limit).copy()

    basename_cache: Dict[str, Optional[str]] = {}
    fallback_hits = 0
    fallback_misses = 0
    resolved_paths: List[Optional[str]] = []
    for model_path in merged["model_path"].astype("string"):
        if pd.isna(model_path) or not model_path:
            resolved_paths.append(None)
            fallback_misses += 1
            continue

        resolved = resolve_model_path(str(model_path), basename_cache)
        if resolved is None:
            fallback_misses += 1
        elif resolved != str(model_path):
            fallback_hits += 1
        resolved_paths.append(resolved)

    merged["resolved_model_path"] = resolved_paths
    return merged, fallback_hits, fallback_misses, {
        "driver_duplicate_rows_dropped": driver_dup_drops,
        "trace_duplicate_rows_dropped": trace_dup_drops,
    }


def init_worker(lp_timeout_seconds: Optional[float]) -> None:
    """Initialize per-process state for oracle extraction."""

    global _WORKER_MODEL_CACHE, _WORKER_LP_TIMEOUT_SECONDS
    _WORKER_MODEL_CACHE = {}
    _WORKER_LP_TIMEOUT_SECONDS = lp_timeout_seconds


def load_workflow_cached(model_path: str):
    """Load a workflow net once per process."""

    if model_path not in _WORKER_MODEL_CACHE:
        wf, _, _, _ = load_model(model_path)
        _WORKER_MODEL_CACHE[model_path] = wf
    return _WORKER_MODEL_CACHE[model_path]


def extract_oracle_row(row: Dict[str, object]) -> Dict[str, object]:
    """Extract oracle features for a single instance."""

    row_start = time.perf_counter()
    result: Dict[str, object] = {
        "_target_row_idx": int(row["_target_row_idx"]),
        "dataset_name": row.get("dataset_name"),
        "model_id": row.get("model_id"),
        "trace_id": row.get("trace_id"),
        "trace_hash": row.get("trace_hash"),
        "model_path": row.get("resolved_model_path") or row.get("model_path"),
        "h_f": np.nan,
        "h_b": np.nan,
        "heuristic_asymmetry": np.nan,
        "normalized_h_f": np.nan,
        "normalized_h_b": np.nan,
        "oracle_extraction_time_ms": np.nan,
        "failure_reason": pd.NA,
    }

    try:
        model_path = row.get("resolved_model_path")
        if pd.isna(model_path) or not model_path:
            raise ValueError("model_path_not_found")

        trace_activities = parse_trace_activities(row.get("trace_activities"))
        if row.get("trace_activities") is not None and not trace_activities:
            raise ValueError("trace_activities_empty")

        trace_length = row.get("trace_length")
        if pd.isna(trace_length):
            trace_length = len(trace_activities)
        trace_length = int(trace_length)

        wf = load_workflow_cached(str(model_path))
        tn = build_trace_net(
            trace_activities,
            trace_id=f"{row.get('trace_id', 'trace')}_{row.get('trace_hash', 'hash')}",
        )
        sp = SynchronousProduct(wf, tn)

        h_forward = create_marking_equation_heuristic(
            sp,
            direction="forward",
            timeout_seconds=_WORKER_LP_TIMEOUT_SECONDS,
        )
        h_backward = create_marking_equation_heuristic(
            sp,
            direction="backward",
            timeout_seconds=_WORKER_LP_TIMEOUT_SECONDS,
        )

        h_f = float(h_forward.estimate(sp.initial_marking))
        h_b = float(h_backward.estimate(sp.final_marking))
        result["h_f"] = h_f
        result["h_b"] = h_b
        result["heuristic_asymmetry"] = abs(h_f - h_b)
        result["normalized_h_f"] = h_f / max(trace_length, 1)
        result["normalized_h_b"] = h_b / max(trace_length, 1)
    except Exception as exc:  # pragma: no cover - exercised in smoke/full runs
        result["failure_reason"] = str(exc)

    result["oracle_extraction_time_ms"] = (
        time.perf_counter() - row_start
    ) * 1000.0
    return result


def write_checkpoint(checkpoint_path: Path, rows: List[Dict[str, object]]) -> None:
    """Write a checkpoint snapshot."""

    checkpoint_df = pd.DataFrame(rows).sort_values("_target_row_idx", kind="stable")
    checkpoint_df.to_csv(checkpoint_path, index=False)


def run_serial(
    rows_to_process: List[Dict[str, object]],
    checkpoint_path: Path,
    checkpoint_every: int,
    lp_timeout_seconds: Optional[float],
    existing_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Run oracle extraction serially."""

    init_worker(lp_timeout_seconds)
    rows = list(existing_rows)
    for idx, row in enumerate(rows_to_process, start=1):
        rows.append(extract_oracle_row(row))
        if checkpoint_every > 0 and idx % checkpoint_every == 0:
            write_checkpoint(checkpoint_path, rows)
    return rows


def run_parallel(
    rows_to_process: List[Dict[str, object]],
    checkpoint_path: Path,
    checkpoint_every: int,
    lp_timeout_seconds: Optional[float],
    jobs: int,
    existing_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Run oracle extraction with process-based parallelism."""

    rows = list(existing_rows)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(lp_timeout_seconds,),
    ) as executor:
        futures = [executor.submit(extract_oracle_row, row) for row in rows_to_process]
        for idx, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if checkpoint_every > 0 and idx % checkpoint_every == 0:
                write_checkpoint(checkpoint_path, rows)
                print(f"Checkpointed {idx} new rows")
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    """Run oracle feature extraction."""

    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_wall_start = time.perf_counter()

    driver_df, fallback_hits, fallback_misses, dedup_meta = prepare_driver(
        features_csv=args.features_csv,
        trace_source_csv=args.trace_source_csv,
        limit=args.limit,
    )

    checkpoint_path = args.out_dir / CHECKPOINT_CSV
    existing_rows: List[Dict[str, object]] = []
    processed_ids: set[int] = set()
    if args.resume and checkpoint_path.exists():
        checkpoint_df = pd.read_csv(checkpoint_path)
        existing_rows = checkpoint_df.to_dict(orient="records")
        processed_ids = set(pd.to_numeric(checkpoint_df["_target_row_idx"]).astype(int))

    rows_to_process = [
        row
        for row in driver_df.to_dict(orient="records")
        if int(row["_target_row_idx"]) not in processed_ids
    ]

    if args.jobs <= 1:
        all_rows = run_serial(
            rows_to_process=rows_to_process,
            checkpoint_path=checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            lp_timeout_seconds=args.lp_timeout_seconds,
            existing_rows=existing_rows,
        )
    else:
        all_rows = run_parallel(
            rows_to_process=rows_to_process,
            checkpoint_path=checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            lp_timeout_seconds=args.lp_timeout_seconds,
            jobs=args.jobs,
            existing_rows=existing_rows,
        )

    oracle_df = (
        pd.DataFrame(all_rows)
        .sort_values("_target_row_idx", kind="stable")
        .reset_index(drop=True)
    )
    key_check = oracle_df.loc[:, ["_target_row_idx", *KEY_COLS]]
    expected_keys = driver_df.loc[:, ["_target_row_idx", *KEY_COLS]].reset_index(drop=True)
    if len(oracle_df) != len(driver_df):
        raise ValueError("Oracle output row count does not match driver row count")
    if not key_check.equals(expected_keys):
        raise ValueError("Oracle output key order does not match driver order")
    if oracle_df.duplicated(subset=KEY_COLS, keep=False).any():
        raise ValueError("Oracle output contains duplicate keys")

    oracle_path = args.out_dir / OUTPUT_CSV
    failures_path = args.out_dir / OUTPUT_FAILURES
    summary_path = args.out_dir / OUTPUT_SUMMARY

    failure_mask = oracle_df["failure_reason"].notna()
    failures_df = pd.DataFrame(
        [
            build_failure_row(row, str(row["failure_reason"]))
            for row in oracle_df.loc[failure_mask].to_dict(orient="records")
        ],
        columns=FAILURE_COLS,
    )

    oracle_output_df = oracle_df.drop(columns=["failure_reason"])
    oracle_output_df.to_csv(oracle_path, index=False)
    failures_df.to_csv(failures_path, index=False)

    extraction_series = pd.to_numeric(
        oracle_output_df["oracle_extraction_time_ms"],
        errors="coerce",
    )
    summary = {
        "features_csv": str(args.features_csv.resolve()),
        "trace_source_csv": str(args.trace_source_csv.resolve()),
        "output_csv": str(oracle_path.resolve()),
        "rows_in_driver": int(len(driver_df)),
        "rows_in_output": int(len(oracle_output_df)),
        "rows_with_failures": int(len(failures_df)),
        **dedup_meta,
        "resolved_model_path_fallback_hits": int(fallback_hits),
        "resolved_model_path_fallback_misses": int(fallback_misses),
        "jobs": int(args.jobs),
        "lp_timeout_seconds": args.lp_timeout_seconds,
        "checkpoint_csv": str(checkpoint_path.resolve()),
        "total_wall_time_ms": float((time.perf_counter() - total_wall_start) * 1000.0),
        "total_oracle_extraction_time_ms": float(extraction_series.sum()),
        "mean_oracle_extraction_time_ms": float(extraction_series.mean()),
        "median_oracle_extraction_time_ms": float(extraction_series.median()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {oracle_path}")
    print(f"Wrote: {failures_path}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
