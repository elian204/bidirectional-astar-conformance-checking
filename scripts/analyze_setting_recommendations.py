#!/usr/bin/env python3
"""Coverage-constrained setting recommender for algorithm-heuristic selection."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.build_selection_analysis_tables import (
        MODEL_FEATURE_COLS,
        MODEL_METADATA_COLS,
        TRACE_FEATURE_COLS,
        enrich_method_rows,
    )
    from scripts.feature_engineering import (
        ModelCacheEntry,
        build_model_cache_entry,
        compute_interaction_features,
        compute_trace_features,
        parse_trace_activities,
        resolve_model_path,
    )
    from scripts.selection_semantics import (
        deduplicate_latest_rows,
        filter_consensus_optimal_rows,
    )
except ImportError:  # pragma: no cover - fallback for direct invocation
    from build_selection_analysis_tables import (  # type: ignore
        MODEL_FEATURE_COLS,
        MODEL_METADATA_COLS,
        TRACE_FEATURE_COLS,
        enrich_method_rows,
    )
    from feature_engineering import (  # type: ignore
        ModelCacheEntry,  # type: ignore
        build_model_cache_entry,
        compute_interaction_features,
        compute_trace_features,
        parse_trace_activities,
        resolve_model_path,
    )
    from selection_semantics import (  # type: ignore
        deduplicate_latest_rows,
        filter_consensus_optimal_rows,
    )


DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments/results_20260310"
INSTANCE_KEYS = [
    "dataset_name",
    "model_id",
    "trace_id",
    "trace_hash",
]
INSTANCE_CONTEXT_COLS = ["model_name", "model_path"]
OUTPUT_INSTANCE_COLS = [
    "dataset_name",
    "model_id",
    "model_name",
    "model_path",
    "trace_id",
    "trace_hash",
]
METHOD_ROW_KEYS = INSTANCE_KEYS + ["method"]
RAW_INSTANCE_COLS = [
    "aggregate_run_name",
    "model_source",
    "trace_activities",
    *TRACE_FEATURE_COLS,
    *MODEL_FEATURE_COLS,
    *MODEL_METADATA_COLS,
]
FAST_FEATURE_COLS = [
    "xor_splits",
    "xor_joins",
    "and_splits",
    "and_joins",
    "tau_ratio",
    "distinct_activities",
    "trace_entropy",
    "trace_max_repetitions",
    "log_only_prop",
    "model_only_prop",
    "alphabet_overlap",
    "branching_imbalance",
    "sp_places",
    "sp_transitions",
    "sp_branching_factor",
]
DEFAULT_ORACLE_FEATURE_COLS = ["normalized_h_f", "normalized_h_b", "h_f", "h_b"]
OPTIONAL_ORACLE_FEATURE_COLS = {"normalized_h_b", "h_b"}
ORACLE_FEATURE_CANDIDATES = [
    "normalized_h_f",
    "normalized_h_b",
    "h_f",
    "h_b",
    "heuristic_asymmetry",
]
ORACLE_NUMERIC_EXCLUDE = {
    "_target_row_idx",
    "oracle_extraction_time_ms",
    *ORACLE_FEATURE_CANDIDATES,
}
NUMERIC_EXCLUDE = {
    "optimal_cost",
    "deviation_cost",
    "time_seconds",
    "cost",
    "sp_nodes",
    "sp_edges",
}
LEAF_ID_COL = "leaf_id"
LABEL_TYPE_METHOD = "method"
LABEL_TYPE_ALGORITHM = "algorithm"
LABEL_TYPE_HEURISTIC = "heuristic"


@dataclass
class SourceFile:
    dataset_name: str
    parent_run_id: str
    source_kind: str
    source_file: Path
    parent_complete: bool
    row_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze coverage-constrained setting recommendations",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing updated experiment results.",
    )
    parser.add_argument(
        "--aggregate-csv",
        type=Path,
        default=None,
        help="Optional prebuilt aggregate CSV path. If missing, it is generated.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for analysis artifacts.",
    )
    parser.add_argument(
        "--group-col",
        default="dataset_name",
        help="Group column for grouped validation.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum tree depth.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=None,
        help="Optional min_samples_leaf override.",
    )
    parser.add_argument(
        "--coverage-tolerance",
        type=float,
        default=0.01,
        help="Absolute coverage tolerance for recommendation candidates.",
    )
    parser.add_argument(
        "--write-appendix-buckets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write descriptive appendix bucket summaries.",
    )
    parser.add_argument(
        "--oracle-features-csv",
        type=Path,
        default=None,
        help="Optional oracle-feature CSV for LP-aware ablation.",
    )
    parser.add_argument(
        "--oracle-feature-cols",
        nargs="+",
        default=DEFAULT_ORACLE_FEATURE_COLS,
        help="Oracle feature columns to include in the optional LP-aware ablation.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=3,
        help="Number of grouped out-of-fold permutation repeats per feature.",
    )
    parser.add_argument(
        "--exclude-predictor-cols",
        nargs="+",
        default=[],
        help="Optional numeric predictor columns to exclude from both the fast-only and oracle-aware variants.",
    )
    parser.add_argument(
        "--primary-metric",
        choices=["time_seconds", "expansions"],
        default="time_seconds",
        help="Primary optimization metric for target construction and recommendation ranking.",
    )
    return parser


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def emit_progress(out_dir: Optional[Path], stage: str, **payload: object) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    message = {"timestamp": timestamp, "stage": stage, **payload}
    print(f"[{timestamp}] {stage}: {json.dumps(payload, sort_keys=True)}", flush=True)
    if out_dir is not None:
        (out_dir / "progress.json").write_text(json.dumps(message, indent=2), encoding="utf-8")


def collect_results_sources(results_root: Path) -> List[SourceFile]:
    sources: List[SourceFile] = []
    for dataset_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        for parent_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            merged = parent_dir / "merged_results.csv"
            if merged.exists():
                sources.append(
                    SourceFile(
                        dataset_name=dataset_dir.name,
                        parent_run_id=parent_dir.name,
                        source_kind="merged",
                        source_file=merged,
                        parent_complete=True,
                        row_count=count_rows(merged),
                    )
                )
                continue

            shard_files = sorted(parent_dir.rglob("results.csv"))
            for shard_csv in shard_files:
                sources.append(
                    SourceFile(
                        dataset_name=dataset_dir.name,
                        parent_run_id=parent_dir.name,
                        source_kind="partial_shard",
                        source_file=shard_csv,
                        parent_complete=False,
                        row_count=count_rows(shard_csv),
                    )
                )
    return sources


def aggregate_sources(sources: Sequence[SourceFile]) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if not sources:
        raise FileNotFoundError("No result CSV sources found")

    frames: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, object]] = []
    for source in sources:
        df = pd.read_csv(source.source_file)
        df["aggregate_run_name"] = source.dataset_name
        df["aggregate_batch_dir"] = str(source.source_file.parent.parent)
        df["aggregate_parent_run_id"] = source.parent_run_id
        df["aggregate_source_kind"] = source.source_kind
        df["aggregate_source_file"] = str(source.source_file)
        df["aggregate_parent_complete"] = int(source.parent_complete)
        frames.append(df)
        manifest_rows.append(
            {
                "dataset_name": source.dataset_name,
                "parent_run_id": source.parent_run_id,
                "source_kind": source.source_kind,
                "source_file": str(source.source_file),
                "parent_complete": int(source.parent_complete),
                "row_count": source.row_count,
            }
        )

    aggregate_df = pd.concat(frames, ignore_index=True, sort=False)
    manifest_df = pd.DataFrame(manifest_rows).sort_values(
        ["dataset_name", "parent_run_id", "source_kind", "source_file"],
        kind="stable",
    )
    summary = {
        "source_files": int(len(sources)),
        "merged_sources": int(sum(s.source_kind == "merged" for s in sources)),
        "partial_shard_sources": int(sum(s.source_kind == "partial_shard" for s in sources)),
        "completed_parent_runs": int(
            len({(s.dataset_name, s.parent_run_id) for s in sources if s.parent_complete})
        ),
        "partial_parent_runs": int(
            len({(s.dataset_name, s.parent_run_id) for s in sources if not s.parent_complete})
        ),
        "total_rows_from_sources": int(sum(s.row_count for s in sources)),
        "datasets": sorted({s.dataset_name for s in sources}),
    }
    return aggregate_df, manifest_df, summary


def load_or_build_aggregate(
    results_root: Path,
    out_dir: Path,
    aggregate_csv: Optional[Path],
) -> tuple[pd.DataFrame, Dict[str, object]]:
    if aggregate_csv is not None and aggregate_csv.exists():
        aggregate_df = pd.read_csv(aggregate_csv, low_memory=False)
        return aggregate_df, {
            "aggregate_csv": str(aggregate_csv.resolve()),
            "source": "prebuilt",
        }

    sources = collect_results_sources(results_root)
    aggregate_df, manifest_df, aggregate_summary = aggregate_sources(sources)
    target_csv = aggregate_csv or (out_dir / "corrected_aggregate.csv")
    target_manifest = target_csv.with_name(target_csv.stem + "_manifest.csv")
    target_summary = target_csv.with_name(target_csv.stem + "_summary.json")
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    aggregate_df.to_csv(target_csv, index=False)
    manifest_df.to_csv(target_manifest, index=False)
    target_summary.write_text(
        json.dumps(
            {
                "aggregate_csv": str(target_csv.resolve()),
                "results_root": str(results_root.resolve()),
                **aggregate_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return aggregate_df, {
        "aggregate_csv": str(target_csv.resolve()),
        "aggregate_manifest_csv": str(target_manifest.resolve()),
        "aggregate_summary_json": str(target_summary.resolve()),
        "source": "generated",
        **aggregate_summary,
    }


def existing_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    return [column for column in columns if column in df.columns]


def build_corrected_frames(aggregate_df: pd.DataFrame) -> Dict[str, object]:
    counts = {
        "rows_aggregated": int(len(aggregate_df)),
    }
    complete_df = aggregate_df.copy()
    if "aggregate_parent_complete" in complete_df.columns:
        complete_flag = pd.to_numeric(complete_df["aggregate_parent_complete"], errors="coerce").fillna(0)
        complete_df = complete_df[complete_flag == 1].copy()
    counts["rows_after_complete_parent_filter"] = int(len(complete_df))

    dedup_df = deduplicate_latest_rows(complete_df, METHOD_ROW_KEYS)
    counts["rows_after_deduplication"] = int(len(dedup_df))

    valid_df = filter_consensus_optimal_rows(dedup_df)
    counts["rows_after_consensus_optimal_filter"] = int(len(valid_df))

    enriched_complete = enrich_method_rows(dedup_df)
    enriched_valid = filter_consensus_optimal_rows(enriched_complete)
    return {
        "counts": counts,
        "complete": enriched_complete,
        "valid": enriched_valid,
    }


def build_instance_base(complete_df: pd.DataFrame, valid_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = existing_columns(complete_df, OUTPUT_INSTANCE_COLS + RAW_INSTANCE_COLS)
    instance_base = (
        complete_df.loc[:, base_cols]
        .drop_duplicates(subset=INSTANCE_KEYS, keep="last")
        .reset_index(drop=True)
    )
    valid_keys = valid_df.loc[:, INSTANCE_KEYS].drop_duplicates()
    instance_base = instance_base.merge(valid_keys, on=INSTANCE_KEYS, how="inner")
    return instance_base.reset_index(drop=True)


def build_fast_features(instance_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if instance_df.empty:
        empty_cols = INSTANCE_KEYS + FAST_FEATURE_COLS
        return (
            pd.DataFrame(columns=empty_cols),
            pd.DataFrame(columns=INSTANCE_KEYS + ["failure_reason"]),
            {"rows_requested": 0, "rows_with_features": 0, "rows_with_failures": 0},
        )

    model_cache: Dict[str, ModelCacheEntry] = {}
    basename_cache: Dict[str, Optional[str]] = {}
    feature_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []

    for row in instance_df.to_dict(orient="records"):
        model_path = row.get("model_path")
        resolved = resolve_model_path(str(model_path), basename_cache) if pd.notna(model_path) else None
        if not resolved:
            failure_rows.append({**{key: row.get(key) for key in INSTANCE_KEYS}, "failure_reason": "model_path_not_found"})
            continue

        try:
            model_entry = model_cache.get(resolved)
            if model_entry is None:
                model_entry = build_model_cache_entry(resolved)
                model_cache[resolved] = model_entry
        except Exception as exc:  # pragma: no cover - exercised in real runs
            failure_rows.append(
                {**{key: row.get(key) for key in INSTANCE_KEYS}, "failure_reason": f"model_load_failed:{exc}"}
            )
            continue

        trace_activities = parse_trace_activities(row.get("trace_activities"))
        trace_features = compute_trace_features(trace_activities)
        interaction_features = compute_interaction_features(
            trace_activities=trace_activities,
            trace_counter=trace_features["trace_counter"],
            trace_alphabet=trace_features["trace_alphabet"],
            model_entry=model_entry,
        )
        feature_rows.append(
            {
                **{key: row.get(key) for key in INSTANCE_KEYS},
                "xor_splits": model_entry.xor_splits,
                "xor_joins": model_entry.xor_joins,
                "and_splits": model_entry.and_splits,
                "and_joins": model_entry.and_joins,
                "tau_ratio": model_entry.tau_ratio,
                "distinct_activities": trace_features["distinct_activities"],
                "trace_entropy": trace_features["trace_entropy"],
                "trace_max_repetitions": trace_features["trace_max_repetitions"],
                **interaction_features,
            }
        )

    features_df = pd.DataFrame(feature_rows)
    failures_df = pd.DataFrame(failure_rows)
    summary = {
        "rows_requested": int(len(instance_df)),
        "rows_with_features": int(len(features_df)),
        "rows_with_failures": int(len(failures_df)),
        "unique_models_loaded": int(len(model_cache)),
    }
    return features_df, failures_df, summary


def load_oracle_features(
    oracle_csv: Optional[Path],
    requested_cols: Sequence[str],
) -> tuple[pd.DataFrame, Dict[str, object]]:
    if oracle_csv is None:
        return pd.DataFrame(columns=INSTANCE_KEYS), {"enabled": False}
    if not oracle_csv.exists():
        raise FileNotFoundError(f"Oracle feature CSV not found: {oracle_csv}")

    oracle_df = pd.read_csv(oracle_csv)
    missing_keys = [column for column in INSTANCE_KEYS if column not in oracle_df.columns]
    if missing_keys:
        raise ValueError(f"Oracle feature CSV is missing key columns: {missing_keys}")
    if oracle_df.duplicated(subset=INSTANCE_KEYS, keep=False).any():
        duplicates = oracle_df.loc[oracle_df.duplicated(subset=INSTANCE_KEYS, keep=False), INSTANCE_KEYS]
        raise ValueError(
            "Oracle feature CSV contains duplicate instance keys: "
            f"{duplicates.drop_duplicates().to_dict(orient='records')[:5]}"
        )

    missing_requested_cols = [column for column in requested_cols if column not in oracle_df.columns]
    missing_required_cols = [
        column for column in missing_requested_cols if column not in OPTIONAL_ORACLE_FEATURE_COLS
    ]
    if missing_required_cols:
        raise ValueError(
            "Oracle feature CSV is missing requested oracle feature columns: "
            f"{missing_required_cols}"
        )
    available_cols = [column for column in requested_cols if column in oracle_df.columns]
    selected_cols = list(dict.fromkeys(list(INSTANCE_KEYS) + available_cols))
    selected_df = oracle_df.loc[:, selected_cols].copy()
    for column in available_cols:
        selected_df[column] = pd.to_numeric(selected_df[column], errors="coerce")

    result_df = selected_df.drop_duplicates(subset=INSTANCE_KEYS, keep="last")
    return result_df, {
        "enabled": True,
        "oracle_features_csv": str(oracle_csv.resolve()),
        "oracle_features_requested": list(requested_cols),
        "oracle_features_available": available_cols,
        "oracle_rows_loaded": int(len(result_df)),
    }


def prepare_numeric_metric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce") if column in df.columns else pd.Series(np.nan, index=df.index)


def secondary_metric(primary_metric: str) -> str:
    return "expansions" if primary_metric == "time_seconds" else "time_seconds"


def target_metric_column(target_prefix: str, primary_metric: str) -> str:
    return f"{target_prefix}_{primary_metric}"


def oracle_metric_column(primary_metric: str) -> str:
    return f"oracle_{primary_metric}"


def metric_display_name(primary_metric: str) -> str:
    return "runtime" if primary_metric == "time_seconds" else "expansions"


def regret_units(primary_metric: str) -> str:
    return "seconds" if primary_metric == "time_seconds" else "expansions"


def select_best_rows(
    df: pd.DataFrame,
    label_col: str,
    primary_metric: str,
    extra_sort_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    work = df.copy()
    work["expansions"] = prepare_numeric_metric(work, "expansions")
    work["time_seconds"] = prepare_numeric_metric(work, "time_seconds")
    secondary = secondary_metric(primary_metric)
    work = work.dropna(subset=[primary_metric]).copy()
    sort_cols = INSTANCE_KEYS + [primary_metric, secondary, label_col]
    if extra_sort_cols:
        sort_cols.extend(extra_sort_cols)
    work = work.sort_values(sort_cols, kind="stable")
    return work.drop_duplicates(subset=INSTANCE_KEYS, keep="first").reset_index(drop=True)


def build_target_tables(
    valid_df: pd.DataFrame,
    primary_metric: str = "expansions",
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    valid = valid_df.copy()
    valid["expansions"] = prepare_numeric_metric(valid, "expansions")
    valid["time_seconds"] = prepare_numeric_metric(valid, "time_seconds")
    secondary = secondary_metric(primary_metric)
    valid = valid.dropna(subset=[primary_metric]).copy()

    best_method_rows = select_best_rows(valid, "method", primary_metric)
    best_method = best_method_rows.loc[:, INSTANCE_KEYS + ["method", "time_seconds", "expansions"]].rename(
        columns={
            "method": "best_method",
            "time_seconds": "best_method_time_seconds",
            "expansions": "best_method_expansions",
        }
    )

    algorithm_valid = (
        valid.sort_values(INSTANCE_KEYS + ["algorithm", primary_metric, secondary, "method"], kind="stable")
        .drop_duplicates(subset=INSTANCE_KEYS + ["algorithm"], keep="first")
        .reset_index(drop=True)
    )
    best_algorithm_rows = select_best_rows(algorithm_valid, "algorithm", primary_metric)
    best_algorithm = best_algorithm_rows.loc[:, INSTANCE_KEYS + ["algorithm", "time_seconds", "expansions"]].rename(
        columns={
            "algorithm": "best_algorithm",
            "time_seconds": "best_algorithm_time_seconds",
            "expansions": "best_algorithm_expansions",
        }
    )

    heuristic_valid = (
        valid.sort_values(INSTANCE_KEYS + ["heuristic", primary_metric, secondary, "method"], kind="stable")
        .drop_duplicates(subset=INSTANCE_KEYS + ["heuristic"], keep="first")
        .reset_index(drop=True)
    )
    best_heuristic_rows = select_best_rows(heuristic_valid, "heuristic", primary_metric)
    best_heuristic = best_heuristic_rows.loc[:, INSTANCE_KEYS + ["heuristic", "time_seconds", "expansions"]].rename(
        columns={
            "heuristic": "best_heuristic",
            "time_seconds": "best_heuristic_time_seconds",
            "expansions": "best_heuristic_expansions",
        }
    )

    targets = best_method.merge(best_algorithm, on=INSTANCE_KEYS, how="left").merge(
        best_heuristic, on=INSTANCE_KEYS, how="left"
    )
    return targets, {
        LABEL_TYPE_METHOD: valid,
        LABEL_TYPE_ALGORITHM: algorithm_valid,
        LABEL_TYPE_HEURISTIC: heuristic_valid,
    }


def build_label_observations(
    complete_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    label_type: str,
    primary_metric: str,
) -> pd.DataFrame:
    oracle_col = oracle_metric_column(primary_metric)
    if label_type == LABEL_TYPE_METHOD:
        label_col = "method"
        obs = complete_df.copy()
        obs["label"] = obs["method"]
        obs["valid"] = False
        if not obs.empty:
            valid_index = filter_consensus_optimal_rows(obs).index
            obs.loc[valid_index, "valid"] = True
        obs["expansions"] = prepare_numeric_metric(obs, "expansions")
        obs["time_seconds"] = prepare_numeric_metric(obs, "time_seconds")
        merged = obs.merge(
            targets_df.loc[:, INSTANCE_KEYS + ["best_method", target_metric_column("best_method", primary_metric)]],
            on=INSTANCE_KEYS,
            how="inner",
        )
        merged[oracle_col] = merged[target_metric_column("best_method", primary_metric)]
        merged["winner_flag"] = merged["label"].eq(merged["best_method"])
        return merged.loc[:, INSTANCE_KEYS + ["label", "valid", "expansions", "time_seconds", oracle_col, "winner_flag"]]

    group_col = "algorithm" if label_type == LABEL_TYPE_ALGORITHM else "heuristic"
    target_col = "best_algorithm" if label_type == LABEL_TYPE_ALGORITHM else "best_heuristic"
    target_metric_col = target_metric_column(target_col, primary_metric)

    complete = complete_df.copy()
    complete["valid"] = False
    if not complete.empty:
        valid_index = filter_consensus_optimal_rows(complete).index
        complete.loc[valid_index, "valid"] = True
    complete["expansions"] = prepare_numeric_metric(complete, "expansions")
    complete["time_seconds"] = prepare_numeric_metric(complete, "time_seconds")

    rows: List[pd.DataFrame] = []
    for _, group in complete.groupby(INSTANCE_KEYS + [group_col], dropna=False):
        valid_group = group[group["valid"] & group[primary_metric].notna()].copy()
        if valid_group.empty:
            rows.append(
                group.iloc[[0]].assign(
                    label=group.iloc[0][group_col],
                    valid=False,
                    expansions=np.nan,
                    time_seconds=np.nan,
                )
            )
        else:
            sorted_group = valid_group.sort_values(
                [primary_metric, secondary_metric(primary_metric), "method"],
                kind="stable",
            )
            best = sorted_group.iloc[[0]].assign(label=sorted_group.iloc[0][group_col])
            rows.append(best)

    obs = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(columns=INSTANCE_KEYS + ["label"])
    merged = obs.merge(targets_df.loc[:, INSTANCE_KEYS + [target_col, target_metric_col]], on=INSTANCE_KEYS, how="inner")
    merged[oracle_col] = merged[target_metric_col]
    merged["winner_flag"] = merged["label"].eq(merged[target_col])
    return merged.loc[:, INSTANCE_KEYS + ["label", "valid", "expansions", "time_seconds", oracle_col, "winner_flag"]]


def compute_predictor_columns(
    modeling_df: pd.DataFrame,
    include_columns: Optional[Sequence[str]] = None,
    exclude_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    include_set = {
        column
        for column in (include_columns or [])
        if column in modeling_df.columns and pd.api.types.is_numeric_dtype(modeling_df[column])
    }
    exclude_set = {
        column
        for column in (exclude_columns or [])
        if column in modeling_df.columns and pd.api.types.is_numeric_dtype(modeling_df[column])
    }
    excluded = set(INSTANCE_KEYS) | {
        "aggregate_run_name",
        "model_source",
        "trace_activities",
        "miner_family",
        "miner_parameter_name",
        "best_method",
        "best_algorithm",
        "best_heuristic",
        "best_method_time_seconds",
        "best_algorithm_time_seconds",
        "best_heuristic_time_seconds",
        "best_method_expansions",
        "best_algorithm_expansions",
        "best_heuristic_expansions",
        LEAF_ID_COL,
    } | NUMERIC_EXCLUDE | ORACLE_NUMERIC_EXCLUDE
    excluded -= include_set
    excluded |= exclude_set

    numeric_cols = [
        column
        for column in modeling_df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(modeling_df[column])
    ]
    return sorted(numeric_cols)


def default_min_samples_leaf(n_rows: int, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    return max(20, int(math.ceil(0.05 * max(n_rows, 1))))


def make_leaf_rule_map(model: DecisionTreeClassifier, feature_names: Sequence[str]) -> Dict[int, str]:
    tree = model.tree_
    feature_name_lookup = [
        feature_names[idx] if idx != -2 else "undefined" for idx in tree.feature
    ]
    rules: Dict[int, str] = {}

    def walk(node_id: int, clauses: List[str]) -> None:
        feature_idx = tree.feature[node_id]
        if feature_idx == -2:
            rules[node_id] = " and ".join(clauses) if clauses else "all instances"
            return

        feature_name = feature_name_lookup[node_id]
        threshold = tree.threshold[node_id]
        walk(tree.children_left[node_id], clauses + [f"{feature_name} <= {threshold:.6f}"])
        walk(tree.children_right[node_id], clauses + [f"{feature_name} > {threshold:.6f}"])

    walk(0, [])
    return rules


def fit_tree_with_guardrail(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: int,
    min_samples_leaf: int,
) -> Dict[str, object]:
    if X.empty:
        raise ValueError("Cannot fit tree on empty dataset")

    feature_columns = [column for column in X.columns if X[column].notna().any()]
    if not feature_columns:
        raise ValueError("No usable feature columns remain after dropping all-NA predictors")
    X_used = X.loc[:, feature_columns]
    imputer = SimpleImputer(strategy="median")
    X_i = imputer.fit_transform(X_used)

    def train_tree(min_leaf: int) -> tuple[DecisionTreeClassifier, np.ndarray]:
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=42,
        )
        tree.fit(X_i, y)
        leaf_ids = tree.apply(X_i)
        return tree, leaf_ids

    initial_tree, initial_leaf_ids = train_tree(min_samples_leaf)
    leaf_support = pd.Series(initial_leaf_ids).value_counts(normalize=True)
    initial_degenerate = len(leaf_support) <= 1 or float(leaf_support.max()) > 0.95

    realized_min_leaf = min_samples_leaf
    guardrail_triggered = False
    model = initial_tree
    leaf_ids = initial_leaf_ids

    if initial_degenerate:
        guardrail_triggered = True
        fallback_min_leaf = max(10, min_samples_leaf // 2)
        if fallback_min_leaf != min_samples_leaf:
            model, leaf_ids = train_tree(fallback_min_leaf)
            realized_min_leaf = fallback_min_leaf

    leaf_rules = make_leaf_rule_map(model, feature_columns)
    return {
        "model": model,
        "imputer": imputer,
        "leaf_ids": leaf_ids,
        "leaf_rules": leaf_rules,
        "feature_columns": feature_columns,
        "initial_leaf_count": int(len(pd.Series(initial_leaf_ids).unique())),
        "initial_max_leaf_support_rate": float(leaf_support.max()) if not leaf_support.empty else 1.0,
        "guardrail_triggered": bool(guardrail_triggered),
        "initial_min_samples_leaf": int(min_samples_leaf),
        "realized_min_samples_leaf": int(realized_min_leaf),
    }


def summarize_partition_recommendations(
    partitions: pd.DataFrame,
    observations: pd.DataFrame,
    partition_cols: Sequence[str],
    coverage_tolerance: float,
    primary_metric: str = "expansions",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    partition_cols = list(partition_cols)
    labels = sorted(observations["label"].dropna().astype(str).unique().tolist())
    partition_key_cols = list(dict.fromkeys(list(INSTANCE_KEYS) + partition_cols))
    partition_keys = partitions.loc[:, partition_key_cols].drop_duplicates().reset_index(drop=True)
    support_df = partition_keys.groupby(partition_cols, dropna=False).size().reset_index(name="partition_support")
    merged = observations.merge(partition_keys, on=INSTANCE_KEYS, how="inner")

    oracle_col = oracle_metric_column(primary_metric)
    rows: List[Dict[str, object]] = []
    recommendations: List[Dict[str, object]] = []
    for support_row in support_df.to_dict(orient="records"):
        support = int(support_row["partition_support"])
        partition_filter = pd.Series(True, index=merged.index)
        for column in partition_cols:
            partition_filter &= merged[column].eq(support_row[column])
        subset = merged[partition_filter].copy()
        for label in labels:
            label_subset = subset[subset["label"] == label]
            valid_mask = label_subset["valid"].fillna(False)
            valid_count = int(valid_mask.sum())
            coverage = float(valid_count / support) if support else 0.0
            regrets = (
                label_subset.loc[valid_mask, primary_metric] - label_subset.loc[valid_mask, oracle_col]
            )
            winner_share = float(label_subset["winner_flag"].fillna(False).sum() / support) if support else 0.0
            rows.append(
                {
                    **{column: support_row[column] for column in partition_cols},
                    "label": label,
                    "partition_support": support,
                    "valid_count": valid_count,
                    "valid_coverage": coverage,
                    "mean_oracle_regret": float(regrets.mean()) if not regrets.empty else np.nan,
                    "median_oracle_regret": float(regrets.median()) if not regrets.empty else np.nan,
                    "winner_share": winner_share,
                    "median_metric_value": float(label_subset.loc[valid_mask, primary_metric].median())
                    if valid_count
                    else np.nan,
                }
            )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df, pd.DataFrame(columns=list(partition_cols))

    for _, group in metrics_df.groupby(partition_cols, dropna=False):
        best_coverage = float(group["valid_coverage"].max())
        candidates = group[group["valid_coverage"] >= best_coverage - coverage_tolerance - 1e-12].copy()
        candidates["mean_oracle_regret_rank"] = candidates["mean_oracle_regret"].fillna(np.inf)
        candidates["median_metric_value_rank"] = candidates["median_metric_value"].fillna(np.inf)
        winner_rank = -candidates["winner_share"]
        best = candidates.assign(_winner_rank=winner_rank).sort_values(
            ["mean_oracle_regret_rank", "_winner_rank", "median_metric_value_rank", "label"],
            kind="stable",
        ).iloc[0]
        recommendations.append(
            {
                **{column: best[column] for column in partition_cols},
                "partition_support": int(best["partition_support"]),
                "leaf_valid_coverage_best": best_coverage,
                "recommended_label": best["label"],
                "recommended_label_valid_coverage": float(best["valid_coverage"]),
                "recommended_label_mean_oracle_regret": float(best["mean_oracle_regret"])
                if pd.notna(best["mean_oracle_regret"])
                else np.nan,
                "recommended_label_median_oracle_regret": float(best["median_oracle_regret"])
                if pd.notna(best["median_oracle_regret"])
                else np.nan,
                "recommended_label_winner_share": float(best["winner_share"]),
                "recommended_label_median_metric_value": float(best["median_metric_value"])
                if pd.notna(best["median_metric_value"])
                else np.nan,
            }
        )

    return metrics_df, pd.DataFrame(recommendations)


def apply_tree_partition(
    fit_result: Dict[str, object],
    X: pd.DataFrame,
) -> np.ndarray:
    imputer: SimpleImputer = fit_result["imputer"]  # type: ignore[assignment]
    model: DecisionTreeClassifier = fit_result["model"]  # type: ignore[assignment]
    feature_columns: List[str] = fit_result["feature_columns"]  # type: ignore[assignment]
    X_i = imputer.transform(X.loc[:, feature_columns])
    return model.apply(X_i)


def evaluate_recommendations(
    assigned_instances: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    observations: pd.DataFrame,
    label_target_col: str,
    primary_metric: str = "expansions",
) -> Dict[str, float]:
    if assigned_instances.empty:
        return {
            "recommended_method_valid_coverage": np.nan,
            "recommended_method_mean_oracle_regret": np.nan,
            "recommended_method_median_oracle_regret": np.nan,
            "recommended_method_winner_share": np.nan,
            "best_method_accuracy": np.nan,
        }

    eval_df = assigned_instances.merge(
        recommendations_df.loc[:, [LEAF_ID_COL, "recommended_label"]],
        on=LEAF_ID_COL,
        how="left",
    )
    obs = observations.rename(columns={"label": "recommended_label"})
    eval_df = eval_df.merge(
        obs.loc[:, INSTANCE_KEYS + ["recommended_label", "valid", primary_metric, oracle_metric_column(primary_metric), "winner_flag"]],
        on=INSTANCE_KEYS + ["recommended_label"],
        how="left",
    )
    # Avoid repeated object-dtype fillna warnings in the hot evaluation loop.
    valid_mask = eval_df["valid"].eq(True)
    regrets = eval_df.loc[valid_mask, primary_metric] - eval_df.loc[valid_mask, oracle_metric_column(primary_metric)]
    winner_share = float(eval_df["winner_flag"].eq(True).mean()) if len(eval_df) else np.nan
    best_accuracy = (
        float(eval_df["recommended_label"].eq(eval_df[label_target_col]).mean())
        if label_target_col in eval_df.columns and len(eval_df)
        else np.nan
    )
    return {
        "recommended_method_valid_coverage": float(valid_mask.mean()) if len(eval_df) else np.nan,
        "recommended_method_mean_oracle_regret": float(regrets.mean()) if not regrets.empty else np.nan,
        "recommended_method_median_oracle_regret": float(regrets.median()) if not regrets.empty else np.nan,
        "recommended_method_winner_share": winner_share,
        "best_method_accuracy": best_accuracy,
    }


def build_grouped_validation_artifacts(
    modeling_df: pd.DataFrame,
    predictor_cols: Sequence[str],
    method_observations: pd.DataFrame,
    coverage_tolerance: float,
    max_depth: int,
    min_samples_leaf: int,
    group_col: str,
    primary_metric: str = "expansions",
    progress_out_dir: Optional[Path] = None,
    progress_label: str = "main",
) -> List[Dict[str, object]]:
    if group_col not in modeling_df.columns:
        return []

    groups = modeling_df[group_col].astype(str)
    unique_groups = groups.nunique()
    if unique_groups < 2:
        return []

    n_splits = min(5, unique_groups)
    splitter = GroupKFold(n_splits=n_splits)
    X = modeling_df.loc[:, predictor_cols]
    y = modeling_df["best_method"].astype(str)
    artifacts: List[Dict[str, object]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        emit_progress(
            progress_out_dir,
            "grouped_validation_fold_start",
            variant=progress_label,
            fold=fold_idx,
            total_folds=n_splits,
        )
        train_df = modeling_df.iloc[train_idx].reset_index(drop=True)
        test_df = modeling_df.iloc[test_idx].reset_index(drop=True)
        fit_result = fit_tree_with_guardrail(
            train_df.loc[:, predictor_cols],
            train_df["best_method"].astype(str),
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        train_assign = train_df.loc[:, INSTANCE_KEYS + ["best_method"]].copy()
        train_assign[LEAF_ID_COL] = fit_result["leaf_ids"]
        _, train_recs = summarize_partition_recommendations(
            train_assign.loc[:, INSTANCE_KEYS + [LEAF_ID_COL]],
            method_observations.merge(train_df.loc[:, INSTANCE_KEYS], on=INSTANCE_KEYS, how="inner"),
            [LEAF_ID_COL],
            coverage_tolerance,
            primary_metric,
        )

        test_assign = test_df.loc[:, INSTANCE_KEYS + ["best_method"]].copy()
        test_assign[LEAF_ID_COL] = apply_tree_partition(fit_result, test_df.loc[:, predictor_cols])
        metrics = evaluate_recommendations(test_assign, train_recs, method_observations, "best_method", primary_metric)
        artifacts.append(
            {
                "fold": fold_idx,
                "train_df": train_df,
                "test_df": test_df,
                "fit_result": fit_result,
                "train_recs": train_recs,
                "base_metrics": metrics,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_groups": int(train_df[group_col].nunique()),
                "test_groups": int(test_df[group_col].nunique()),
                "realized_min_samples_leaf": int(fit_result["realized_min_samples_leaf"]),
                "leaf_count_total": int(len(pd.unique(train_assign[LEAF_ID_COL]))),
            }
        )
        emit_progress(
            progress_out_dir,
            "grouped_validation_fold_done",
            variant=progress_label,
            fold=fold_idx,
            total_folds=n_splits,
            test_rows=int(len(test_df)),
        )

    return artifacts


def artifacts_to_validation_df(artifacts: Sequence[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for artifact in artifacts:
        rows.append(
            {
                "fold": int(artifact["fold"]),
                "train_rows": int(artifact["train_rows"]),
                "test_rows": int(artifact["test_rows"]),
                "train_groups": int(artifact["train_groups"]),
                "test_groups": int(artifact["test_groups"]),
                "realized_min_samples_leaf": int(artifact["realized_min_samples_leaf"]),
                "leaf_count_total": int(artifact["leaf_count_total"]),
                **artifact["base_metrics"],  # type: ignore[arg-type]
            }
        )
    return pd.DataFrame(rows)


def run_grouped_validation(
    modeling_df: pd.DataFrame,
    predictor_cols: Sequence[str],
    method_observations: pd.DataFrame,
    coverage_tolerance: float,
    max_depth: int,
    min_samples_leaf: int,
    group_col: str,
    primary_metric: str = "expansions",
) -> pd.DataFrame:
    artifacts = build_grouped_validation_artifacts(
        modeling_df=modeling_df,
        predictor_cols=predictor_cols,
        method_observations=method_observations,
        coverage_tolerance=coverage_tolerance,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        group_col=group_col,
        primary_metric=primary_metric,
    )
    return artifacts_to_validation_df(artifacts)


def build_tree_feature_importance(fit_result: Dict[str, object]) -> pd.DataFrame:
    model: DecisionTreeClassifier = fit_result["model"]  # type: ignore[assignment]
    feature_columns: List[str] = fit_result["feature_columns"]  # type: ignore[assignment]
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "tree_importance": model.feature_importances_,
        }
    )
    return importance_df.sort_values(["tree_importance", "feature"], ascending=[False, True], kind="stable").reset_index(drop=True)


def run_grouped_permutation_importance(
    fold_artifacts: Sequence[Dict[str, object]],
    predictor_cols: Sequence[str],
    method_observations: pd.DataFrame,
    n_repeats: int,
    primary_metric: str = "expansions",
    random_state: int = 42,
    progress_out_dir: Optional[Path] = None,
    progress_label: str = "main",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_cols = [
        "feature",
        "fold",
        "repeat",
        "base_valid_coverage",
        "permuted_valid_coverage",
        "valid_coverage_drop",
        "base_mean_oracle_regret",
        "permuted_mean_oracle_regret",
        "mean_regret_increase",
        "base_median_oracle_regret",
        "permuted_median_oracle_regret",
        "median_regret_increase",
        "base_winner_share",
        "permuted_winner_share",
        "winner_share_drop",
        "base_best_method_accuracy",
        "permuted_best_method_accuracy",
        "best_method_accuracy_drop",
    ]
    if not fold_artifacts or n_repeats <= 0:
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=["feature"])

    rng = np.random.default_rng(random_state)
    detail_rows: List[Dict[str, object]] = []
    total_folds = len(fold_artifacts)
    total_features = len(predictor_cols)
    for fold_position, artifact in enumerate(fold_artifacts, start=1):
        fit_result: Dict[str, object] = artifact["fit_result"]  # type: ignore[assignment]
        train_recs: pd.DataFrame = artifact["train_recs"]  # type: ignore[assignment]
        test_df: pd.DataFrame = artifact["test_df"]  # type: ignore[assignment]
        base_metrics: Dict[str, float] = artifact["base_metrics"]  # type: ignore[assignment]
        if test_df.empty:
            continue

        emit_progress(
            progress_out_dir,
            "permutation_fold_start",
            variant=progress_label,
            fold=int(artifact["fold"]),
            fold_position=fold_position,
            total_folds=total_folds,
            total_features=total_features,
            repeats=n_repeats,
        )
        base_predictors = test_df.loc[:, predictor_cols].copy()
        for feature_idx, feature in enumerate(predictor_cols, start=1):
            if feature_idx == 1 or feature_idx == total_features or feature_idx % 5 == 0:
                emit_progress(
                    progress_out_dir,
                    "permutation_feature_progress",
                    variant=progress_label,
                    fold=int(artifact["fold"]),
                    fold_position=fold_position,
                    total_folds=total_folds,
                    feature_index=feature_idx,
                    total_features=total_features,
                    feature=feature,
                    repeats=n_repeats,
                )
            feature_values = base_predictors[feature].to_numpy(copy=True)
            for repeat in range(1, n_repeats + 1):
                permuted_predictors = base_predictors.copy()
                permuted_predictors[feature] = rng.permutation(feature_values)
                test_assign = test_df.loc[:, INSTANCE_KEYS + ["best_method"]].copy()
                test_assign[LEAF_ID_COL] = apply_tree_partition(fit_result, permuted_predictors)
                permuted_metrics = evaluate_recommendations(
                    test_assign,
                    train_recs,
                    method_observations,
                    "best_method",
                    primary_metric,
                )
                detail_rows.append(
                    {
                        "feature": feature,
                        "fold": int(artifact["fold"]),
                        "repeat": repeat,
                        "base_valid_coverage": base_metrics["recommended_method_valid_coverage"],
                        "permuted_valid_coverage": permuted_metrics["recommended_method_valid_coverage"],
                        "valid_coverage_drop": base_metrics["recommended_method_valid_coverage"]
                        - permuted_metrics["recommended_method_valid_coverage"],
                        "base_mean_oracle_regret": base_metrics["recommended_method_mean_oracle_regret"],
                        "permuted_mean_oracle_regret": permuted_metrics["recommended_method_mean_oracle_regret"],
                        "mean_regret_increase": permuted_metrics["recommended_method_mean_oracle_regret"]
                        - base_metrics["recommended_method_mean_oracle_regret"],
                        "base_median_oracle_regret": base_metrics["recommended_method_median_oracle_regret"],
                        "permuted_median_oracle_regret": permuted_metrics["recommended_method_median_oracle_regret"],
                        "median_regret_increase": permuted_metrics["recommended_method_median_oracle_regret"]
                        - base_metrics["recommended_method_median_oracle_regret"],
                        "base_winner_share": base_metrics["recommended_method_winner_share"],
                        "permuted_winner_share": permuted_metrics["recommended_method_winner_share"],
                        "winner_share_drop": base_metrics["recommended_method_winner_share"]
                        - permuted_metrics["recommended_method_winner_share"],
                        "base_best_method_accuracy": base_metrics["best_method_accuracy"],
                        "permuted_best_method_accuracy": permuted_metrics["best_method_accuracy"],
                        "best_method_accuracy_drop": base_metrics["best_method_accuracy"]
                        - permuted_metrics["best_method_accuracy"],
                    }
                )
        emit_progress(
            progress_out_dir,
            "permutation_fold_done",
            variant=progress_label,
            fold=int(artifact["fold"]),
            fold_position=fold_position,
            total_folds=total_folds,
        )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return detail_df, pd.DataFrame(columns=["feature"])

    summary_df = (
        detail_df.groupby("feature", dropna=False)
        .agg(
            folds=("fold", "nunique"),
            repeats=("repeat", "max"),
            mean_valid_coverage_drop=("valid_coverage_drop", "mean"),
            std_valid_coverage_drop=("valid_coverage_drop", "std"),
            mean_regret_increase=("mean_regret_increase", "mean"),
            std_regret_increase=("mean_regret_increase", "std"),
            mean_median_regret_increase=("median_regret_increase", "mean"),
            std_median_regret_increase=("median_regret_increase", "std"),
            mean_winner_share_drop=("winner_share_drop", "mean"),
            std_winner_share_drop=("winner_share_drop", "std"),
            mean_best_method_accuracy_drop=("best_method_accuracy_drop", "mean"),
            std_best_method_accuracy_drop=("best_method_accuracy_drop", "std"),
        )
        .reset_index()
        .sort_values(["mean_regret_increase", "mean_valid_coverage_drop", "feature"], ascending=[False, False, True], kind="stable")
        .reset_index(drop=True)
    )
    return detail_df, summary_df


def bucketize_feature(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty or valid.nunique() <= 1:
        return pd.Series("all", index=values.index, dtype="object")
    try:
        ranked = valid.rank(method="first")
        binned = pd.qcut(ranked, q=3, labels=["low", "mid", "high"])
    except ValueError:
        return pd.Series("all", index=values.index, dtype="object")

    out = pd.Series("all", index=values.index, dtype="object")
    out.loc[valid.index] = binned.astype(str)
    return out


def build_appendix_buckets(
    modeling_df: pd.DataFrame,
    predictor_cols: Sequence[str],
    method_observations: pd.DataFrame,
    coverage_tolerance: float,
    primary_metric: str = "expansions",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    for feature in predictor_cols:
        feature_df = modeling_df.loc[:, INSTANCE_KEYS + [feature]].copy()
        feature_df["bucket_feature"] = feature
        feature_df["bucket_label"] = (
            feature_df.groupby("dataset_name", dropna=False)[feature]
            .transform(bucketize_feature)
        )
        rows.append(feature_df.loc[:, INSTANCE_KEYS + ["bucket_feature", "bucket_label"]])

    partitions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if partitions.empty:
        return (
            pd.DataFrame(columns=["dataset_name", "bucket_feature", "bucket_label", "label"]),
            pd.DataFrame(columns=["dataset_name", "bucket_feature", "bucket_label"]),
        )
    return summarize_partition_recommendations(
        partitions,
        method_observations,
        ["dataset_name", "bucket_feature", "bucket_label"],
        coverage_tolerance,
        primary_metric,
    )


def build_oracle_ready_modeling_df(
    modeling_df: pd.DataFrame,
    oracle_feature_cols: Sequence[str],
) -> tuple[pd.DataFrame, Dict[str, object]]:
    available_cols = [column for column in oracle_feature_cols if column in modeling_df.columns]
    if not available_cols:
        return modeling_df.iloc[0:0].copy(), {
            "oracle_features_available": [],
            "oracle_rows_before_complete_filter": int(len(modeling_df)),
            "oracle_rows_after_complete_filter": 0,
        }

    complete_mask = modeling_df.loc[:, available_cols].notna().all(axis=1)
    filtered_df = modeling_df.loc[complete_mask].copy().reset_index(drop=True)
    return filtered_df, {
        "oracle_features_available": available_cols,
        "oracle_rows_before_complete_filter": int(len(modeling_df)),
        "oracle_rows_after_complete_filter": int(len(filtered_df)),
        "oracle_rows_dropped_for_missing_features": int((~complete_mask).sum()),
    }


def run_analysis_variant(
    modeling_df: pd.DataFrame,
    predictor_cols: Sequence[str],
    method_observations: pd.DataFrame,
    algorithm_observations: pd.DataFrame,
    heuristic_observations: pd.DataFrame,
    coverage_tolerance: float,
    max_depth: int,
    min_samples_leaf: int,
    group_col: str,
    permutation_repeats: int,
    primary_metric: str,
    out_dir: Path,
    progress_label: str,
) -> Dict[str, object]:
    emit_progress(
        out_dir,
        "variant_start",
        variant=progress_label,
        rows=int(len(modeling_df)),
        predictors=int(len(predictor_cols)),
        permutation_repeats=int(permutation_repeats),
    )
    variant_df = modeling_df.copy()
    fit_result = fit_tree_with_guardrail(
        variant_df.loc[:, predictor_cols],
        variant_df["best_method"].astype(str),
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    variant_df[LEAF_ID_COL] = fit_result["leaf_ids"]
    emit_progress(
        out_dir,
        "variant_tree_fit_done",
        variant=progress_label,
        leaf_count=int(len(pd.unique(variant_df[LEAF_ID_COL]))),
        realized_min_samples_leaf=int(fit_result["realized_min_samples_leaf"]),
        guardrail_triggered=bool(fit_result["guardrail_triggered"]),
    )

    leaf_method_metrics, leaf_method_recs = summarize_partition_recommendations(
        variant_df.loc[:, INSTANCE_KEYS + [LEAF_ID_COL]],
        method_observations,
        [LEAF_ID_COL],
        coverage_tolerance,
        primary_metric,
    )
    leaf_algorithm_metrics, leaf_algorithm_recs = summarize_partition_recommendations(
        variant_df.loc[:, INSTANCE_KEYS + [LEAF_ID_COL]],
        algorithm_observations,
        [LEAF_ID_COL],
        coverage_tolerance,
        primary_metric,
    )
    leaf_heuristic_metrics, leaf_heuristic_recs = summarize_partition_recommendations(
        variant_df.loc[:, INSTANCE_KEYS + [LEAF_ID_COL]],
        heuristic_observations,
        [LEAF_ID_COL],
        coverage_tolerance,
        primary_metric,
    )

    leaf_summary = rename_recommendation_columns(leaf_method_recs, "method").merge(
        rename_recommendation_columns(leaf_algorithm_recs, "algorithm").loc[:, [LEAF_ID_COL, "recommended_algorithm"]],
        on=LEAF_ID_COL,
        how="left",
    ).merge(
        rename_recommendation_columns(leaf_heuristic_recs, "heuristic").loc[:, [LEAF_ID_COL, "recommended_heuristic"]],
        on=LEAF_ID_COL,
        how="left",
    )
    leaf_summary = leaf_summary.rename(columns={"partition_support": "leaf_support"})
    leaf_summary["leaf_support_rate"] = leaf_summary["leaf_support"] / max(len(variant_df), 1)
    leaf_summary["realized_min_samples_leaf"] = int(fit_result["realized_min_samples_leaf"])
    leaf_summary["leaf_count_total"] = int(len(pd.unique(variant_df[LEAF_ID_COL])))
    rule_map: Dict[int, str] = fit_result["leaf_rules"]  # type: ignore[assignment]
    leaf_summary["leaf_rule"] = leaf_summary[LEAF_ID_COL].map(rule_map)
    leaf_summary = leaf_summary.sort_values([LEAF_ID_COL], kind="stable").reset_index(drop=True)
    emit_progress(
        out_dir,
        "variant_leaf_summary_done",
        variant=progress_label,
        leaf_rows=int(len(leaf_summary)),
    )

    fold_artifacts = build_grouped_validation_artifacts(
        modeling_df=variant_df,
        predictor_cols=predictor_cols,
        method_observations=method_observations,
        coverage_tolerance=coverage_tolerance,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        group_col=group_col,
        primary_metric=primary_metric,
        progress_out_dir=out_dir,
        progress_label=progress_label,
    )
    validation_df = artifacts_to_validation_df(fold_artifacts)
    emit_progress(
        out_dir,
        "variant_validation_done",
        variant=progress_label,
        validation_rows=int(len(validation_df)),
    )
    permutation_detail_df, permutation_summary_df = run_grouped_permutation_importance(
        fold_artifacts=fold_artifacts,
        predictor_cols=predictor_cols,
        method_observations=method_observations,
        n_repeats=permutation_repeats,
        primary_metric=primary_metric,
        progress_out_dir=out_dir,
        progress_label=progress_label,
    )
    tree_importance_df = build_tree_feature_importance(fit_result)
    emit_progress(
        out_dir,
        "variant_permutation_done",
        variant=progress_label,
        permutation_rows=int(len(permutation_detail_df)),
        importance_rows=int(len(permutation_summary_df)),
    )

    return {
        "modeling_df": variant_df,
        "fit_result": fit_result,
        "rule_map": rule_map,
        "leaf_summary": leaf_summary,
        "leaf_method_metrics": leaf_method_metrics,
        "leaf_algorithm_metrics": leaf_algorithm_metrics,
        "leaf_heuristic_metrics": leaf_heuristic_metrics,
        "validation_df": validation_df,
        "permutation_detail_df": permutation_detail_df,
        "permutation_summary_df": permutation_summary_df,
        "tree_importance_df": tree_importance_df,
    }


def build_markdown_summary(
    summary_json: Dict[str, object],
    leaf_summary: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    oracle_ablation_summary: Optional[Dict[str, object]] = None,
) -> str:
    primary_metric = str(summary_json.get("primary_optimization_metric", "time_seconds"))
    regret_label = "mean_runtime_regret_seconds" if primary_metric == "time_seconds" else "mean_expansion_regret"
    importance_label = "runtime_regret_increase_seconds" if primary_metric == "time_seconds" else "expansion_regret_increase"
    oracle_label = (
        "Oracle variant validation mean runtime regret (seconds)"
        if primary_metric == "time_seconds"
        else "Oracle variant validation mean expansion regret"
    )
    lines = [
        "# Setting Recommendation Summary",
        "",
        f"- Instances modeled: {summary_json['instances_modeled']}",
        f"- Predictor columns: {summary_json['predictor_count']}",
        f"- Primary optimization metric: {summary_json.get('primary_optimization_metric', 'time_seconds')}",
        f"- Regret units: {summary_json.get('regret_units', 'seconds')}",
        f"- Leaves: {summary_json['leaf_count_total']}",
        f"- Guardrail triggered: {summary_json['tree_guardrail_triggered']}",
        "",
        "## Top Leaves",
        "",
    ]
    if leaf_summary.empty:
        lines.append("No leaf summary available.")
    else:
        top = leaf_summary.sort_values("leaf_support", ascending=False, kind="stable").head(8)
        for _, row in top.iterrows():
            lines.append(
                f"- Leaf {int(row['leaf_id'])}: support={int(row['leaf_support'])}, "
                f"method={row['recommended_method']}, coverage={row['recommended_method_valid_coverage']:.3f}, "
                f"{regret_label}={row['recommended_method_mean_oracle_regret']:.3f}"
            )
    lines.extend(["", "## Top Importance Signals", ""])
    if permutation_summary.empty:
        lines.append("No permutation importance available.")
    else:
        top_features = permutation_summary.head(8)
        for _, row in top_features.iterrows():
            lines.append(
                f"- {row['feature']}: {importance_label}={row['mean_regret_increase']:.4f}, "
                f"coverage_drop={row['mean_valid_coverage_drop']:.4f}, "
                f"winner_share_drop={row['mean_winner_share_drop']:.4f}"
            )
    if oracle_ablation_summary:
        lines.extend(["", "## Oracle Ablation", ""])
        lines.append(
            f"- Oracle features available: {', '.join(oracle_ablation_summary.get('oracle_features_available', [])) or 'none'}"
        )
        lines.append(
            f"- Oracle variant predictors: {oracle_ablation_summary.get('predictor_count', 0)}"
        )
        lines.append(
            f"- {oracle_label}: {oracle_ablation_summary.get('validation_mean_oracle_regret', np.nan):.4f}"
        )
    return "\n".join(lines) + "\n"


def write_tree_rules(out_path: Path, fit_result: Dict[str, object], predictor_cols: Sequence[str]) -> None:
    model: DecisionTreeClassifier = fit_result["model"]  # type: ignore[assignment]
    rule_map: Dict[int, str] = fit_result["leaf_rules"]  # type: ignore[assignment]
    feature_columns: List[str] = fit_result["feature_columns"]  # type: ignore[assignment]
    lines = [
        export_text(model, feature_names=feature_columns),
        "",
        "Leaf Rules:",
    ]
    for leaf_id in sorted(rule_map):
        lines.append(f"{leaf_id}: {rule_map[leaf_id]}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rename_recommendation_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename_map = {
        "recommended_label": f"recommended_{prefix}",
        "recommended_label_valid_coverage": f"recommended_{prefix}_valid_coverage",
        "recommended_label_mean_oracle_regret": f"recommended_{prefix}_mean_oracle_regret",
        "recommended_label_median_oracle_regret": f"recommended_{prefix}_median_oracle_regret",
        "recommended_label_winner_share": f"recommended_{prefix}_winner_share",
        "recommended_label_median_metric_value": f"recommended_{prefix}_median_metric_value",
    }
    return df.rename(columns=rename_map)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(args.out_dir, "startup", output_dir=str(args.out_dir), permutation_repeats=int(args.permutation_repeats))

    aggregate_df, aggregate_meta = load_or_build_aggregate(
        results_root=args.results_root,
        out_dir=args.out_dir,
        aggregate_csv=args.aggregate_csv,
    )
    emit_progress(args.out_dir, "aggregate_ready", rows=int(len(aggregate_df)))

    corrected = build_corrected_frames(aggregate_df)
    counts = corrected["counts"]
    complete_df: pd.DataFrame = corrected["complete"]  # type: ignore[assignment]
    valid_df: pd.DataFrame = corrected["valid"]  # type: ignore[assignment]
    emit_progress(
        args.out_dir,
        "corrected_frames_ready",
        rows_complete=int(len(complete_df)),
        rows_valid=int(len(valid_df)),
    )

    instance_base = build_instance_base(complete_df, valid_df)
    fast_features_df, feature_failures_df, feature_summary = build_fast_features(instance_base)
    fast_instance_with_features = instance_base.merge(fast_features_df, on=INSTANCE_KEYS, how="inner")
    emit_progress(
        args.out_dir,
        "fast_features_ready",
        instance_rows=int(len(instance_base)),
        feature_rows=int(len(fast_features_df)),
        failures=int(len(feature_failures_df)),
    )
    oracle_features_df, oracle_feature_meta = load_oracle_features(args.oracle_features_csv, args.oracle_feature_cols)
    oracle_instance_with_features = fast_instance_with_features
    if not oracle_features_df.empty:
        oracle_instance_with_features = fast_instance_with_features.merge(oracle_features_df, on=INSTANCE_KEYS, how="left")
    emit_progress(
        args.out_dir,
        "oracle_features_ready",
        oracle_rows=int(len(oracle_features_df)),
        oracle_enabled=bool(not oracle_features_df.empty),
    )

    targets_df, _ = build_target_tables(valid_df, args.primary_metric)
    modeling_df = fast_instance_with_features.merge(targets_df, on=INSTANCE_KEYS, how="inner")
    oracle_modeling_input_df = oracle_instance_with_features.merge(targets_df, on=INSTANCE_KEYS, how="inner")
    method_observations = build_label_observations(complete_df, targets_df, LABEL_TYPE_METHOD, args.primary_metric)
    algorithm_observations = build_label_observations(complete_df, targets_df, LABEL_TYPE_ALGORITHM, args.primary_metric)
    heuristic_observations = build_label_observations(complete_df, targets_df, LABEL_TYPE_HEURISTIC, args.primary_metric)
    emit_progress(
        args.out_dir,
        "observations_ready",
        target_rows=int(len(targets_df)),
        method_observations=int(len(method_observations)),
    )

    predictor_cols = compute_predictor_columns(
        modeling_df,
        exclude_columns=args.exclude_predictor_cols,
    )
    if not predictor_cols:
        raise ValueError("No numeric predictor columns available for modeling")

    min_leaf = default_min_samples_leaf(len(modeling_df), args.min_samples_leaf)
    main_variant = run_analysis_variant(
        modeling_df=modeling_df,
        predictor_cols=predictor_cols,
        method_observations=method_observations,
        algorithm_observations=algorithm_observations,
        heuristic_observations=heuristic_observations,
        coverage_tolerance=args.coverage_tolerance,
        max_depth=args.max_depth,
        min_samples_leaf=min_leaf,
        group_col=args.group_col,
        permutation_repeats=args.permutation_repeats,
        primary_metric=args.primary_metric,
        out_dir=args.out_dir,
        progress_label="main",
    )
    modeling_with_leaves: pd.DataFrame = main_variant["modeling_df"]  # type: ignore[assignment]
    fit_result: Dict[str, object] = main_variant["fit_result"]  # type: ignore[assignment]
    leaf_summary: pd.DataFrame = main_variant["leaf_summary"]  # type: ignore[assignment]
    leaf_method_metrics: pd.DataFrame = main_variant["leaf_method_metrics"]  # type: ignore[assignment]
    leaf_algorithm_metrics: pd.DataFrame = main_variant["leaf_algorithm_metrics"]  # type: ignore[assignment]
    leaf_heuristic_metrics: pd.DataFrame = main_variant["leaf_heuristic_metrics"]  # type: ignore[assignment]
    validation_df: pd.DataFrame = main_variant["validation_df"]  # type: ignore[assignment]
    permutation_detail_df: pd.DataFrame = main_variant["permutation_detail_df"]  # type: ignore[assignment]
    permutation_summary_df: pd.DataFrame = main_variant["permutation_summary_df"]  # type: ignore[assignment]
    tree_importance_df: pd.DataFrame = main_variant["tree_importance_df"]  # type: ignore[assignment]
    rule_map: Dict[int, str] = main_variant["rule_map"]  # type: ignore[assignment]

    appendix_method_metrics = pd.DataFrame()
    appendix_method_recs = pd.DataFrame()
    if args.write_appendix_buckets:
        appendix_method_metrics, appendix_method_recs = build_appendix_buckets(
            modeling_with_leaves,
            predictor_cols,
            method_observations,
            args.coverage_tolerance,
            args.primary_metric,
        )
        emit_progress(
            args.out_dir,
            "appendix_ready",
            appendix_metric_rows=int(len(appendix_method_metrics)),
            appendix_rec_rows=int(len(appendix_method_recs)),
        )

    oracle_ablation_summary: Optional[Dict[str, object]] = None
    oracle_variant: Optional[Dict[str, object]] = None
    oracle_modeling_df, oracle_ready_meta = build_oracle_ready_modeling_df(oracle_modeling_input_df, args.oracle_feature_cols)
    oracle_predictor_cols = compute_predictor_columns(
        oracle_modeling_df,
        include_columns=args.oracle_feature_cols,
        exclude_columns=args.exclude_predictor_cols,
    )
    oracle_available_predictors = [column for column in args.oracle_feature_cols if column in oracle_predictor_cols]
    if args.oracle_features_csv is not None and oracle_available_predictors and not oracle_modeling_df.empty:
        oracle_variant = run_analysis_variant(
            modeling_df=oracle_modeling_df,
            predictor_cols=oracle_predictor_cols,
            method_observations=method_observations,
            algorithm_observations=algorithm_observations,
            heuristic_observations=heuristic_observations,
            coverage_tolerance=args.coverage_tolerance,
            max_depth=args.max_depth,
            min_samples_leaf=min_leaf,
            group_col=args.group_col,
            permutation_repeats=args.permutation_repeats,
            primary_metric=args.primary_metric,
            out_dir=args.out_dir,
            progress_label="oracle",
        )
        oracle_validation_df: pd.DataFrame = oracle_variant["validation_df"]  # type: ignore[assignment]
        oracle_ablation_summary = {
            **oracle_feature_meta,
            **oracle_ready_meta,
            "predictor_columns": oracle_predictor_cols,
            "predictor_count": int(len(oracle_predictor_cols)),
            "excluded_predictor_columns": list(args.exclude_predictor_cols),
            "instances_modeled": int(len(oracle_modeling_df)),
            "validation_rows": int(len(oracle_validation_df)),
            "validation_mean_oracle_regret": float(oracle_validation_df["recommended_method_mean_oracle_regret"].mean())
            if not oracle_validation_df.empty
            else np.nan,
            "validation_mean_valid_coverage": float(oracle_validation_df["recommended_method_valid_coverage"].mean())
            if not oracle_validation_df.empty
            else np.nan,
            "validation_mean_winner_share": float(oracle_validation_df["recommended_method_winner_share"].mean())
            if not oracle_validation_df.empty
            else np.nan,
        }
    elif args.oracle_features_csv is not None:
        oracle_ablation_summary = {
            **oracle_feature_meta,
            **oracle_ready_meta,
            "enabled": False,
            "reason": "requested oracle features were not available with complete coverage",
            "predictor_columns": oracle_predictor_cols,
            "predictor_count": int(len(oracle_predictor_cols)),
            "excluded_predictor_columns": list(args.exclude_predictor_cols),
            "instances_modeled": int(len(oracle_modeling_df)),
        }

    assignments_df = modeling_with_leaves.loc[:, OUTPUT_INSTANCE_COLS + [LEAF_ID_COL, "best_method", "best_algorithm", "best_heuristic"]].copy()
    assignments_df["leaf_rule"] = assignments_df[LEAF_ID_COL].map(rule_map)

    targets_out = modeling_with_leaves.loc[
        :,
        OUTPUT_INSTANCE_COLS
        + [
            "best_method",
            "best_method_time_seconds",
            "best_method_expansions",
            "best_algorithm",
            "best_algorithm_time_seconds",
            "best_algorithm_expansions",
            "best_heuristic",
            "best_heuristic_time_seconds",
            "best_heuristic_expansions",
        ],
    ].copy()

    summary = {
        **aggregate_meta,
        **counts,
        **feature_summary,
        **oracle_feature_meta,
        "instances_with_valid_targets": int(len(targets_df)),
        "instances_modeled": int(len(modeling_with_leaves)),
        "instances_dropped_for_feature_failures": int(len(feature_failures_df)),
        "rows_after_feature_merge": int(len(fast_instance_with_features)),
        "primary_optimization_metric": args.primary_metric,
        "regret_units": regret_units(args.primary_metric),
        "predictor_columns": predictor_cols,
        "predictor_count": int(len(predictor_cols)),
        "excluded_predictor_columns": list(args.exclude_predictor_cols),
        "coverage_tolerance": float(args.coverage_tolerance),
        "permutation_repeats": int(args.permutation_repeats),
        "tree_max_depth": int(args.max_depth),
        "tree_guardrail_triggered": bool(fit_result["guardrail_triggered"]),
        "tree_initial_leaf_count": int(fit_result["initial_leaf_count"]),
        "tree_initial_max_leaf_support_rate": float(fit_result["initial_max_leaf_support_rate"]),
        "tree_initial_min_samples_leaf": int(fit_result["initial_min_samples_leaf"]),
        "realized_min_samples_leaf": int(fit_result["realized_min_samples_leaf"]),
        "leaf_count_total": int(len(pd.unique(modeling_with_leaves[LEAF_ID_COL]))),
        "validation_rows": int(len(validation_df)),
        "validation_mean_oracle_regret": float(validation_df["recommended_method_mean_oracle_regret"].mean())
        if not validation_df.empty
        else np.nan,
        "validation_mean_valid_coverage": float(validation_df["recommended_method_valid_coverage"].mean())
        if not validation_df.empty
        else np.nan,
        "validation_mean_winner_share": float(validation_df["recommended_method_winner_share"].mean())
        if not validation_df.empty
        else np.nan,
        "oracle_ablation": oracle_ablation_summary,
    }

    modeling_path = args.out_dir / "corrected_instance_modeling_table.csv"
    targets_path = args.out_dir / "corrected_instance_targets.csv"
    assignments_path = args.out_dir / "setting_leaf_assignments.csv"
    rules_path = args.out_dir / "setting_leaf_rules.txt"
    leaf_summary_path = args.out_dir / "setting_leaf_summary.csv"
    leaf_method_metrics_path = args.out_dir / "setting_leaf_method_summary.csv"
    leaf_algorithm_metrics_path = args.out_dir / "setting_leaf_algorithm_summary.csv"
    leaf_heuristic_metrics_path = args.out_dir / "setting_leaf_heuristic_summary.csv"
    validation_path = args.out_dir / "setting_model_validation.csv"
    tree_importance_path = args.out_dir / "setting_tree_feature_importance.csv"
    permutation_path = args.out_dir / "setting_permutation_importance.csv"
    permutation_detail_path = args.out_dir / "setting_permutation_importance_folds.csv"
    summary_json_path = args.out_dir / "setting_analysis_summary.json"
    summary_md_path = args.out_dir / "setting_analysis_summary.md"
    failures_path = args.out_dir / "setting_feature_failures.csv"
    appendix_metrics_path = args.out_dir / "appendix_feature_bucket_method_summary.csv"
    appendix_recs_path = args.out_dir / "appendix_feature_bucket_recommendations.csv"
    oracle_leaf_summary_path = args.out_dir / "setting_oracle_ablation_leaf_summary.csv"
    oracle_rules_path = args.out_dir / "setting_oracle_ablation_rules.txt"
    oracle_validation_path = args.out_dir / "setting_oracle_ablation_validation.csv"
    oracle_tree_importance_path = args.out_dir / "setting_oracle_ablation_tree_feature_importance.csv"
    oracle_permutation_path = args.out_dir / "setting_oracle_ablation_permutation_importance.csv"
    oracle_permutation_detail_path = args.out_dir / "setting_oracle_ablation_permutation_importance_folds.csv"
    oracle_summary_path = args.out_dir / "setting_oracle_ablation_summary.json"

    leaf_method_metrics_out = leaf_method_metrics.rename(columns={"label": "method"})
    leaf_algorithm_metrics_out = leaf_algorithm_metrics.rename(columns={"label": "algorithm"})
    leaf_heuristic_metrics_out = leaf_heuristic_metrics.rename(columns={"label": "heuristic"})
    appendix_method_metrics_out = appendix_method_metrics.rename(columns={"label": "method"})
    appendix_method_recs_out = rename_recommendation_columns(appendix_method_recs, "method")

    modeling_with_leaves.drop(columns=[LEAF_ID_COL]).to_csv(modeling_path, index=False)
    targets_out.to_csv(targets_path, index=False)
    assignments_df.to_csv(assignments_path, index=False)
    write_tree_rules(rules_path, fit_result, predictor_cols)
    leaf_summary.to_csv(leaf_summary_path, index=False)
    leaf_method_metrics_out.to_csv(leaf_method_metrics_path, index=False)
    leaf_algorithm_metrics_out.to_csv(leaf_algorithm_metrics_path, index=False)
    leaf_heuristic_metrics_out.to_csv(leaf_heuristic_metrics_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    tree_importance_df.to_csv(tree_importance_path, index=False)
    permutation_summary_df.to_csv(permutation_path, index=False)
    permutation_detail_df.to_csv(permutation_detail_path, index=False)
    feature_failures_df.to_csv(failures_path, index=False)
    if args.write_appendix_buckets:
        appendix_method_metrics_out.to_csv(appendix_metrics_path, index=False)
        appendix_method_recs_out.to_csv(appendix_recs_path, index=False)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md_path.write_text(
        build_markdown_summary(summary, leaf_summary, permutation_summary_df, oracle_ablation_summary),
        encoding="utf-8",
    )
    if oracle_ablation_summary is not None:
        oracle_summary_path.write_text(json.dumps(oracle_ablation_summary, indent=2), encoding="utf-8")
    if oracle_variant is not None and oracle_ablation_summary is not None:
        oracle_leaf_summary: pd.DataFrame = oracle_variant["leaf_summary"]  # type: ignore[assignment]
        oracle_fit_result: Dict[str, object] = oracle_variant["fit_result"]  # type: ignore[assignment]
        oracle_validation_df = oracle_variant["validation_df"]  # type: ignore[assignment]
        oracle_tree_importance_df = oracle_variant["tree_importance_df"]  # type: ignore[assignment]
        oracle_permutation_summary_df = oracle_variant["permutation_summary_df"]  # type: ignore[assignment]
        oracle_permutation_detail_df = oracle_variant["permutation_detail_df"]  # type: ignore[assignment]
        oracle_leaf_summary.to_csv(oracle_leaf_summary_path, index=False)
        write_tree_rules(oracle_rules_path, oracle_fit_result, oracle_predictor_cols)
        oracle_validation_df.to_csv(oracle_validation_path, index=False)
        oracle_tree_importance_df.to_csv(oracle_tree_importance_path, index=False)
        oracle_permutation_summary_df.to_csv(oracle_permutation_path, index=False)
        oracle_permutation_detail_df.to_csv(oracle_permutation_detail_path, index=False)
    emit_progress(args.out_dir, "outputs_written", output_dir=str(args.out_dir))

    print(f"Wrote: {modeling_path}")
    print(f"Wrote: {targets_path}")
    print(f"Wrote: {assignments_path}")
    print(f"Wrote: {rules_path}")
    print(f"Wrote: {leaf_summary_path}")
    print(f"Wrote: {leaf_method_metrics_path}")
    print(f"Wrote: {leaf_algorithm_metrics_path}")
    print(f"Wrote: {leaf_heuristic_metrics_path}")
    print(f"Wrote: {validation_path}")
    print(f"Wrote: {summary_json_path}")
    print(f"Wrote: {summary_md_path}")
    if args.write_appendix_buckets:
        print(f"Wrote: {appendix_metrics_path}")
        print(f"Wrote: {appendix_recs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
