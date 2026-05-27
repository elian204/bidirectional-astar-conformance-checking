#!/usr/bin/env python3
"""Analyze positional split/join features against forward/backward ME search."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmark_loader import load_model
from scripts.positional_structure import POSITIONAL_FEATURE_NAMES, compute_positional_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-csv", type=Path, required=True)
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def resolve_model_path(model_path: str, basename_cache: Dict[str, str]) -> str:
    candidate = Path(model_path)
    if candidate.exists():
        return str(candidate)

    repo_candidate = REPO_ROOT / model_path
    if repo_candidate.exists():
        return str(repo_candidate)

    basename = candidate.name
    if basename in basename_cache:
        return basename_cache[basename]

    matches = sorted(REPO_ROOT.rglob(basename))
    if not matches:
        raise FileNotFoundError(model_path)
    resolved = str(matches[0])
    basename_cache[basename] = resolved
    return resolved


def _safe_spearman(df: pd.DataFrame, feature: str, target: str) -> float:
    subset = df[[feature, target]].dropna()
    if len(subset) < 3:
        return math.nan
    if subset[feature].nunique() < 2 or subset[target].nunique() < 2:
        return math.nan
    return float(subset[feature].corr(subset[target], method="spearman"))


def _winner_share(df: pd.DataFrame, winner: str) -> float:
    if df.empty:
        return math.nan
    return float((df["winner"] == winner).mean())


def _median_value(df: pd.DataFrame, column: str) -> float:
    if df.empty:
        return math.nan
    return float(df[column].median())


def compute_model_feature_table(instance_df: pd.DataFrame) -> pd.DataFrame:
    model_rows = (
        instance_df.loc[:, ["dataset_name", "model_id", "model_name", "model_path"]]
        .drop_duplicates(subset=["model_id"])
        .sort_values(["dataset_name", "model_name"], kind="stable")
    )

    records: List[Dict[str, object]] = []
    basename_cache: Dict[str, str] = {}
    for row in model_rows.to_dict(orient="records"):
        model_path = resolve_model_path(str(row["model_path"]), basename_cache)
        wf, _, _, _ = load_model(model_path)
        features = compute_positional_features(wf)
        record = dict(row)
        record["model_path"] = model_path
        record.update(features)
        records.append(record)

    return pd.DataFrame.from_records(records)


def compute_pairwise_table(
    instance_df: pd.DataFrame,
    aggregate_csv: Path,
    model_features_df: pd.DataFrame,
) -> pd.DataFrame:
    aggregate_df = pd.read_csv(
        aggregate_csv,
        usecols=["model_id", "trace_id", "method", "time_seconds", "status"],
        low_memory=False,
    )
    aggregate_df = aggregate_df[
        (aggregate_df["status"] == "ok")
        & (aggregate_df["method"].isin(["forward_me", "backward_me"]))
    ].copy()

    pairwise = (
        aggregate_df.pivot_table(
            index=["model_id", "trace_id"],
            columns="method",
            values="time_seconds",
            aggfunc="first",
        )
        .reset_index()
        .dropna(subset=["forward_me", "backward_me"])
    )

    pairwise = pairwise.rename(
        columns={
            "forward_me": "forward_runtime_seconds",
            "backward_me": "backward_runtime_seconds",
        }
    )
    pairwise["forward_runtime_seconds"] = pairwise["forward_runtime_seconds"].astype(float)
    pairwise["backward_runtime_seconds"] = pairwise["backward_runtime_seconds"].astype(float)
    eps = 1e-9
    pairwise["log_fb_ratio"] = np.log(
        (pairwise["forward_runtime_seconds"] + eps) / (pairwise["backward_runtime_seconds"] + eps)
    )
    pairwise["runtime_ratio"] = (
        pairwise["forward_runtime_seconds"] + eps
    ) / (pairwise["backward_runtime_seconds"] + eps)
    pairwise["winner"] = np.where(
        pairwise["forward_runtime_seconds"] < pairwise["backward_runtime_seconds"],
        "forward",
        np.where(
            pairwise["backward_runtime_seconds"] < pairwise["forward_runtime_seconds"],
            "backward",
            "tie",
        ),
    )

    instance_cols = [
        "dataset_name",
        "model_id",
        "model_name",
        "trace_id",
        "tau_ratio",
        "and_splits",
        "and_joins",
        "xor_splits",
        "xor_joins",
    ]
    instance_base = instance_df.loc[:, instance_cols].drop_duplicates(subset=["model_id", "trace_id"])
    pairwise = pairwise.merge(instance_base, on=["model_id", "trace_id"], how="left")
    pairwise = pairwise.merge(
        model_features_df.drop(columns=["dataset_name", "model_name", "model_path"]),
        on="model_id",
        how="left",
    )
    return pairwise


def build_feature_summary(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    mid_tau = pairwise_df[(pairwise_df["tau_ratio"] >= 0.05) & (pairwise_df["tau_ratio"] < 0.30)]
    high_tau = pairwise_df[pairwise_df["tau_ratio"] >= 0.50]

    rows: List[Dict[str, object]] = []
    for feature in ["tau_ratio", *POSITIONAL_FEATURE_NAMES]:
        feature_values = pairwise_df[feature].dropna()
        if feature_values.empty:
            continue

        q1 = float(feature_values.quantile(0.25))
        q3 = float(feature_values.quantile(0.75))
        low = pairwise_df[pairwise_df[feature] <= q1]
        high = pairwise_df[pairwise_df[feature] >= q3]

        rows.append(
            {
                "feature": feature,
                "coverage": int(pairwise_df[feature].notna().sum()),
                "spearman_all": _safe_spearman(pairwise_df, feature, "log_fb_ratio"),
                "spearman_mid_tau": _safe_spearman(mid_tau, feature, "log_fb_ratio"),
                "spearman_high_tau": _safe_spearman(high_tau, feature, "log_fb_ratio"),
                "q1": q1,
                "q3": q3,
                "low_q_forward_win": _winner_share(low, "forward"),
                "low_q_backward_win": _winner_share(low, "backward"),
                "high_q_forward_win": _winner_share(high, "forward"),
                "high_q_backward_win": _winner_share(high, "backward"),
                "low_q_median_runtime_ratio": _median_value(low, "runtime_ratio"),
                "high_q_median_runtime_ratio": _median_value(high, "runtime_ratio"),
            }
        )

    summary_df = pd.DataFrame.from_records(rows)
    summary_df = summary_df.sort_values(
        by="spearman_all",
        key=lambda series: series.abs(),
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    return summary_df


def build_subset_feature_summary(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    subset_specs = [
        ("mid_tau", (pairwise_df["tau_ratio"] >= 0.05) & (pairwise_df["tau_ratio"] < 0.30)),
        ("high_tau", pairwise_df["tau_ratio"] >= 0.50),
    ]
    features = [
        "first_and_split_pos",
        "last_and_join_pos",
        "prefix_and_split_load_30",
        "suffix_and_join_load_30",
        "prefix_tau_density_30",
        "suffix_tau_density_30",
    ]

    rows: List[Dict[str, object]] = []
    for subset_name, mask in subset_specs:
        subset_df = pairwise_df[mask].copy()
        for feature in features:
            values = subset_df[feature].dropna()
            if len(values) < 10 or values.nunique() < 2:
                continue
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            low = subset_df[subset_df[feature] <= q1]
            high = subset_df[subset_df[feature] >= q3]
            rows.append(
                {
                    "subset": subset_name,
                    "feature": feature,
                    "coverage": int(subset_df[feature].notna().sum()),
                    "spearman": _safe_spearman(subset_df, feature, "log_fb_ratio"),
                    "q1": q1,
                    "q3": q3,
                    "low_q_forward_win": _winner_share(low, "forward"),
                    "low_q_backward_win": _winner_share(low, "backward"),
                    "high_q_forward_win": _winner_share(high, "forward"),
                    "high_q_backward_win": _winner_share(high, "backward"),
                    "low_q_median_runtime_ratio": _median_value(low, "runtime_ratio"),
                    "high_q_median_runtime_ratio": _median_value(high, "runtime_ratio"),
                }
            )

    return pd.DataFrame.from_records(rows)


def format_pct(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{100.0 * value:.1f}%"


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.3f}"


def write_markdown_summary(
    out_path: Path,
    pairwise_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    subset_summary_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    n = len(pairwise_df)
    forward_share = _winner_share(pairwise_df, "forward")
    backward_share = _winner_share(pairwise_df, "backward")
    tie_share = _winner_share(pairwise_df, "tie")

    lines.append("# Positional Structure Analysis")
    lines.append("")
    lines.append("- Primary comparison metric: runtime (`time_seconds`)")
    lines.append(f"- Pairwise instances with both `forward_me` and `backward_me` solved: `{n}`")
    lines.append(f"- `forward_me` wins: `{format_pct(forward_share)}`")
    lines.append(f"- `backward_me` wins: `{format_pct(backward_share)}`")
    lines.append(f"- ties: `{format_pct(tie_share)}`")
    lines.append("")

    lines.append("## Strongest Global Signals")
    lines.append("")
    for _, row in summary_df.head(8).iterrows():
        lines.append(
            f"- `{row['feature']}`: "
            f"Spearman(all) `{format_float(row['spearman_all'])}`, "
            f"mid-tau `{format_float(row['spearman_mid_tau'])}`, "
            f"high-tau `{format_float(row['spearman_high_tau'])}`; "
            f"low quartile backward win `{format_pct(row['low_q_backward_win'])}` vs "
            f"high quartile `{format_pct(row['high_q_backward_win'])}`; "
            f"median runtime ratio low `{format_float(row['low_q_median_runtime_ratio'])}` vs "
            f"high `{format_float(row['high_q_median_runtime_ratio'])}`"
        )
    lines.append("")

    lines.append("## Split/Join Position Focus")
    lines.append("")
    for feature in [
        "first_and_split_pos",
        "last_and_join_pos",
        "prefix_and_split_load_30",
        "suffix_and_join_load_30",
        "first_unavoidable_and_split_pos",
        "last_unavoidable_and_join_pos",
        "prefix_tau_density_30",
        "suffix_tau_density_30",
    ]:
        row = summary_df[summary_df["feature"] == feature]
        if row.empty:
            continue
        record = row.iloc[0]
        lines.append(
            f"- `{feature}`: "
            f"Spearman(all) `{format_float(record['spearman_all'])}`, "
            f"mid-tau `{format_float(record['spearman_mid_tau'])}`, "
            f"high-tau `{format_float(record['spearman_high_tau'])}`; "
            f"backward win low quartile `{format_pct(record['low_q_backward_win'])}` vs "
            f"high quartile `{format_pct(record['high_q_backward_win'])}`"
        )
    lines.append("")

    lines.append("## Tau-Controlled Split/Join Signals")
    lines.append("")
    for subset in ["mid_tau", "high_tau"]:
        lines.append(f"### {subset}")
        lines.append("")
        subset_rows = subset_summary_df[subset_summary_df["subset"] == subset]
        for _, record in subset_rows.iterrows():
            lines.append(
                f"- `{record['feature']}`: "
                f"Spearman `{format_float(record['spearman'])}`; "
                f"backward win low quartile `{format_pct(record['low_q_backward_win'])}` vs "
                f"high quartile `{format_pct(record['high_q_backward_win'])}`; "
                f"forward win low quartile `{format_pct(record['low_q_forward_win'])}` vs "
                f"high quartile `{format_pct(record['high_q_forward_win'])}`; "
                f"median runtime ratio low `{format_float(record['low_q_median_runtime_ratio'])}` vs "
                f"high `{format_float(record['high_q_median_runtime_ratio'])}`"
            )
        lines.append("")

    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- Positive Spearman means higher feature values push the pairwise balance toward `backward_me`.")
    lines.append("- For `first_*_pos`, smaller values mean earlier structure; for `last_*_pos`, larger values mean later structure.")
    lines.append("- Median runtime ratio is `forward_runtime_seconds / backward_runtime_seconds`, so values above `1` mean forward is slower.")
    lines.append("- Mid-tau means `0.05 <= tau_ratio < 0.30`; high-tau means `tau_ratio >= 0.50`.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    instance_df = pd.read_csv(args.instance_csv, low_memory=False)
    model_features_df = compute_model_feature_table(instance_df)
    pairwise_df = compute_pairwise_table(instance_df, args.aggregate_csv, model_features_df)
    summary_df = build_feature_summary(pairwise_df)
    subset_summary_df = build_subset_feature_summary(pairwise_df)

    model_features_path = args.out_dir / "model_positional_features.csv"
    pairwise_path = args.out_dir / "forward_backward_pairwise_with_positional.csv"
    summary_csv_path = args.out_dir / "positional_feature_summary.csv"
    subset_summary_csv_path = args.out_dir / "positional_subset_feature_summary.csv"
    summary_md_path = args.out_dir / "positional_analysis_summary.md"

    model_features_df.to_csv(model_features_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    subset_summary_df.to_csv(subset_summary_csv_path, index=False)
    write_markdown_summary(summary_md_path, pairwise_df, summary_df, subset_summary_df)

    print(f"Wrote: {model_features_path}")
    print(f"Wrote: {pairwise_path}")
    print(f"Wrote: {summary_csv_path}")
    print(f"Wrote: {subset_summary_csv_path}")
    print(f"Wrote: {summary_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
