#!/usr/bin/env python3
"""
Precision-first inductive-miner sweeps for the video activity datasets.

Preprocessing:
1. Collapse consecutive duplicate labels per case.
2. Drop null / boundary labels.
3. Collapse again to remove duplicates created by the dropped labels.

This keeps the discovered model focused on segment-to-segment control flow
instead of frame persistence.
"""

from __future__ import annotations

import argparse
import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.obj import EventLog
from pm4py.util import constants as pm4py_constants


pm4py_constants.SHOW_PROGRESS_BAR = False


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    log_path: Path
    mapping_path: Path
    drop_activity_ids: tuple[int, ...]
    recipe_source_dir: Path | None = None
    recipe_format: str | None = None


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT.parent

DATASETS: dict[str, DatasetConfig] = {
    "breakfast": DatasetConfig(
        name="breakfast",
        log_path=ROOT / "kari/datasets/breakfast_asformer/breakfast_asformer_log.csv",
        mapping_path=ROOT / "kari/datasets/breakfast_asformer/mapping.txt",
        drop_activity_ids=(0,),
        recipe_source_dir=ROOT / "data/data/breakfast/groundTruth",
        recipe_format="breakfast",
    ),
    "50salads": DatasetConfig(
        name="50salads",
        log_path=ROOT / "datasets/50salads_asformer/50salads_asformer_log.csv",
        mapping_path=ROOT / "data/data/50salads/mapping.txt",
        drop_activity_ids=(17, 18),
    ),
    "gtea": DatasetConfig(
        name="gtea",
        log_path=ROOT / "datasets/gtea_asformer/gtea_asformer_log.csv",
        mapping_path=ROOT / "data/data/gtea/mapping.txt",
        drop_activity_ids=(10,),
        recipe_source_dir=ROOT / "data/data/gtea/groundTruth",
        recipe_format="gtea",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["breakfast", "50salads", "gtea"],
        choices=sorted(DATASETS),
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--noise-grid",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help="Noise thresholds to test for the inductive miner.",
    )
    parser.add_argument(
        "--metric",
        choices=["token", "alignment"],
        default="token",
        help="Conformance metric family for fitness / precision.",
    )
    parser.add_argument(
        "--partition-mode",
        choices=["global", "recipe", "both"],
        default="global",
        help="Evaluate only global logs, only recipe partitions, or both.",
    )
    parser.add_argument(
        "--sampling-fractions",
        nargs="*",
        type=float,
        default=[],
        help="Optional trace fractions for sampled discovery evaluated on the full partition.",
    )
    parser.add_argument(
        "--sampling-seeds",
        nargs="*",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Random seeds for sampled discovery.",
    )
    parser.add_argument(
        "--variant-selection",
        choices=["off", "auto", "exhaustive", "beam"],
        default="off",
        help="Optional search over subsets of collapsed trace variants.",
    )
    parser.add_argument(
        "--variant-max-exhaustive",
        type=int,
        default=12,
        help="Use exhaustive variant search only up to this many unique variants when selection=auto.",
    )
    parser.add_argument(
        "--variant-beam-width",
        type=int,
        default=25,
        help="Beam width for variant subset search when selection=beam or auto falls back to beam.",
    )
    parser.add_argument(
        "--variant-max-subset-size",
        type=int,
        default=4,
        help="Maximum number of unique variants to include in a searched discovery subset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "discovery/models/video_im_collapsed",
        help="Directory for CSV summaries.",
    )
    parser.add_argument(
        "--require-sound",
        action="store_true",
        help="Keep only sound WF-net candidates in the best-per-partition outputs.",
    )
    parser.add_argument(
        "--require-1-safe",
        action="store_true",
        help="Keep only 1-safe candidates in the best-per-partition outputs.",
    )
    return parser.parse_args()


def load_mapping(mapping_path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with mapping_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            idx, label = line.split(maxsplit=1)
            mapping[int(idx)] = label
    return mapping


def collapse_consecutive_runs(df: pd.DataFrame) -> pd.DataFrame:
    shifted = df.groupby("case:concept:name", sort=False)["concept:name"].shift()
    return df.loc[df["concept:name"].ne(shifted)].copy()


def preprocess_log(df: pd.DataFrame, drop_activity_ids: Iterable[int]) -> pd.DataFrame:
    collapsed = collapse_consecutive_runs(df)
    filtered = collapsed.loc[~collapsed["concept:name"].isin(drop_activity_ids)].copy()
    collapsed_again = collapse_consecutive_runs(filtered)
    collapsed_again["case:concept:name"] = collapsed_again["case:concept:name"].astype(str)
    collapsed_again["concept:name"] = collapsed_again["concept:name"].astype(str)
    collapsed_again["event_idx"] = (
        collapsed_again.groupby("case:concept:name", sort=False).cumcount()
    )
    collapsed_again["time:timestamp"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        collapsed_again["event_idx"], unit="s"
    )
    return pm4py.format_dataframe(
        collapsed_again,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )


def build_recipe_map(config: DatasetConfig) -> dict[str, str]:
    if config.recipe_source_dir is None or config.recipe_format is None:
        return {}

    files = sorted(config.recipe_source_dir.glob("*.txt"))
    recipe_by_case: dict[str, str] = {}
    for i, path in enumerate(files):
        stem = path.stem
        if config.recipe_format == "breakfast":
            recipe = stem.split("_")[-1]
        elif config.recipe_format == "gtea":
            recipe = stem.split("_")[1]
        else:
            raise ValueError(f"Unsupported recipe format: {config.recipe_format}")
        recipe_by_case[str(i)] = recipe
    return recipe_by_case


def add_recipe_column(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    recipe_by_case = build_recipe_map(config)
    enriched = df.copy()
    enriched["partition_label"] = enriched["case:concept:name"].map(recipe_by_case)
    return enriched


def discover_inductive_model(df: pd.DataFrame | EventLog, noise_threshold: float):
    return pm4py.discover_petri_net_inductive(
        df, noise_threshold=noise_threshold
    )


def analyze_model_constraints(net, initial_marking, final_marking) -> dict[str, bool]:
    try:
        is_sound_wfnet, _diagnostics = pm4py.check_soundness(
            net, initial_marking, final_marking
        )
    except Exception:
        is_sound_wfnet = False

    is_one_safe = False
    if is_sound_wfnet:
        try:
            from pm4py.objects.petri_net.utils import reachability_graph

            incoming_transitions, _outgoing_transitions, _eventually_enabled = (
                reachability_graph.marking_flow_petri(net, initial_marking)
            )
            is_one_safe = all(
                all(tokens <= 1 for tokens in marking.values())
                for marking in incoming_transitions
            )
        except Exception:
            is_one_safe = False

    return {
        "is_sound_wfnet": bool(is_sound_wfnet),
        "is_one_safe": bool(is_one_safe),
    }


def score_model(
    df: pd.DataFrame, net, initial_marking, final_marking, metric: str
) -> dict[str, float | int]:
    if metric == "alignment":
        fitness = pm4py.fitness_alignments(df, net, initial_marking, final_marking)[
            "averageFitness"
        ]
        precision = pm4py.precision_alignments(df, net, initial_marking, final_marking)
    else:
        fitness = pm4py.fitness_token_based_replay(df, net, initial_marking, final_marking)[
            "average_trace_fitness"
        ]
        precision = pm4py.precision_token_based_replay(df, net, initial_marking, final_marking)

    return {
        "fitness": float(fitness),
        "precision": float(precision),
        "places": len(net.places),
        "transitions": len(net.transitions),
        "arcs": len(net.arcs),
        "invisible_transitions": sum(1 for t in net.transitions if t.label is None),
    }


def evaluate_model(df: pd.DataFrame, noise_threshold: float, metric: str) -> dict[str, float | int | bool]:
    net, initial_marking, final_marking = discover_inductive_model(df, noise_threshold)
    metrics = score_model(df, net, initial_marking, final_marking, metric)
    constraints = analyze_model_constraints(net, initial_marking, final_marking)
    return {**metrics, **constraints}


def build_base_row(
    dataset_name: str,
    partition_name: str,
    metric: str,
    n_traces: int,
    n_events: int,
    n_activities: int,
) -> dict[str, float | int | str | bool | None]:
    return {
        "dataset": dataset_name,
        "partition": partition_name,
        "metric_family": metric,
        "traces": n_traces,
        "events": n_events,
        "activities": n_activities,
        "sample_fraction": None,
        "sample_seed": None,
        "variant_strategy": None,
        "variant_indices": None,
        "variant_subset_size": None,
        "discovery_traces": None,
        "is_sound_wfnet": None,
        "is_one_safe": None,
        "passes_hard_constraints": None,
    }


def finalize_row(
    base_row: dict[str, float | int | str | bool | None],
    discovered_row: dict[str, float | int | str | bool],
    require_sound: bool,
    require_one_safe: bool,
) -> dict[str, float | int | str | bool]:
    passes_hard_constraints = True
    if require_sound:
        passes_hard_constraints = passes_hard_constraints and bool(
            discovered_row["is_sound_wfnet"]
        )
    if require_one_safe:
        passes_hard_constraints = passes_hard_constraints and bool(
            discovered_row["is_one_safe"]
        )

    return {
        **base_row,
        **discovered_row,
        "passes_hard_constraints": passes_hard_constraints,
    }


def sweep_partition_full(
    dataset_name: str,
    partition_name: str,
    df: pd.DataFrame,
    noise_grid: list[float],
    metric: str,
    require_sound: bool,
    require_one_safe: bool,
) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    n_traces = df["case:concept:name"].nunique()
    n_events = len(df)
    n_activities = df["concept:name"].nunique()
    base_row = build_base_row(
        dataset_name=dataset_name,
        partition_name=partition_name,
        metric=metric,
        n_traces=n_traces,
        n_events=n_events,
        n_activities=n_activities,
    )

    for noise in noise_grid:
        metrics = evaluate_model(df, noise, metric)
        rows.append(
            finalize_row(
                base_row,
                {
                    "discovery_mode": "full",
                    "noise_threshold": noise,
                    "discovery_traces": n_traces,
                    **metrics,
                    "meets_target": metrics["fitness"] > 0.8 and metrics["precision"] > 0.8,
                    "score_min_f_p": min(metrics["fitness"], metrics["precision"]),
                },
                require_sound=require_sound,
                require_one_safe=require_one_safe,
            )
        )

    return rows


def event_log_from_dataframe(df: pd.DataFrame) -> EventLog:
    return log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG)


def sample_event_log(log: EventLog, fraction: float, seed: int) -> EventLog:
    n_traces = max(1, int(len(log) * fraction))
    if n_traces >= len(log):
        return log

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(log)), n_traces))
    sampled = EventLog()
    for idx in indices:
        sampled.append(log[idx])
    return sampled


def sweep_partition_sampled(
    dataset_name: str,
    partition_name: str,
    df: pd.DataFrame,
    noise_grid: list[float],
    metric: str,
    sample_fractions: list[float],
    sample_seeds: list[int],
    require_sound: bool,
    require_one_safe: bool,
) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    n_traces = df["case:concept:name"].nunique()
    n_events = len(df)
    n_activities = df["concept:name"].nunique()
    full_log = event_log_from_dataframe(df)
    base_row = build_base_row(
        dataset_name=dataset_name,
        partition_name=partition_name,
        metric=metric,
        n_traces=n_traces,
        n_events=n_events,
        n_activities=n_activities,
    )

    for fraction in sample_fractions:
        for seed in sample_seeds:
            sampled_log = sample_event_log(full_log, fraction, seed)
            for noise in noise_grid:
                net, initial_marking, final_marking = discover_inductive_model(sampled_log, noise)
                metrics = score_model(df, net, initial_marking, final_marking, metric)
                constraints = analyze_model_constraints(net, initial_marking, final_marking)
                rows.append(
                    finalize_row(
                        base_row,
                        {
                            "discovery_mode": "sampled",
                            "noise_threshold": noise,
                            "sample_fraction": fraction,
                            "sample_seed": seed,
                            "discovery_traces": len(sampled_log),
                            **metrics,
                            **constraints,
                            "meets_target": metrics["fitness"] > 0.8 and metrics["precision"] > 0.8,
                            "score_min_f_p": min(metrics["fitness"], metrics["precision"]),
                        },
                        require_sound=require_sound,
                        require_one_safe=require_one_safe,
                    )
                )

    return rows


def get_variant_case_groups(df: pd.DataFrame) -> tuple[list[tuple[str, ...]], dict[tuple[str, ...], list[str]]]:
    variant_series = df.groupby("case:concept:name", sort=False)["concept:name"].apply(tuple)
    variant_to_cases: dict[tuple[str, ...], list[str]] = {}
    variants: list[tuple[str, ...]] = []
    for case_id, variant in variant_series.items():
        if variant not in variant_to_cases:
            variants.append(variant)
            variant_to_cases[variant] = []
        variant_to_cases[variant].append(case_id)
    return variants, variant_to_cases


def evaluate_best_noise_for_discovery_df(
    full_df: pd.DataFrame,
    discovery_df: pd.DataFrame,
    noise_grid: list[float],
    metric: str,
    require_sound: bool,
    require_one_safe: bool,
) -> dict[str, float | int | bool]:
    best_row: dict[str, float | int | bool] | None = None
    for noise in noise_grid:
        net, initial_marking, final_marking = discover_inductive_model(discovery_df, noise)
        metrics = score_model(full_df, net, initial_marking, final_marking, metric)
        constraints = analyze_model_constraints(net, initial_marking, final_marking)
        passes_hard_constraints = True
        if require_sound:
            passes_hard_constraints = passes_hard_constraints and constraints["is_sound_wfnet"]
        if require_one_safe:
            passes_hard_constraints = passes_hard_constraints and constraints["is_one_safe"]
        row = {
            "noise_threshold": noise,
            **metrics,
            **constraints,
            "meets_target": metrics["fitness"] > 0.8 and metrics["precision"] > 0.8,
            "score_min_f_p": min(metrics["fitness"], metrics["precision"]),
            "passes_hard_constraints": passes_hard_constraints,
        }
        if best_row is None or (
            row["passes_hard_constraints"],
            row["score_min_f_p"],
            row["fitness"],
            row["precision"],
        ) > (
            best_row["passes_hard_constraints"],
            best_row["score_min_f_p"],
            best_row["fitness"],
            best_row["precision"],
        ):
            best_row = row

    assert best_row is not None
    return best_row


def sweep_partition_variants(
    dataset_name: str,
    partition_name: str,
    df: pd.DataFrame,
    noise_grid: list[float],
    metric: str,
    selection_mode: str,
    max_exhaustive: int,
    beam_width: int,
    max_subset_size: int,
    require_sound: bool,
    require_one_safe: bool,
) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    n_traces = df["case:concept:name"].nunique()
    n_events = len(df)
    n_activities = df["concept:name"].nunique()
    base_row = build_base_row(
        dataset_name=dataset_name,
        partition_name=partition_name,
        metric=metric,
        n_traces=n_traces,
        n_events=n_events,
        n_activities=n_activities,
    )

    variants, variant_to_cases = get_variant_case_groups(df)
    if not variants:
        return rows

    effective_mode = selection_mode
    if selection_mode == "auto":
        effective_mode = "exhaustive" if len(variants) <= max_exhaustive else "beam"

    visited: set[tuple[int, ...]] = set()
    max_size = min(max_subset_size, len(variants))

    def evaluate_combo(combo: tuple[int, ...]) -> dict[str, float | int | str | bool]:
        selected_cases: list[str] = []
        for idx in combo:
            selected_cases.extend(variant_to_cases[variants[idx]])
        discovery_df = df[df["case:concept:name"].isin(selected_cases)].copy()
        best = evaluate_best_noise_for_discovery_df(
            df,
            discovery_df,
            noise_grid,
            metric,
            require_sound=require_sound,
            require_one_safe=require_one_safe,
        )
        return finalize_row(
            base_row,
            {
                "discovery_mode": f"variants_{effective_mode}",
                "variant_strategy": effective_mode,
                "variant_indices": ",".join(str(idx) for idx in combo),
                "variant_subset_size": len(combo),
                "discovery_traces": discovery_df["case:concept:name"].nunique(),
                **best,
            },
            require_sound=require_sound,
            require_one_safe=require_one_safe,
        )

    if effective_mode == "exhaustive":
        for size in range(1, max_size + 1):
            for combo in itertools.combinations(range(len(variants)), size):
                visited.add(combo)
                rows.append(evaluate_combo(combo))
        return rows

    frontier: list[tuple[tuple[int, ...], dict[str, float | int | str | bool]]] = []
    for idx in range(len(variants)):
        combo = (idx,)
        visited.add(combo)
        row = evaluate_combo(combo)
        rows.append(row)
        frontier.append((combo, row))

    frontier.sort(
        key=lambda item: (
            item[1]["score_min_f_p"],
            item[1]["fitness"],
            item[1]["precision"],
        ),
        reverse=True,
    )
    frontier = frontier[:beam_width]

    for size in range(2, max_size + 1):
        candidates: list[tuple[tuple[int, ...], dict[str, float | int | str | bool]]] = []
        for combo, _ in frontier:
            start_idx = combo[-1] + 1
            for next_idx in range(start_idx, len(variants)):
                next_combo = combo + (next_idx,)
                if next_combo in visited:
                    continue
                visited.add(next_combo)
                row = evaluate_combo(next_combo)
                rows.append(row)
                candidates.append((next_combo, row))

        if not candidates:
            break

        candidates.sort(
            key=lambda item: (
                item[1]["score_min_f_p"],
                item[1]["fitness"],
                item[1]["precision"],
            ),
            reverse=True,
        )
        frontier = candidates[:beam_width]

    return rows


def print_best_rows(summary_df: pd.DataFrame) -> None:
    for (dataset, partition), group in summary_df.groupby(["dataset", "partition"], sort=False):
        best = group.sort_values(["score_min_f_p", "fitness", "precision"], ascending=False).iloc[0]
        sampled_suffix = ""
        if best["discovery_mode"] == "sampled":
            sampled_suffix = (
                f" sample_fraction={best['sample_fraction']:.2f}"
                f" sample_seed={int(best['sample_seed'])}"
            )
        variant_suffix = ""
        if str(best["discovery_mode"]).startswith("variants_"):
            variant_suffix = (
                f" variants={best['variant_indices']}"
                f" discovery_traces={int(best['discovery_traces'])}"
            )
        print(
            f"{dataset:10s} {partition:16s} "
            f"mode={best['discovery_mode']:7s} "
            f"noise={best['noise_threshold']:.2f} "
            f"fitness={best['fitness']:.4f} "
            f"precision={best['precision']:.4f} "
            f"target={bool(best['meets_target'])} "
            f"sound={bool(best['is_sound_wfnet'])} "
            f"one_safe={bool(best['is_one_safe'])}"
            f"{sampled_suffix}"
            f"{variant_suffix}"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, float | int | str | bool]] = []

    for dataset_name in args.datasets:
        config = DATASETS[dataset_name]
        print(f"\n=== {dataset_name} ===")
        raw_df = pd.read_csv(config.log_path)
        processed_df = preprocess_log(raw_df, config.drop_activity_ids)

        if args.partition_mode in {"global", "both"}:
            all_rows.extend(
                sweep_partition_full(
                    dataset_name=dataset_name,
                    partition_name="global",
                    df=processed_df,
                    noise_grid=args.noise_grid,
                    metric=args.metric,
                    require_sound=args.require_sound,
                    require_one_safe=args.require_1_safe,
                )
            )
            if args.sampling_fractions:
                all_rows.extend(
                    sweep_partition_sampled(
                        dataset_name=dataset_name,
                        partition_name="global",
                        df=processed_df,
                        noise_grid=args.noise_grid,
                        metric=args.metric,
                        sample_fractions=args.sampling_fractions,
                        sample_seeds=args.sampling_seeds,
                        require_sound=args.require_sound,
                        require_one_safe=args.require_1_safe,
                    )
                )
            if args.variant_selection != "off":
                all_rows.extend(
                    sweep_partition_variants(
                        dataset_name=dataset_name,
                        partition_name="global",
                        df=processed_df,
                        noise_grid=args.noise_grid,
                        metric=args.metric,
                        selection_mode=args.variant_selection,
                        max_exhaustive=args.variant_max_exhaustive,
                        beam_width=args.variant_beam_width,
                        max_subset_size=args.variant_max_subset_size,
                        require_sound=args.require_sound,
                        require_one_safe=args.require_1_safe,
                    )
                )

        if args.partition_mode in {"recipe", "both"} and config.recipe_source_dir is not None:
            partitioned_df = add_recipe_column(processed_df, config)
            for partition_label, partition_df in partitioned_df.groupby("partition_label", sort=True):
                all_rows.extend(
                    sweep_partition_full(
                        dataset_name=dataset_name,
                        partition_name=f"recipe:{partition_label}",
                        df=partition_df,
                        noise_grid=args.noise_grid,
                        metric=args.metric,
                        require_sound=args.require_sound,
                        require_one_safe=args.require_1_safe,
                    )
                )
                if args.variant_selection != "off":
                    all_rows.extend(
                        sweep_partition_variants(
                            dataset_name=dataset_name,
                            partition_name=f"recipe:{partition_label}",
                            df=partition_df,
                            noise_grid=args.noise_grid,
                            metric=args.metric,
                            selection_mode=args.variant_selection,
                            max_exhaustive=args.variant_max_exhaustive,
                            beam_width=args.variant_beam_width,
                            max_subset_size=args.variant_max_subset_size,
                            require_sound=args.require_sound,
                            require_one_safe=args.require_1_safe,
                        )
                    )

    summary_df = pd.DataFrame(all_rows)
    summary_path = args.output_dir / f"video_im_collapsed_{args.metric}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    selection_df = summary_df
    if args.require_sound or args.require_1_safe:
        selection_df = selection_df[selection_df["passes_hard_constraints"]].copy()

    best_df = (
        selection_df.sort_values(
            ["dataset", "partition", "score_min_f_p", "fitness", "precision"],
            ascending=[True, True, False, False, False],
        )
        .groupby(["dataset", "partition"], sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_path = args.output_dir / f"video_im_collapsed_{args.metric}_best.csv"
    best_df.to_csv(best_path, index=False)

    print("\nBest settings per partition")
    print_best_rows(best_df)
    print(f"\nWrote {summary_path}")
    print(f"Wrote {best_path}")


if __name__ == "__main__":
    main()
