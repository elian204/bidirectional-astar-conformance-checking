#!/usr/bin/env python3
"""
Search for deliberately weak but valid inductive-miner models.

The script discovers IM Petri nets from small case subsets, scores them on the
full collapsed log, and keeps only sound WF-nets that are also 1-safe.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import math
import multiprocessing
import queue as queue_mod
import random
import signal
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT.parent
DISCOVERY_DIR = REPO_ROOT / "discovery"
BASE_SCRIPT = DISCOVERY_DIR / "video_im_collapsed_sweep.py"
WORKER_MODULE = None
WORKER_FULL_DF = None


class EvaluationTimeout(Exception):
    pass


def handle_evaluation_timeout(signum, frame):
    raise EvaluationTimeout("candidate evaluation timed out")


@contextmanager
def evaluation_timeout(seconds: int):
    if seconds <= 0:
        yield
        return

    old_handler = signal.signal(signal.SIGALRM, handle_evaluation_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def evaluate_candidate_worker(case_ids: tuple[str, ...], noise: float, output_queue):
    try:
        discovery_df = WORKER_FULL_DF[
            WORKER_FULL_DF["case:concept:name"].isin(case_ids)
        ].copy()
        net, initial_marking, final_marking = WORKER_MODULE.discover_inductive_model(
            discovery_df, noise
        )
        metrics = WORKER_MODULE.score_model(
            WORKER_FULL_DF, net, initial_marking, final_marking, "token"
        )
        constraints = WORKER_MODULE.analyze_model_constraints(
            net, initial_marking, final_marking
        )
        output_queue.put(
            {
                "status": "ok",
                "metrics": metrics,
                "constraints": constraints,
            }
        )
    except Exception as exc:
        output_queue.put({"status": "error", "error": repr(exc)})


def evaluate_candidate_isolated(
    case_ids: tuple[str, ...], noise: float, timeout_seconds: int
) -> dict[str, object]:
    ctx = multiprocessing.get_context("fork")
    output_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=evaluate_candidate_worker,
        args=(case_ids, noise, output_queue),
    )
    process.start()
    process.join(timeout_seconds if timeout_seconds > 0 else None)

    if process.is_alive():
        process.terminate()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join()
        return {"status": "timeout"}

    try:
        return output_queue.get_nowait()
    except queue_mod.Empty:
        return {"status": "error", "error": f"exitcode={process.exitcode}"}


def load_base_module():
    spec = importlib.util.spec_from_file_location("video_im_collapsed_sweep", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["50salads", "gtea"],
        default=["50salads", "gtea"],
        help="Datasets to search.",
    )
    parser.add_argument(
        "--subset-sizes",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Case subset sizes to evaluate.",
    )
    parser.add_argument(
        "--noise-grid",
        nargs="+",
        type=float,
        default=[0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Noise thresholds to test.",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.4,
        help="Target metric value for both fitness and precision.",
    )
    parser.add_argument(
        "--target-max",
        type=float,
        default=0.45,
        help="Maximum fitness/precision preferred for the bad-model tier.",
    )
    parser.add_argument(
        "--max-combos-per-size",
        type=int,
        default=0,
        help="Optional random cap per subset size; 0 means exhaustive.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=7,
        help="Seed used when max-combos-per-size limits the search.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "discovery/models",
        help="Base output directory.",
    )
    parser.add_argument(
        "--eval-timeout-seconds",
        type=int,
        default=60,
        help="Skip a candidate if PM4Py scoring/soundness exceeds this many seconds; 0 disables.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print and checkpoint progress every N discovery subsets.",
    )
    parser.add_argument(
        "--isolate-evaluations",
        action="store_true",
        help="Run each candidate in a child process so hard PM4Py hangs can be killed.",
    )
    return parser.parse_args()


def iter_case_combos(
    cases: list[str], subset_size: int, max_combos_per_size: int, seed: int
):
    combos = list(itertools.combinations(cases, subset_size))
    if max_combos_per_size > 0 and len(combos) > max_combos_per_size:
        rng = random.Random(seed + subset_size)
        combos = rng.sample(combos, max_combos_per_size)
        combos.sort()
    return combos


def score_sort_frame(
    df: pd.DataFrame, target: float, target_max: float
) -> pd.DataFrame:
    ranked = df.copy()
    ranked["within_target_cap"] = (
        (ranked["fitness"] <= target_max) & (ranked["precision"] <= target_max)
    )
    ranked["target_distance"] = (
        (ranked["fitness"] - target).abs() + (ranked["precision"] - target).abs()
    )
    ranked["target_cap_overflow"] = (
        (ranked["fitness"] - target_max).clip(lower=0.0)
        + (ranked["precision"] - target_max).clip(lower=0.0)
    )
    ranked["metric_max"] = ranked[["fitness", "precision"]].max(axis=1)
    ranked["metric_min"] = ranked[["fitness", "precision"]].min(axis=1)
    return ranked.sort_values(
        [
            "within_target_cap",
            "target_cap_overflow",
            "target_distance",
            "metric_max",
            "metric_min",
            "subset_size",
            "noise_threshold",
        ],
        ascending=[False, True, True, True, True, True, True],
    )


def main() -> None:
    args = parse_args()
    if args.isolate_evaluations:
        multiprocessing.set_start_method("fork", force=True)

    module = load_base_module()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"video_im_low_quality_case_search_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for dataset_name in args.datasets:
        config = module.DATASETS[dataset_name]
        raw_df = pd.read_csv(config.log_path)
        full_df = module.preprocess_log(raw_df, config.drop_activity_ids)
        global WORKER_MODULE, WORKER_FULL_DF
        WORKER_MODULE = module
        WORKER_FULL_DF = full_df

        cases = sorted(full_df["case:concept:name"].astype(str).unique())
        dataset_rows: list[dict[str, object]] = []
        timeout_count = 0
        error_count = 0

        print(
            f"[{dataset_name}] traces={len(cases)} events={len(full_df)} "
            f"subset_sizes={args.subset_sizes} noise_grid={args.noise_grid}",
            flush=True,
        )

        total_combos = 0
        for subset_size in args.subset_sizes:
            combos = iter_case_combos(
                cases,
                subset_size=subset_size,
                max_combos_per_size=args.max_combos_per_size,
                seed=args.random_seed,
            )
            total_combos += len(combos)
            print(
                f"[{dataset_name}] subset_size={subset_size} combos={len(combos)}",
                flush=True,
            )

            for combo_idx, combo in enumerate(combos, start=1):
                discovery_df = full_df[full_df["case:concept:name"].isin(combo)].copy()
                for noise in args.noise_grid:
                    if args.isolate_evaluations:
                        result = evaluate_candidate_isolated(
                            tuple(combo), noise, args.eval_timeout_seconds
                        )
                        if result["status"] == "timeout":
                            timeout_count += 1
                            continue
                        if result["status"] != "ok":
                            error_count += 1
                            continue
                        metrics = result["metrics"]
                        constraints = result["constraints"]
                    else:
                        try:
                            with evaluation_timeout(args.eval_timeout_seconds):
                                net, initial_marking, final_marking = (
                                    module.discover_inductive_model(discovery_df, noise)
                                )
                                metrics = module.score_model(
                                    full_df, net, initial_marking, final_marking, "token"
                                )
                                constraints = module.analyze_model_constraints(
                                    net, initial_marking, final_marking
                                )
                        except EvaluationTimeout:
                            timeout_count += 1
                            continue
                        except Exception:
                            error_count += 1
                            continue

                    if not (
                        constraints["is_sound_wfnet"] and constraints["is_one_safe"]
                    ):
                        continue
                    dataset_rows.append(
                        {
                            "dataset": dataset_name,
                            "subset_size": subset_size,
                            "case_ids": ",".join(combo),
                            "discovery_traces": discovery_df["case:concept:name"].nunique(),
                            "noise_threshold": noise,
                            **metrics,
                            **constraints,
                        }
                    )

                should_report = (
                    combo_idx % args.progress_every == 0 or combo_idx == len(combos)
                )
                if should_report:
                    if dataset_rows:
                        partial_df = pd.DataFrame(dataset_rows)
                        partial_df = score_sort_frame(
                            partial_df, args.target, args.target_max
                        )
                        partial_df.to_csv(
                            run_dir / f"{dataset_name}_partial_results.csv",
                            index=False,
                        )
                    print(
                        f"[{dataset_name}] subset_size={subset_size} "
                        f"processed={combo_idx}/{len(combos)} "
                        f"valid_rows={len(dataset_rows)} "
                        f"timeouts={timeout_count} errors={error_count}",
                        flush=True,
                    )

        if not dataset_rows:
            print(f"[{dataset_name}] no valid sound+1-safe rows found", flush=True)
            continue

        dataset_df = pd.DataFrame(dataset_rows)
        dataset_df = score_sort_frame(dataset_df, args.target, args.target_max)
        all_path = run_dir / f"{dataset_name}_all_results.csv"
        top_path = run_dir / f"{dataset_name}_top50.csv"
        best_path = run_dir / f"{dataset_name}_best.csv"
        dataset_df.to_csv(all_path, index=False)
        dataset_df.head(50).to_csv(top_path, index=False)
        dataset_df.head(1).to_csv(best_path, index=False)

        best = dataset_df.iloc[0].to_dict()
        summary_rows.append(
            {
                "dataset": dataset_name,
                "search_combos": total_combos,
                "evaluated_rows": len(dataset_df),
                "best_case_ids": best["case_ids"],
                "best_subset_size": int(best["subset_size"]),
                "best_noise_threshold": float(best["noise_threshold"]),
                "best_fitness": float(best["fitness"]),
                "best_precision": float(best["precision"]),
                "within_target_cap": bool(best["within_target_cap"]),
                "target_distance": float(best["target_distance"]),
            }
        )
        print(
            f"[{dataset_name}] best fitness={best['fitness']:.6f} "
            f"precision={best['precision']:.6f} cases={best['case_ids']} "
            f"noise={best['noise_threshold']:.2f}",
            flush=True,
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(run_dir / "summary.csv", index=False)
        print("\nSummary")
        print(summary_df.to_string(index=False))
        print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
