#!/usr/bin/env python3
"""
main.py  —  Bidirectional A* for Conformance Checking
======================================================

Modes
-----

1. Built-in toy example (no files needed):
     python main.py --mode toy

2. Single trace against a given model (quick test):
     python main.py --mode single \\
                    --model data/sepsis.pnml \\
                    --trace "A,B,C,D"

3. Dataset experiment — given model + log:
     python main.py --mode dataset \\
                    --model data/model.pnml \\
                    --log   data/log.xes

4. Dataset experiment with PICKLE model:
     python main.py --mode dataset \\
                    --model data/BPI_Challenge_2012_model.pkl \\
                    --log   data/BPI_Challenge_2012.xes

5. Dataset experiment — discover model with Inductive Miner:
     python main.py --mode dataset \\
                    --log   data/log.xes \\
                    --discover \\
                    --noise-threshold 0.2

6. Dataset with filtering and output control:
     python main.py --mode dataset \\
                    --model  data/model.pnml \\
                    --log    data/log.xes \\
                    --max-traces 200 \\
                    --algorithms forward bidir_mm \\
                    --heuristics me mmr \\
                    --max-expansions 500000 \\
                    --timeout 30 \\
                    --output-dir results

Algorithms (5):
  - forward    : Standard forward A*
  - backward   : Backward A* on reversed graph
  - bidir_std  : Bidirectional A* (Front-to-End)
  - bidir_mm   : Bidirectional A* (Meet-in-the-Middle / MM algorithm)
  - dibbs      : DIBBS bidirectional search

Heuristics (3):
  - zero       : h ≡ 0 (Dijkstra)
  - me         : Marking equation (LP relaxation)
  - mmr        : MMR/REACH heuristic (structural)

Total: 5 × 3 = 15 method combinations

Output (mode=dataset):
  results/<dataset>_<timestamp>/
    ├── results.json   (full structured log)
    ├── results.csv    (flat table)
    └── run.log        (execution log)

Cost model (τ-epsilon):
  - Synchronous moves:  cost = 0
  - Silent (τ) moves:   cost = ε = 10⁻⁶
  - Deviation moves:    cost = 1

  Deviation cost = round(total_cost) extracts the integer alignment cost.
  This ensures that among alignments with equal deviation count,
  those with fewer τ-moves are preferred (parsimonious alignments).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# =============================================================================
# Argument parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bidirectional A* for Conformance Checking",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- Mode ---
    p.add_argument(
        "--mode", choices=["toy", "single", "dataset"], default="toy",
        help=(
            "toy     : run built-in paper example\n"
            "single  : one trace against a model\n"
            "dataset : full benchmark experiment"
        ),
    )

    # --- Model / Log ---
    p.add_argument("--model", type=str, default=None,
                   help="Path to process model (.pnml or .pkl)")
    p.add_argument("--log", type=str, default=None,
                   help="Path to event log (.xes)")
    p.add_argument("--discover", action="store_true",
                   help="Discover model with Inductive Miner (requires --log)")
    p.add_argument("--noise-threshold", type=float, default=0.0,
                   help="Inductive Miner noise threshold [0,1] (default: 0.0)")
    p.add_argument("--no-save-model", action="store_true",
                   help="Do not save discovered model to disk")

    # --- Single-trace mode ---
    p.add_argument("--trace", type=str, default=None,
                   help="Comma-separated trace for --mode single (e.g. 'A,B,C')")

    # --- Filtering ---
    p.add_argument("--max-traces", type=int, default=None,
                   help="Cap on number of traces to process")
    p.add_argument(
        "--trace-hash-allowlist",
        type=str,
        default=None,
        help="Optional file with one trace hash per line; only those unique traces are processed",
    )
    p.add_argument("--trace-shard-count", type=int, default=1,
                   help="Number of deterministic unique-trace shards (default: 1)")
    p.add_argument("--trace-shard-index", type=int, default=0,
                   help="Which unique-trace shard to execute (default: 0)")
    p.add_argument(
        "--algorithms", nargs="+",
        choices=["forward", "backward", "bidir_std", "bidir_mm", "dibbs", "all"],
        default=["all"],
        help="Algorithms to run (default: all five)"
    )
    p.add_argument(
        "--heuristics", nargs="+",
        choices=["zero", "h0", "me", "marking_eq", "mmr", "reach", "all"],
        default=["all"],
        help="Heuristics to use (default: all three)"
    )

    # --- Search parameters ---
    p.add_argument("--max-expansions", type=int, default=1_000_000,
                   help="Expansion cap per (trace, method) (default: 1,000,000)")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="Timeout per (trace, method) in seconds (default: 30)")

    # --- Output ---
    p.add_argument("--output-dir", type=str, default="results",
                   help="Directory for experiment output (default: results/)")
    p.add_argument("--sp-stats", action="store_true",
                   help="Count SP nodes/edges per trace (slow for large models)")
    p.add_argument("--no-quality", action="store_true",
                   help="Skip fitness/precision computation")

    # --- Toy benchmark ---
    p.add_argument("--toy-benchmark",
                   choices=["paper_example", "simple_sequence"],
                   default="paper_example",
                   help="Which built-in benchmark for --mode toy")

    return p


# =============================================================================
# Normalize algorithm/heuristic names
# =============================================================================

def normalize_algorithms(args_algorithms):
    """Convert CLI algorithm names to internal names."""
    if "all" in args_algorithms:
        return None  # None means all
    return args_algorithms


def normalize_heuristics(args_heuristics):
    """Convert CLI heuristic names to internal names."""
    if "all" in args_heuristics:
        return None  # None means all

    # Map aliases
    mapping = {
        "h0": "zero",
        "marking_eq": "me",
        "reach": "mmr",
    }
    return [mapping.get(h, h) for h in args_heuristics]


# =============================================================================
# Cost comparison utilities for τ-epsilon
# =============================================================================

def deviation_cost(total_cost: float) -> int:
    """
    Extract integer deviation cost from total alignment cost.

    With τ-epsilon costs, deviation_cost = round(total_cost).
    """
    return round(total_cost)


def costs_equal(cost1: float, cost2: float) -> bool:
    """
    Check if two alignment costs represent the same deviation cost.

    With τ-epsilon costs, two alignments have equal deviation cost
    if round(cost1) == round(cost2).
    """
    return round(cost1) == round(cost2)


# =============================================================================
# Mode: toy
# =============================================================================

def run_toy(args):
    """Run built-in toy example."""
    from experiments.benchmark_loader import build_paper_example, build_simple_sequence
    from core.trace_model import build_trace_net
    from core.synchronous_product import SynchronousProduct
    from experiments.methods_config import ALL_METHODS, get_methods
    from experiments.method_dispatcher import run_method

    print("=" * 85)
    print("TOY MODE: Built-in Example")
    print("=" * 85)

    # Load example
    if args.toy_benchmark == "paper_example":
        wf, trace = build_paper_example()
        expected_cost = 2
        print("Benchmark: Paper example (Figure 1)")
        print(f"Trace: {trace}")
        print(f"Expected optimal deviation cost: {expected_cost}")
    else:
        wf, trace = build_simple_sequence()
        expected_cost = 1
        print("Benchmark: Simple sequence")
        print(f"Trace: {trace}")
        print(f"Expected optimal deviation cost: {expected_cost}")

    print("-" * 85)

    # Build SP
    tn = build_trace_net(trace)
    sp = SynchronousProduct(wf, tn)

    # Get methods to run
    algorithms = normalize_algorithms(args.algorithms)
    heuristics = normalize_heuristics(args.heuristics)
    methods = get_methods(algorithms=algorithms, heuristics=heuristics)

    print(f"Running {len(methods)} methods...")
    print("-" * 85)
    print(
        f"{'Method':<25} {'DevCost':>8} {'RawCost':>10} {'Expansions':>10} {'Generations':>12} {'Time (ms)':>10} {'Status':>8}")
    print("-" * 85)

    all_correct = True
    for method in methods:
        try:
            result = run_method(method, sp, max_expansions=args.max_expansions)

            # Use round() to extract deviation cost for comparison with τ-epsilon
            computed_deviation = round(result.optimal_cost)
            correct = (computed_deviation == expected_cost)
            status = "✓" if correct else "✗ WRONG"
            if not correct:
                all_correct = False

            print(f"{method.name:<25} {computed_deviation:>8d} "
                  f"{result.optimal_cost:>10.6f} "
                  f"{result.expansions:>10d} {result.generations:>12d} "
                  f"{result.time_seconds * 1000:>10.2f} {status:>8}")

        except Exception as e:
            print(f"{method.name:<25} {'ERROR':>8} {0:>10} {0:>10} {0:>12} {0:>10.2f} {str(e)[:20]:>8}")
            all_correct = False

    print("-" * 85)
    if all_correct:
        print("✓ All methods found correct optimal deviation cost!")
    else:
        print("✗ Some methods returned incorrect results!")
    print("=" * 85)

    return 0 if all_correct else 1


# =============================================================================
# Mode: single
# =============================================================================

def run_single(args):
    """Run single trace against a model."""
    from experiments.benchmark_loader import load_model
    from core.trace_model import build_trace_net
    from core.synchronous_product import SynchronousProduct
    from experiments.methods_config import get_methods
    from experiments.method_dispatcher import run_method

    if not args.model:
        print("ERROR: --model required for --mode single")
        return 1
    if not args.trace:
        print("ERROR: --trace required for --mode single")
        return 1

    print("=" * 85)
    print("SINGLE TRACE MODE")
    print("=" * 85)

    # Load model
    print(f"Loading model: {args.model}")
    wf, _ = load_model(args.model)
    print(f"  Places: {len(wf.net.places)}, Transitions: {len(wf.net.transitions)}")

    # Parse trace
    trace = [a.strip() for a in args.trace.split(",")]
    print(f"Trace: {trace} (length={len(trace)})")
    print("-" * 85)

    # Build SP
    tn = build_trace_net(trace)
    sp = SynchronousProduct(wf, tn)

    # Get methods
    algorithms = normalize_algorithms(args.algorithms)
    heuristics = normalize_heuristics(args.heuristics)
    methods = get_methods(algorithms=algorithms, heuristics=heuristics)

    print(f"Running {len(methods)} methods...")
    print("-" * 85)
    print(f"{'Method':<25} {'DevCost':>8} {'RawCost':>12} {'Expansions':>12} {'Time (ms)':>12}")
    print("-" * 85)

    costs = []
    for method in methods:
        try:
            result = run_method(method, sp, max_expansions=args.max_expansions)
            costs.append(result.optimal_cost)

            dev_cost = round(result.optimal_cost)
            print(f"{method.name:<25} {dev_cost:>8d} {result.optimal_cost:>12.6f} "
                  f"{result.expansions:>12d} {result.time_seconds * 1000:>12.2f}")

            # Print bidirectional stats
            if method.is_bidirectional and result.stats:
                print(f"  {'└─ F:':<3}{result.stats.expansions_fwd}, "
                      f"B:{result.stats.expansions_bwd}, "
                      f"wasted:{result.stats.wasted_expansions}, "
                      f"asym:{result.stats.asymmetry_ratio:.2f}")

        except Exception as e:
            print(f"{method.name:<25} {'ERROR':>8} {0:>12} {0:>12} {0:>12.2f}  {e}")

    print("-" * 85)

    # Check agreement (using deviation cost for τ-epsilon)
    finite_costs = [c for c in costs if c < float('inf')]
    if finite_costs:
        deviation_costs = [round(c) for c in finite_costs]
        if len(set(deviation_costs)) == 1:
            print(f"✓ All methods agree: deviation_cost = {deviation_costs[0]}")
        else:
            print(f"✗ MISMATCH: deviation_costs = {set(deviation_costs)}")

    print("=" * 85)
    return 0


# =============================================================================
# Mode: dataset
# =============================================================================

def run_dataset(args):
    """Run full dataset experiment."""
    from experiments.benchmark_loader import load_dataset
    from experiments.runner import run_dataset_experiment

    print("=" * 70)
    print("DATASET MODE")
    print("=" * 70)

    # Validate arguments
    if not args.log and not args.model:
        print("ERROR: --log or --model required for --mode dataset")
        return 1

    if getattr(args, 'discover', False) and not args.log:
        print("ERROR: --discover requires --log")
        return 1

    # Load dataset
    print(f"Loading dataset...")
    print(f"  Log: {args.log}")
    print(f"  Model: {args.model or '(will discover)'}")

    try:
        # Call load_dataset with the correct signature for your codebase
        result = load_dataset(
            log_path=args.log,
            model_path=args.model,
            discover_model=args.discover if hasattr(args, 'discover') else False,
            noise_threshold=args.noise_threshold if hasattr(args, 'noise_threshold') else 0.0,
            max_traces=args.max_traces,
            compute_quality=not getattr(args, 'no_quality', False),
        )

        # Handle different return formats
        if len(result) == 4:
            wf, traces, case_ids, model_info = result
            # Convert traces to (trace_id, trace) format if needed
            if case_ids:
                traces = list(zip(case_ids, traces))
        elif len(result) == 3:
            wf, traces, model_info = result
            # traces might already be in the right format
            if traces and isinstance(traces[0], (list, tuple)) and len(traces[0]) == 2:
                pass  # Already (id, trace) format
            else:
                traces = [(f"trace_{i}", t) for i, t in enumerate(traces)]
        else:
            wf, traces = result[:2]
            model_info = {}
            traces = [(f"trace_{i}", t) for i, t in enumerate(traces)]

    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Dataset name from log file
    if args.log:
        dataset_name = Path(args.log).stem
    elif args.model:
        dataset_name = Path(args.model).stem
    else:
        dataset_name = "unknown"

    print(f"  Dataset: {dataset_name}")
    print(f"  Traces: {len(traces)}")

    # Handle different key names in model_info
    places = model_info.get('places', model_info.get('model_places', '?'))
    transitions = model_info.get('transitions', model_info.get('model_transitions', '?'))
    print(f"  Model: {places} places, {transitions} transitions")

    if 'fitness' in model_info:
        print(f"  Fitness: {model_info['fitness']:.4f}, "
              f"Precision: {model_info.get('precision', 'N/A')}")

    # Get methods
    algorithms = normalize_algorithms(args.algorithms)
    heuristics = normalize_heuristics(args.heuristics)

    print("-" * 70)

    # Run experiment
    result = run_dataset_experiment(
        wf=wf,
        traces=traces,
        dataset_name=dataset_name,
        model_path=args.model or "",
        log_path=args.log or "",
        algorithms=algorithms,
        heuristics=heuristics,
        max_traces=args.max_traces,
        max_expansions=args.max_expansions,
        timeout_seconds=args.timeout,
        trace_hash_allowlist_path=args.trace_hash_allowlist,
        trace_shard_count=args.trace_shard_count,
        trace_shard_index=args.trace_shard_index,
        output_dir=args.output_dir,
        compute_sp_stats=getattr(args, 'sp_stats', False),
        compute_quality=not getattr(args, 'no_quality', False),
        model_info=model_info,
        command_args=sys.argv[1:],
    )

    print("=" * 70)
    return 0


# =============================================================================
# Main entry point
# =============================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.trace_shard_count < 1:
        parser.error("--trace-shard-count must be >= 1")
    if args.trace_shard_index < 0:
        parser.error("--trace-shard-index must be >= 0")
    if args.trace_shard_index >= args.trace_shard_count:
        parser.error("--trace-shard-index must be < --trace-shard-count")

    # Dispatch to mode
    if args.mode == "toy":
        return run_toy(args)
    elif args.mode == "single":
        return run_single(args)
    elif args.mode == "dataset":
        return run_dataset(args)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
