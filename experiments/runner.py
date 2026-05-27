"""
experiments/runner.py
---------------------
Dataset-level experiment runner for conformance checking.

Runs 15 algorithm × heuristic combinations:
  - 5 Algorithms: forward, backward, bidir_std, bidir_mm, dibbs
  - 3 Heuristics: zero (h=0), me (marking equation), mmr (REACH)

Output structure:
  results/<dataset>_<timestamp>/
    ├── results.json   (full structured log)
    ├── results.csv    (flat table, one row per UNIQUE trace × method)
    └── run.log        (timestamped execution log)

Cost comparison with τ-epsilon:
  All cost comparisons use round() to extract the integer deviation cost.
  This ensures correct handling of τ-epsilon costs where two alignments
  with the same deviation count may have slightly different raw costs.

Winner selection:
  The winner for each trace is the method with minimum CPU time
  (among those finding the optimal deviation cost).
"""

from __future__ import annotations

import csv
import json
import logging
import multiprocessing as mp
import os
import platform
import signal
import sys
import time
import hashlib
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Local imports
from core.synchronous_product import SynchronousProduct
from core.trace_model import build_trace_net
from core.petri_net import WorkflowNet

from experiments.methods_config import (
    ALL_METHODS,
    MethodConfig,
    get_methods,
    Algorithm,
    Heuristic,
)
from experiments.method_dispatcher import run_method, find_winner

# Try to detect Gurobi availability
try:
    import gurobipy
    _gurobi_available = True
    try:
        # Test if license is valid
        env = gurobipy.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        env.dispose()
        SOLVER_INFO = "Gurobi (licensed)"
    except gurobipy.GurobiError as e:
        SOLVER_INFO = f"scipy (Gurobi error: {e})"
        _gurobi_available = False
except ImportError:
    SOLVER_INFO = "scipy (gurobipy not installed)"
    _gurobi_available = False


# =============================================================================
# Cost comparison utilities for τ-epsilon
# =============================================================================

def costs_equal(cost1: float, cost2: float) -> bool:
    """
    Check if two alignment costs represent the same deviation cost.

    With τ-epsilon costs, two alignments have equal deviation cost
    if round(cost1) == round(cost2).
    """
    return round(cost1) == round(cost2)


def deviation_cost(total_cost: float) -> int:
    """
    Extract integer deviation cost from total alignment cost.

    With τ-epsilon costs, deviation_cost = round(total_cost).
    """
    return round(total_cost)


# =============================================================================
# Timeout handling
# =============================================================================

class TimeoutError(Exception):
    """Raised when a search exceeds its time limit."""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Search timed out")


# =============================================================================
# Result structures
# =============================================================================

@dataclass
class MethodResult:
    """Result for a single method on a single trace."""
    method: str
    algorithm: str
    heuristic: str
    optimal_cost: float
    expansions: int
    generations: int
    time_seconds: float
    status: str  # "ok", "timeout", "error", "max_expansions"
    error_message: Optional[str] = None

    # Bidirectional-specific stats
    expansions_fwd: Optional[int] = None
    expansions_bwd: Optional[int] = None
    wasted_expansions: Optional[int] = None
    asymmetry_ratio: Optional[float] = None
    meeting_g_fwd: Optional[float] = None
    meeting_g_bwd: Optional[float] = None
    tau_transitions_fwd: Optional[int] = None
    tau_transitions_bwd: Optional[int] = None

    @property
    def deviation_cost(self) -> int:
        """Integer deviation cost (round of optimal_cost for τ-epsilon)."""
        return round(self.optimal_cost) if self.optimal_cost < float('inf') else -1


@dataclass
class TraceResult:
    """Result for all methods on a single trace."""
    trace_id: str
    trace_hash: str
    trace_length: int
    trace_activities: List[str]
    optimal_cost: float  # Consensus optimal cost (raw, may include ε)
    cost_agreement: bool  # All methods found same deviation cost?
    winner: Optional[str]  # Method with minimum CPU time (among optimal)
    winner_time_seconds: float  # Winner's CPU time
    winner_expansions: int  # Winner's expansion count (for backwards compatibility)
    methods: Dict[str, MethodResult] = field(default_factory=dict)
    sp_nodes: Optional[int] = None
    sp_edges: Optional[int] = None
    trace_unique_activities: Optional[int] = None
    trace_repetition_ratio: Optional[float] = None
    trace_unique_dfg_edges: Optional[int] = None
    trace_self_loops: Optional[int] = None
    trace_variant_frequency: Optional[int] = None
    trace_impossible_activities: Optional[int] = None

    @property
    def deviation_cost(self) -> int:
        """Integer deviation cost (round of optimal_cost for τ-epsilon)."""
        return round(self.optimal_cost) if self.optimal_cost < float('inf') else -1


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    experiment_id: str
    dataset_name: str
    model_path: str
    log_path: str
    algorithms: List[str]
    heuristics: List[str]
    max_traces: Optional[int]
    trace_shard_count: int
    trace_shard_index: int
    max_expansions: int
    timeout_seconds: float
    compute_sp_stats: bool
    compute_quality: bool
    gurobi_available: bool
    lp_solver: str
    timestamp: str


@dataclass
class ExperimentResult:
    """Complete result for an experiment."""
    config: ExperimentConfig
    model_info: Dict[str, Any]
    summary: Dict[str, Any]
    per_trace: List[TraceResult]


# =============================================================================
# Trace caching
# =============================================================================

def _trace_hash(trace: List[str]) -> str:
    """Compute hash for trace to detect duplicates."""
    return hashlib.md5(",".join(trace).encode()).hexdigest()[:12]


def _load_trace_hash_allowlist(path: Optional[str]) -> Optional[set]:
    """Load an optional newline-delimited set of permitted trace hashes."""
    if not path:
        return None
    allowlist_path = Path(path)
    return {
        line.strip()
        for line in allowlist_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _compute_trace_features(trace: List[str]) -> Dict[str, Any]:
    """Compute lightweight trace-level features used for runtime analysis."""
    n = len(trace)
    if n == 0:
        return {
            "trace_unique_activities": 0,
            "trace_repetition_ratio": 0.0,
            "trace_unique_dfg_edges": 0,
            "trace_self_loops": 0,
        }

    unique_activities = len(set(trace))
    unique_dfg_edges = len(set(zip(trace, trace[1:]))) if n > 1 else 0
    self_loops = sum(1 for i in range(n - 1) if trace[i] == trace[i + 1])

    return {
        "trace_unique_activities": unique_activities,
        "trace_repetition_ratio": 1.0 - (unique_activities / n),
        "trace_unique_dfg_edges": unique_dfg_edges,
        "trace_self_loops": self_loops,
    }


def _compute_model_features(wf: WorkflowNet) -> Dict[str, Any]:
    """Extract process-model structural features from the workflow net."""
    net = wf.net if hasattr(wf, "net") else wf

    features = {
        "model_places": None,
        "model_transitions": None,
        "model_arcs": None,
        "model_silent_transitions": None,
        "model_visible_transitions": None,
        "model_place_in_degree_avg": None,
        "model_place_out_degree_avg": None,
        "model_place_in_degree_max": None,
        "model_place_out_degree_max": None,
        "model_transition_in_degree_avg": None,
        "model_transition_out_degree_avg": None,
        "model_transition_in_degree_max": None,
        "model_transition_out_degree_max": None,
    }

    try:
        places = list(getattr(net, "places", []))
        transitions = list(getattr(net, "transitions", []))
        preset = getattr(net, "_preset", {})
        postset = getattr(net, "_postset", {})

        if hasattr(net, "arcs"):
            n_arcs = len(getattr(net, "arcs", []))
        else:
            # Internal PetriNet does not expose net.arcs.
            n_arcs = sum(len(preset.get(t, ())) + len(postset.get(t, ())) for t in transitions)

        silent = 0
        for t in transitions:
            label = getattr(t, "label", None)
            if label is None or (isinstance(label, str) and label.strip() in ("", "tau")):
                silent += 1

        transition_in = [len(preset.get(t, ())) for t in transitions]
        transition_out = [len(postset.get(t, ())) for t in transitions]

        place_in_map = {p: 0 for p in places}
        place_out_map = {p: 0 for p in places}
        for t in transitions:
            for p in preset.get(t, ()):
                place_out_map[p] = place_out_map.get(p, 0) + 1
            for p in postset.get(t, ()):
                place_in_map[p] = place_in_map.get(p, 0) + 1

        place_in = list(place_in_map.values())
        place_out = list(place_out_map.values())

        def _avg(values: List[int]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        def _max(values: List[int]) -> int:
            return max(values) if values else 0

        features["model_places"] = len(places)
        features["model_transitions"] = len(transitions)
        features["model_arcs"] = n_arcs
        features["model_silent_transitions"] = silent
        features["model_visible_transitions"] = len(transitions) - silent
        features["model_place_in_degree_avg"] = _avg(place_in)
        features["model_place_out_degree_avg"] = _avg(place_out)
        features["model_place_in_degree_max"] = _max(place_in)
        features["model_place_out_degree_max"] = _max(place_out)
        features["model_transition_in_degree_avg"] = _avg(transition_in)
        features["model_transition_out_degree_avg"] = _avg(transition_out)
        features["model_transition_in_degree_max"] = _max(transition_in)
        features["model_transition_out_degree_max"] = _max(transition_out)
    except Exception:
        pass

    return features


def _visible_model_labels(wf: WorkflowNet) -> set:
    """Collect visible activity labels from model transitions."""
    net = wf.net if hasattr(wf, "net") else wf
    labels = set()
    for t in getattr(net, "transitions", []):
        label = getattr(t, "label", None)
        if label is None:
            continue
        s = str(label).strip()
        if s and s.lower() != "tau":
            labels.add(s)
    return labels


def _count_trace_impossible_activities(trace: List[str], visible_labels: set) -> int:
    """Count events in a trace that are not present as visible model labels."""
    return sum(1 for act in trace if act not in visible_labels)


def _apply_trace_context_features(
    trace_result: TraceResult,
    trace_activities: List[str],
    trace_variant_frequency: Counter,
    visible_labels: set,
) -> None:
    """Set trace-context features independent of the search method internals."""
    t_hash = _trace_hash(trace_activities)
    trace_result.trace_variant_frequency = trace_variant_frequency.get(t_hash, 1)
    trace_result.trace_impossible_activities = _count_trace_impossible_activities(
        trace_activities, visible_labels
    )


def _select_shard_trace_hashes(
    traces: List[Tuple[str, List[str]]],
    trace_shard_count: int = 1,
    trace_shard_index: int = 0,
) -> Tuple[set, Counter, int]:
    """Compute deterministic unique-trace shard ownership and variant frequencies."""
    if trace_shard_count < 1:
        raise ValueError("trace_shard_count must be >= 1")
    if trace_shard_index < 0 or trace_shard_index >= trace_shard_count:
        raise ValueError("trace_shard_index must satisfy 0 <= index < count")

    trace_variant_frequency: Counter = Counter()
    first_occurrence_by_hash: Dict[str, Tuple[str, List[str]]] = {}

    for trace_id, trace_activities in traces:
        t_hash = _trace_hash(trace_activities)
        trace_variant_frequency[t_hash] += 1
        if t_hash not in first_occurrence_by_hash:
            first_occurrence_by_hash[t_hash] = (trace_id, trace_activities)

    unique_hashes_sorted = sorted(first_occurrence_by_hash.keys())
    if trace_shard_count == 1:
        return set(unique_hashes_sorted), trace_variant_frequency, len(unique_hashes_sorted)

    selected_hashes = {
        t_hash
        for idx, t_hash in enumerate(unique_hashes_sorted)
        if idx % trace_shard_count == trace_shard_index
    }
    return selected_hashes, trace_variant_frequency, len(unique_hashes_sorted)


def _runtime_environment(command_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Collect runtime metadata for reproducibility."""
    pm4py_version = None
    try:
        import pm4py as _pm4py  # type: ignore
        pm4py_version = getattr(_pm4py, "__version__", None)
    except Exception:
        pm4py_version = None

    gurobi_version = None
    try:
        import gurobipy as _gurobipy  # type: ignore
        if hasattr(_gurobipy, "gurobi") and hasattr(_gurobipy.gurobi, "version"):
            gurobi_version = ".".join(str(v) for v in _gurobipy.gurobi.version())
        else:
            gurobi_version = getattr(_gurobipy, "__version__", None)
    except Exception:
        gurobi_version = None

    return {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pm4py_version": pm4py_version,
        "gurobipy_version": gurobi_version,
        "solver_info": SOLVER_INFO,
        "cwd": os.getcwd(),
        "command_args": command_args or [],
    }


def _compute_model_identity(
    dataset_name: str,
    model_path: str,
    model_features: Dict[str, Any],
) -> Dict[str, str]:
    """Build stable identifiers for the process model used in a run."""
    if model_path:
        path_obj = Path(model_path)
        model_name = path_obj.name
        model_source = "file"
    else:
        model_name = f"{dataset_name}_discovered"
        model_source = "discovered"

    fingerprint = (
        f"{model_name}|{model_source}|"
        f"{model_features.get('model_places')}|"
        f"{model_features.get('model_transitions')}|"
        f"{model_features.get('model_arcs')}|"
        f"{model_features.get('model_silent_transitions')}"
    )
    model_id = hashlib.md5(fingerprint.encode()).hexdigest()[:12]

    return {
        "model_id": model_id,
        "model_name": model_name,
        "model_source": model_source,
    }


def _write_trace_result_rows(
    csv_writer,
    trace_result: TraceResult,
    trace_id: str,
    trace_hash: str,
    run_context: Dict[str, Any],
    model_features: Dict[str, Any],
) -> None:
    """Write one CSV row per method for a trace result."""
    for method_name, method_result in trace_result.methods.items():
        row = {
            "experiment_id": run_context["experiment_id"],
            "dataset_name": run_context["dataset_name"],
            "log_path": run_context["log_path"],
            "model_path": run_context["model_path"],
            "model_id": run_context["model_id"],
            "model_name": run_context["model_name"],
            "model_source": run_context["model_source"],
            "trace_id": trace_id,
            "trace_hash": trace_hash,
            "trace_activities": "|".join(trace_result.trace_activities),
            "trace_length": trace_result.trace_length,
            "trace_unique_activities": trace_result.trace_unique_activities,
            "trace_repetition_ratio": trace_result.trace_repetition_ratio,
            "trace_unique_dfg_edges": trace_result.trace_unique_dfg_edges,
            "trace_self_loops": trace_result.trace_self_loops,
            "trace_variant_frequency": trace_result.trace_variant_frequency,
            "trace_impossible_activities": trace_result.trace_impossible_activities,
            "trace_shard_count": run_context["trace_shard_count"],
            "trace_shard_index": run_context["trace_shard_index"],
            "sp_nodes": trace_result.sp_nodes,
            "sp_edges": trace_result.sp_edges,
            "optimal_cost": trace_result.optimal_cost,
            "deviation_cost": trace_result.deviation_cost,
            "method": method_name,
            "algorithm": method_result.algorithm,
            "heuristic": method_result.heuristic,
            "cost": method_result.optimal_cost,
            "expansions": method_result.expansions,
            "generations": method_result.generations,
            "time_seconds": method_result.time_seconds,
            "status": method_result.status,
            "expansions_fwd": method_result.expansions_fwd,
            "expansions_bwd": method_result.expansions_bwd,
            "wasted_expansions": method_result.wasted_expansions,
            "asymmetry_ratio": method_result.asymmetry_ratio,
            "model_places": model_features.get("model_places"),
            "model_transitions": model_features.get("model_transitions"),
            "model_arcs": model_features.get("model_arcs"),
            "model_silent_transitions": model_features.get("model_silent_transitions"),
            "model_visible_transitions": model_features.get("model_visible_transitions"),
            "model_place_in_degree_avg": model_features.get("model_place_in_degree_avg"),
            "model_place_out_degree_avg": model_features.get("model_place_out_degree_avg"),
            "model_place_in_degree_max": model_features.get("model_place_in_degree_max"),
            "model_place_out_degree_max": model_features.get("model_place_out_degree_max"),
            "model_transition_in_degree_avg": model_features.get("model_transition_in_degree_avg"),
            "model_transition_out_degree_avg": model_features.get("model_transition_out_degree_avg"),
            "model_transition_in_degree_max": model_features.get("model_transition_in_degree_max"),
            "model_transition_out_degree_max": model_features.get("model_transition_out_degree_max"),
        }
        csv_writer.writerow(row)


# =============================================================================
# Hard timeout execution
# =============================================================================

def _serialize_search_result(result: Any) -> Dict[str, Any]:
    """Convert algorithm SearchResult into a JSON/pickle-safe payload."""
    stats = getattr(result, "stats", None)
    return {
        "optimal_cost": getattr(result, "optimal_cost", float("inf")),
        "expansions": getattr(result, "expansions", 0),
        "generations": getattr(result, "generations", 0),
        "expansions_fwd": getattr(stats, "expansions_fwd", None) if stats else None,
        "expansions_bwd": getattr(stats, "expansions_bwd", None) if stats else None,
        "wasted_expansions": getattr(stats, "wasted_expansions", None) if stats else None,
        "asymmetry_ratio": getattr(stats, "asymmetry_ratio", None) if stats else None,
        "meeting_g_fwd": getattr(stats, "meeting_g_fwd", None) if stats else None,
        "meeting_g_bwd": getattr(stats, "meeting_g_bwd", None) if stats else None,
        "tau_transitions_fwd": getattr(stats, "tau_transitions_fwd", None) if stats else None,
        "tau_transitions_bwd": getattr(stats, "tau_transitions_bwd", None) if stats else None,
    }


def _run_method_worker(
    method: MethodConfig,
    sp: SynchronousProduct,
    max_expansions: int,
    collect_stats: bool,
    timeout_seconds: float,
    out_queue: mp.Queue,
) -> None:
    """Child-process worker for hard timeout execution of one method."""
    try:
        result = run_method(
            method=method,
            sp=sp,
            max_expansions=max_expansions,
            collect_stats=collect_stats,
            timeout_seconds=timeout_seconds,
        )
        out_queue.put({"ok": True, "result": _serialize_search_result(result)})
    except Exception as e:
        out_queue.put({"ok": False, "error": str(e)})


def _run_method_with_hard_timeout(
    method: MethodConfig,
    sp: SynchronousProduct,
    max_expansions: int,
    collect_stats: bool,
    timeout_seconds: float,
) -> Tuple[str, float, Optional[Dict[str, Any]], Optional[str]]:
    """
    Execute one method in a child process and kill it at timeout.

    Returns
    -------
    (status, elapsed, payload, error_message)
        status in {"ok", "timeout", "error"}
    """
    ctx = mp.get_context("fork")
    out_queue: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_run_method_worker,
        args=(method, sp, max_expansions, collect_stats, timeout_seconds, out_queue),
    )

    t_start = time.perf_counter()
    proc.start()
    proc.join(timeout_seconds)
    elapsed = time.perf_counter() - t_start

    try:
        if proc.is_alive():
            proc.terminate()
            proc.join(1.0)
            if proc.is_alive() and hasattr(proc, "kill"):
                proc.kill()
                proc.join(1.0)
            return ("timeout", elapsed, None, f"Exceeded {timeout_seconds}s timeout")

        payload = None
        try:
            payload = out_queue.get_nowait()
        except Exception:
            payload = None

        if payload is None:
            return (
                "error",
                elapsed,
                None,
                f"Worker exited with code {proc.exitcode} and returned no payload",
            )

        if not payload.get("ok", False):
            return ("error", elapsed, None, payload.get("error", "Unknown worker error"))

        return ("ok", elapsed, payload["result"], None)
    finally:
        try:
            out_queue.close()
            out_queue.join_thread()
        except Exception:
            pass


# =============================================================================
# Single trace execution
# =============================================================================

def _run_single_trace(
    sp: SynchronousProduct,
    trace: List[str],
    trace_id: str,
    methods: List[MethodConfig],
    max_expansions: int,
    timeout_seconds: float,
    collect_stats: bool = True,
) -> TraceResult:
    """
    Run all methods on a single trace.

    Parameters
    ----------
    sp : SynchronousProduct
        Pre-built synchronous product
    trace : List[str]
        The trace activities
    trace_id : str
        Identifier for this trace
    methods : List[MethodConfig]
        Methods to run
    max_expansions : int
        Expansion limit per method
    timeout_seconds : float
        Time limit per method
    collect_stats : bool
        Whether to collect detailed bidirectional stats

    Returns
    -------
    TraceResult
        Results for all methods
    """
    log = logging.getLogger("runner")

    trace_result = TraceResult(
        trace_id=trace_id,
        trace_hash=_trace_hash(trace),
        trace_length=len(trace),
        trace_activities=trace,
        optimal_cost=float('inf'),
        cost_agreement=True,
        winner=None,
        winner_time_seconds=0.0,
        winner_expansions=0,
    )
    trace_result.__dict__.update(_compute_trace_features(trace))

    costs_found = []
    use_hard_timeout = hasattr(os, "fork")

    for method in methods:
        method_name = method.name
        if use_hard_timeout:
            status_name, elapsed, payload, err = _run_method_with_hard_timeout(
                method=method,
                sp=sp,
                max_expansions=max_expansions,
                collect_stats=collect_stats,
                timeout_seconds=timeout_seconds,
            )

            if status_name == "timeout":
                method_result = MethodResult(
                    method=method_name,
                    algorithm=method.algorithm,
                    heuristic=method.heuristic,
                    optimal_cost=float("inf"),
                    expansions=0,
                    generations=0,
                    time_seconds=elapsed,
                    status="timeout",
                    error_message=err or f"Exceeded {timeout_seconds}s timeout",
                )
                log.warning(f"  {method_name}: TIMEOUT after {elapsed:.1f}s")
            elif status_name == "error":
                method_result = MethodResult(
                    method=method_name,
                    algorithm=method.algorithm,
                    heuristic=method.heuristic,
                    optimal_cost=float("inf"),
                    expansions=0,
                    generations=0,
                    time_seconds=elapsed,
                    status="error",
                    error_message=err or "Unknown method error",
                )
                log.error(f"  {method_name}: ERROR - {method_result.error_message}")
            else:
                method_payload = payload or {}
                result_optimal_cost = method_payload.get("optimal_cost", float("inf"))
                result_expansions = method_payload.get("expansions", 0)
                result_generations = method_payload.get("generations", 0)

                if result_optimal_cost == float("inf"):
                    status = "max_expansions" if result_expansions >= max_expansions else "no_solution"
                else:
                    status = "ok"

                method_result = MethodResult(
                    method=method_name,
                    algorithm=method.algorithm,
                    heuristic=method.heuristic,
                    optimal_cost=result_optimal_cost,
                    expansions=result_expansions,
                    generations=result_generations,
                    time_seconds=elapsed,
                    status=status,
                    expansions_fwd=method_payload.get("expansions_fwd"),
                    expansions_bwd=method_payload.get("expansions_bwd"),
                    wasted_expansions=method_payload.get("wasted_expansions"),
                    asymmetry_ratio=method_payload.get("asymmetry_ratio"),
                    meeting_g_fwd=method_payload.get("meeting_g_fwd"),
                    meeting_g_bwd=method_payload.get("meeting_g_bwd"),
                    tau_transitions_fwd=method_payload.get("tau_transitions_fwd"),
                    tau_transitions_bwd=method_payload.get("tau_transitions_bwd"),
                )
                costs_found.append(result_optimal_cost)
        else:
            # Fallback path for platforms without os.fork.
            use_signal_timeout = hasattr(signal, "SIGALRM")
            if use_signal_timeout:
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, timeout_seconds)

            t_start = time.perf_counter()

            try:
                result = run_method(
                    method=method,
                    sp=sp,
                    max_expansions=max_expansions,
                    collect_stats=collect_stats,
                    timeout_seconds=timeout_seconds,
                )

                elapsed = time.perf_counter() - t_start

                # Determine status
                if result.optimal_cost == float("inf"):
                    if result.expansions >= max_expansions:
                        status = "max_expansions"
                    else:
                        status = "no_solution"
                else:
                    status = "ok"

                method_result = MethodResult(
                    method=method_name,
                    algorithm=method.algorithm,
                    heuristic=method.heuristic,
                    optimal_cost=result.optimal_cost,
                    expansions=result.expansions,
                    generations=result.generations,
                    time_seconds=elapsed,
                    status=status,
                )

                # Add bidirectional stats if available
                if method.is_bidirectional and result.stats:
                    method_result.expansions_fwd = result.stats.expansions_fwd
                    method_result.expansions_bwd = result.stats.expansions_bwd
                    method_result.wasted_expansions = result.stats.wasted_expansions
                    method_result.asymmetry_ratio = result.stats.asymmetry_ratio
                    method_result.meeting_g_fwd = result.stats.meeting_g_fwd
                    method_result.meeting_g_bwd = result.stats.meeting_g_bwd
                    method_result.tau_transitions_fwd = result.stats.tau_transitions_fwd
                    method_result.tau_transitions_bwd = result.stats.tau_transitions_bwd

                costs_found.append(result.optimal_cost)

            except TimeoutError:
                elapsed = time.perf_counter() - t_start
                method_result = MethodResult(
                    method=method_name,
                    algorithm=method.algorithm,
                    heuristic=method.heuristic,
                    optimal_cost=float("inf"),
                    expansions=0,
                    generations=0,
                    time_seconds=elapsed,
                    status="timeout",
                    error_message=f"Exceeded {timeout_seconds}s timeout",
                )
                log.warning(f"  {method_name}: TIMEOUT after {elapsed:.1f}s")

            except Exception as e:
                elapsed = time.perf_counter() - t_start
                method_result = MethodResult(
                    method=method_name,
                    algorithm=method.algorithm,
                    heuristic=method.heuristic,
                    optimal_cost=float("inf"),
                    expansions=0,
                    generations=0,
                    time_seconds=elapsed,
                    status="error",
                    error_message=str(e),
                )
                log.error(f"  {method_name}: ERROR - {e}")

            finally:
                if use_signal_timeout:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)

        trace_result.methods[method_name] = method_result

    # Compute consensus optimal cost (minimum among finite costs)
    finite_costs = [c for c in costs_found if c < float('inf')]
    if finite_costs:
        trace_result.optimal_cost = min(finite_costs)

        # Cost agreement: all methods found same DEVIATION cost (use round() for τ-epsilon)
        optimal_deviation = round(trace_result.optimal_cost)
        trace_result.cost_agreement = all(
            round(c) == optimal_deviation for c in finite_costs
        )

    # Find winner: minimum CPU TIME among those finding optimal deviation cost
    optimal_deviation = round(trace_result.optimal_cost) if trace_result.optimal_cost < float('inf') else -1
    optimal_methods = [
        (name, r) for name, r in trace_result.methods.items()
        if r.status == "ok" and round(r.optimal_cost) == optimal_deviation
    ]
    if optimal_methods:
        # Winner is the method with minimum CPU time
        winner_name, winner_result = min(optimal_methods, key=lambda x: x[1].time_seconds)
        trace_result.winner = winner_name
        trace_result.winner_time_seconds = winner_result.time_seconds
        trace_result.winner_expansions = winner_result.expansions

    return trace_result


# =============================================================================
# Main experiment runner
# =============================================================================

def run_dataset_experiment(
    wf: WorkflowNet,
    traces: List[Tuple[str, List[str]]],  # List of (trace_id, trace_activities)
    dataset_name: str,
    model_path: str = "",
    log_path: str = "",
    algorithms: List[Algorithm] = None,
    heuristics: List[Heuristic] = None,
    max_traces: Optional[int] = None,
    trace_hash_allowlist_path: Optional[str] = None,
    trace_shard_count: int = 1,
    trace_shard_index: int = 0,
    max_expansions: int = 1_000_000,
    timeout_seconds: float = 30.0,
    output_dir: str = "results",
    compute_sp_stats: bool = False,
    compute_quality: bool = True,
    model_info: Dict[str, Any] = None,
    command_args: Optional[List[str]] = None,
) -> ExperimentResult:
    """
    Run a complete dataset experiment.

    Parameters
    ----------
    wf : WorkflowNet
        The workflow net (process model)
    traces : List[Tuple[str, List[str]]]
        List of (trace_id, trace_activities) tuples
    dataset_name : str
        Name for the dataset (used in output directory)
    model_path : str
        Path to model file (for logging)
    log_path : str
        Path to log file (for logging)
    algorithms : List[Algorithm]
        Algorithms to run (default: all 4)
    heuristics : List[Heuristic]
        Heuristics to use (default: all 3)
    max_traces : int
        Maximum number of traces to process
    max_expansions : int
        Expansion limit per method
    timeout_seconds : float
        Time limit per method
    output_dir : str
        Base directory for results
    compute_sp_stats : bool
        Whether to count SP nodes/edges (slow)
    compute_quality : bool
        Whether to compute fitness/precision
    model_info : dict
        Pre-computed model metadata

    Returns
    -------
    ExperimentResult
        Complete experiment results
    """
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{dataset_name}_{timestamp}"
    exp_dir = Path(output_dir) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to both file and console
    log = logging.getLogger("runner")
    log.setLevel(logging.INFO)
    log.handlers.clear()

    file_handler = logging.FileHandler(exp_dir / "run.log")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    log.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    log.addHandler(console_handler)

    # Get methods to run
    methods = get_methods(algorithms=algorithms, heuristics=heuristics)
    n_methods = len(methods)

    log.info("=" * 70)
    log.info(f"Experiment: {exp_id}")
    log.info(f"Dataset: {dataset_name}")
    log.info(f"Methods: {n_methods} ({[m.name for m in methods]})")
    log.info(f"Traces: {len(traces)} (max: {max_traces or 'all'})")
    log.info(f"Timeout: {timeout_seconds}s | Max expansions: {max_expansions:,}")
    log.info(f"LP solver: {SOLVER_INFO}")
    log.info(f"Winner criterion: minimum CPU time")
    log.info("=" * 70)
    runtime_env = _runtime_environment(command_args=command_args)
    log.info(f"Runtime: python={runtime_env['python_version']}")
    log.info(f"Runtime: platform={runtime_env['platform']}")
    log.info(f"Runtime: pm4py={runtime_env['pm4py_version']}, gurobipy={runtime_env['gurobipy_version']}")
    log.info(f"Runtime: args={runtime_env['command_args']}")

    # Limit traces if requested
    if max_traces is not None:
        traces = traces[:max_traces]

    selected_hashes, trace_variant_frequency, total_unique_traces = _select_shard_trace_hashes(
        traces,
        trace_shard_count=trace_shard_count,
        trace_shard_index=trace_shard_index,
    )
    allowed_hashes = _load_trace_hash_allowlist(trace_hash_allowlist_path)
    if allowed_hashes is not None:
        selected_hashes = selected_hashes & allowed_hashes
    if trace_shard_count > 1:
        traces = [
            (trace_id, trace_activities)
            for trace_id, trace_activities in traces
            if _trace_hash(trace_activities) in selected_hashes
        ]
    elif allowed_hashes is not None:
        traces = [
            (trace_id, trace_activities)
            for trace_id, trace_activities in traces
            if _trace_hash(trace_activities) in selected_hashes
        ]
    selected_unique_traces = len(selected_hashes)

    # Build config
    config = ExperimentConfig(
        experiment_id=exp_id,
        dataset_name=dataset_name,
        model_path=str(model_path),
        log_path=str(log_path),
        algorithms=[m.algorithm for m in methods],
        heuristics=list(set(m.heuristic for m in methods)),
        max_traces=max_traces,
        trace_shard_count=trace_shard_count,
        trace_shard_index=trace_shard_index,
        max_expansions=max_expansions,
        timeout_seconds=timeout_seconds,
        compute_sp_stats=compute_sp_stats,
        compute_quality=compute_quality,
        gurobi_available=_gurobi_available,
        lp_solver=SOLVER_INFO,
        timestamp=timestamp,
    )

    # Model info
    if model_info is None:
        model_info = {}
        try:
            if hasattr(wf, 'net'):
                model_info["places"] = len(wf.net.places)
                model_info["transitions"] = len(wf.net.transitions)
            elif hasattr(wf, 'places'):
                model_info["places"] = len(wf.places)
                model_info["transitions"] = len(wf.transitions)
        except Exception:
            model_info["places"] = "?"
            model_info["transitions"] = "?"
    model_features = _compute_model_features(wf)
    if model_features["model_places"] is not None:
        model_info["places"] = model_features["model_places"]
    if model_features["model_transitions"] is not None:
        model_info["transitions"] = model_features["model_transitions"]
    model_info["arcs"] = model_features["model_arcs"]
    model_info["silent_transitions"] = model_features["model_silent_transitions"]
    model_info["visible_transitions"] = model_features["model_visible_transitions"]
    model_info["place_in_degree_avg"] = model_features["model_place_in_degree_avg"]
    model_info["place_out_degree_avg"] = model_features["model_place_out_degree_avg"]
    model_info["place_in_degree_max"] = model_features["model_place_in_degree_max"]
    model_info["place_out_degree_max"] = model_features["model_place_out_degree_max"]
    model_info["transition_in_degree_avg"] = model_features["model_transition_in_degree_avg"]
    model_info["transition_out_degree_avg"] = model_features["model_transition_out_degree_avg"]
    model_info["transition_in_degree_max"] = model_features["model_transition_in_degree_max"]
    model_info["transition_out_degree_max"] = model_features["model_transition_out_degree_max"]
    model_identity = _compute_model_identity(
        dataset_name=dataset_name,
        model_path=str(model_path),
        model_features=model_features,
    )
    model_info.update(model_identity)
    run_context = {
        "experiment_id": exp_id,
        "dataset_name": dataset_name,
        "log_path": str(log_path),
        "model_path": str(model_path),
        "model_id": model_identity["model_id"],
        "model_name": model_identity["model_name"],
        "model_source": model_identity["model_source"],
        "trace_shard_count": trace_shard_count,
        "trace_shard_index": trace_shard_index,
    }

    # Prepare CSV output
    csv_path = exp_dir / "results.csv"
    csv_fields = [
        "experiment_id", "dataset_name", "log_path", "model_path",
        "model_id", "model_name", "model_source",
        "trace_id", "trace_hash", "trace_length",
        "trace_activities",
        "trace_unique_activities", "trace_repetition_ratio", "trace_unique_dfg_edges",
        "trace_self_loops", "trace_variant_frequency", "trace_impossible_activities",
        "trace_shard_count", "trace_shard_index",
        "sp_nodes", "sp_edges",
        "optimal_cost", "deviation_cost",
        "method", "algorithm", "heuristic",
        "cost", "expansions", "generations", "time_seconds", "status",
        "expansions_fwd", "expansions_bwd", "wasted_expansions", "asymmetry_ratio",
        "model_places", "model_transitions", "model_arcs",
        "model_silent_transitions", "model_visible_transitions",
        "model_place_in_degree_avg", "model_place_out_degree_avg",
        "model_place_in_degree_max", "model_place_out_degree_max",
        "model_transition_in_degree_avg", "model_transition_out_degree_avg",
        "model_transition_in_degree_max", "model_transition_out_degree_max",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Trace cache for identical traces
    trace_cache: Dict[str, TraceResult] = {}

    # Run experiment
    all_results: List[TraceResult] = []
    n_total = len(traces) * n_methods
    n_done = 0
    t_exp_start = time.perf_counter()

    # Progress tracking - log every 10%
    n_traces = len(traces)
    progress_interval = max(1, n_traces // 10)
    last_progress = -1

    if trace_shard_count > 1:
        log.info(f"Shard: {trace_shard_index + 1}/{trace_shard_count}")
        log.info(f"Unique traces in shard: {selected_unique_traces}/{total_unique_traces}")
    elif allowed_hashes is not None:
        log.info(f"Unique traces selected by allowlist: {selected_unique_traces}/{total_unique_traces}")

    log.info(f"Starting experiment: {n_traces} traces × {n_methods} methods = {n_traces * n_methods} runs")
    visible_labels = _visible_model_labels(wf)

    for i, (trace_id, trace_activities) in enumerate(traces):
        t_hash = _trace_hash(trace_activities)

        # Check cache
        if t_hash in trace_cache:
            cached = trace_cache[t_hash]

            # Create copy with new trace_id
            result = TraceResult(
                trace_id=trace_id,
                trace_hash=t_hash,
                trace_length=cached.trace_length,
                trace_activities=trace_activities,
                optimal_cost=cached.optimal_cost,
                cost_agreement=cached.cost_agreement,
                winner=cached.winner,
                winner_time_seconds=cached.winner_time_seconds,
                winner_expansions=cached.winner_expansions,
                methods=cached.methods,
                sp_nodes=cached.sp_nodes,
                sp_edges=cached.sp_edges,
                trace_unique_activities=cached.trace_unique_activities,
                trace_repetition_ratio=cached.trace_repetition_ratio,
                trace_unique_dfg_edges=cached.trace_unique_dfg_edges,
                trace_self_loops=cached.trace_self_loops,
            )
            _apply_trace_context_features(
                trace_result=result,
                trace_activities=trace_activities,
                trace_variant_frequency=trace_variant_frequency,
                visible_labels=visible_labels,
            )
            all_results.append(result)
            n_done += n_methods

            # Progress update at 10% intervals
            current_progress = (i + 1) * 10 // n_traces
            if current_progress > last_progress:
                elapsed = time.perf_counter() - t_exp_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (n_traces - i - 1) / rate if rate > 0 else 0
                log.info(f"Progress: {(i+1)*100//n_traces}% ({i+1}/{n_traces}) | "
                         f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
                last_progress = current_progress
            continue

        # Build synchronous product
        try:
            tn = build_trace_net(trace_activities)
            sp = SynchronousProduct(wf, tn)
        except Exception as e:
            log.error(f"[{i+1}/{n_traces}] Failed to build SP for {trace_id}: {e}")
            continue

        # Optional SP stats
        sp_nodes = None
        sp_edges = None
        if compute_sp_stats:
            try:
                sp_nodes = sp.count_reachable_markings()
                sp_edges = sp.count_edges()
            except Exception:
                pass

        # Run all methods
        result = _run_single_trace(
            sp=sp,
            trace=trace_activities,
            trace_id=trace_id,
            methods=methods,
            max_expansions=max_expansions,
            timeout_seconds=timeout_seconds,
            collect_stats=True,
        )
        result.sp_nodes = sp_nodes
        result.sp_edges = sp_edges
        _apply_trace_context_features(
            trace_result=result,
            trace_activities=trace_activities,
            trace_variant_frequency=trace_variant_frequency,
            visible_labels=visible_labels,
        )

        # Cache result
        trace_cache[t_hash] = result
        all_results.append(result)

        # Write to CSV incrementally
        _write_trace_result_rows(
            csv_writer=csv_writer,
            trace_result=result,
            trace_id=trace_id,
            trace_hash=t_hash,
            run_context=run_context,
            model_features=model_features,
        )
        csv_file.flush()

        # Progress update at 10% intervals
        n_done += n_methods
        current_progress = (i + 1) * 10 // n_traces
        if current_progress > last_progress:
            elapsed = time.perf_counter() - t_exp_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_traces - i - 1) / rate if rate > 0 else 0
            log.info(f"Progress: {(i+1)*100//n_traces}% ({i+1}/{n_traces}) | "
                     f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | "
                     f"Last: C*={result.deviation_cost}, winner={result.winner}")
            last_progress = current_progress

    csv_file.close()

    # Log 100% completion
    elapsed = time.perf_counter() - t_exp_start
    log.info(f"Progress: 100% ({n_traces}/{n_traces}) | Total time: {elapsed:.1f}s")

    # Compute summary statistics
    t_total = time.perf_counter() - t_exp_start

    summary = _compute_summary(all_results, methods, t_total)

    # Build final result
    experiment_result = ExperimentResult(
        config=config,
        model_info=model_info,
        summary=summary,
        per_trace=[],  # Will be converted to dicts for JSON
    )

    # Write JSON
    json_path = exp_dir / "results.json"
    json_data = {
        "config": asdict(config),
        "run_context": run_context,
        "runtime_environment": runtime_env,
        "model_info": model_info,
        "summary": summary,
        "per_trace": [_trace_result_to_dict(r) for r in all_results],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    log.info("=" * 70)
    log.info(f"Experiment complete in {t_total:.1f}s")
    log.info(f"Results: {exp_dir}")
    log.info(f"  - {json_path.name}")
    log.info(f"  - {csv_path.name}")
    log.info("=" * 70)

    # Print summary table
    _print_summary_table(summary, methods, log)

    return experiment_result


def _trace_result_to_dict(result: TraceResult) -> Dict:
    """Convert TraceResult to JSON-serializable dict."""
    return {
        "trace_id": result.trace_id,
        "trace_hash": result.trace_hash,
        "trace_length": result.trace_length,
        "trace_activities": result.trace_activities,
        "optimal_cost": result.optimal_cost,
        "deviation_cost": result.deviation_cost,
        "cost_agreement": result.cost_agreement,
        "winner": result.winner,
        "winner_time_seconds": result.winner_time_seconds,
        "winner_expansions": result.winner_expansions,
        "sp_nodes": result.sp_nodes,
        "sp_edges": result.sp_edges,
        "trace_unique_activities": result.trace_unique_activities,
        "trace_repetition_ratio": result.trace_repetition_ratio,
        "trace_unique_dfg_edges": result.trace_unique_dfg_edges,
        "trace_self_loops": result.trace_self_loops,
        "trace_variant_frequency": result.trace_variant_frequency,
        "trace_impossible_activities": result.trace_impossible_activities,
        "methods": {
            name: asdict(method_result)
            for name, method_result in result.methods.items()
        },
    }


def _compute_summary(
    results: List[TraceResult],
    methods: List[MethodConfig],
    total_time: float
) -> Dict[str, Any]:
    """Compute summary statistics across all traces."""
    n_traces = len(results)

    if n_traces == 0:
        return {"error": "No traces processed"}

    # Win counts (by CPU time)
    win_counts = {m.name: 0 for m in methods}
    for r in results:
        if r.winner:
            win_counts[r.winner] = win_counts.get(r.winner, 0) + 1

    # Per-method statistics
    method_stats = {}
    for method in methods:
        method_results = [
            r.methods.get(method.name)
            for r in results
            if method.name in r.methods
        ]

        valid = [m for m in method_results if m and m.status == "ok"]

        if valid:
            method_stats[method.name] = {
                "count": len(valid),
                "timeouts": sum(1 for m in method_results if m and m.status == "timeout"),
                "errors": sum(1 for m in method_results if m and m.status == "error"),
                "avg_expansions": sum(m.expansions for m in valid) / len(valid),
                "avg_generations": sum(m.generations for m in valid) / len(valid),
                "total_expansions": sum(m.expansions for m in valid),
                "total_generations": sum(m.generations for m in valid),
                "avg_time_seconds": sum(m.time_seconds for m in valid) / len(valid),
                "avg_time_ms": sum(m.time_seconds for m in valid) * 1000 / len(valid),
                "total_time_seconds": sum(m.time_seconds for m in valid),
                "wins": win_counts.get(method.name, 0),
                "win_rate": win_counts.get(method.name, 0) / n_traces,
            }
        else:
            method_stats[method.name] = {
                "count": 0,
                "timeouts": sum(1 for m in method_results if m and m.status == "timeout"),
                "errors": sum(1 for m in method_results if m and m.status == "error"),
            }

    # Cost agreement (using deviation cost for τ-epsilon)
    n_agree = sum(1 for r in results if r.cost_agreement)

    return {
        "n_traces": n_traces,
        "n_unique_traces": len(set(r.trace_hash for r in results)),
        "cost_agreement_rate": n_agree / n_traces if n_traces > 0 else 0,
        "total_time_seconds": total_time,
        "winner_criterion": "minimum_cpu_time",
        "win_counts": win_counts,
        "method_stats": method_stats,
    }


def _print_summary_table(summary: Dict, methods: List[MethodConfig], log):
    """Print a summary table to the log."""
    log.info("")
    log.info("Summary by Method (Winner = min CPU time):")
    log.info("-" * 110)
    log.info(f"{'Method':<25} {'OK':>6} {'TO':>4} {'Err':>4} {'Wins':>6} {'WinRate':>8} {'AvgExp':>12} {'AvgGen':>12} {'AvgCpu(ms)':>12}")
    log.info("-" * 110)

    for method in methods:
        stats = summary.get("method_stats", {}).get(method.name, {})
        log.info(
            f"{method.name:<25} "
            f"{stats.get('count', 0):>6} "
            f"{stats.get('timeouts', 0):>4} "
            f"{stats.get('errors', 0):>4} "
            f"{stats.get('wins', 0):>6} "
            f"{stats.get('win_rate', 0)*100:>7.1f}% "
            f"{stats.get('avg_expansions', 0):>12.0f} "
            f"{stats.get('avg_generations', 0):>12.0f} "
            f"{stats.get('avg_time_ms', 0):>12.2f}"
        )

    log.info("-" * 110)
