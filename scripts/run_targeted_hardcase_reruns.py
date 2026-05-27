#!/usr/bin/env python3
"""
Run targeted hard-case reruns on a selected subset of trace-model pairs.

The script builds per-model trace-hash allowlists from an aggregate CSV and
launches only those unique traces against the original model/log pair with a
longer timeout. This is intended for survivorship-bias checks on timeout-heavy
or high-conformance-cost slices.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


STATUS_FIELDS = [
    "timestamp",
    "run_id",
    "parent_run_id",
    "dataset_name",
    "model_path",
    "log_path",
    "status",
    "duration_seconds",
    "return_code",
    "results_dir",
    "command",
    "message",
]


@dataclass
class TargetedModelRun:
    dataset_name: str
    model_path: str
    log_path: str
    allowlist_path: str
    n_hashes: int
    parent_run_id: str
    shard_count: int


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run targeted hard-case reruns")
    p.add_argument("--aggregate-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--trace-shard-count", type=int, default=60)
    p.add_argument("--max-expansions", type=int, default=1_000_000)
    p.add_argument("--cost-threshold", type=int, default=30)
    p.add_argument("--include-any-timeout", action="store_true", default=True)
    p.add_argument(
        "--model-path-contains",
        type=str,
        default=None,
        help="Optional substring filter to restrict which model paths are rerun",
    )
    p.add_argument("--dry-run", action="store_true")
    return p


def _append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def _parse_results_dir(output: str) -> str:
    for line in output.splitlines():
        marker = "Results written to:"
        if marker in line:
            return line.split(marker, 1)[1].strip()
    return ""


def _build_parent_run_id(model_path: str, log_path: str, timeout: float, shard_count: int) -> str:
    payload = "|".join([model_path, log_path, str(timeout), str(shard_count)])
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def _prepare_targets(args: argparse.Namespace) -> Tuple[List[TargetedModelRun], Dict[str, object]]:
    usecols = [
        "dataset_name",
        "log_path",
        "model_path",
        "model_name",
        "trace_hash",
        "deviation_cost",
        "status",
    ]
    df = pd.read_csv(args.aggregate_csv, usecols=usecols, low_memory=False)
    if args.model_path_contains:
        df = df[df["model_path"].astype(str).str.contains(args.model_path_contains, regex=False)].copy()

    key_cols = ["dataset_name", "log_path", "model_path", "trace_hash"]
    grouped = (
        df.groupby(key_cols, as_index=False)
        .agg(
            any_timeout=("status", lambda s: (s.astype(str) == "timeout").any()),
            any_ok=("status", lambda s: (s.astype(str) == "ok").any()),
            min_cost=("deviation_cost", lambda s: pd.to_numeric(s, errors="coerce").dropna().min()),
        )
    )
    grouped["high_cost"] = grouped["min_cost"].fillna(-1) >= args.cost_threshold
    grouped["target"] = grouped["high_cost"] | grouped["any_timeout"]

    target_df = grouped[grouped["target"]].copy()
    out_dir = Path(args.output_dir)
    allowlists_dir = out_dir / "allowlists"
    allowlists_dir.mkdir(parents=True, exist_ok=True)

    targets: List[TargetedModelRun] = []
    per_model = (
        target_df.groupby(["dataset_name", "log_path", "model_path"], as_index=False)
        .agg(
            n_hashes=("trace_hash", "nunique"),
            timeout_hashes=("any_timeout", "sum"),
            high_cost_hashes=("high_cost", "sum"),
        )
    )

    for row in per_model.itertuples(index=False):
        dataset_name = row.dataset_name
        model_path = row.model_path
        log_path = row.log_path
        hashes = sorted(
            target_df[
                (target_df["dataset_name"] == dataset_name)
                & (target_df["model_path"] == model_path)
                & (target_df["log_path"] == log_path)
            ]["trace_hash"].astype(str).unique().tolist()
        )
        if not hashes:
            continue
        allowlist_name = hashlib.md5(f"{dataset_name}|{model_path}".encode("utf-8")).hexdigest()[:12]
        allowlist_path = allowlists_dir / f"{allowlist_name}.txt"
        allowlist_path.write_text("\n".join(hashes) + "\n", encoding="utf-8")
        shard_count = max(1, min(args.trace_shard_count, len(hashes)))
        parent_run_id = _build_parent_run_id(model_path, log_path, args.timeout, shard_count)
        targets.append(
            TargetedModelRun(
                dataset_name=dataset_name,
                model_path=model_path,
                log_path=log_path,
                allowlist_path=str(allowlist_path),
                n_hashes=len(hashes),
                parent_run_id=parent_run_id,
                shard_count=shard_count,
            )
        )

    summary = {
        "aggregate_csv": args.aggregate_csv,
        "model_path_contains": args.model_path_contains,
        "timeout_seconds": args.timeout,
        "cost_threshold": args.cost_threshold,
        "target_instances": int(target_df.shape[0]),
        "target_models": len(targets),
        "timeout_target_instances": int(target_df["any_timeout"].sum()),
        "high_cost_target_instances": int(target_df["high_cost"].sum()),
        "target_instances_both": int((target_df["any_timeout"] & target_df["high_cost"]).sum()),
        "per_model": [
            {
                "dataset_name": t.dataset_name,
                "model_path": t.model_path,
                "log_path": t.log_path,
                "n_hashes": t.n_hashes,
                "shard_count": t.shard_count,
                "allowlist_path": t.allowlist_path,
            }
            for t in targets
        ],
    }
    (out_dir / "target_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary["per_model"]).to_csv(out_dir / "target_models.csv", index=False)
    return targets, summary


def _interleave_runs(targets: List[TargetedModelRun]) -> List[Tuple[TargetedModelRun, int]]:
    queues = {target.parent_run_id: list(range(target.shard_count)) for target in targets}
    by_id = {target.parent_run_id: target for target in targets}
    order = [target.parent_run_id for target in targets]
    interleaved: List[Tuple[TargetedModelRun, int]] = []
    while order:
        next_order: List[str] = []
        for parent_id in order:
            shard_queue = queues[parent_id]
            if not shard_queue:
                continue
            shard_idx = shard_queue.pop(0)
            interleaved.append((by_id[parent_id], shard_idx))
            if shard_queue:
                next_order.append(parent_id)
        order = next_order
    return interleaved


def _launch_command(
    main_py: Path,
    target: TargetedModelRun,
    shard_idx: int,
    timeout: float,
    max_expansions: int,
    out_root: Path,
) -> Tuple[List[str], Path]:
    shard_root = out_root / target.parent_run_id / f"shard_{shard_idx:02d}"
    cmd = [
        sys.executable,
        str(main_py),
        "--mode",
        "dataset",
        "--model",
        target.model_path,
        "--log",
        target.log_path,
        "--output-dir",
        str(shard_root),
        "--timeout",
        str(timeout),
        "--max-expansions",
        str(max_expansions),
        "--algorithms",
        "all",
        "--heuristics",
        "all",
        "--trace-hash-allowlist",
        target.allowlist_path,
        "--trace-shard-count",
        str(target.shard_count),
        "--trace-shard-index",
        str(shard_idx),
    ]
    return cmd, shard_root


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_csv = out_dir / "batch_status.csv"
    with status_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        writer.writeheader()

    targets, summary = _prepare_targets(args)
    if not targets:
        print("No targeted models matched the selection criteria.")
        return 0

    repo_root = Path(__file__).resolve().parents[1]
    main_py = repo_root / "main.py"
    queue = _interleave_runs(targets)
    print(
        f"Prepared {summary['target_instances']} targeted trace-model hashes "
        f"across {summary['target_models']} models."
    )

    if args.dry_run:
        return 0

    running: List[Tuple[subprocess.Popen, float, TargetedModelRun, int, List[str]]] = []
    queue_index = 0
    worker_env = os.environ.copy()
    worker_env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    while queue_index < len(queue) or running:
        while queue_index < len(queue) and len(running) < args.jobs:
            target, shard_idx = queue[queue_index]
            queue_index += 1
            cmd, shard_root = _launch_command(
                main_py=main_py,
                target=target,
                shard_idx=shard_idx,
                timeout=args.timeout,
                max_expansions=args.max_expansions,
                out_root=out_dir,
            )
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                cwd=repo_root,
                env=worker_env,
                start_new_session=True,
            )
            running.append((proc, time.perf_counter(), target, shard_idx, cmd))

        next_running: List[Tuple[subprocess.Popen, float, TargetedModelRun, int, List[str]]] = []
        for proc, started_at, target, shard_idx, cmd in running:
            ret = proc.poll()
            if ret is None:
                next_running.append((proc, started_at, target, shard_idx, cmd))
                continue
            elapsed = time.perf_counter() - started_at
            row = {
                "timestamp": datetime.utcnow().isoformat(),
                "run_id": f"{target.parent_run_id}_s{shard_idx:02d}",
                "parent_run_id": target.parent_run_id,
                "dataset_name": target.dataset_name,
                "model_path": target.model_path,
                "log_path": target.log_path,
                "status": "success" if ret == 0 else "failed",
                "duration_seconds": f"{elapsed:.6f}",
                "return_code": ret,
                "results_dir": str(out_dir / target.parent_run_id / f"shard_{shard_idx:02d}"),
                "command": " ".join(cmd),
                "message": f"shard {shard_idx}/{target.shard_count}",
            }
            _append_csv_row(status_csv, STATUS_FIELDS, row)
        running = next_running
        if running:
            time.sleep(0.5)

    print("Targeted rerun batch finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
