#!/usr/bin/env python3
"""Aggregate canonical and in-progress A* experiment CSVs into one analysis-ready file.

Rules:
- Prefer parent-level ``merged_results.csv`` when it exists.
- Otherwise include all shard-level ``results.csv`` files under that parent run.
- Reruns simply overwrite the aggregate outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]

META_COLUMNS = [
    "aggregate_run_name",
    "aggregate_batch_dir",
    "aggregate_parent_run_id",
    "aggregate_source_kind",
    "aggregate_source_file",
    "aggregate_parent_complete",
]


@dataclass
class SourceFile:
    run_name: str
    batch_dir: Path
    parent_run_id: str
    source_kind: str
    source_file: Path
    parent_complete: bool
    row_count: int


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate A* CSV outputs into one file")
    p.add_argument(
        "--root-dir",
        type=Path,
        default=REPO_ROOT / "tmp_smoke",
        help="Root directory to scan for batch_results directories",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to the aggregate CSV to write",
    )
    p.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Optional manifest CSV path (default: alongside output)",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary JSON path (default: alongside output)",
    )
    p.add_argument(
        "--run-name-contains",
        nargs="*",
        default=[],
        help="Optional substrings; if provided, only include runs whose root names contain one of them",
    )
    p.add_argument(
        "--include-noncanonical-archives",
        action="store_true",
        help="Include runs stored under noncanonical_archive_* directories",
    )
    return p


def count_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def iter_batch_dirs(root_dir: Path, include_noncanonical_archives: bool) -> Iterable[Path]:
    for batch_dir in sorted(root_dir.rglob("batch_results")):
        parts = batch_dir.parts
        if (not include_noncanonical_archives) and any(
            part.startswith("noncanonical_archive_") for part in parts
        ):
            continue
        yield batch_dir


def run_name_allowed(run_name: str, filters: List[str]) -> bool:
    if not filters:
        return True
    return any(token in run_name for token in filters)


def collect_sources(
    root_dir: Path,
    run_name_filters: List[str],
    include_noncanonical_archives: bool,
) -> List[SourceFile]:
    sources: List[SourceFile] = []
    for batch_dir in iter_batch_dirs(root_dir, include_noncanonical_archives):
        run_name = batch_dir.parent.name
        if not run_name_allowed(run_name, run_name_filters):
            continue

        parent_dirs = sorted(p for p in batch_dir.iterdir() if p.is_dir())
        for parent_dir in parent_dirs:
            merged = parent_dir / "merged_results.csv"
            if merged.exists():
                sources.append(
                    SourceFile(
                        run_name=run_name,
                        batch_dir=batch_dir,
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
                        run_name=run_name,
                        batch_dir=batch_dir,
                        parent_run_id=parent_dir.name,
                        source_kind="partial_shard",
                        source_file=shard_csv,
                        parent_complete=False,
                        row_count=count_rows(shard_csv),
                    )
                )
    return sources


def determine_fieldnames(sources: List[SourceFile]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for source in sources:
        with source.source_file.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in seen:
                    ordered.append(field)
                    seen.add(field)
    return META_COLUMNS + ordered


def write_aggregate(output_csv: Path, sources: List[SourceFile], fieldnames: List[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for source in sources:
            with source.source_file.open(newline="") as src_handle:
                reader = csv.DictReader(src_handle)
                for row in reader:
                    row.update(
                        {
                            "aggregate_run_name": source.run_name,
                            "aggregate_batch_dir": str(source.batch_dir),
                            "aggregate_parent_run_id": source.parent_run_id,
                            "aggregate_source_kind": source.source_kind,
                            "aggregate_source_file": str(source.source_file),
                            "aggregate_parent_complete": int(source.parent_complete),
                        }
                    )
                    writer.writerow(row)


def write_manifest(manifest_csv: Path, sources: List[SourceFile]) -> None:
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "batch_dir",
                "parent_run_id",
                "source_kind",
                "source_file",
                "parent_complete",
                "row_count",
            ],
        )
        writer.writeheader()
        for source in sources:
            writer.writerow(
                {
                    "run_name": source.run_name,
                    "batch_dir": str(source.batch_dir),
                    "parent_run_id": source.parent_run_id,
                    "source_kind": source.source_kind,
                    "source_file": str(source.source_file),
                    "parent_complete": int(source.parent_complete),
                    "row_count": source.row_count,
                }
            )


def write_summary(summary_json: Path, sources: List[SourceFile], output_csv: Path) -> None:
    summary = {
        "output_csv": str(output_csv),
        "source_files": len(sources),
        "merged_sources": sum(s.source_kind == "merged" for s in sources),
        "partial_shard_sources": sum(s.source_kind == "partial_shard" for s in sources),
        "completed_parent_runs": len({(s.run_name, s.parent_run_id) for s in sources if s.parent_complete}),
        "partial_parent_runs": len({(s.run_name, s.parent_run_id) for s in sources if not s.parent_complete}),
        "total_rows_from_sources": sum(s.row_count for s in sources),
        "runs": sorted({s.run_name for s in sources}),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2))


def main() -> int:
    args = build_parser().parse_args()
    manifest_csv = args.manifest_csv or args.output_csv.with_name(args.output_csv.stem + "_manifest.csv")
    summary_json = args.summary_json or args.output_csv.with_name(args.output_csv.stem + "_summary.json")

    sources = collect_sources(
        args.root_dir,
        args.run_name_contains,
        args.include_noncanonical_archives,
    )
    if not sources:
        raise FileNotFoundError(f"No result CSV sources found under {args.root_dir}")

    fieldnames = determine_fieldnames(sources)
    write_aggregate(args.output_csv, sources, fieldnames)
    write_manifest(manifest_csv, sources)
    write_summary(summary_json, sources, args.output_csv)

    print(f"Wrote aggregate CSV: {args.output_csv}")
    print(f"Wrote manifest CSV: {manifest_csv}")
    print(f"Wrote summary JSON: {summary_json}")
    print(f"Included {len(sources)} source files across {len({s.run_name for s in sources})} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
