"""Tests for coverage-constrained setting recommendations."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts import analyze_setting_recommendations as setting_rec


def _fake_model_entry() -> SimpleNamespace:
    return SimpleNamespace(
        xor_splits=1,
        xor_joins=1,
        and_splits=0,
        and_joins=0,
        tau_ratio=0.1,
        model_places=5,
        model_transitions=4,
        visible_alphabet={"A", "B", "C"},
        visible_transition_count_by_label={"A": 1, "B": 1, "C": 1},
        enabled_initial_count=2,
        enabled_final_count=1,
    )


def _make_method_row(
    *,
    experiment_id: str,
    dataset_name: str,
    model_id: str,
    model_name: str,
    model_path: str,
    trace_id: str,
    trace_hash: str,
    trace_length: int,
    method: str,
    algorithm: str,
    heuristic: str,
    expansions: float | None,
    time_seconds: float | None,
    status: str,
    cost: float | None,
    optimal_cost: float,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "dataset_name": dataset_name,
        "model_id": model_id,
        "model_name": model_name,
        "model_path": model_path,
        "model_source": "file",
        "trace_id": trace_id,
        "trace_hash": trace_hash,
        "trace_activities": "|".join(["A"] * trace_length),
        "trace_length": trace_length,
        "trace_unique_activities": 1,
        "trace_repetition_ratio": 1.0 - (1.0 / trace_length),
        "trace_unique_dfg_edges": 1 if trace_length > 1 else 0,
        "trace_self_loops": max(trace_length - 1, 0),
        "trace_variant_frequency": 1,
        "trace_impossible_activities": 0,
        "model_places": 5,
        "model_transitions": 4,
        "model_arcs": 8,
        "model_silent_transitions": 1,
        "model_visible_transitions": 3,
        "model_place_in_degree_avg": 1.0,
        "model_place_out_degree_avg": 1.0,
        "model_place_in_degree_max": 1,
        "model_place_out_degree_max": 1,
        "model_transition_in_degree_avg": 1.0,
        "model_transition_out_degree_avg": 1.0,
        "model_transition_in_degree_max": 1,
        "model_transition_out_degree_max": 1,
        "method": method,
        "algorithm": algorithm,
        "heuristic": heuristic,
        "expansions": expansions,
        "time_seconds": time_seconds,
        "status": status,
        "cost": cost,
        "optimal_cost": optimal_cost,
    }


def _write_pipeline_root(results_root: Path, model_path: Path) -> None:
    rows_by_dataset: dict[str, list[dict[str, object]]] = {"alpha": [], "beta": []}
    rows_old: list[dict[str, object]] = []

    instances = [
        ("alpha", "model-a", "1", "h1", 8),
        ("alpha", "model-a", "2", "h2", 9),
        ("alpha", "model-a", "3", "l1", 2),
        ("beta", "model-b", "1", "h3", 8),
        ("beta", "model-b", "2", "l2", 2),
        ("beta", "model-b", "3", "l3", 1),
    ]

    for dataset_name, model_id, trace_id, trace_hash, trace_length in instances:
        model_name = f"{dataset_name}_f0.9000_p0.8000_IMf_n0.10_model.pkl"
        exp = f"exp_{dataset_name}_{trace_id}_20260312_120000"
        if trace_hash.startswith("h"):
            rows_by_dataset[dataset_name].extend(
                [
                    _make_method_row(
                        experiment_id=exp,
                        dataset_name=dataset_name,
                        model_id=model_id,
                        model_name=model_name,
                        model_path=str(model_path),
                        trace_id=trace_id,
                        trace_hash=trace_hash,
                        trace_length=trace_length,
                        method="forward_zero",
                        algorithm="forward",
                        heuristic="zero",
                        expansions=10,
                        time_seconds=1.0,
                        status="ok",
                        cost=1.0,
                        optimal_cost=1.0,
                    ),
                    _make_method_row(
                        experiment_id=exp,
                        dataset_name=dataset_name,
                        model_id=model_id,
                        model_name=model_name,
                        model_path=str(model_path),
                        trace_id=trace_id,
                        trace_hash=trace_hash,
                        trace_length=trace_length,
                        method="dibbs_zero",
                        algorithm="dibbs",
                        heuristic="zero",
                        expansions=1 if trace_hash != "h3" else None,
                        time_seconds=0.2 if trace_hash != "h3" else None,
                        status="ok" if trace_hash != "h3" else "timeout",
                        cost=1.0 if trace_hash != "h3" else None,
                        optimal_cost=1.0,
                    ),
                    _make_method_row(
                        experiment_id=exp,
                        dataset_name=dataset_name,
                        model_id=model_id,
                        model_name=model_name,
                        model_path=str(model_path),
                        trace_id=trace_id,
                        trace_hash=trace_hash,
                        trace_length=trace_length,
                        method="backward_me",
                        algorithm="backward",
                        heuristic="me",
                        expansions=12,
                        time_seconds=0.9,
                        status="ok",
                        cost=1.0,
                        optimal_cost=1.0,
                    ),
                ]
            )
        else:
            rows_by_dataset[dataset_name].extend(
                [
                    _make_method_row(
                        experiment_id=exp,
                        dataset_name=dataset_name,
                        model_id=model_id,
                        model_name=model_name,
                        model_path=str(model_path),
                        trace_id=trace_id,
                        trace_hash=trace_hash,
                        trace_length=trace_length,
                        method="forward_zero",
                        algorithm="forward",
                        heuristic="zero",
                        expansions=2,
                        time_seconds=0.2,
                        status="ok",
                        cost=1.0,
                        optimal_cost=1.0,
                    ),
                    _make_method_row(
                        experiment_id=exp,
                        dataset_name=dataset_name,
                        model_id=model_id,
                        model_name=model_name,
                        model_path=str(model_path),
                        trace_id=trace_id,
                        trace_hash=trace_hash,
                        trace_length=trace_length,
                        method="dibbs_zero",
                        algorithm="dibbs",
                        heuristic="zero",
                        expansions=6,
                        time_seconds=0.3,
                        status="ok",
                        cost=1.0,
                        optimal_cost=1.0,
                    ),
                    _make_method_row(
                        experiment_id=exp,
                        dataset_name=dataset_name,
                        model_id=model_id,
                        model_name=model_name,
                        model_path=str(model_path),
                        trace_id=trace_id,
                        trace_hash=trace_hash,
                        trace_length=trace_length,
                        method="backward_me",
                        algorithm="backward",
                        heuristic="me",
                        expansions=0.5,
                        time_seconds=0.1,
                        status="ok",
                        cost=2.0,
                        optimal_cost=1.0,
                    ),
                ]
            )

    rows_old.append(
        _make_method_row(
            experiment_id="exp_alpha_old_20260311_110000",
            dataset_name="alpha",
            model_id="model-a",
            model_name="alpha_f0.9000_p0.8000_IMf_n0.10_model.pkl",
            model_path=str(model_path),
            trace_id="3",
            trace_hash="l1",
            trace_length=2,
            method="forward_zero",
            algorithm="forward",
            heuristic="zero",
            expansions=99,
            time_seconds=9.9,
            status="ok",
            cost=1.0,
            optimal_cost=1.0,
        )
    )

    for dataset_name, dataset_rows in rows_by_dataset.items():
        merged_parent = results_root / dataset_name / "parent_new"
        merged_parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dataset_rows).to_csv(merged_parent / "merged_results.csv", index=False)

    old_parent = results_root / "alpha" / "parent_old"
    old_parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_old).to_csv(old_parent / "merged_results.csv", index=False)

    partial_parent = results_root / "beta" / "partial_parent" / "shard_00" / "run_1"
    partial_parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            _make_method_row(
                experiment_id="exp_partial_20260312_130000",
                dataset_name="beta",
                model_id="model-b",
                model_name="beta_f0.9000_p0.8000_IMf_n0.10_model.pkl",
                model_path=str(model_path),
                trace_id="999",
                trace_hash="partial",
                trace_length=5,
                method="forward_zero",
                algorithm="forward",
                heuristic="zero",
                expansions=1,
                time_seconds=0.1,
                status="ok",
                cost=1.0,
                optimal_cost=1.0,
            )
        ]
    ).to_csv(partial_parent / "results.csv", index=False)


def test_setting_recommendations_pipeline_outputs(tmp_path: Path, monkeypatch):
    results_root = tmp_path / "results_20260310"
    out_dir = tmp_path / "out"
    oracle_csv = tmp_path / "oracle.csv"
    model_path = tmp_path / "selected_models_20260310" / "demo" / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"placeholder")
    _write_pipeline_root(results_root, model_path)
    pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "model_id": model_id,
                "trace_id": trace_id,
                "trace_hash": trace_hash,
                "normalized_h_f": trace_length / 10.0,
                "h_f": float(trace_length),
                "oracle_extraction_time_ms": 12.5,
            }
            for dataset_name, model_id, trace_id, trace_hash, trace_length in [
                ("alpha", "model-a", "1", "h1", 8),
                ("alpha", "model-a", "2", "h2", 9),
                ("alpha", "model-a", "3", "l1", 2),
                ("beta", "model-b", "1", "h3", 8),
                ("beta", "model-b", "2", "l2", 2),
                ("beta", "model-b", "3", "l3", 1),
            ]
        ]
    ).to_csv(oracle_csv, index=False)

    monkeypatch.setattr(setting_rec, "build_model_cache_entry", lambda _: _fake_model_entry())
    monkeypatch.setattr(setting_rec, "resolve_model_path", lambda model_path, _: model_path)

    rc = setting_rec.main(
        [
            "--results-root",
            str(results_root),
            "--out-dir",
            str(out_dir),
            "--min-samples-leaf",
            "1",
            "--oracle-features-csv",
            str(oracle_csv),
        ]
    )

    assert rc == 0

    summary = json.loads((out_dir / "setting_analysis_summary.json").read_text())
    assert summary["rows_aggregated"] == 20
    assert summary["rows_after_complete_parent_filter"] == 19
    assert summary["rows_after_deduplication"] == 18
    assert summary["rows_after_consensus_optimal_filter"] == 14
    assert summary["instances_modeled"] == 6
    assert summary["primary_optimization_metric"] == "time_seconds"
    assert summary["tree_guardrail_triggered"] is False
    assert "optimal_cost" not in summary["predictor_columns"]
    assert "sp_nodes" not in summary["predictor_columns"]
    assert "dataset_name" not in summary["predictor_columns"]
    assert summary["oracle_features_available"] == ["normalized_h_f", "h_f"]
    assert summary["oracle_ablation"]["predictor_columns"].count("normalized_h_f") == 1

    modeling = pd.read_csv(out_dir / "corrected_instance_modeling_table.csv")
    assert "optimal_cost" not in modeling.columns
    assert "sp_nodes" not in modeling.columns
    assert "normalized_h_f" not in modeling.columns
    assert "h_f" not in modeling.columns
    assert setting_rec.LEAF_ID_COL not in modeling.columns
    assert "dataset_name" in modeling.columns
    assert "model_name" in modeling.columns
    assert "model_path" in modeling.columns
    assert "sp_places" in modeling.columns

    targets = pd.read_csv(out_dir / "corrected_instance_targets.csv")
    assert targets.loc[targets["trace_hash"] == "l1", "best_method"].iloc[0] == "forward_zero"

    leaf_summary = pd.read_csv(out_dir / "setting_leaf_summary.csv")
    assignments = pd.read_csv(out_dir / "setting_leaf_assignments.csv")
    h3_leaf_id = assignments.loc[assignments["trace_hash"] == "h3", setting_rec.LEAF_ID_COL].iloc[0]
    h3_leaf = leaf_summary[leaf_summary["leaf_id"] == h3_leaf_id].iloc[0]
    assert h3_leaf["recommended_method"] == "backward_me"
    assert h3_leaf["recommended_method_valid_coverage"] == 1.0
    assert int(leaf_summary["leaf_count_total"].max()) >= 2

    validation = pd.read_csv(out_dir / "setting_model_validation.csv")
    assert not validation.empty
    assert {"recommended_method_valid_coverage", "recommended_method_mean_oracle_regret"}.issubset(validation.columns)
    permutation = pd.read_csv(out_dir / "setting_permutation_importance.csv")
    assert {"feature", "mean_regret_increase", "mean_valid_coverage_drop"}.issubset(permutation.columns)
    oracle_validation = pd.read_csv(out_dir / "setting_oracle_ablation_validation.csv")
    assert not oracle_validation.empty
    oracle_importance = pd.read_csv(out_dir / "setting_oracle_ablation_tree_feature_importance.csv")
    assert "normalized_h_f" in oracle_importance["feature"].tolist()
    assert "oracle_extraction_time_ms" not in oracle_importance["feature"].tolist()

    appendix = pd.read_csv(out_dir / "appendix_feature_bucket_recommendations.csv")
    assert "recommended_method" in appendix.columns


def test_coverage_tolerance_boundary_allows_exact_difference():
    partitions = pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": str(i),
                "trace_hash": f"h{i}",
                setting_rec.LEAF_ID_COL: 1,
            }
            for i in range(100)
        ]
    )
    rows = []
    for i in range(100):
        base = {
            "dataset_name": "demo",
            "model_id": "m",
            "model_name": "m.pkl",
            "model_path": "/tmp/m.pkl",
            "trace_id": str(i),
            "trace_hash": f"h{i}",
            "oracle_expansions": 1.0,
        }
        rows.append({**base, "label": "forward_zero", "valid": True, "expansions": 10.0, "time_seconds": 1.0, "winner_flag": False})
        rows.append(
            {
                **base,
                "label": "dibbs_zero",
                "valid": i < 99,
                "expansions": 1.0 if i < 99 else np.nan,
                "time_seconds": 0.1 if i < 99 else np.nan,
                "winner_flag": i < 99,
            }
        )
    observations = pd.DataFrame(rows)

    _, recs = setting_rec.summarize_partition_recommendations(
        partitions,
        observations,
        [setting_rec.LEAF_ID_COL],
        0.01,
    )
    assert recs.loc[0, "recommended_label"] == "dibbs_zero"


def test_lower_coverage_candidate_is_excluded_outside_tolerance():
    partitions = pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "model_id": "m",
                "trace_id": str(i),
                "trace_hash": f"h{i}",
                setting_rec.LEAF_ID_COL: 1,
            }
            for i in range(10)
        ]
    )
    rows = []
    for i in range(10):
        base = {
            "dataset_name": "demo",
            "model_id": "m",
            "trace_id": str(i),
            "trace_hash": f"h{i}",
            "oracle_expansions": 1.0,
        }
        rows.append(
            {
                **base,
                "label": "forward_zero",
                "valid": True,
                "expansions": 5.0,
                "time_seconds": 0.5,
                "winner_flag": False,
            }
        )
        rows.append(
            {
                **base,
                "label": "dibbs_zero",
                "valid": i < 5,
                "expansions": 1.0 if i < 5 else np.nan,
                "time_seconds": 0.1 if i < 5 else np.nan,
                "winner_flag": i < 5,
            }
        )

    observations = pd.DataFrame(rows)
    _, recs = setting_rec.summarize_partition_recommendations(
        partitions,
        observations,
        [setting_rec.LEAF_ID_COL],
        0.01,
    )
    assert recs.loc[0, "recommended_label"] == "forward_zero"


def test_tie_break_prefers_lower_time_then_lexicographic_label():
    valid_df = pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "model_id": "m",
                "trace_id": "1",
                "trace_hash": "h1",
                "method": "forward_zero",
                "algorithm": "forward",
                "heuristic": "zero",
                "expansions": 5,
                "time_seconds": 0.5,
            },
            {
                "dataset_name": "demo",
                "model_id": "m",
                "trace_id": "1",
                "trace_hash": "h1",
                "method": "dibbs_zero",
                "algorithm": "dibbs",
                "heuristic": "zero",
                "expansions": 5,
                "time_seconds": 0.2,
            },
            {
                "dataset_name": "demo",
                "model_id": "m",
                "trace_id": "2",
                "trace_hash": "h2",
                "method": "forward_zero",
                "algorithm": "forward",
                "heuristic": "zero",
                "expansions": 4,
                "time_seconds": 0.2,
            },
            {
                "dataset_name": "demo",
                "model_id": "m",
                "trace_id": "2",
                "trace_hash": "h2",
                "method": "backward_zero",
                "algorithm": "backward",
                "heuristic": "zero",
                "expansions": 4,
                "time_seconds": 0.2,
            },
        ]
    )

    targets, _ = setting_rec.build_target_tables(valid_df)
    assert targets.loc[targets["trace_hash"] == "h1", "best_method"].iloc[0] == "dibbs_zero"
    assert targets.loc[targets["trace_hash"] == "h2", "best_method"].iloc[0] == "backward_zero"


def test_instance_base_uses_logical_instance_key_when_path_drifts():
    complete_df = pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "model_id": "m",
                "model_name": "old.pkl",
                "model_path": "/tmp/old.pkl",
                "trace_id": "1",
                "trace_hash": "h1",
                "trace_activities": "A|B",
                "trace_length": 2,
                "trace_unique_activities": 2,
                "trace_repetition_ratio": 0.0,
                "trace_unique_dfg_edges": 1,
                "trace_self_loops": 0,
                "trace_variant_frequency": 1,
                "trace_impossible_activities": 0,
            },
            {
                "dataset_name": "demo",
                "model_id": "m",
                "model_name": "new.pkl",
                "model_path": "/tmp/new.pkl",
                "trace_id": "1",
                "trace_hash": "h1",
                "trace_activities": "A|B",
                "trace_length": 2,
                "trace_unique_activities": 2,
                "trace_repetition_ratio": 0.0,
                "trace_unique_dfg_edges": 1,
                "trace_self_loops": 0,
                "trace_variant_frequency": 1,
                "trace_impossible_activities": 0,
            },
        ]
    )
    valid_df = complete_df.copy()

    instance_base = setting_rec.build_instance_base(complete_df, valid_df)
    assert len(instance_base) == 1
    assert instance_base.loc[0, "model_name"] == "new.pkl"
    assert instance_base.loc[0, "model_path"] == "/tmp/new.pkl"


def test_grouped_validation_uses_train_only_leaf_recommendations():
    modeling_df = pd.DataFrame(
        [
            {
                "dataset_name": "alpha",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "1",
                "trace_hash": "a1",
                "x": 0.0,
                "best_method": "forward_zero",
            },
            {
                "dataset_name": "alpha",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "2",
                "trace_hash": "a2",
                "x": 0.0,
                "best_method": "forward_zero",
            },
            {
                "dataset_name": "beta",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "3",
                "trace_hash": "b1",
                "x": 0.0,
                "best_method": "dibbs_zero",
            },
            {
                "dataset_name": "beta",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "4",
                "trace_hash": "b2",
                "x": 0.0,
                "best_method": "dibbs_zero",
            },
        ]
    )
    observations = pd.DataFrame(
        [
            {
                "dataset_name": "alpha",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "1",
                "trace_hash": "a1",
                "label": "forward_zero",
                "valid": True,
                "expansions": 1.0,
                "time_seconds": 0.1,
                "oracle_expansions": 1.0,
                "winner_flag": True,
            },
            {
                "dataset_name": "alpha",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "1",
                "trace_hash": "a1",
                "label": "dibbs_zero",
                "valid": False,
                "expansions": None,
                "time_seconds": None,
                "oracle_expansions": 1.0,
                "winner_flag": False,
            },
            {
                "dataset_name": "alpha",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "2",
                "trace_hash": "a2",
                "label": "forward_zero",
                "valid": True,
                "expansions": 1.0,
                "time_seconds": 0.1,
                "oracle_expansions": 1.0,
                "winner_flag": True,
            },
            {
                "dataset_name": "alpha",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "2",
                "trace_hash": "a2",
                "label": "dibbs_zero",
                "valid": False,
                "expansions": None,
                "time_seconds": None,
                "oracle_expansions": 1.0,
                "winner_flag": False,
            },
            {
                "dataset_name": "beta",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "3",
                "trace_hash": "b1",
                "label": "forward_zero",
                "valid": False,
                "expansions": None,
                "time_seconds": None,
                "oracle_expansions": 1.0,
                "winner_flag": False,
            },
            {
                "dataset_name": "beta",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "3",
                "trace_hash": "b1",
                "label": "dibbs_zero",
                "valid": True,
                "expansions": 1.0,
                "time_seconds": 0.1,
                "oracle_expansions": 1.0,
                "winner_flag": True,
            },
            {
                "dataset_name": "beta",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "4",
                "trace_hash": "b2",
                "label": "forward_zero",
                "valid": False,
                "expansions": None,
                "time_seconds": None,
                "oracle_expansions": 1.0,
                "winner_flag": False,
            },
            {
                "dataset_name": "beta",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "4",
                "trace_hash": "b2",
                "label": "dibbs_zero",
                "valid": True,
                "expansions": 1.0,
                "time_seconds": 0.1,
                "oracle_expansions": 1.0,
                "winner_flag": True,
            },
        ]
    )

    validation = setting_rec.run_grouped_validation(
        modeling_df=modeling_df,
        predictor_cols=["x"],
        method_observations=observations,
        coverage_tolerance=0.01,
        max_depth=1,
        min_samples_leaf=1,
        group_col="dataset_name",
    )

    assert len(validation) == 2
    assert validation["recommended_method_valid_coverage"].tolist() == [0.0, 0.0]


def test_tree_guardrail_reports_degenerate_tree(tmp_path: Path, monkeypatch):
    results_root = tmp_path / "results_20260310"
    out_dir = tmp_path / "out"
    model_path = tmp_path / "selected_models_20260310" / "demo" / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"x")

    rows = []
    for dataset_name in ["alpha", "beta"]:
        merged_parent = results_root / dataset_name / "parent"
        merged_parent.mkdir(parents=True, exist_ok=True)
        rows.clear()
        for trace_id, trace_hash in [("1", f"{dataset_name}-1"), ("2", f"{dataset_name}-2")]:
            rows.extend(
                [
                    _make_method_row(
                        experiment_id=f"exp_{dataset_name}_{trace_id}_20260312_120000",
                        dataset_name=dataset_name,
                        model_id="m",
                        model_name="demo_f0.9000_p0.8000_IMf_n0.10_model.pkl",
                        model_path=str(model_path),
                        trace_id=trace_id,
                        trace_hash=trace_hash,
                        trace_length=5,
                        method="forward_zero",
                        algorithm="forward",
                        heuristic="zero",
                        expansions=1,
                        time_seconds=0.1,
                        status="ok",
                        cost=1.0,
                        optimal_cost=1.0,
                    ),
                    _make_method_row(
                        experiment_id=f"exp_{dataset_name}_{trace_id}_20260312_120000",
                        dataset_name=dataset_name,
                        model_id="m",
                        model_name="demo_f0.9000_p0.8000_IMf_n0.10_model.pkl",
                        model_path=str(model_path),
                        trace_id=trace_id,
                        trace_hash=trace_hash,
                        trace_length=5,
                        method="dibbs_zero",
                        algorithm="dibbs",
                        heuristic="zero",
                        expansions=2,
                        time_seconds=0.2,
                        status="ok",
                        cost=1.0,
                        optimal_cost=1.0,
                    ),
                ]
            )
        pd.DataFrame(rows).to_csv(merged_parent / "merged_results.csv", index=False)

    monkeypatch.setattr(setting_rec, "build_model_cache_entry", lambda _: _fake_model_entry())
    monkeypatch.setattr(setting_rec, "resolve_model_path", lambda model_path, _: model_path)

    rc = setting_rec.main(
        [
            "--results-root",
            str(results_root),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    summary = json.loads((out_dir / "setting_analysis_summary.json").read_text())
    assert summary["tree_guardrail_triggered"] is True
    assert summary["leaf_count_total"] == 1
    assert summary["realized_min_samples_leaf"] == 10


def test_compute_predictor_columns_excludes_oracle_by_default_and_allows_selected():
    modeling_df = pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "model_id": "m",
                "trace_id": "1",
                "trace_hash": "h1",
                "trace_length": 5,
                "tau_ratio": 0.2,
                "normalized_h_f": 0.8,
                "h_f": 4.0,
                "oracle_extraction_time_ms": 12.0,
                "best_method": "forward_zero",
                "best_algorithm": "forward",
                "best_heuristic": "zero",
                "best_method_expansions": 1.0,
                "best_algorithm_expansions": 1.0,
                "best_heuristic_expansions": 1.0,
            }
        ]
    )

    default_predictors = setting_rec.compute_predictor_columns(modeling_df)
    assert "trace_length" in default_predictors
    assert "tau_ratio" in default_predictors
    assert "normalized_h_f" not in default_predictors
    assert "h_f" not in default_predictors
    assert "oracle_extraction_time_ms" not in default_predictors

    oracle_predictors = setting_rec.compute_predictor_columns(
        modeling_df,
        include_columns=["normalized_h_f"],
    )
    assert "normalized_h_f" in oracle_predictors
    assert "h_f" not in oracle_predictors
    assert "oracle_extraction_time_ms" not in oracle_predictors

    stripped_predictors = setting_rec.compute_predictor_columns(
        modeling_df.assign(
            model_fitness=0.95,
            model_precision=0.81,
            miner_parameter_value=0.6,
        ),
        exclude_columns=["model_fitness", "model_precision", "miner_parameter_value"],
    )
    assert "trace_length" in stripped_predictors
    assert "tau_ratio" in stripped_predictors
    assert "model_fitness" not in stripped_predictors
    assert "model_precision" not in stripped_predictors
    assert "miner_parameter_value" not in stripped_predictors


def test_grouped_permutation_importance_returns_feature_deltas():
    modeling_df = pd.DataFrame(
        [
            {
                "dataset_name": "alpha",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "1",
                "trace_hash": "a1",
                "signal": 0.0,
                "best_method": "forward_zero",
            },
            {
                "dataset_name": "alpha",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "2",
                "trace_hash": "a2",
                "signal": 0.0,
                "best_method": "forward_zero",
            },
            {
                "dataset_name": "beta",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "3",
                "trace_hash": "b1",
                "signal": 1.0,
                "best_method": "dibbs_zero",
            },
            {
                "dataset_name": "beta",
                "model_id": "m",
                "model_name": "m.pkl",
                "model_path": "/tmp/m.pkl",
                "trace_id": "4",
                "trace_hash": "b2",
                "signal": 1.0,
                "best_method": "dibbs_zero",
            },
        ]
    )
    observations = pd.DataFrame(
        [
            {"dataset_name": "alpha", "model_id": "m", "trace_id": "1", "trace_hash": "a1", "label": "forward_zero", "valid": True, "expansions": 1.0, "time_seconds": 0.1, "oracle_expansions": 1.0, "winner_flag": True},
            {"dataset_name": "alpha", "model_id": "m", "trace_id": "1", "trace_hash": "a1", "label": "dibbs_zero", "valid": False, "expansions": np.nan, "time_seconds": np.nan, "oracle_expansions": 1.0, "winner_flag": False},
            {"dataset_name": "alpha", "model_id": "m", "trace_id": "2", "trace_hash": "a2", "label": "forward_zero", "valid": True, "expansions": 1.0, "time_seconds": 0.1, "oracle_expansions": 1.0, "winner_flag": True},
            {"dataset_name": "alpha", "model_id": "m", "trace_id": "2", "trace_hash": "a2", "label": "dibbs_zero", "valid": False, "expansions": np.nan, "time_seconds": np.nan, "oracle_expansions": 1.0, "winner_flag": False},
            {"dataset_name": "beta", "model_id": "m", "trace_id": "3", "trace_hash": "b1", "label": "forward_zero", "valid": False, "expansions": np.nan, "time_seconds": np.nan, "oracle_expansions": 1.0, "winner_flag": False},
            {"dataset_name": "beta", "model_id": "m", "trace_id": "3", "trace_hash": "b1", "label": "dibbs_zero", "valid": True, "expansions": 1.0, "time_seconds": 0.1, "oracle_expansions": 1.0, "winner_flag": True},
            {"dataset_name": "beta", "model_id": "m", "trace_id": "4", "trace_hash": "b2", "label": "forward_zero", "valid": False, "expansions": np.nan, "time_seconds": np.nan, "oracle_expansions": 1.0, "winner_flag": False},
            {"dataset_name": "beta", "model_id": "m", "trace_id": "4", "trace_hash": "b2", "label": "dibbs_zero", "valid": True, "expansions": 1.0, "time_seconds": 0.1, "oracle_expansions": 1.0, "winner_flag": True},
        ]
    )

    fold_artifacts = setting_rec.build_grouped_validation_artifacts(
        modeling_df=modeling_df,
        predictor_cols=["signal"],
        method_observations=observations,
        coverage_tolerance=0.01,
        max_depth=1,
        min_samples_leaf=1,
        group_col="dataset_name",
    )
    detail_df, summary_df = setting_rec.run_grouped_permutation_importance(
        fold_artifacts=fold_artifacts,
        predictor_cols=["signal"],
        method_observations=observations,
        n_repeats=1,
    )

    assert not detail_df.empty
    assert summary_df["feature"].tolist() == ["signal"]
    assert {
        "mean_regret_increase",
        "mean_valid_coverage_drop",
        "mean_winner_share_drop",
    }.issubset(summary_df.columns)


def test_oracle_ablation_outputs_and_predictors(tmp_path: Path, monkeypatch):
    results_root = tmp_path / "results_20260310"
    out_dir = tmp_path / "out"
    oracle_csv = tmp_path / "oracle.csv"
    model_path = tmp_path / "selected_models_20260310" / "demo" / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"placeholder")
    _write_pipeline_root(results_root, model_path)

    oracle_rows = [
        {
            "dataset_name": dataset_name,
            "model_id": model_id,
            "trace_id": trace_id,
            "trace_hash": trace_hash,
            "normalized_h_f": float(trace_length) / 10.0,
            "h_f": float(trace_length),
            "oracle_extraction_time_ms": 3.0,
        }
        for dataset_name, model_id, trace_id, trace_hash, trace_length in [
            ("alpha", "model-a", "1", "h1", 8),
            ("alpha", "model-a", "2", "h2", 9),
            ("alpha", "model-a", "3", "l1", 2),
            ("beta", "model-b", "1", "h3", 8),
            ("beta", "model-b", "2", "l2", 2),
            ("beta", "model-b", "3", "l3", 1),
        ]
    ]
    pd.DataFrame(oracle_rows).to_csv(oracle_csv, index=False)

    monkeypatch.setattr(setting_rec, "build_model_cache_entry", lambda _: _fake_model_entry())
    monkeypatch.setattr(setting_rec, "resolve_model_path", lambda model_path, _: model_path)

    rc = setting_rec.main(
        [
            "--results-root",
            str(results_root),
            "--out-dir",
            str(out_dir),
            "--min-samples-leaf",
            "1",
            "--oracle-features-csv",
            str(oracle_csv),
            "--oracle-feature-cols",
            "normalized_h_f",
            "h_f",
            "--permutation-repeats",
            "1",
        ]
    )

    assert rc == 0
    oracle_summary = json.loads((out_dir / "setting_oracle_ablation_summary.json").read_text())
    assert "normalized_h_f" in oracle_summary["predictor_columns"]
    assert "h_f" in oracle_summary["predictor_columns"]
    assert "oracle_extraction_time_ms" not in oracle_summary["predictor_columns"]
    assert (out_dir / "setting_oracle_ablation_validation.csv").exists()
    assert (out_dir / "setting_oracle_ablation_permutation_importance.csv").exists()
    assert (out_dir / "setting_permutation_importance.csv").exists()


def test_oracle_ablation_drops_rows_with_missing_oracle_features(tmp_path: Path, monkeypatch):
    results_root = tmp_path / "results_20260310"
    out_dir = tmp_path / "out"
    oracle_csv = tmp_path / "oracle.csv"
    model_path = tmp_path / "selected_models_20260310" / "demo" / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"placeholder")
    _write_pipeline_root(results_root, model_path)

    oracle_rows = [
        {
            "dataset_name": dataset_name,
            "model_id": model_id,
            "trace_id": trace_id,
            "trace_hash": trace_hash,
            "normalized_h_f": float(trace_length) / 10.0,
            "h_f": float(trace_length),
        }
        for dataset_name, model_id, trace_id, trace_hash, trace_length in [
            ("alpha", "model-a", "1", "h1", 8),
            ("alpha", "model-a", "2", "h2", 9),
            ("alpha", "model-a", "3", "l1", 2),
            ("beta", "model-b", "1", "h3", 8),
            ("beta", "model-b", "2", "l2", 2),
        ]
    ]
    oracle_rows[1]["normalized_h_f"] = np.nan
    pd.DataFrame(oracle_rows).to_csv(oracle_csv, index=False)

    monkeypatch.setattr(setting_rec, "build_model_cache_entry", lambda _: _fake_model_entry())
    monkeypatch.setattr(setting_rec, "resolve_model_path", lambda model_path, _: model_path)

    rc = setting_rec.main(
        [
            "--results-root",
            str(results_root),
            "--out-dir",
            str(out_dir),
            "--min-samples-leaf",
            "1",
            "--oracle-features-csv",
            str(oracle_csv),
            "--oracle-feature-cols",
            "normalized_h_f",
            "h_f",
            "--permutation-repeats",
            "1",
        ]
    )

    assert rc == 0
    oracle_summary = json.loads((out_dir / "setting_oracle_ablation_summary.json").read_text())
    assert oracle_summary["oracle_rows_before_complete_filter"] == 6
    assert oracle_summary["oracle_rows_after_complete_filter"] == 4
    assert oracle_summary["oracle_rows_dropped_for_missing_features"] == 2
    assert oracle_summary["instances_modeled"] == 4


def test_load_oracle_features_rejects_duplicate_keys(tmp_path: Path):
    oracle_csv = tmp_path / "oracle.csv"
    pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "model_id": "m",
                "trace_id": "1",
                "trace_hash": "h1",
                "normalized_h_f": 1.0,
                "h_f": 4.0,
            },
            {
                "dataset_name": "demo",
                "model_id": "m",
                "trace_id": "1",
                "trace_hash": "h1",
                "normalized_h_f": 9.0,
                "h_f": 4.0,
            },
        ]
    ).to_csv(oracle_csv, index=False)

    try:
        setting_rec.load_oracle_features(oracle_csv, ["normalized_h_f", "h_f"])
    except ValueError as exc:
        assert "duplicate instance keys" in str(exc)
    else:
        raise AssertionError("Expected duplicate oracle keys to raise ValueError")


def test_load_oracle_features_rejects_missing_requested_columns(tmp_path: Path):
    oracle_csv = tmp_path / "oracle.csv"
    pd.DataFrame(
        [
            {
                "dataset_name": "demo",
                "model_id": "m",
                "trace_id": "1",
                "trace_hash": "h1",
                "normalized_h_f": 1.0,
            }
        ]
    ).to_csv(oracle_csv, index=False)

    try:
        setting_rec.load_oracle_features(oracle_csv, ["normalized_h_f", "h_f"])
    except ValueError as exc:
        assert "missing requested oracle feature columns" in str(exc)
    else:
        raise AssertionError("Expected missing requested oracle columns to raise ValueError")
