#!/usr/bin/env python3
"""Run the Oracle Stage C pipeline with fast + LP features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


RANDOM_STATE = 42
KEY_COLS = ["dataset_name", "model_id", "trace_id", "trace_hash"]
FAST_FEATURES = [
    "xor_splits",
    "xor_joins",
    "and_splits",
    "and_joins",
    "tau_ratio",
    "trace_length",
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
ORACLE_FEATURES = [
    "h_f",
    "normalized_h_f",
    "h_b",
    "normalized_h_b",
]
REQUIRED_ORACLE_FEATURES = [
    "h_f",
    "normalized_h_f",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description="Run the Oracle Stage C pipeline")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("tmp_smoke/stage_b_features_current/selection_features_full.csv"),
    )
    parser.add_argument(
        "--oracle-csv",
        type=Path,
        default=Path("tmp_smoke/stage_b_features_current/selection_features_oracle.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp_smoke/stage_c_oracle"),
    )
    parser.add_argument(
        "--target-col",
        default="dibbs_vs_forward_expansion_ratio",
    )
    parser.add_argument(
        "--group-col",
        default="dataset_name",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--target-transform",
        choices=["raw", "log", "log1p", "clip"],
        default="raw",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
    )
    parser.add_argument(
        "--shap-background-max",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--shap-sample-max",
        type=int,
        default=1200,
    )
    return parser


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """Ensure a frame contains the expected columns."""

    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def infer_consistency_policy(target_col: str) -> str:
    """Infer the target-specific consistency policy."""

    if target_col.startswith("dibbs_zero_vs_forward_zero"):
        return "none"
    if target_col.startswith("dibbs_vs_forward"):
        return "dibbs_only"
    return "none"


def build_consistency_mask(df: pd.DataFrame, policy: str) -> pd.Series:
    """Build the consistency mask for the selected target policy."""

    if policy == "none":
        return pd.Series(True, index=df.index)
    if policy == "dibbs_only":
        validate_columns(df, ["uses_potentially_inconsistent_dibbs_target"])
        return (
            pd.to_numeric(
                df["uses_potentially_inconsistent_dibbs_target"],
                errors="coerce",
            )
            .fillna(0)
            .eq(0)
        )
    raise ValueError(f"Unsupported consistency policy: {policy}")


def build_target_transform(
    transform: str,
    clip_min: float,
    clip_max: float,
) -> tuple[
    Callable[[pd.Series], pd.Series],
    Callable[[np.ndarray], np.ndarray],
    Dict[str, float | str],
]:
    """Build forward and inverse transforms for the target."""

    if transform == "raw":
        return (
            lambda s: s.astype(float),
            lambda arr: np.asarray(arr, dtype=float),
            {"target_transform": "raw"},
        )

    if transform == "log":
        eps = 1e-12
        return (
            lambda s: np.log(np.clip(s.astype(float), eps, None)),
            lambda arr: np.exp(np.asarray(arr, dtype=float)),
            {"target_transform": "log", "log_epsilon": eps},
        )

    if transform == "log1p":
        return (
            lambda s: np.log1p(np.clip(s.astype(float), 0.0, None)),
            lambda arr: np.expm1(np.asarray(arr, dtype=float)),
            {"target_transform": "log1p"},
        )

    if transform == "clip":
        if clip_min <= 0 or clip_max <= 0 or clip_min >= clip_max:
            raise ValueError("clip_min and clip_max must satisfy 0 < clip_min < clip_max")
        return (
            lambda s: np.clip(s.astype(float), clip_min, clip_max),
            lambda arr: np.asarray(arr, dtype=float),
            {
                "target_transform": "clip",
                "clip_min": float(clip_min),
                "clip_max": float(clip_max),
            },
        )

    raise ValueError(f"Unsupported target transform: {transform}")


def prepare_merged_dataset(
    features_csv: Path,
    oracle_csv: Path,
    target_col: str,
    group_col: str,
    require_complete_oracle: bool = True,
) -> tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """Merge fast and oracle features and apply the Oracle target contract."""

    fast_df = pd.read_csv(features_csv)
    oracle_df = pd.read_csv(oracle_csv)

    validate_columns(fast_df, KEY_COLS + [target_col, group_col])
    validate_columns(oracle_df, KEY_COLS + REQUIRED_ORACLE_FEATURES)

    available_oracle_features = [column for column in ORACLE_FEATURES if column in oracle_df.columns]
    merge_cols = KEY_COLS + available_oracle_features
    merged = fast_df.merge(oracle_df.loc[:, merge_cols], on=KEY_COLS, how="inner")

    feature_cols = FAST_FEATURES + available_oracle_features
    validate_columns(merged, feature_cols)

    work = merged.dropna(subset=[target_col, group_col]).copy()
    rows_after_target_filter = int(len(work))
    consistency_policy = infer_consistency_policy(target_col)
    consistent_mask = build_consistency_mask(work, consistency_policy)
    work = work[consistent_mask].copy()
    rows_after_consistency_filter = int(len(work))
    rows_missing_any_oracle = int(work[available_oracle_features].isna().any(axis=1).sum())
    if require_complete_oracle:
        work = work.dropna(subset=available_oracle_features, how="any").copy()
    work = work.dropna(subset=feature_cols, how="all").reset_index(drop=True)
    return work, feature_cols, {
        "rows_after_target_filter": rows_after_target_filter,
        "rows_after_consistency_filter": rows_after_consistency_filter,
        "rows_missing_any_oracle": rows_missing_any_oracle,
        "consistency_policy": consistency_policy,
    }


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[SimpleImputer, HistGradientBoostingRegressor]:
    """Fit the oracle regressor."""

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_i, y_train)
    return imputer, model


def run_grouped_cv(
    X: pd.DataFrame,
    y_model: pd.Series,
    y_raw: pd.Series,
    groups: pd.Series,
    n_splits: int,
    inverse_transform: Callable[[np.ndarray], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run grouped cross-validation and collect metrics plus permutation importances."""

    if groups.nunique() < n_splits:
        raise ValueError(
            f"Need at least {n_splits} groups for GroupKFold, found {groups.nunique()}"
        )

    splitter = GroupKFold(n_splits=n_splits)
    fold_rows: List[Dict[str, float]] = []
    importance_rows: List[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y_model, groups),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_model.iloc[train_idx]
        y_test = y_model.iloc[test_idx]
        y_test_raw = y_raw.iloc[test_idx]

        imputer, model = fit_model(X_train, y_train)
        X_test_i = imputer.transform(X_test)
        predictions_model = model.predict(X_test_i)
        predictions_raw = inverse_transform(predictions_model)

        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_groups": int(groups.iloc[train_idx].nunique()),
                "test_groups": int(groups.iloc[test_idx].nunique()),
                "mae_transformed": float(mean_absolute_error(y_test, predictions_model)),
                "r2_transformed": float(r2_score(y_test, predictions_model)),
                "mae_raw": float(mean_absolute_error(y_test_raw, predictions_raw)),
                "r2_raw": float(r2_score(y_test_raw, predictions_raw)),
                "y_test_mean_transformed": float(y_test.mean()),
                "prediction_mean_transformed": float(np.mean(predictions_model)),
                "y_test_mean_raw": float(y_test_raw.mean()),
                "prediction_mean_raw": float(np.mean(predictions_raw)),
            }
        )

        perm = permutation_importance(
            model,
            X_test_i,
            y_test,
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="neg_mean_absolute_error",
        )
        importance_rows.append(
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance_mean": perm.importances_mean,
                    "importance_std": perm.importances_std,
                    "fold": fold,
                }
            )
        )

    importance_df = pd.concat(importance_rows, ignore_index=True)
    return pd.DataFrame(fold_rows), importance_df


def compute_shap_outputs(
    model: HistGradientBoostingRegressor,
    X_full: pd.DataFrame,
    out_dir: Path,
    background_max: int,
    sample_max: int,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Compute SHAP outputs for the fitted final model."""

    if shap is None:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"]), {
            "shap_available": False,
            "shap_rows_used": 0,
            "shap_background_rows": 0,
            "shap_explainer": None,
        }

    background_n = min(background_max, len(X_full))
    sample_n = min(sample_max, len(X_full))
    background = X_full.sample(background_n, random_state=RANDOM_STATE)
    sample = X_full.sample(sample_n, random_state=RANDOM_STATE)

    explainer_name = None
    shap_values = None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(sample)
        explainer_name = "TreeExplainer"
    except Exception:
        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer(sample)
        explainer_name = "Explainer"

    shap_df = (
        pd.DataFrame(
            {
                "feature": sample.columns.tolist(),
                "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    shap_df.to_csv(out_dir / "oracle_shap_global.csv", index=False)

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "oracle_shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_dir / "oracle_shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    return shap_df, {
        "shap_available": True,
        "shap_rows_used": int(sample_n),
        "shap_background_rows": int(background_n),
        "shap_explainer": explainer_name,
    }


def main(argv: List[str] | None = None) -> int:
    """Run the Oracle ML pipeline."""

    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_df, feature_cols, prep_stats = prepare_merged_dataset(
        features_csv=args.features_csv,
        oracle_csv=args.oracle_csv,
        target_col=args.target_col,
        group_col=args.group_col,
    )

    X = model_df.loc[:, feature_cols]
    y_raw = pd.to_numeric(model_df[args.target_col], errors="coerce")
    transform_target, inverse_transform, transform_meta = build_target_transform(
        transform=args.target_transform,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )
    y_model = pd.Series(transform_target(y_raw), index=y_raw.index, dtype=float)
    groups = model_df[args.group_col].astype(str)

    fold_df, fold_importance_df = run_grouped_cv(
        X=X,
        y_model=y_model,
        y_raw=y_raw,
        groups=groups,
        n_splits=args.n_splits,
        inverse_transform=inverse_transform,
    )

    cv_importance_df = (
        fold_importance_df.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
        )
        .sort_values("importance_mean", ascending=False, kind="stable")
        .reset_index(drop=True)
    )

    final_imputer, final_model = fit_model(X, y_model)
    X_full_i = pd.DataFrame(final_imputer.transform(X), columns=feature_cols)

    shap_df = pd.DataFrame(columns=["feature", "mean_abs_shap"])
    shap_meta: Dict[str, object] = {
        "shap_available": False,
        "shap_rows_used": 0,
        "shap_background_rows": 0,
        "shap_explainer": None,
    }
    if not args.skip_shap:
        shap_df, shap_meta = compute_shap_outputs(
            model=final_model,
            X_full=X_full_i,
            out_dir=args.out_dir,
            background_max=args.shap_background_max,
            sample_max=args.shap_sample_max,
        )

    fold_path = args.out_dir / "oracle_cv_metrics.csv"
    importance_path = args.out_dir / "oracle_cv_permutation_importance.csv"
    summary_path = args.out_dir / "oracle_summary.json"

    fold_df.to_csv(fold_path, index=False)
    cv_importance_df.to_csv(importance_path, index=False)

    summary = {
        "features_csv": str(args.features_csv.resolve()),
        "oracle_csv": str(args.oracle_csv.resolve()),
        "target_col": args.target_col,
        "group_col": args.group_col,
        "n_splits": int(args.n_splits),
        **transform_meta,
        "rows_used": int(len(model_df)),
        "rows_after_target_filter": int(prep_stats["rows_after_target_filter"]),
        "rows_after_consistency_filter": int(prep_stats["rows_after_consistency_filter"]),
        "rows_missing_any_oracle_before_drop": int(prep_stats["rows_missing_any_oracle"]),
        "groups_used": int(groups.nunique()),
        "consistency_policy": str(prep_stats["consistency_policy"]),
        "oracle_feature_set_name": "reduced_oracle_initial_me",
        "feature_cols": feature_cols,
        "target_mean_raw": float(y_raw.mean()),
        "target_median_raw": float(y_raw.median()),
        "target_mean_transformed": float(y_model.mean()),
        "target_median_transformed": float(y_model.median()),
        "target_bidirectional_win_rate": float((y_raw < 1.0).mean()),
        "cv_mae_transformed_mean": float(fold_df["mae_transformed"].mean()),
        "cv_mae_transformed_std": float(fold_df["mae_transformed"].std(ddof=0)),
        "cv_r2_transformed_mean": float(fold_df["r2_transformed"].mean()),
        "cv_r2_transformed_std": float(fold_df["r2_transformed"].std(ddof=0)),
        "cv_mae_raw_mean": float(fold_df["mae_raw"].mean()),
        "cv_mae_raw_std": float(fold_df["mae_raw"].std(ddof=0)),
        "cv_r2_raw_mean": float(fold_df["r2_raw"].mean()),
        "cv_r2_raw_std": float(fold_df["r2_raw"].std(ddof=0)),
        "top_cv_permutation_features": cv_importance_df.head(10).to_dict(orient="records"),
        **shap_meta,
        "top_shap_features": shap_df.head(10).to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Rows used: {len(model_df)}")
    print(f"Groups used: {groups.nunique()}")
    print(f"Target transform: {args.target_transform}")
    print(f"CV mean MAE (transformed): {fold_df['mae_transformed'].mean():.6f}")
    print(f"CV mean R2 (transformed): {fold_df['r2_transformed'].mean():.6f}")
    print(f"CV mean MAE (raw): {fold_df['mae_raw'].mean():.6f}")
    print(f"CV mean R2 (raw): {fold_df['r2_raw'].mean():.6f}")
    print(f"Wrote: {fold_path}")
    print(f"Wrote: {importance_path}")
    print(f"Wrote: {summary_path}")
    if not args.skip_shap and shap_meta["shap_available"]:
        print(f"Wrote: {args.out_dir / 'oracle_shap_global.csv'}")
        print(f"Wrote: {args.out_dir / 'oracle_shap_bar.png'}")
        print(f"Wrote: {args.out_dir / 'oracle_shap_beeswarm.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
