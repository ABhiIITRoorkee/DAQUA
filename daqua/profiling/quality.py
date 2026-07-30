from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from daqua.loaders.defect_loader import ProjectData


logger = logging.getLogger(__name__)


QUALITY_COLUMNS = [
    "dataset",
    "project",
    "path",
    "raw_rows",
    "raw_columns",
    "clean_rows",
    "clean_columns",
    "n_features",
    "row_retention_rate",
    "missing_value_rate",
    "duplicate_rate",
    "conflicting_duplicate_rate",
    "constant_feature_rate",
    "near_constant_feature_rate",
    "high_correlation_feature_rate",
    "outlier_instance_rate",
    "outlier_cell_rate",
    "class_0_count",
    "class_1_count",
    "minority_class_count",
    "majority_class_count",
    "minority_class_percentage",
    "class_imbalance_ratio",
    "label_entropy",
    "quality_score",
    "quality_level",
    "warnings",
]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0

    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def feature_columns(df: pd.DataFrame, label_col: str = "label") -> List[str]:
    return [col for col in df.columns if col != label_col]


def compute_missing_value_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0

    return float(df.isna().mean().mean())


def compute_duplicate_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0

    return float(df.duplicated().mean())


def compute_conflicting_duplicate_rate(
    df: pd.DataFrame,
    label_col: str = "label",
) -> float:
    if df.empty or label_col not in df.columns:
        return 0.0

    features = feature_columns(df, label_col)

    if not features:
        return 0.0

    grouped = df.groupby(features, dropna=False)[label_col].nunique()
    conflicting_keys = grouped[grouped > 1].index

    if len(conflicting_keys) == 0:
        return 0.0

    feature_index = df.set_index(features).index
    conflicting_rows = feature_index.isin(conflicting_keys)

    return float(np.mean(conflicting_rows))


def compute_constant_feature_rate(
    df: pd.DataFrame,
    label_col: str = "label",
) -> float:
    features = feature_columns(df, label_col)

    if not features:
        return 0.0

    constant_count = 0

    for col in features:
        if df[col].nunique(dropna=False) <= 1:
            constant_count += 1

    return safe_divide(constant_count, len(features))


def compute_near_constant_feature_rate(
    df: pd.DataFrame,
    label_col: str = "label",
    dominance_threshold: float = 0.95,
) -> float:
    features = feature_columns(df, label_col)

    if not features:
        return 0.0

    near_constant_count = 0

    for col in features:
        value_counts = df[col].value_counts(dropna=False, normalize=True)

        if not value_counts.empty and value_counts.iloc[0] >= dominance_threshold:
            near_constant_count += 1

    return safe_divide(near_constant_count, len(features))


def compute_high_correlation_feature_rate(
    df: pd.DataFrame,
    label_col: str = "label",
    threshold: float = 0.95,
) -> float:
    features = feature_columns(df, label_col)

    if len(features) < 2:
        return 0.0

    corr = df[features].corr(method="spearman").abs()
    upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    highly_correlated_features = set()

    for col in upper_triangle.columns:
        correlated = upper_triangle.index[upper_triangle[col] >= threshold].tolist()

        if correlated:
            highly_correlated_features.add(col)
            highly_correlated_features.update(correlated)

    return safe_divide(len(highly_correlated_features), len(features))


def compute_outlier_rates(
    df: pd.DataFrame,
    label_col: str = "label",
    z_threshold: float = 3.5,
) -> Tuple[float, float]:
    features = feature_columns(df, label_col)

    if not features or df.empty:
        return 0.0, 0.0

    x = df[features].astype(float)

    med = x.median(axis=0)
    mad = (x - med).abs().median(axis=0)

    usable = mad > 0

    if usable.sum() == 0:
        return 0.0, 0.0

    modified_z = pd.DataFrame(index=x.index)

    for col in x.columns[usable]:
        modified_z[col] = 0.6745 * (x[col] - med[col]).abs() / mad[col]

    outlier_mask = modified_z > z_threshold

    outlier_instance_rate = float(outlier_mask.any(axis=1).mean())
    outlier_cell_rate = float(outlier_mask.mean().mean())

    return outlier_instance_rate, outlier_cell_rate


def compute_class_distribution(
    df: pd.DataFrame,
    label_col: str = "label",
) -> Dict[str, float]:
    if df.empty or label_col not in df.columns:
        return {
            "class_0_count": 0,
            "class_1_count": 0,
            "minority_class_count": 0,
            "majority_class_count": 0,
            "minority_class_percentage": 0.0,
            "class_imbalance_ratio": np.nan,
            "label_entropy": 0.0,
        }

    counts = df[label_col].value_counts().to_dict()

    class_0 = int(counts.get(0, 0))
    class_1 = int(counts.get(1, 0))

    minority = min(class_0, class_1)
    majority = max(class_0, class_1)

    total = class_0 + class_1
    minority_percentage = safe_divide(minority, total)
    imbalance_ratio = safe_divide(majority, minority, default=np.inf)

    positive_ratio = safe_divide(class_1, total)
    entropy = binary_entropy(positive_ratio)

    return {
        "class_0_count": class_0,
        "class_1_count": class_1,
        "minority_class_count": minority,
        "majority_class_count": majority,
        "minority_class_percentage": minority_percentage,
        "class_imbalance_ratio": imbalance_ratio,
        "label_entropy": entropy,
    }


def penalty_from_rate(rate: float, tolerance: float, severe: float) -> float:
    if pd.isna(rate):
        return 1.0

    if rate <= tolerance:
        return 0.0

    if rate >= severe:
        return 1.0

    return safe_divide(rate - tolerance, severe - tolerance)


def imbalance_penalty(imbalance_ratio: float) -> float:
    if pd.isna(imbalance_ratio):
        return 1.0

    if np.isinf(imbalance_ratio):
        return 1.0

    if imbalance_ratio <= 3:
        return 0.0

    if imbalance_ratio >= 20:
        return 1.0

    return safe_divide(imbalance_ratio - 3, 17)


def retention_penalty(row_retention_rate: float) -> float:
    if row_retention_rate >= 0.95:
        return 0.0

    if row_retention_rate <= 0.50:
        return 1.0

    return safe_divide(0.95 - row_retention_rate, 0.45)


def compute_quality_score(metrics: Dict[str, float]) -> float:
    penalties = {
        "missing": penalty_from_rate(metrics["missing_value_rate"], 0.00, 0.20),
        "duplicates": penalty_from_rate(metrics["duplicate_rate"], 0.00, 0.30),
        "conflicts": penalty_from_rate(metrics["conflicting_duplicate_rate"], 0.00, 0.10),
        "constant": penalty_from_rate(metrics["constant_feature_rate"], 0.00, 0.20),
        "near_constant": penalty_from_rate(metrics["near_constant_feature_rate"], 0.05, 0.50),
        "correlation": penalty_from_rate(metrics["high_correlation_feature_rate"], 0.10, 0.70),
        "outlier_instances": penalty_from_rate(metrics["outlier_instance_rate"], 0.05, 0.50),
        "imbalance": imbalance_penalty(metrics["class_imbalance_ratio"]),
        "retention": retention_penalty(metrics["row_retention_rate"]),
    }

    weights = {
        "missing": 0.15,
        "duplicates": 0.10,
        "conflicts": 0.15,
        "constant": 0.08,
        "near_constant": 0.08,
        "correlation": 0.10,
        "outlier_instances": 0.10,
        "imbalance": 0.14,
        "retention": 0.10,
    }

    total_penalty = sum(weights[key] * penalties[key] for key in weights)
    score = 100.0 * (1.0 - total_penalty)

    return round(float(np.clip(score, 0.0, 100.0)), 2)


def quality_level(score: float) -> str:
    if score >= 85:
        return "high"
    if score >= 70:
        return "moderate"
    if score >= 50:
        return "limited"
    return "low"


def build_quality_warnings(metrics: Dict[str, float]) -> List[str]:
    warnings: List[str] = []

    if metrics["row_retention_rate"] < 0.80:
        warnings.append("low_row_retention")

    if metrics["missing_value_rate"] > 0.05:
        warnings.append("missing_values")

    if metrics["duplicate_rate"] > 0.05:
        warnings.append("duplicate_instances")

    if metrics["conflicting_duplicate_rate"] > 0.0:
        warnings.append("conflicting_duplicates")

    if metrics["constant_feature_rate"] > 0.0:
        warnings.append("constant_features")

    if metrics["near_constant_feature_rate"] > 0.20:
        warnings.append("near_constant_features")

    if metrics["high_correlation_feature_rate"] > 0.40:
        warnings.append("high_feature_redundancy")

    if metrics["outlier_instance_rate"] > 0.20:
        warnings.append("many_outlier_instances")

    if metrics["minority_class_percentage"] < 0.10:
        warnings.append("severe_class_imbalance")
    elif metrics["minority_class_percentage"] < 0.20:
        warnings.append("moderate_class_imbalance")

    return warnings


def profile_project_quality(project: ProjectData) -> Dict[str, object]:
    df = project.df.copy()

    raw_rows, raw_columns = project.raw_shape
    clean_rows, clean_columns = df.shape

    n_features = max(clean_columns - 1, 0)

    missing_value_rate = compute_missing_value_rate(df)
    duplicate_rate = compute_duplicate_rate(df)
    conflicting_duplicate_rate = compute_conflicting_duplicate_rate(df)
    constant_feature_rate = compute_constant_feature_rate(df)
    near_constant_feature_rate = compute_near_constant_feature_rate(df)
    high_correlation_feature_rate = compute_high_correlation_feature_rate(df)
    outlier_instance_rate, outlier_cell_rate = compute_outlier_rates(df)

    class_metrics = compute_class_distribution(df)

    metrics: Dict[str, object] = {
        "dataset": project.dataset,
        "project": project.project,
        "path": project.path,
        "raw_rows": int(raw_rows),
        "raw_columns": int(raw_columns),
        "clean_rows": int(clean_rows),
        "clean_columns": int(clean_columns),
        "n_features": int(n_features),
        "row_retention_rate": safe_divide(clean_rows, raw_rows),
        "missing_value_rate": missing_value_rate,
        "duplicate_rate": duplicate_rate,
        "conflicting_duplicate_rate": conflicting_duplicate_rate,
        "constant_feature_rate": constant_feature_rate,
        "near_constant_feature_rate": near_constant_feature_rate,
        "high_correlation_feature_rate": high_correlation_feature_rate,
        "outlier_instance_rate": outlier_instance_rate,
        "outlier_cell_rate": outlier_cell_rate,
        **class_metrics,
    }

    score = compute_quality_score(metrics)
    warnings = build_quality_warnings(metrics)

    metrics["quality_score"] = score
    metrics["quality_level"] = quality_level(score)
    metrics["warnings"] = ";".join(warnings) if warnings else "none"

    return metrics


def profile_quality(projects: Sequence[ProjectData]) -> pd.DataFrame:
    rows = []

    for project in projects:
        try:
            rows.append(profile_project_quality(project))
        except Exception as exc:
            logger.exception(
                "Quality profiling failed for %s/%s: %s",
                project.dataset,
                project.project,
                exc,
            )

    profile = pd.DataFrame(rows)

    if profile.empty:
        return pd.DataFrame(columns=QUALITY_COLUMNS)

    for column in QUALITY_COLUMNS:
        if column not in profile.columns:
            profile[column] = np.nan

    return profile[QUALITY_COLUMNS]


def summarize_quality_by_dataset(quality_profile: pd.DataFrame) -> pd.DataFrame:
    if quality_profile.empty:
        return pd.DataFrame()

    numeric_columns = [
        "raw_rows",
        "clean_rows",
        "n_features",
        "row_retention_rate",
        "missing_value_rate",
        "duplicate_rate",
        "conflicting_duplicate_rate",
        "constant_feature_rate",
        "near_constant_feature_rate",
        "high_correlation_feature_rate",
        "outlier_instance_rate",
        "outlier_cell_rate",
        "minority_class_percentage",
        "class_imbalance_ratio",
        "label_entropy",
        "quality_score",
    ]

    summary = (
        quality_profile
        .groupby("dataset", as_index=False)[numeric_columns]
        .mean(numeric_only=True)
    )

    summary["n_projects"] = quality_profile.groupby("dataset")["project"].count().values

    ordered_columns = ["dataset", "n_projects"] + numeric_columns
    return summary[ordered_columns]


if __name__ == "__main__":
    import argparse

    from daqua.loaders.defect_loader import load_all_projects

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="Data-set")
    parser.add_argument("--out", type=str, default="outputs/profiles/DAQUA_quality_profile.csv")
    parser.add_argument(
        "--summary-out",
        type=str,
        default="outputs/profiles/DAQUA_quality_summary_by_dataset.csv",
    )

    args = parser.parse_args()

    projects = load_all_projects(args.root)
    quality = profile_quality(projects)
    summary = summarize_quality_by_dataset(quality)

    quality.to_csv(args.out, index=False)
    summary.to_csv(args.summary_out, index=False)

    print(f"Saved project quality profile to: {args.out}")
    print(f"Saved dataset quality summary to: {args.summary_out}")
    print(summary.round(4).to_string(index=False))