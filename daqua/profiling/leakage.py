from __future__ import annotations

import argparse
import itertools
import logging
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from daqua.loaders.defect_loader import ProjectData, load_all_projects


logger = logging.getLogger(__name__)


SUSPICIOUS_LABEL_TERMS = [
    "bug",
    "bugs",
    "defect",
    "defects",
    "defective",
    "fault",
    "faults",
    "failure",
    "failures",
    "fix",
    "fixed",
    "repair",
    "label",
    "class",
    "realbug",
    "bugcount",
    "defectcount",
]

TEMPORAL_TERMS = [
    "date",
    "time",
    "timestamp",
    "commitdate",
    "committime",
    "created",
    "resolved",
    "closed",
    "modified",
]

POST_RELEASE_TERMS = [
    "post",
    "after",
    "future",
    "resolved",
    "closed",
    "fix",
    "fixed",
    "bugfix",
]


PROJECT_LEAKAGE_COLUMNS = [
    "dataset",
    "project",
    "path",
    "n_instances",
    "n_features",
    "suspicious_feature_count",
    "suspicious_feature_rate",
    "suspicious_features",
    "temporal_feature_count",
    "temporal_features",
    "post_release_feature_count",
    "post_release_features",
    "perfect_label_correlation_count",
    "high_label_correlation_count",
    "high_label_correlation_features",
    "duplicate_instance_rate",
    "cross_project_duplicate_rate",
    "leakage_score",
    "leakage_level",
    "warnings",
]


DATASET_LEAKAGE_COLUMNS = [
    "dataset",
    "n_projects",
    "mean_suspicious_feature_rate",
    "mean_temporal_feature_count",
    "mean_post_release_feature_count",
    "mean_high_label_correlation_count",
    "mean_duplicate_instance_rate",
    "mean_cross_project_duplicate_rate",
    "mean_leakage_score",
    "leakage_level",
    "warnings",
]


def norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def feature_columns(df: pd.DataFrame, label_col: str = "label") -> List[str]:
    return [col for col in df.columns if col != label_col]


def contains_any_term(column: str, terms: Sequence[str]) -> bool:
    normalized = norm_text(column)

    return any(norm_text(term) in normalized for term in terms)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def duplicate_instance_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0

    return float(df.duplicated().mean())


def high_label_correlation_features(
    df: pd.DataFrame,
    label_col: str = "label",
    high_threshold: float = 0.95,
    perfect_threshold: float = 0.999999,
) -> Tuple[int, int, List[str]]:
    features = feature_columns(df, label_col)

    if not features or label_col not in df.columns:
        return 0, 0, []

    y = df[label_col].astype(float)

    high_features: List[str] = []
    high_count = 0
    perfect_count = 0

    for col in features:
        x = pd.to_numeric(df[col], errors="coerce")

        if x.nunique(dropna=True) <= 1:
            continue

        corr = x.corr(y, method="spearman")

        if pd.isna(corr):
            continue

        abs_corr = abs(float(corr))

        if abs_corr >= perfect_threshold:
            perfect_count += 1

        if abs_corr >= high_threshold:
            high_count += 1
            high_features.append(col)

    return perfect_count, high_count, high_features


def compute_cross_project_duplicate_rates(projects: Sequence[ProjectData]) -> Dict[Tuple[str, str], float]:
    rates: Dict[Tuple[str, str], float] = {}

    by_dataset: Dict[str, List[ProjectData]] = {}

    for project in projects:
        by_dataset.setdefault(project.dataset, []).append(project)

    for _, dataset_projects in by_dataset.items():
        for project_a, project_b in itertools.combinations(dataset_projects, 2):
            features_a = set(feature_columns(project_a.df))
            features_b = set(feature_columns(project_b.df))
            common_features = sorted(features_a.intersection(features_b))

            if not common_features:
                rates[(project_a.dataset, project_a.project)] = max(
                    rates.get((project_a.dataset, project_a.project), 0.0),
                    0.0,
                )
                rates[(project_b.dataset, project_b.project)] = max(
                    rates.get((project_b.dataset, project_b.project), 0.0),
                    0.0,
                )
                continue

            

            a_keys = project_a.df[common_features].copy()
            b_keys = project_b.df[common_features].copy()

            for col in common_features:
                a_keys[col] = pd.to_numeric(a_keys[col], errors="coerce").astype(float).round(10)
                b_keys[col] = pd.to_numeric(b_keys[col], errors="coerce").astype(float).round(10)

            a_keys = a_keys.dropna().drop_duplicates()
            b_keys = b_keys.dropna().drop_duplicates()

            merged = a_keys.merge(b_keys, on=common_features, how="inner")

            rate_a = safe_divide(len(merged), len(a_keys))
            rate_b = safe_divide(len(merged), len(b_keys))

            key_a = (project_a.dataset, project_a.project)
            key_b = (project_b.dataset, project_b.project)

            rates[key_a] = max(rates.get(key_a, 0.0), rate_a)
            rates[key_b] = max(rates.get(key_b, 0.0), rate_b)

    return rates


def leakage_score_from_metrics(metrics: Dict[str, object]) -> float:
    suspicious_feature_rate = float(metrics["suspicious_feature_rate"])
    post_release_feature_count = float(metrics["post_release_feature_count"])
    perfect_label_correlation_count = float(metrics["perfect_label_correlation_count"])
    high_label_correlation_count = float(metrics["high_label_correlation_count"])
    duplicate_rate = float(metrics["duplicate_instance_rate"])
    cross_project_duplicate_rate = float(metrics["cross_project_duplicate_rate"])

    penalties = {
        "suspicious_names": min(suspicious_feature_rate / 0.20, 1.0),
        "post_release_names": min(post_release_feature_count / 3.0, 1.0),
        "perfect_label_corr": min(perfect_label_correlation_count / 1.0, 1.0),
        "high_label_corr": min(high_label_correlation_count / 3.0, 1.0),
        "duplicates": min(duplicate_rate / 0.20, 1.0),
        "cross_project_duplicates": min(cross_project_duplicate_rate / 0.10, 1.0),
    }

    weights = {
        "suspicious_names": 0.15,
        "post_release_names": 0.15,
        "perfect_label_corr": 0.25,
        "high_label_corr": 0.15,
        "duplicates": 0.10,
        "cross_project_duplicates": 0.20,
    }

    total_penalty = sum(weights[key] * penalties[key] for key in weights)
    score = 100.0 * (1.0 - total_penalty)

    return round(float(np.clip(score, 0.0, 100.0)), 2)


def leakage_level(score: float) -> str:
    if score >= 90:
        return "low_leakage_risk"
    if score >= 75:
        return "moderate_leakage_risk"
    if score >= 50:
        return "high_leakage_risk"
    return "severe_leakage_risk"


def build_project_warnings(metrics: Dict[str, object]) -> List[str]:
    warnings: List[str] = []

    if float(metrics["suspicious_feature_rate"]) > 0:
        warnings.append("suspicious_feature_names")

    if int(metrics["post_release_feature_count"]) > 0:
        warnings.append("possible_post_release_features")

    if int(metrics["perfect_label_correlation_count"]) > 0:
        warnings.append("perfect_label_correlation")

    if int(metrics["high_label_correlation_count"]) > 0:
        warnings.append("high_label_correlation")

    if float(metrics["duplicate_instance_rate"]) > 0.05:
        warnings.append("duplicate_instances")

    if float(metrics["cross_project_duplicate_rate"]) > 0.01:
        warnings.append("possible_cross_project_contamination")

    return warnings


def profile_project_leakage(
    project: ProjectData,
    cross_project_duplicate_rate: float,
) -> Dict[str, object]:
    df = project.df.copy()
    features = feature_columns(df)

    suspicious_features = [
        col for col in features if contains_any_term(col, SUSPICIOUS_LABEL_TERMS)
    ]

    temporal_features = [
        col for col in features if contains_any_term(col, TEMPORAL_TERMS)
    ]

    post_release_features = [
        col for col in features if contains_any_term(col, POST_RELEASE_TERMS)
    ]

    perfect_corr_count, high_corr_count, high_corr_features = high_label_correlation_features(df)

    metrics: Dict[str, object] = {
        "dataset": project.dataset,
        "project": project.project,
        "path": project.path,
        "n_instances": int(len(df)),
        "n_features": int(len(features)),
        "suspicious_feature_count": int(len(suspicious_features)),
        "suspicious_feature_rate": safe_divide(len(suspicious_features), len(features)),
        "suspicious_features": ";".join(suspicious_features) if suspicious_features else "none",
        "temporal_feature_count": int(len(temporal_features)),
        "temporal_features": ";".join(temporal_features) if temporal_features else "none",
        "post_release_feature_count": int(len(post_release_features)),
        "post_release_features": ";".join(post_release_features) if post_release_features else "none",
        "perfect_label_correlation_count": int(perfect_corr_count),
        "high_label_correlation_count": int(high_corr_count),
        "high_label_correlation_features": ";".join(high_corr_features) if high_corr_features else "none",
        "duplicate_instance_rate": duplicate_instance_rate(df),
        "cross_project_duplicate_rate": float(cross_project_duplicate_rate),
    }

    score = leakage_score_from_metrics(metrics)
    warnings = build_project_warnings(metrics)

    metrics["leakage_score"] = score
    metrics["leakage_level"] = leakage_level(score)
    metrics["warnings"] = ";".join(warnings) if warnings else "none"

    return metrics


def profile_leakage(projects: Sequence[ProjectData]) -> pd.DataFrame:
    cross_project_rates = compute_cross_project_duplicate_rates(projects)

    rows: List[Dict[str, object]] = []

    for project in projects:
        key = (project.dataset, project.project)
        rate = cross_project_rates.get(key, 0.0)

        try:
            rows.append(profile_project_leakage(project, rate))
        except Exception as exc:
            logger.exception(
                "Leakage profiling failed for %s/%s: %s",
                project.dataset,
                project.project,
                exc,
            )

    profile = pd.DataFrame(rows)

    if profile.empty:
        return pd.DataFrame(columns=PROJECT_LEAKAGE_COLUMNS)

    for column in PROJECT_LEAKAGE_COLUMNS:
        if column not in profile.columns:
            profile[column] = np.nan

    return profile[PROJECT_LEAKAGE_COLUMNS]


def summarize_leakage_by_dataset(leakage_profile: pd.DataFrame) -> pd.DataFrame:
    if leakage_profile.empty:
        return pd.DataFrame(columns=DATASET_LEAKAGE_COLUMNS)

    numeric_columns = [
        "suspicious_feature_rate",
        "temporal_feature_count",
        "post_release_feature_count",
        "high_label_correlation_count",
        "duplicate_instance_rate",
        "cross_project_duplicate_rate",
        "leakage_score",
    ]

    summary = (
        leakage_profile
        .groupby("dataset", as_index=False)[numeric_columns]
        .mean(numeric_only=True)
    )

    summary = summary.rename(
        columns={
            "suspicious_feature_rate": "mean_suspicious_feature_rate",
            "temporal_feature_count": "mean_temporal_feature_count",
            "post_release_feature_count": "mean_post_release_feature_count",
            "high_label_correlation_count": "mean_high_label_correlation_count",
            "duplicate_instance_rate": "mean_duplicate_instance_rate",
            "cross_project_duplicate_rate": "mean_cross_project_duplicate_rate",
            "leakage_score": "mean_leakage_score",
        }
    )

    summary["n_projects"] = leakage_profile.groupby("dataset")["project"].count().values
    summary["leakage_level"] = summary["mean_leakage_score"].apply(leakage_level)

    warnings_by_dataset = []

    for dataset in summary["dataset"]:
        dataset_rows = leakage_profile[leakage_profile["dataset"] == dataset]
        warnings = []

        for value in dataset_rows["warnings"].tolist():
            if str(value).strip().lower() == "none":
                continue
            warnings.extend([x.strip() for x in str(value).split(";") if x.strip()])

        warnings_by_dataset.append(";".join(sorted(set(warnings))) if warnings else "none")

    summary["warnings"] = warnings_by_dataset

    for column in DATASET_LEAKAGE_COLUMNS:
        if column not in summary.columns:
            summary[column] = np.nan

    return summary[DATASET_LEAKAGE_COLUMNS]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="Data-set")
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/profiles/DAQUA_leakage_profile.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="outputs/profiles/DAQUA_leakage_summary_by_dataset.csv",
    )

    args = parser.parse_args()

    projects = load_all_projects(args.root)
    leakage = profile_leakage(projects)
    summary = summarize_leakage_by_dataset(leakage)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    leakage.to_csv(args.out, index=False)
    summary.to_csv(args.summary_out, index=False)

    print(f"Saved leakage profile to: {args.out}")
    print(f"Saved leakage summary to: {args.summary_out}")
    print(summary.round(4).to_string(index=False))