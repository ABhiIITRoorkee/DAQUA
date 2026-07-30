from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.preprocessing import MinMaxScaler

from daqua.loaders.defect_loader import ProjectData, load_all_projects


logger = logging.getLogger(__name__)


PAIRWISE_STABILITY_COLUMNS = [
    "dataset",
    "project_a",
    "project_b",
    "n_a",
    "n_b",
    "n_common_features",
    "feature_set_jaccard",
    "label_prevalence_a",
    "label_prevalence_b",
    "label_prevalence_abs_diff",
    "mean_ks_statistic",
    "median_ks_statistic",
    "max_ks_statistic",
    "mean_ks_pvalue",
    "mean_wasserstein_distance",
    "median_wasserstein_distance",
    "max_wasserstein_distance",
    "mean_feature_mean_shift",
    "mean_feature_std_shift",
    "project_size_ratio",
    "pairwise_stability_score",
    "pairwise_stability_level",
    "warnings",
]


DATASET_STABILITY_COLUMNS = [
    "dataset",
    "n_projects",
    "mean_project_size",
    "project_size_cv",
    "mean_feature_count",
    "feature_count_cv",
    "feature_set_consistency",
    "mean_label_prevalence",
    "label_prevalence_std",
    "label_prevalence_range",
    "mean_pairwise_ks",
    "mean_pairwise_wasserstein",
    "mean_pairwise_feature_mean_shift",
    "mean_pairwise_feature_std_shift",
    "mean_pairwise_stability_score",
    "stability_score",
    "stability_level",
    "warnings",
]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def coefficient_of_variation(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        return np.nan

    mean = np.nanmean(arr)

    if mean == 0:
        return 0.0

    return float(np.nanstd(arr) / mean)


def feature_columns(df: pd.DataFrame, label_col: str = "label") -> List[str]:
    return [col for col in df.columns if col != label_col]


def label_prevalence(df: pd.DataFrame, label_col: str = "label") -> float:
    if df.empty or label_col not in df.columns:
        return np.nan

    return float(df[label_col].astype(int).mean())


def normalize_pairwise_features(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x_a = df_a[features].astype(float).replace([np.inf, -np.inf], np.nan)
    x_b = df_b[features].astype(float).replace([np.inf, -np.inf], np.nan)

    combined = pd.concat([x_a, x_b], axis=0, ignore_index=True)
    combined = combined.fillna(combined.median(numeric_only=True))

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(combined)

    scaled_a = pd.DataFrame(scaled[: len(x_a)], columns=features)
    scaled_b = pd.DataFrame(scaled[len(x_a) :], columns=features)

    return scaled_a, scaled_b


def compute_feature_set_jaccard(features_a: Sequence[str], features_b: Sequence[str]) -> float:
    set_a = set(features_a)
    set_b = set(features_b)

    union = set_a.union(set_b)

    if not union:
        return 1.0

    return safe_divide(len(set_a.intersection(set_b)), len(union))


def compute_distribution_shift(
    x_a: pd.DataFrame,
    x_b: pd.DataFrame,
    features: Sequence[str],
) -> Dict[str, float]:
    ks_stats: List[float] = []
    ks_pvalues: List[float] = []
    wasserstein_values: List[float] = []
    mean_shifts: List[float] = []
    std_shifts: List[float] = []

    for feature in features:
        a = x_a[feature].dropna().to_numpy(dtype=float)
        b = x_b[feature].dropna().to_numpy(dtype=float)

        if len(a) == 0 or len(b) == 0:
            continue

        try:
            ks = ks_2samp(a, b, alternative="two-sided", mode="auto")
            ks_stats.append(float(ks.statistic))
            ks_pvalues.append(float(ks.pvalue))
        except Exception:
            ks_stats.append(np.nan)
            ks_pvalues.append(np.nan)

        try:
            wasserstein_values.append(float(wasserstein_distance(a, b)))
        except Exception:
            wasserstein_values.append(np.nan)

        mean_shifts.append(float(abs(np.mean(a) - np.mean(b))))
        std_shifts.append(float(abs(np.std(a) - np.std(b))))

    return {
        "mean_ks_statistic": float(np.nanmean(ks_stats)) if ks_stats else np.nan,
        "median_ks_statistic": float(np.nanmedian(ks_stats)) if ks_stats else np.nan,
        "max_ks_statistic": float(np.nanmax(ks_stats)) if ks_stats else np.nan,
        "mean_ks_pvalue": float(np.nanmean(ks_pvalues)) if ks_pvalues else np.nan,
        "mean_wasserstein_distance": float(np.nanmean(wasserstein_values)) if wasserstein_values else np.nan,
        "median_wasserstein_distance": float(np.nanmedian(wasserstein_values)) if wasserstein_values else np.nan,
        "max_wasserstein_distance": float(np.nanmax(wasserstein_values)) if wasserstein_values else np.nan,
        "mean_feature_mean_shift": float(np.nanmean(mean_shifts)) if mean_shifts else np.nan,
        "mean_feature_std_shift": float(np.nanmean(std_shifts)) if std_shifts else np.nan,
    }


def clamp01(value: float) -> float:
    if pd.isna(value):
        return 1.0

    return float(np.clip(value, 0.0, 1.0))


def compute_pairwise_stability_score(metrics: Dict[str, float]) -> float:
    penalties = {
        "feature_set": 1.0 - clamp01(metrics["feature_set_jaccard"]),
        "label_shift": clamp01(safe_divide(metrics["label_prevalence_abs_diff"], 0.50)),
        "ks_shift": clamp01(metrics["mean_ks_statistic"]),
        "wasserstein_shift": clamp01(metrics["mean_wasserstein_distance"]),
        "mean_shift": clamp01(metrics["mean_feature_mean_shift"]),
        "std_shift": clamp01(metrics["mean_feature_std_shift"]),
        "size_shift": clamp01(abs(1.0 - metrics["project_size_ratio"])),
    }

    weights = {
        "feature_set": 0.15,
        "label_shift": 0.20,
        "ks_shift": 0.25,
        "wasserstein_shift": 0.15,
        "mean_shift": 0.10,
        "std_shift": 0.05,
        "size_shift": 0.10,
    }

    total_penalty = sum(weights[key] * penalties[key] for key in weights)
    score = 100.0 * (1.0 - total_penalty)

    return round(float(np.clip(score, 0.0, 100.0)), 2)


def stability_level(score: float) -> str:
    if score >= 85:
        return "high"
    if score >= 70:
        return "moderate"
    if score >= 50:
        return "limited"
    return "low"


def build_pairwise_warnings(metrics: Dict[str, float]) -> List[str]:
    warnings: List[str] = []

    if metrics["feature_set_jaccard"] < 0.80:
        warnings.append("inconsistent_feature_sets")

    if metrics["label_prevalence_abs_diff"] > 0.20:
        warnings.append("large_label_distribution_shift")
    elif metrics["label_prevalence_abs_diff"] > 0.10:
        warnings.append("moderate_label_distribution_shift")

    if metrics["mean_ks_statistic"] > 0.40:
        warnings.append("large_feature_distribution_shift")
    elif metrics["mean_ks_statistic"] > 0.25:
        warnings.append("moderate_feature_distribution_shift")

    if metrics["mean_wasserstein_distance"] > 0.30:
        warnings.append("large_feature_magnitude_shift")

    if metrics["project_size_ratio"] < 0.25:
        warnings.append("large_project_size_mismatch")

    return warnings


def profile_project_pair_stability(
    project_a: ProjectData,
    project_b: ProjectData,
) -> Dict[str, object]:
    df_a = project_a.df.copy()
    df_b = project_b.df.copy()

    features_a = feature_columns(df_a)
    features_b = feature_columns(df_b)
    common_features = sorted(set(features_a).intersection(features_b))

    feature_set_jaccard = compute_feature_set_jaccard(features_a, features_b)

    prevalence_a = label_prevalence(df_a)
    prevalence_b = label_prevalence(df_b)

    metrics: Dict[str, object] = {
        "dataset": project_a.dataset,
        "project_a": project_a.project,
        "project_b": project_b.project,
        "n_a": int(len(df_a)),
        "n_b": int(len(df_b)),
        "n_common_features": int(len(common_features)),
        "feature_set_jaccard": feature_set_jaccard,
        "label_prevalence_a": prevalence_a,
        "label_prevalence_b": prevalence_b,
        "label_prevalence_abs_diff": abs(prevalence_a - prevalence_b),
        "project_size_ratio": safe_divide(min(len(df_a), len(df_b)), max(len(df_a), len(df_b))),
    }

    if common_features:
        x_a, x_b = normalize_pairwise_features(df_a, df_b, common_features)
        metrics.update(compute_distribution_shift(x_a, x_b, common_features))
    else:
        metrics.update(
            {
                "mean_ks_statistic": np.nan,
                "median_ks_statistic": np.nan,
                "max_ks_statistic": np.nan,
                "mean_ks_pvalue": np.nan,
                "mean_wasserstein_distance": np.nan,
                "median_wasserstein_distance": np.nan,
                "max_wasserstein_distance": np.nan,
                "mean_feature_mean_shift": np.nan,
                "mean_feature_std_shift": np.nan,
            }
        )

    score = compute_pairwise_stability_score(metrics)
    warnings = build_pairwise_warnings(metrics)

    metrics["pairwise_stability_score"] = score
    metrics["pairwise_stability_level"] = stability_level(score)
    metrics["warnings"] = ";".join(warnings) if warnings else "none"

    return metrics


def profile_pairwise_stability(projects: Sequence[ProjectData]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    by_dataset: Dict[str, List[ProjectData]] = {}

    for project in projects:
        by_dataset.setdefault(project.dataset, []).append(project)

    for dataset, dataset_projects in sorted(by_dataset.items()):
        if len(dataset_projects) < 2:
            logger.warning("Dataset '%s' has fewer than two projects; skipping pairwise stability.", dataset)
            continue

        logger.info(
            "Computing pairwise stability for dataset='%s' | projects=%d",
            dataset,
            len(dataset_projects),
        )

        for project_a, project_b in itertools.combinations(dataset_projects, 2):
            try:
                rows.append(profile_project_pair_stability(project_a, project_b))
            except Exception as exc:
                logger.exception(
                    "Pairwise stability failed for %s/%s vs %s/%s: %s",
                    project_a.dataset,
                    project_a.project,
                    project_b.dataset,
                    project_b.project,
                    exc,
                )

    profile = pd.DataFrame(rows)

    if profile.empty:
        return pd.DataFrame(columns=PAIRWISE_STABILITY_COLUMNS)

    for column in PAIRWISE_STABILITY_COLUMNS:
        if column not in profile.columns:
            profile[column] = np.nan

    return profile[PAIRWISE_STABILITY_COLUMNS]


def feature_set_consistency(projects: Sequence[ProjectData]) -> float:
    if len(projects) < 2:
        return 1.0

    jaccards: List[float] = []

    for project_a, project_b in itertools.combinations(projects, 2):
        features_a = feature_columns(project_a.df)
        features_b = feature_columns(project_b.df)
        jaccards.append(compute_feature_set_jaccard(features_a, features_b))

    return float(np.nanmean(jaccards)) if jaccards else 1.0


def dataset_level_warnings(metrics: Dict[str, float]) -> List[str]:
    warnings: List[str] = []

    if metrics["feature_set_consistency"] < 0.80:
        warnings.append("inconsistent_feature_sets")

    if metrics["project_size_cv"] > 1.0:
        warnings.append("unstable_project_sizes")

    if metrics["label_prevalence_range"] > 0.30:
        warnings.append("large_label_prevalence_range")
    elif metrics["label_prevalence_range"] > 0.20:
        warnings.append("moderate_label_prevalence_range")

    if metrics["mean_pairwise_ks"] > 0.40:
        warnings.append("large_feature_distribution_shift")
    elif metrics["mean_pairwise_ks"] > 0.25:
        warnings.append("moderate_feature_distribution_shift")

    if metrics["mean_pairwise_stability_score"] < 60:
        warnings.append("low_pairwise_stability")

    return warnings


def compute_dataset_stability_score(metrics: Dict[str, float]) -> float:
    penalties = {
        "pairwise": 1.0 - clamp01(safe_divide(metrics["mean_pairwise_stability_score"], 100.0)),
        "label_range": clamp01(safe_divide(metrics["label_prevalence_range"], 0.50)),
        "project_size_cv": clamp01(safe_divide(metrics["project_size_cv"], 2.0)),
        "feature_count_cv": clamp01(safe_divide(metrics["feature_count_cv"], 1.0)),
        "feature_consistency": 1.0 - clamp01(metrics["feature_set_consistency"]),
        "ks_shift": clamp01(metrics["mean_pairwise_ks"]),
        "wasserstein_shift": clamp01(metrics["mean_pairwise_wasserstein"]),
    }

    weights = {
        "pairwise": 0.30,
        "label_range": 0.20,
        "project_size_cv": 0.10,
        "feature_count_cv": 0.05,
        "feature_consistency": 0.15,
        "ks_shift": 0.15,
        "wasserstein_shift": 0.05,
    }

    total_penalty = sum(weights[key] * penalties[key] for key in weights)
    score = 100.0 * (1.0 - total_penalty)

    return round(float(np.clip(score, 0.0, 100.0)), 2)


def summarize_stability_by_dataset(
    projects: Sequence[ProjectData],
    pairwise_profile: pd.DataFrame,
) -> pd.DataFrame:
    by_dataset: Dict[str, List[ProjectData]] = {}

    for project in projects:
        by_dataset.setdefault(project.dataset, []).append(project)

    rows: List[Dict[str, object]] = []

    for dataset, dataset_projects in sorted(by_dataset.items()):
        dataset_pairs = pairwise_profile[pairwise_profile["dataset"] == dataset]

        project_sizes = [len(project.df) for project in dataset_projects]
        feature_counts = [len(feature_columns(project.df)) for project in dataset_projects]
        prevalences = [label_prevalence(project.df) for project in dataset_projects]

        metrics: Dict[str, object] = {
            "dataset": dataset,
            "n_projects": int(len(dataset_projects)),
            "mean_project_size": float(np.mean(project_sizes)),
            "project_size_cv": coefficient_of_variation(project_sizes),
            "mean_feature_count": float(np.mean(feature_counts)),
            "feature_count_cv": coefficient_of_variation(feature_counts),
            "feature_set_consistency": feature_set_consistency(dataset_projects),
            "mean_label_prevalence": float(np.nanmean(prevalences)),
            "label_prevalence_std": float(np.nanstd(prevalences)),
            "label_prevalence_range": float(np.nanmax(prevalences) - np.nanmin(prevalences)),
            "mean_pairwise_ks": float(dataset_pairs["mean_ks_statistic"].mean()) if not dataset_pairs.empty else np.nan,
            "mean_pairwise_wasserstein": float(dataset_pairs["mean_wasserstein_distance"].mean()) if not dataset_pairs.empty else np.nan,
            "mean_pairwise_feature_mean_shift": float(dataset_pairs["mean_feature_mean_shift"].mean()) if not dataset_pairs.empty else np.nan,
            "mean_pairwise_feature_std_shift": float(dataset_pairs["mean_feature_std_shift"].mean()) if not dataset_pairs.empty else np.nan,
            "mean_pairwise_stability_score": float(dataset_pairs["pairwise_stability_score"].mean()) if not dataset_pairs.empty else np.nan,
        }

        score = compute_dataset_stability_score(metrics)
        warnings = dataset_level_warnings(metrics)

        metrics["stability_score"] = score
        metrics["stability_level"] = stability_level(score)
        metrics["warnings"] = ";".join(warnings) if warnings else "none"

        rows.append(metrics)

    summary = pd.DataFrame(rows)

    if summary.empty:
        return pd.DataFrame(columns=DATASET_STABILITY_COLUMNS)

    for column in DATASET_STABILITY_COLUMNS:
        if column not in summary.columns:
            summary[column] = np.nan

    return summary[DATASET_STABILITY_COLUMNS]


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="Data-set")
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/profiles/DAQUA_stability_pairwise_profile.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="outputs/profiles/DAQUA_stability_summary_by_dataset.csv",
    )

    args = parser.parse_args()

    projects = load_all_projects(args.root)

    pairwise = profile_pairwise_stability(projects)
    summary = summarize_stability_by_dataset(projects, pairwise)

    pairwise.to_csv(args.out, index=False)
    summary.to_csv(args.summary_out, index=False)

    print(f"Saved pairwise stability profile to: {args.out}")
    print(f"Saved dataset stability summary to: {args.summary_out}")
    print(summary.round(4).to_string(index=False))