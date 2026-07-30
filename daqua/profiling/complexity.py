from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from daqua.loaders.defect_loader import ProjectData, load_all_projects


logger = logging.getLogger(__name__)


COMPLEXITY_COLUMNS = [
    "dataset",
    "project",
    "n_instances",
    "n_features",
    "minority_class",
    "minority_class_percentage",
    "nn_same_class_ratio",
    "minority_nn_same_class_ratio",
    "majority_nn_same_class_ratio",
    "nn_opposite_class_ratio",
    "borderline_instance_rate",
    "minority_borderline_rate",
    "class_centroid_distance",
    "intra_class_distance_mean",
    "inter_class_distance_mean",
    "distance_separation_ratio",
    "mean_feature_overlap",
    "high_overlap_feature_rate",
    "mean_mutual_information",
    "zero_mi_feature_rate",
    "pca_components_95",
    "pca_components_95_ratio",
    "complexity_score",
    "complexity_level",
    "warnings",
]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def feature_columns(df: pd.DataFrame, label_col: str = "label") -> List[str]:
    return [col for col in df.columns if col != label_col]


def prepare_xy(
    df: pd.DataFrame,
    label_col: str = "label",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    features = feature_columns(df, label_col)

    x = df[features].astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))

    y = df[label_col].astype(int).to_numpy()

    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)

    return x_scaled, y, features


def class_distribution(y: np.ndarray) -> Dict[str, float]:
    labels, counts = np.unique(y, return_counts=True)
    count_map = dict(zip(labels.tolist(), counts.tolist()))

    class_0 = int(count_map.get(0, 0))
    class_1 = int(count_map.get(1, 0))

    if class_0 <= class_1:
        minority_class = 0
        minority_count = class_0
        majority_count = class_1
    else:
        minority_class = 1
        minority_count = class_1
        majority_count = class_0

    total = len(y)

    return {
        "minority_class": minority_class,
        "minority_class_count": minority_count,
        "majority_class_count": majority_count,
        "minority_class_percentage": safe_divide(minority_count, total),
    }


def nearest_neighbor_complexity(
    x: np.ndarray,
    y: np.ndarray,
    k_borderline: int = 5,
) -> Dict[str, float]:
    if len(y) < 3 or len(np.unique(y)) < 2:
        return {
            "nn_same_class_ratio": np.nan,
            "minority_nn_same_class_ratio": np.nan,
            "majority_nn_same_class_ratio": np.nan,
            "nn_opposite_class_ratio": np.nan,
            "borderline_instance_rate": np.nan,
            "minority_borderline_rate": np.nan,
        }

    dist = pairwise_distances(x, metric="euclidean")
    np.fill_diagonal(dist, np.inf)

    nearest_idx = np.argmin(dist, axis=1)
    nearest_same = y[nearest_idx] == y

    dist_info = class_distribution(y)
    minority_class = int(dist_info["minority_class"])
    majority_class = 1 - minority_class

    minority_mask = y == minority_class
    majority_mask = y == majority_class

    nn_same_class_ratio = float(np.mean(nearest_same))
    minority_nn_same_class_ratio = float(np.mean(nearest_same[minority_mask])) if minority_mask.any() else np.nan
    majority_nn_same_class_ratio = float(np.mean(nearest_same[majority_mask])) if majority_mask.any() else np.nan

    k = min(k_borderline + 1, len(y))

    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean")
    neighbors.fit(x)

    _, indices = neighbors.kneighbors(x)

    neighbor_indices = indices[:, 1:]
    neighbor_labels = y[neighbor_indices]

    opposite_ratio = np.mean(neighbor_labels != y[:, None], axis=1)

    borderline = opposite_ratio >= 0.50

    return {
        "nn_same_class_ratio": nn_same_class_ratio,
        "minority_nn_same_class_ratio": minority_nn_same_class_ratio,
        "majority_nn_same_class_ratio": majority_nn_same_class_ratio,
        "nn_opposite_class_ratio": float(np.mean(opposite_ratio)),
        "borderline_instance_rate": float(np.mean(borderline)),
        "minority_borderline_rate": float(np.mean(borderline[minority_mask])) if minority_mask.any() else np.nan,
    }


def distance_complexity(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if len(y) < 3 or len(np.unique(y)) < 2:
        return {
            "class_centroid_distance": np.nan,
            "intra_class_distance_mean": np.nan,
            "inter_class_distance_mean": np.nan,
            "distance_separation_ratio": np.nan,
        }

    x0 = x[y == 0]
    x1 = x[y == 1]

    if len(x0) == 0 or len(x1) == 0:
        return {
            "class_centroid_distance": np.nan,
            "intra_class_distance_mean": np.nan,
            "inter_class_distance_mean": np.nan,
            "distance_separation_ratio": np.nan,
        }

    centroid_0 = x0.mean(axis=0)
    centroid_1 = x1.mean(axis=0)

    class_centroid_distance = float(np.linalg.norm(centroid_0 - centroid_1))

    intra_0 = np.linalg.norm(x0 - centroid_0, axis=1).mean() if len(x0) > 0 else np.nan
    intra_1 = np.linalg.norm(x1 - centroid_1, axis=1).mean() if len(x1) > 0 else np.nan

    intra_class_distance_mean = float(np.nanmean([intra_0, intra_1]))

    sample_limit = 2000

    if len(x0) * len(x1) > sample_limit * sample_limit:
        rng = np.random.default_rng(42)
        x0_sample = x0[rng.choice(len(x0), size=min(sample_limit, len(x0)), replace=False)]
        x1_sample = x1[rng.choice(len(x1), size=min(sample_limit, len(x1)), replace=False)]
    else:
        x0_sample = x0
        x1_sample = x1

    inter_dist = pairwise_distances(x0_sample, x1_sample, metric="euclidean")
    inter_class_distance_mean = float(inter_dist.mean())

    distance_separation_ratio = safe_divide(
        inter_class_distance_mean,
        intra_class_distance_mean,
        default=np.nan,
    )

    return {
        "class_centroid_distance": class_centroid_distance,
        "intra_class_distance_mean": intra_class_distance_mean,
        "inter_class_distance_mean": inter_class_distance_mean,
        "distance_separation_ratio": distance_separation_ratio,
    }


def feature_overlap_complexity(
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    if len(y) < 3 or len(np.unique(y)) < 2:
        return {
            "mean_feature_overlap": np.nan,
            "high_overlap_feature_rate": np.nan,
        }

    x0 = x[y == 0]
    x1 = x[y == 1]

    overlaps = []

    for j in range(x.shape[1]):
        min_0, max_0 = np.min(x0[:, j]), np.max(x0[:, j])
        min_1, max_1 = np.min(x1[:, j]), np.max(x1[:, j])

        intersection = max(0.0, min(max_0, max_1) - max(min_0, min_1))
        union = max(max_0, max_1) - min(min_0, min_1)

        overlap = safe_divide(intersection, union, default=0.0)
        overlaps.append(overlap)

    overlaps_array = np.asarray(overlaps, dtype=float)

    return {
        "mean_feature_overlap": float(np.mean(overlaps_array)),
        "high_overlap_feature_rate": float(np.mean(overlaps_array >= 0.80)),
    }


def mutual_information_complexity(
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    if len(y) < 3 or len(np.unique(y)) < 2:
        return {
            "mean_mutual_information": np.nan,
            "zero_mi_feature_rate": np.nan,
        }

    try:
        mi = mutual_info_classif(x, y, random_state=42)
    except Exception as exc:
        logger.warning("Mutual information failed: %s", exc)
        return {
            "mean_mutual_information": np.nan,
            "zero_mi_feature_rate": np.nan,
        }

    return {
        "mean_mutual_information": float(np.mean(mi)),
        "zero_mi_feature_rate": float(np.mean(mi <= 1e-8)),
    }


def pca_complexity(x: np.ndarray) -> Dict[str, float]:
    if x.shape[0] < 2 or x.shape[1] < 2:
        return {
            "pca_components_95": 1,
            "pca_components_95_ratio": 1.0,
        }

    max_components = min(x.shape[0], x.shape[1])

    try:
        pca = PCA(n_components=max_components, random_state=42)
        pca.fit(x)

        cumulative = np.cumsum(pca.explained_variance_ratio_)
        components_95 = int(np.searchsorted(cumulative, 0.95) + 1)

        return {
            "pca_components_95": components_95,
            "pca_components_95_ratio": safe_divide(components_95, x.shape[1]),
        }

    except Exception as exc:
        logger.warning("PCA complexity failed: %s", exc)
        return {
            "pca_components_95": np.nan,
            "pca_components_95_ratio": np.nan,
        }


def clamp01(value: float) -> float:
    if pd.isna(value):
        return 1.0

    return float(np.clip(value, 0.0, 1.0))


def compute_complexity_score(metrics: Dict[str, float]) -> float:
    """
    Higher score means easier / more prediction-ready.
    Lower score means more complex, overlapped, or unstable for learning.
    """

    penalties = {
        "nn_overlap": 1.0 - clamp01(metrics["nn_same_class_ratio"]),
        "minority_overlap": 1.0 - clamp01(metrics["minority_nn_same_class_ratio"]),
        "borderline": clamp01(metrics["borderline_instance_rate"]),
        "minority_borderline": clamp01(metrics["minority_borderline_rate"]),
        "feature_overlap": clamp01(metrics["mean_feature_overlap"]),
        "high_overlap_features": clamp01(metrics["high_overlap_feature_rate"]),
        "zero_mi": clamp01(metrics["zero_mi_feature_rate"]),
        "pca_dimensionality": clamp01(metrics["pca_components_95_ratio"]),
        "distance_separation": 1.0 - clamp01(safe_divide(metrics["distance_separation_ratio"], 2.0)),
        "imbalance": imbalance_complexity_penalty(metrics["minority_class_percentage"]),
    }

    weights = {
        "nn_overlap": 0.15,
        "minority_overlap": 0.15,
        "borderline": 0.12,
        "minority_borderline": 0.12,
        "feature_overlap": 0.10,
        "high_overlap_features": 0.08,
        "zero_mi": 0.08,
        "pca_dimensionality": 0.06,
        "distance_separation": 0.08,
        "imbalance": 0.06,
    }

    total_penalty = sum(weights[key] * penalties[key] for key in weights)
    score = 100.0 * (1.0 - total_penalty)

    return round(float(np.clip(score, 0.0, 100.0)), 2)


def imbalance_complexity_penalty(minority_percentage: float) -> float:
    if pd.isna(minority_percentage):
        return 1.0

    if minority_percentage >= 0.30:
        return 0.0

    if minority_percentage <= 0.05:
        return 1.0

    return safe_divide(0.30 - minority_percentage, 0.25)


def complexity_level(score: float) -> str:
    if score >= 80:
        return "low_complexity"
    if score >= 65:
        return "moderate_complexity"
    if score >= 50:
        return "high_complexity"
    return "very_high_complexity"


def build_complexity_warnings(metrics: Dict[str, float]) -> List[str]:
    warnings: List[str] = []

    if metrics["minority_class_percentage"] < 0.10:
        warnings.append("severe_minority_difficulty")
    elif metrics["minority_class_percentage"] < 0.20:
        warnings.append("moderate_minority_difficulty")

    if metrics["nn_same_class_ratio"] < 0.65:
        warnings.append("low_neighborhood_separability")

    if metrics["minority_nn_same_class_ratio"] < 0.50:
        warnings.append("minority_class_poorly_separated")

    if metrics["borderline_instance_rate"] > 0.30:
        warnings.append("many_borderline_instances")

    if metrics["minority_borderline_rate"] > 0.40:
        warnings.append("minority_class_borderline_heavy")

    if metrics["mean_feature_overlap"] > 0.75:
        warnings.append("high_class_overlap")

    if metrics["zero_mi_feature_rate"] > 0.50:
        warnings.append("many_uninformative_features")

    if metrics["pca_components_95_ratio"] > 0.80:
        warnings.append("high_intrinsic_dimensionality")

    if metrics["distance_separation_ratio"] < 1.0:
        warnings.append("weak_distance_separation")

    return warnings


def profile_project_complexity(project: ProjectData) -> Dict[str, object]:
    df = project.df.copy()
    x, y, features = prepare_xy(df)

    distribution = class_distribution(y)

    metrics: Dict[str, object] = {
        "dataset": project.dataset,
        "project": project.project,
        "n_instances": int(len(y)),
        "n_features": int(len(features)),
        "minority_class": int(distribution["minority_class"]),
        "minority_class_percentage": distribution["minority_class_percentage"],
    }

    metrics.update(nearest_neighbor_complexity(x, y))
    metrics.update(distance_complexity(x, y))
    metrics.update(feature_overlap_complexity(x, y))
    metrics.update(mutual_information_complexity(x, y))
    metrics.update(pca_complexity(x))

    score = compute_complexity_score(metrics)
    warnings = build_complexity_warnings(metrics)

    metrics["complexity_score"] = score
    metrics["complexity_level"] = complexity_level(score)
    metrics["warnings"] = ";".join(warnings) if warnings else "none"

    return metrics


def profile_complexity(projects: Sequence[ProjectData]) -> pd.DataFrame:
    rows = []

    for project in projects:
        try:
            rows.append(profile_project_complexity(project))
        except Exception as exc:
            logger.exception(
                "Complexity profiling failed for %s/%s: %s",
                project.dataset,
                project.project,
                exc,
            )

    profile = pd.DataFrame(rows)

    if profile.empty:
        return pd.DataFrame(columns=COMPLEXITY_COLUMNS)

    for column in COMPLEXITY_COLUMNS:
        if column not in profile.columns:
            profile[column] = np.nan

    return profile[COMPLEXITY_COLUMNS]


def summarize_complexity_by_dataset(complexity_profile: pd.DataFrame) -> pd.DataFrame:
    if complexity_profile.empty:
        return pd.DataFrame()

    numeric_columns = [
        "n_instances",
        "n_features",
        "minority_class_percentage",
        "nn_same_class_ratio",
        "minority_nn_same_class_ratio",
        "majority_nn_same_class_ratio",
        "nn_opposite_class_ratio",
        "borderline_instance_rate",
        "minority_borderline_rate",
        "class_centroid_distance",
        "intra_class_distance_mean",
        "inter_class_distance_mean",
        "distance_separation_ratio",
        "mean_feature_overlap",
        "high_overlap_feature_rate",
        "mean_mutual_information",
        "zero_mi_feature_rate",
        "pca_components_95",
        "pca_components_95_ratio",
        "complexity_score",
    ]

    summary = (
        complexity_profile
        .groupby("dataset", as_index=False)[numeric_columns]
        .mean(numeric_only=True)
    )

    summary["n_projects"] = complexity_profile.groupby("dataset")["project"].count().values

    ordered_columns = ["dataset", "n_projects"] + numeric_columns
    return summary[ordered_columns]


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
        default="outputs/profiles/DAQUA_complexity_profile.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="outputs/profiles/DAQUA_complexity_summary_by_dataset.csv",
    )

    args = parser.parse_args()

    projects = load_all_projects(args.root)
    complexity = profile_complexity(projects)
    summary = summarize_complexity_by_dataset(complexity)

    complexity.to_csv(args.out, index=False)
    summary.to_csv(args.summary_out, index=False)

    print(f"Saved project complexity profile to: {args.out}")
    print(f"Saved dataset complexity summary to: {args.summary_out}")
    print(summary.round(4).to_string(index=False))