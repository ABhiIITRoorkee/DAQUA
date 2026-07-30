#daqua/explainability/feature_importance.py

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from daqua.loaders.defect_loader import ProjectData, load_all_projects


logger = logging.getLogger(__name__)


FEATURE_IMPORTANCE_COLUMNS = [
    "dataset",
    "project",
    "method",
    "repeat",
    "fold",
    "feature",
    "importance",
    "rank",
    "n_instances",
    "n_features",
    "minority_class_percentage",
]


PROJECT_STABILITY_COLUMNS = [
    "dataset",
    "project",
    "n_features",
    "n_methods",
    "n_importance_runs",
    "mean_pairwise_spearman",
    "mean_pairwise_kendall",
    "mean_top5_jaccard",
    "mean_top10_jaccard",
    "mean_top20_jaccard",
    "feature_importance_stability_score",
    "feature_importance_stability_level",
    "warnings",
]


DATASET_STABILITY_COLUMNS = [
    "dataset",
    "n_projects",
    "mean_project_feature_importance_stability",
    "std_project_feature_importance_stability",
    "mean_pairwise_project_top10_jaccard",
    "mean_pairwise_project_spearman",
    "dataset_feature_importance_stability_score",
    "dataset_feature_importance_stability_level",
    "warnings",
]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def feature_columns(df: pd.DataFrame, label_col: str = "label") -> List[str]:
    return [col for col in df.columns if col != label_col]


def class_distribution(y: np.ndarray) -> Dict[str, float]:
    labels, counts = np.unique(y, return_counts=True)
    count_map = dict(zip(labels.tolist(), counts.tolist()))

    class_0 = int(count_map.get(0, 0))
    class_1 = int(count_map.get(1, 0))

    minority = min(class_0, class_1)
    total = class_0 + class_1

    return {
        "class_0_count": class_0,
        "class_1_count": class_1,
        "minority_class_count": minority,
        "minority_class_percentage": safe_divide(minority, total),
    }


def prepare_project_xy(project: ProjectData) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    df = project.df.copy()
    features = feature_columns(df)

    x = df[features].astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))

    y = df["label"].astype(int).to_numpy()

    return x, y, features


def feasible_cv_splits(y: np.ndarray, requested_splits: int) -> int:
    _, counts = np.unique(y, return_counts=True)

    if len(counts) < 2:
        return 0

    min_class_count = int(np.min(counts))

    if min_class_count < 2:
        return 0

    return min(requested_splits, min_class_count)


def normalize_importance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    values = np.abs(values)
    total = values.sum()

    if total <= 0:
        return np.zeros_like(values, dtype=float)

    return values / total


def ranked_importance_frame(
    dataset: str,
    project: str,
    method: str,
    repeat: int,
    fold: int,
    features: List[str],
    importances: np.ndarray,
    n_instances: int,
    minority_class_percentage: float,
) -> pd.DataFrame:
    normalized = normalize_importance(importances)

    out = pd.DataFrame(
        {
            "dataset": dataset,
            "project": project,
            "method": method,
            "repeat": repeat,
            "fold": fold,
            "feature": features,
            "importance": normalized,
            "n_instances": n_instances,
            "n_features": len(features),
            "minority_class_percentage": minority_class_percentage,
        }
    )

    out["rank"] = out["importance"].rank(ascending=False, method="average")
    return out[FEATURE_IMPORTANCE_COLUMNS]


def random_forest_importance(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
) -> np.ndarray:
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    model.fit(x_train, y_train)
    return model.feature_importances_


def permutation_importance_rf(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    random_state: int,
) -> np.ndarray:
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    model.fit(x_train, y_train)

    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=5,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )

    return result.importances_mean


def logistic_regression_importance(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
) -> np.ndarray:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    coef = model.named_steps["clf"].coef_.reshape(-1)
    return np.abs(coef)


def mutual_information_importance(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mutual_info_classif(x_train, y_train, random_state=random_state)


def compute_fold_importances(
    project: ProjectData,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    x, y, features = prepare_project_xy(project)
    dist = class_distribution(y)

    effective_splits = feasible_cv_splits(y, n_splits)

    if effective_splits < 2:
        logger.warning(
            "Skipping feature importance for %s/%s because at least one class has fewer than two instances.",
            project.dataset,
            project.project,
        )
        return pd.DataFrame(columns=FEATURE_IMPORTANCE_COLUMNS)

    frames: List[pd.DataFrame] = []

    for repeat in range(1, n_repeats + 1):
        cv = StratifiedKFold(
            n_splits=effective_splits,
            shuffle=True,
            random_state=random_state + repeat,
        )

        for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
            x_train = x.iloc[train_idx]
            x_test = x.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            methods = {
                "random_forest_gini": None,
                "permutation_auc_rf": None,
                "logistic_abs_coef": None,
                "mutual_information": None,
            }

            for method in methods:
                try:
                    if method == "random_forest_gini":
                        importances = random_forest_importance(
                            x_train=x_train,
                            y_train=y_train,
                            random_state=random_state + repeat + fold,
                        )
                    elif method == "permutation_auc_rf":
                        importances = permutation_importance_rf(
                            x_train=x_train,
                            y_train=y_train,
                            x_test=x_test,
                            y_test=y_test,
                            random_state=random_state + repeat + fold,
                        )
                    elif method == "logistic_abs_coef":
                        importances = logistic_regression_importance(
                            x_train=x_train,
                            y_train=y_train,
                            random_state=random_state + repeat + fold,
                        )
                    elif method == "mutual_information":
                        importances = mutual_information_importance(
                            x_train=x_train,
                            y_train=y_train,
                            random_state=random_state + repeat + fold,
                        )
                    else:
                        continue

                    frames.append(
                        ranked_importance_frame(
                            dataset=project.dataset,
                            project=project.project,
                            method=method,
                            repeat=repeat,
                            fold=fold,
                            features=features,
                            importances=importances,
                            n_instances=len(y),
                            minority_class_percentage=dist["minority_class_percentage"],
                        )
                    )

                except Exception as exc:
                    logger.warning(
                        "Feature importance failed for %s/%s method=%s repeat=%d fold=%d: %s",
                        project.dataset,
                        project.project,
                        method,
                        repeat,
                        fold,
                        exc,
                    )

    if not frames:
        return pd.DataFrame(columns=FEATURE_IMPORTANCE_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def compute_all_feature_importances(
    projects: Sequence[ProjectData],
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for project in projects:
        logger.info("Computing feature importance for %s/%s", project.dataset, project.project)

        frame = compute_fold_importances(
            project=project,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=FEATURE_IMPORTANCE_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def importance_vector(frame: pd.DataFrame, features: List[str]) -> np.ndarray:
    feature_to_importance = dict(zip(frame["feature"], frame["importance"]))
    return np.asarray([feature_to_importance.get(feature, 0.0) for feature in features], dtype=float)


def topk_set(frame: pd.DataFrame, k: int) -> set:
    return set(frame.sort_values("rank", ascending=True).head(k)["feature"].tolist())


def jaccard(a: set, b: set) -> float:
    union = a.union(b)

    if not union:
        return 1.0

    return len(a.intersection(b)) / len(union)


def pairwise_importance_stability(frames: List[pd.DataFrame], features: List[str]) -> Dict[str, float]:
    if len(frames) < 2:
        return {
            "mean_pairwise_spearman": np.nan,
            "mean_pairwise_kendall": np.nan,
            "mean_top5_jaccard": np.nan,
            "mean_top10_jaccard": np.nan,
            "mean_top20_jaccard": np.nan,
        }

    spearman_values: List[float] = []
    kendall_values: List[float] = []
    top5_values: List[float] = []
    top10_values: List[float] = []
    top20_values: List[float] = []

    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            a = importance_vector(frames[i], features)
            b = importance_vector(frames[j], features)

            valid = np.isfinite(a) & np.isfinite(b)

            if valid.sum() >= 2:
                s = spearmanr(a[valid], b[valid]).correlation
                k = kendalltau(a[valid], b[valid]).correlation

                if not pd.isna(s):
                    spearman_values.append(float(s))

                if not pd.isna(k):
                    kendall_values.append(float(k))

            top5_values.append(jaccard(topk_set(frames[i], 5), topk_set(frames[j], 5)))
            top10_values.append(jaccard(topk_set(frames[i], 10), topk_set(frames[j], 10)))
            top20_values.append(jaccard(topk_set(frames[i], 20), topk_set(frames[j], 20)))

    return {
        "mean_pairwise_spearman": float(np.nanmean(spearman_values)) if spearman_values else np.nan,
        "mean_pairwise_kendall": float(np.nanmean(kendall_values)) if kendall_values else np.nan,
        "mean_top5_jaccard": float(np.nanmean(top5_values)) if top5_values else np.nan,
        "mean_top10_jaccard": float(np.nanmean(top10_values)) if top10_values else np.nan,
        "mean_top20_jaccard": float(np.nanmean(top20_values)) if top20_values else np.nan,
    }


def stability_score_from_metrics(metrics: Dict[str, float]) -> float:
    values = [
        metrics.get("mean_pairwise_spearman", np.nan),
        metrics.get("mean_pairwise_kendall", np.nan),
        metrics.get("mean_top5_jaccard", np.nan),
        metrics.get("mean_top10_jaccard", np.nan),
        metrics.get("mean_top20_jaccard", np.nan),
    ]

    normalized = []

    for value in values:
        if pd.isna(value):
            continue

        if value < -1:
            value = -1

        if value > 1:
            value = 1

        if value < 0:
            value = (value + 1.0) / 2.0

        normalized.append(float(value))

    if not normalized:
        return np.nan

    return round(float(100.0 * np.mean(normalized)), 2)


def stability_level(score: float) -> str:
    if pd.isna(score):
        return "not_available"
    if score >= 85:
        return "high"
    if score >= 70:
        return "moderate"
    if score >= 50:
        return "limited"
    return "low"


def stability_warning(score: float) -> str:
    if pd.isna(score):
        return "insufficient_feature_importance_runs"
    if score < 50:
        return "unstable_feature_importance"
    if score < 70:
        return "limited_feature_importance_stability"
    return "none"


def summarize_project_feature_importance_stability(importances: pd.DataFrame) -> pd.DataFrame:
    if importances.empty:
        return pd.DataFrame(columns=PROJECT_STABILITY_COLUMNS)

    rows: List[Dict[str, object]] = []

    for (dataset, project), group in importances.groupby(["dataset", "project"]):
        features = sorted(group["feature"].unique().tolist())

        frames = [
            frame
            for _, frame in group.groupby(["method", "repeat", "fold"])
            if not frame.empty
        ]

        metrics = pairwise_importance_stability(frames, features)
        score = stability_score_from_metrics(metrics)

        rows.append(
            {
                "dataset": dataset,
                "project": project,
                "n_features": int(len(features)),
                "n_methods": int(group["method"].nunique()),
                "n_importance_runs": int(len(frames)),
                **metrics,
                "feature_importance_stability_score": score,
                "feature_importance_stability_level": stability_level(score),
                "warnings": stability_warning(score),
            }
        )

    out = pd.DataFrame(rows)

    for column in PROJECT_STABILITY_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    return out[PROJECT_STABILITY_COLUMNS]


def mean_project_importance_frame(importances: pd.DataFrame, dataset: str, project: str) -> pd.DataFrame:
    group = importances[(importances["dataset"] == dataset) & (importances["project"] == project)]

    out = (
        group.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )

    out["rank"] = out["importance"].rank(ascending=False, method="average")
    return out


def summarize_dataset_feature_importance_stability(
    importances: pd.DataFrame,
    project_stability: pd.DataFrame,
) -> pd.DataFrame:
    if importances.empty or project_stability.empty:
        return pd.DataFrame(columns=DATASET_STABILITY_COLUMNS)

    rows: List[Dict[str, object]] = []

    for dataset, dataset_projects in project_stability.groupby("dataset"):
        project_names = sorted(dataset_projects["project"].unique().tolist())

        project_frames = [
            mean_project_importance_frame(importances, dataset, project)
            for project in project_names
        ]

        all_features = sorted(
            set().union(*[set(frame["feature"].tolist()) for frame in project_frames if not frame.empty])
        )

        spearman_values: List[float] = []
        top10_values: List[float] = []

        for i in range(len(project_frames)):
            for j in range(i + 1, len(project_frames)):
                a = importance_vector(project_frames[i], all_features)
                b = importance_vector(project_frames[j], all_features)

                valid = np.isfinite(a) & np.isfinite(b)

                if valid.sum() >= 2:
                    s = spearmanr(a[valid], b[valid]).correlation
                    if not pd.isna(s):
                        spearman_values.append(float(s))

                top10_values.append(jaccard(topk_set(project_frames[i], 10), topk_set(project_frames[j], 10)))

        mean_project_score = float(dataset_projects["feature_importance_stability_score"].mean())
        mean_top10 = float(np.nanmean(top10_values)) if top10_values else np.nan
        mean_spearman = float(np.nanmean(spearman_values)) if spearman_values else np.nan

        score_parts = [mean_project_score]

        if not pd.isna(mean_top10):
            score_parts.append(100.0 * mean_top10)

        if not pd.isna(mean_spearman):
            score_parts.append(100.0 * ((mean_spearman + 1.0) / 2.0))

        dataset_score = round(float(np.nanmean(score_parts)), 2)

        rows.append(
            {
                "dataset": dataset,
                "n_projects": int(len(project_names)),
                "mean_project_feature_importance_stability": mean_project_score,
                "std_project_feature_importance_stability": float(dataset_projects["feature_importance_stability_score"].std()),
                "mean_pairwise_project_top10_jaccard": mean_top10,
                "mean_pairwise_project_spearman": mean_spearman,
                "dataset_feature_importance_stability_score": dataset_score,
                "dataset_feature_importance_stability_level": stability_level(dataset_score),
                "warnings": stability_warning(dataset_score),
            }
        )

    out = pd.DataFrame(rows)

    for column in DATASET_STABILITY_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    return out[DATASET_STABILITY_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="Data-set")
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/profiles/DAQUA_feature_importance_profile.csv",
    )
    parser.add_argument(
        "--project-summary-out",
        type=str,
        default="outputs/profiles/DAQUA_feature_importance_project_stability.csv",
    )
    parser.add_argument(
        "--dataset-summary-out",
        type=str,
        default="outputs/profiles/DAQUA_feature_importance_dataset_stability.csv",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    projects = load_all_projects(args.root)

    importances = compute_all_feature_importances(
        projects=projects,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )

    project_stability = summarize_project_feature_importance_stability(importances)
    dataset_stability = summarize_dataset_feature_importance_stability(
        importances=importances,
        project_stability=project_stability,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    importances.to_csv(args.out, index=False)
    project_stability.to_csv(args.project_summary_out, index=False)
    dataset_stability.to_csv(args.dataset_summary_out, index=False)

    print(f"Saved feature importance profile to: {args.out}")
    print(f"Saved project feature importance stability to: {args.project_summary_out}")
    print(f"Saved dataset feature importance stability to: {args.dataset_summary_out}")
    print("")
    print(dataset_stability.round(4).to_string(index=False))


if __name__ == "__main__":
    main()