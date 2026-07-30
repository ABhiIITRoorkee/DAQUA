#daqua/models/baseline_models.py

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from daqua.loaders.defect_loader import ProjectData, load_all_projects


logger = logging.getLogger(__name__)


BASELINE_RESULT_COLUMNS = [
    "dataset",
    "project",
    "model",
    "n_instances",
    "n_features",
    "minority_class_percentage",
    "repeat",
    "fold",
    "precision",
    "recall",
    "f1_score",
    "g_mean",
    "accuracy",
    "auc",
    "mcc",
    "PD",
    "PF",
]


MODEL_SUMMARY_COLUMNS = [
    "dataset",
    "project",
    "model",
    "n_runs",
    "mean_auc",
    "std_auc",
    "mean_mcc",
    "std_mcc",
    "mean_f1_score",
    "std_f1_score",
    "mean_g_mean",
    "std_g_mean",
    "mean_accuracy",
    "std_accuracy",
    "rank_by_auc",
    "rank_by_mcc",
]


RANKING_STABILITY_COLUMNS = [
    "dataset",
    "n_projects",
    "n_models",
    "mean_pairwise_spearman_auc",
    "mean_pairwise_kendall_auc",
    "mean_pairwise_spearman_mcc",
    "mean_pairwise_kendall_mcc",
    "model_ranking_stability_score",
    "model_ranking_stability_level",
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
    majority = max(class_0, class_1)
    total = class_0 + class_1

    return {
        "class_0_count": class_0,
        "class_1_count": class_1,
        "minority_class_count": minority,
        "majority_class_count": majority,
        "minority_class_percentage": safe_divide(minority, total),
    }


def get_baseline_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "Dummy": DummyClassifier(strategy="stratified", random_state=random_state),
        "LogisticRegression": Pipeline(
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
        ),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
            min_samples_leaf=2,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "KNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    }


def predicted_scores(model: object, x_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        scores = np.asarray(scores, dtype=float)
        min_score = np.min(scores)
        max_score = np.max(scores)
        return (scores - min_score) / (max_score - min_score + 1e-10)

    return model.predict(x_test).astype(float)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    y_pred = (y_score >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        return {
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
            "g_mean": np.nan,
            "accuracy": np.nan,
            "auc": np.nan,
            "mcc": np.nan,
            "PD": np.nan,
            "PF": np.nan,
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pd_value = safe_divide(tp, tp + fn)
    pf_value = safe_divide(fp, fp + tn)
    specificity = safe_divide(tn, tn + fp)
    g_mean = float(np.sqrt(max(pd_value * specificity, 0.0)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = np.nan

        return {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "g_mean": g_mean,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "auc": float(auc) if not pd.isna(auc) else np.nan,
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "PD": float(pd_value),
            "PF": float(pf_value),
        }


def prepare_project_xy(project: ProjectData) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = project.df.copy()
    features = feature_columns(df)

    x = df[features].astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))

    y = df["label"].astype(int).to_numpy()

    return x.to_numpy(dtype=float), y, features


def feasible_cv_splits(y: np.ndarray, requested_splits: int) -> int:
    _, counts = np.unique(y, return_counts=True)

    if len(counts) < 2:
        return 0

    min_class_count = int(np.min(counts))

    if min_class_count < 2:
        return 0

    return min(requested_splits, min_class_count)


def evaluate_project_baselines(
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
            "Skipping %s/%s because at least one class has fewer than two instances.",
            project.dataset,
            project.project,
        )
        return pd.DataFrame(columns=BASELINE_RESULT_COLUMNS)

    models = get_baseline_models(random_state=random_state)

    cv = RepeatedStratifiedKFold(
        n_splits=effective_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    rows: List[Dict[str, object]] = []

    for split_index, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        repeat = split_index // effective_splits + 1
        fold = split_index % effective_splits + 1

        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for model_name, model_template in models.items():
            model = clone(model_template)

            try:
                model.fit(x_train, y_train)
                y_score = predicted_scores(model, x_test)
                metrics = compute_metrics(y_test, y_score)

                rows.append(
                    {
                        "dataset": project.dataset,
                        "project": project.project,
                        "model": model_name,
                        "n_instances": int(len(y)),
                        "n_features": int(len(features)),
                        "minority_class_percentage": dist["minority_class_percentage"],
                        "repeat": int(repeat),
                        "fold": int(fold),
                        **metrics,
                    }
                )

            except Exception as exc:
                logger.warning(
                    "Model failed for %s/%s/%s repeat=%d fold=%d: %s",
                    project.dataset,
                    project.project,
                    model_name,
                    repeat,
                    fold,
                    exc,
                )

    result = pd.DataFrame(rows)

    if result.empty:
        return pd.DataFrame(columns=BASELINE_RESULT_COLUMNS)

    for column in BASELINE_RESULT_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan

    return result[BASELINE_RESULT_COLUMNS]


def evaluate_all_baselines(
    projects: Sequence[ProjectData],
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for project in projects:
        logger.info("Evaluating baseline models for %s/%s", project.dataset, project.project)

        frame = evaluate_project_baselines(
            project=project,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=BASELINE_RESULT_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def summarize_model_performance(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=MODEL_SUMMARY_COLUMNS)

    grouped = (
        results.groupby(["dataset", "project", "model"], as_index=False)
        .agg(
            n_runs=("auc", "count"),
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            mean_mcc=("mcc", "mean"),
            std_mcc=("mcc", "std"),
            mean_f1_score=("f1_score", "mean"),
            std_f1_score=("f1_score", "std"),
            mean_g_mean=("g_mean", "mean"),
            std_g_mean=("g_mean", "std"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
        )
    )

    grouped["rank_by_auc"] = grouped.groupby(["dataset", "project"])["mean_auc"].rank(
        ascending=False,
        method="average",
    )

    grouped["rank_by_mcc"] = grouped.groupby(["dataset", "project"])["mean_mcc"].rank(
        ascending=False,
        method="average",
    )

    for column in MODEL_SUMMARY_COLUMNS:
        if column not in grouped.columns:
            grouped[column] = np.nan

    return grouped[MODEL_SUMMARY_COLUMNS]


def ranking_vector(project_summary: pd.DataFrame, metric: str, model_order: List[str]) -> np.ndarray:
    values = []

    for model in model_order:
        row = project_summary[project_summary["model"] == model]

        if row.empty:
            values.append(np.nan)
        else:
            values.append(float(row.iloc[0][metric]))

    return np.asarray(values, dtype=float)


def pairwise_rank_correlations(
    summary: pd.DataFrame,
    metric: str,
) -> Tuple[float, float]:
    if summary.empty:
        return np.nan, np.nan

    models = sorted(summary["model"].dropna().unique().tolist())
    project_keys = summary[["dataset", "project"]].drop_duplicates()

    vectors = []

    for _, key in project_keys.iterrows():
        project_summary = summary[
            (summary["dataset"] == key["dataset"])
            & (summary["project"] == key["project"])
        ]

        vec = ranking_vector(project_summary, metric, models)

        if np.isfinite(vec).sum() >= 2:
            vectors.append(vec)

    if len(vectors) < 2:
        return np.nan, np.nan

    spearman_values = []
    kendall_values = []

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a = vectors[i]
            b = vectors[j]

            valid = np.isfinite(a) & np.isfinite(b)

            if valid.sum() < 2:
                continue

            s = spearmanr(a[valid], b[valid]).correlation
            k = kendalltau(a[valid], b[valid]).correlation

            if not pd.isna(s):
                spearman_values.append(float(s))

            if not pd.isna(k):
                kendall_values.append(float(k))

    mean_spearman = float(np.nanmean(spearman_values)) if spearman_values else np.nan
    mean_kendall = float(np.nanmean(kendall_values)) if kendall_values else np.nan

    return mean_spearman, mean_kendall


def ranking_stability_score(spearman_auc: float, kendall_auc: float, spearman_mcc: float, kendall_mcc: float) -> float:
    values = [spearman_auc, kendall_auc, spearman_mcc, kendall_mcc]
    valid = [v for v in values if not pd.isna(v)]

    if not valid:
        return np.nan

    normalized = [(v + 1.0) / 2.0 for v in valid]
    return round(float(100.0 * np.mean(normalized)), 2)


def ranking_stability_level(score: float) -> str:
    if pd.isna(score):
        return "not_available"
    if score >= 85:
        return "high"
    if score >= 70:
        return "moderate"
    if score >= 50:
        return "limited"
    return "low"


def ranking_warnings(score: float) -> str:
    if pd.isna(score):
        return "insufficient_projects_for_ranking_stability"

    if score < 50:
        return "unstable_model_ranking"

    if score < 70:
        return "limited_model_ranking_stability"

    return "none"


def summarize_ranking_stability(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=RANKING_STABILITY_COLUMNS)

    rows: List[Dict[str, object]] = []

    for dataset, dataset_summary in summary.groupby("dataset"):
        spearman_auc, kendall_auc = pairwise_rank_correlations(dataset_summary, "mean_auc")
        spearman_mcc, kendall_mcc = pairwise_rank_correlations(dataset_summary, "mean_mcc")

        score = ranking_stability_score(
            spearman_auc=spearman_auc,
            kendall_auc=kendall_auc,
            spearman_mcc=spearman_mcc,
            kendall_mcc=kendall_mcc,
        )

        rows.append(
            {
                "dataset": dataset,
                "n_projects": int(dataset_summary["project"].nunique()),
                "n_models": int(dataset_summary["model"].nunique()),
                "mean_pairwise_spearman_auc": spearman_auc,
                "mean_pairwise_kendall_auc": kendall_auc,
                "mean_pairwise_spearman_mcc": spearman_mcc,
                "mean_pairwise_kendall_mcc": kendall_mcc,
                "model_ranking_stability_score": score,
                "model_ranking_stability_level": ranking_stability_level(score),
                "warnings": ranking_warnings(score),
            }
        )

    out = pd.DataFrame(rows)

    for column in RANKING_STABILITY_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    return out[RANKING_STABILITY_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="Data-set")
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/profiles/DAQUA_baseline_model_results.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="outputs/profiles/DAQUA_baseline_model_summary.csv",
    )
    parser.add_argument(
        "--ranking-out",
        type=str,
        default="outputs/profiles/DAQUA_model_ranking_stability.csv",
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

    results = evaluate_all_baselines(
        projects=projects,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )

    summary = summarize_model_performance(results)
    ranking = summarize_ranking_stability(summary)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(args.out, index=False)
    summary.to_csv(args.summary_out, index=False)
    ranking.to_csv(args.ranking_out, index=False)

    print(f"Saved baseline model results to: {args.out}")
    print(f"Saved baseline model summary to: {args.summary_out}")
    print(f"Saved model ranking stability to: {args.ranking_out}")
    print("")
    print(ranking.round(4).to_string(index=False))


if __name__ == "__main__":
    main()