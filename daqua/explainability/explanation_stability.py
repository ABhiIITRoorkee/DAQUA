#daqua/explainability/explanation_stability.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


EXPLANATION_READINESS_COLUMNS = [
    "dataset",
    "n_projects",
    "mean_project_feature_importance_stability",
    "std_project_feature_importance_stability",
    "mean_pairwise_project_top10_jaccard",
    "mean_pairwise_project_spearman",
    "dataset_feature_importance_stability_score",
    "explanation_readiness_score",
    "explanation_readiness_level",
    "primary_risks",
    "recommended_protocol",
    "recommended_metrics",
    "warnings",
]


def score_level(score: float) -> str:
    if pd.isna(score):
        return "not_available"
    if score >= 85:
        return "high"
    if score >= 70:
        return "moderate"
    if score >= 50:
        return "limited"
    return "low"


def unique_join(values: List[str]) -> str:
    seen = set()
    ordered: List[str] = []

    for value in values:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)

    return ";".join(ordered) if ordered else "none"


def split_warnings(value: object) -> List[str]:
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()

    if not text or text.lower() == "none":
        return []

    return [item.strip() for item in text.split(";") if item.strip()]


def clamp01(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def clamp_score(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(np.clip(value, 0.0, 100.0))


def compute_explanation_readiness_score(row: pd.Series) -> float:
    """
    Explanation readiness is deliberately conservative.

    It combines:
    - within-project feature-importance stability,
    - cross-project top-k explanation overlap,
    - cross-project feature-rank correlation,
    - penalty for high variance across projects.
    """

    project_stability = clamp_score(
        row.get("mean_project_feature_importance_stability", np.nan)
    )

    dataset_stability = clamp_score(
        row.get("dataset_feature_importance_stability_score", np.nan)
    )

    top10_jaccard = 100.0 * clamp01(
        row.get("mean_pairwise_project_top10_jaccard", np.nan)
    )

    spearman = row.get("mean_pairwise_project_spearman", np.nan)
    if pd.isna(spearman):
        spearman_score = 0.0
    else:
        spearman_score = 100.0 * ((float(np.clip(spearman, -1.0, 1.0)) + 1.0) / 2.0)

    std_project = row.get("std_project_feature_importance_stability", np.nan)
    if pd.isna(std_project):
        variance_penalty = 0.0
    else:
        variance_penalty = min(float(std_project) / 30.0, 1.0) * 100.0

    score = (
        0.35 * project_stability
        + 0.30 * dataset_stability
        + 0.20 * top10_jaccard
        + 0.15 * spearman_score
        - 0.10 * variance_penalty
    )

    return round(float(np.clip(score, 0.0, 100.0)), 2)


def infer_explanation_risks(row: pd.Series) -> List[str]:
    risks: List[str] = []

    score = row.get("explanation_readiness_score", np.nan)
    project_stability = row.get("mean_project_feature_importance_stability", np.nan)
    dataset_stability = row.get("dataset_feature_importance_stability_score", np.nan)
    top10_jaccard = row.get("mean_pairwise_project_top10_jaccard", np.nan)
    spearman = row.get("mean_pairwise_project_spearman", np.nan)
    std_project = row.get("std_project_feature_importance_stability", np.nan)

    if not pd.isna(score):
        if score < 50:
            risks.append("low_explanation_readiness")
        elif score < 70:
            risks.append("limited_explanation_readiness")

    if not pd.isna(project_stability) and project_stability < 50:
        risks.append("unstable_within_project_feature_importance")

    if not pd.isna(dataset_stability) and dataset_stability < 50:
        risks.append("unstable_dataset_level_feature_importance")

    if not pd.isna(top10_jaccard) and top10_jaccard < 0.40:
        risks.append("low_top10_explanation_overlap")

    if not pd.isna(spearman) and spearman < 0.50:
        risks.append("low_cross_project_feature_rank_correlation")

    if not pd.isna(std_project) and std_project > 10:
        risks.append("heterogeneous_explanation_stability_across_projects")

    return risks


def recommend_explanation_protocol(row: pd.Series) -> str:
    risks = set(infer_explanation_risks(row))
    protocol: List[str] = []

    protocol.append("report_feature_importance_stability")

    if (
        "low_explanation_readiness" in risks
        or "limited_explanation_readiness" in risks
    ):
        protocol.append("avoid_strong_global_explanation_claims")

    if "unstable_within_project_feature_importance" in risks:
        protocol.append("use_repeated_explanation_runs")

    if "unstable_dataset_level_feature_importance" in risks:
        protocol.append("report_project_specific_explanations")

    if "low_top10_explanation_overlap" in risks:
        protocol.append("avoid_single_top_feature_interpretation")

    if "low_cross_project_feature_rank_correlation" in risks:
        protocol.append("compare_explanations_across_projects")

    if "heterogeneous_explanation_stability_across_projects" in risks:
        protocol.append("report_explanation_stability_variance")

    protocol.append("treat_explanations_as_dataset_conditional")

    return unique_join(protocol)


def recommend_explanation_metrics(row: pd.Series) -> str:
    metrics = [
        "top_k_feature_overlap",
        "spearman_feature_rank_correlation",
        "kendall_feature_rank_correlation",
        "feature_importance_variance",
    ]

    risks = set(infer_explanation_risks(row))

    if "low_top10_explanation_overlap" in risks:
        metrics.append("top5_top10_top20_jaccard")

    if "heterogeneous_explanation_stability_across_projects" in risks:
        metrics.append("per_project_explanation_stability")

    return unique_join(metrics)


def build_warnings(row: pd.Series) -> str:
    warnings: List[str] = []

    warnings.extend(split_warnings(row.get("warnings", "none")))
    warnings.extend(infer_explanation_risks(row))

    return unique_join(warnings)


def compute_explanation_readiness(
    feature_importance_dataset_stability: pd.DataFrame,
) -> pd.DataFrame:
    if feature_importance_dataset_stability.empty:
        return pd.DataFrame(columns=EXPLANATION_READINESS_COLUMNS)

    rows: List[pd.Series] = []

    for _, input_row in feature_importance_dataset_stability.iterrows():
        row = input_row.copy()

        score = compute_explanation_readiness_score(row)
        row["explanation_readiness_score"] = score
        row["explanation_readiness_level"] = score_level(score)
        row["primary_risks"] = unique_join(infer_explanation_risks(row))
        row["recommended_protocol"] = recommend_explanation_protocol(row)
        row["recommended_metrics"] = recommend_explanation_metrics(row)
        row["warnings"] = build_warnings(row)

        rows.append(row)

    out = pd.DataFrame(rows)

    for column in EXPLANATION_READINESS_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    return out[EXPLANATION_READINESS_COLUMNS].sort_values(
        by="explanation_readiness_score",
        ascending=False,
    )


def load_csv(path: str, kind: str) -> pd.DataFrame:
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"{kind} file not found: {path}")

    return pd.read_csv(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature-importance-dataset",
        type=str,
        default="outputs/profiles/DAQUA_feature_importance_dataset_stability.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/profiles/DAQUA_explanation_readiness.csv",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    feature_dataset = load_csv(
        args.feature_importance_dataset,
        "Feature importance dataset stability",
    )

    explanation_readiness = compute_explanation_readiness(feature_dataset)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    explanation_readiness.to_csv(args.out, index=False)

    print(f"Saved explanation readiness to: {args.out}")
    print("")
    print(
        explanation_readiness[
            [
                "dataset",
                "explanation_readiness_score",
                "explanation_readiness_level",
                "primary_risks",
            ]
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()