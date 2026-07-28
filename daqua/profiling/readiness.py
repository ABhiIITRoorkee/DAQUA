#daqua/profiling/readiness.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


READINESS_COLUMNS = [
    "dataset",
    "quality_score",
    "complexity_score",
    "stability_score",
    "leakage_score",
    "model_stability_score",
    "explanation_readiness_score",
    "overall_readiness_score",
    "readiness_level",
    "quality_level",
    "complexity_level",
    "stability_level",
    "leakage_level",
    "model_stability_level",
    "explanation_readiness_level",
    "prediction_readiness",
    "recommended_protocol",
    "recommended_metrics",
    "primary_risks",
    "warnings",
]


def clamp_score(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(np.clip(value, 0.0, 100.0))


def score_level(score: float) -> str:
    if score >= 85:
        return "high"
    if score >= 70:
        return "moderate"
    if score >= 50:
        return "limited"
    return "low"


def normalize_complexity_level(score: float) -> str:
    if score >= 80:
        return "low_complexity"
    if score >= 65:
        return "moderate_complexity"
    if score >= 50:
        return "high_complexity"
    return "very_high_complexity"


def leakage_level_from_score(score: float) -> str:
    if score >= 90:
        return "low_leakage_risk"
    if score >= 75:
        return "moderate_leakage_risk"
    if score >= 50:
        return "high_leakage_risk"
    return "severe_leakage_risk"


def model_stability_level_from_score(score: float) -> str:
    if pd.isna(score):
        return "not_evaluated"
    if score >= 85:
        return "high"
    if score >= 70:
        return "moderate"
    if score >= 50:
        return "limited"
    return "low"


def explanation_readiness_level_from_score(score: float) -> str:
    if pd.isna(score):
        return "not_evaluated"
    if score >= 85:
        return "high"
    if score >= 70:
        return "moderate"
    if score >= 50:
        return "limited"
    return "low"


def compute_overall_readiness_score(
    quality_score: float,
    complexity_score: float,
    stability_score: float,
    leakage_score: float,
    model_stability_score: float,
    explanation_readiness_score: float,
) -> float:
    """
    DAQUA-V3 score.

    Higher scores indicate stronger readiness across prediction,
    transfer, leakage, model-ranking, and explanation dimensions.
    """

    weights = {
        "quality": 0.20,
        "complexity": 0.18,
        "stability": 0.18,
        "leakage": 0.17,
        "model_stability": 0.12,
        "explanation_readiness": 0.15,
    }

    score = (
        weights["quality"] * clamp_score(quality_score)
        + weights["complexity"] * clamp_score(complexity_score)
        + weights["stability"] * clamp_score(stability_score)
        + weights["leakage"] * clamp_score(leakage_score)
        + weights["model_stability"] * clamp_score(model_stability_score)
        + weights["explanation_readiness"] * clamp_score(explanation_readiness_score)
    )

    return round(float(np.clip(score, 0.0, 100.0)), 2)


def split_warnings(value: object) -> List[str]:
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()

    if not text or text.lower() == "none":
        return []

    return [item.strip() for item in text.split(";") if item.strip()]


def unique_join(values: List[str]) -> str:
    seen = set()
    ordered: List[str] = []

    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)

    return ";".join(ordered) if ordered else "none"


def infer_primary_risks(row: pd.Series) -> List[str]:
    risks: List[str] = []

    quality_score = row.get("quality_score", np.nan)
    complexity_score = row.get("complexity_score", np.nan)
    stability_score = row.get("stability_score", np.nan)
    leakage_score = row.get("leakage_score", np.nan)
    model_stability_score = row.get("model_stability_score", np.nan)
    explanation_score = row.get("explanation_readiness_score", np.nan)

    if quality_score < 70:
        risks.append("data_quality_risk")

    if complexity_score < 65:
        risks.append("prediction_complexity_risk")

    if stability_score < 70:
        risks.append("dataset_stability_risk")

    if leakage_score < 75:
        risks.append("leakage_risk")
    elif leakage_score < 90:
        risks.append("moderate_leakage_risk")

    if not pd.isna(model_stability_score):
        if model_stability_score < 70:
            risks.append("model_ranking_instability_risk")
        elif model_stability_score < 85:
            risks.append("moderate_model_ranking_stability_risk")

    if not pd.isna(explanation_score):
        if explanation_score < 50:
            risks.append("low_explanation_readiness_risk")
        elif explanation_score < 70:
            risks.append("limited_explanation_readiness_risk")

    minority_percentage = row.get("minority_class_percentage", np.nan)
    if not pd.isna(minority_percentage) and minority_percentage < 0.20:
        risks.append("class_imbalance_risk")

    high_corr = row.get("high_correlation_feature_rate", np.nan)
    if not pd.isna(high_corr) and high_corr > 0.40:
        risks.append("feature_redundancy_risk")

    row_retention = row.get("row_retention_rate", np.nan)
    if not pd.isna(row_retention) and row_retention < 0.80:
        risks.append("data_loss_after_cleaning_risk")

    label_range = row.get("label_prevalence_range", np.nan)
    if not pd.isna(label_range) and label_range > 0.20:
        risks.append("label_distribution_shift_risk")

    mean_ks = row.get("mean_pairwise_ks", np.nan)
    if not pd.isna(mean_ks) and mean_ks > 0.25:
        risks.append("feature_distribution_shift_risk")

    minority_nn = row.get("minority_nn_same_class_ratio", np.nan)
    if not pd.isna(minority_nn) and minority_nn < 0.50:
        risks.append("minority_class_overlap_risk")

    borderline = row.get("borderline_instance_rate", np.nan)
    if not pd.isna(borderline) and borderline > 0.30:
        risks.append("borderline_instance_risk")

    project_size_cv = row.get("project_size_cv", np.nan)
    if not pd.isna(project_size_cv) and project_size_cv > 1.0:
        risks.append("project_size_instability_risk")

    suspicious_rate = row.get("mean_suspicious_feature_rate", np.nan)
    if not pd.isna(suspicious_rate) and suspicious_rate > 0.0:
        risks.append("suspicious_feature_name_risk")

    post_release_count = row.get("mean_post_release_feature_count", np.nan)
    if not pd.isna(post_release_count) and post_release_count > 0.0:
        risks.append("possible_post_release_feature_risk")

    cross_project_dup = row.get("mean_cross_project_duplicate_rate", np.nan)
    if not pd.isna(cross_project_dup) and cross_project_dup > 0.01:
        risks.append("possible_cross_project_contamination_risk")

    return risks


def recommend_metrics(row: pd.Series) -> List[str]:
    metrics = ["AUC", "MCC", "F1", "G-mean"]

    minority_percentage = row.get("minority_class_percentage", np.nan)
    if pd.isna(minority_percentage) or minority_percentage < 0.30:
        metrics.extend(["P@20", "R@20", "IFA", "TopK_AUC"])

    stability_score = row.get("stability_score", np.nan)
    if pd.isna(stability_score) or stability_score < 75:
        metrics.extend(["confidence_intervals", "run_variance"])

    complexity_score = row.get("complexity_score", np.nan)
    if pd.isna(complexity_score) or complexity_score < 65:
        metrics.extend(["per_project_results", "failure_rate"])

    leakage_score = row.get("leakage_score", np.nan)
    if pd.isna(leakage_score) or leakage_score < 90:
        metrics.extend(["leakage_sensitivity_analysis"])

    model_stability_score = row.get("model_stability_score", np.nan)
    if pd.isna(model_stability_score) or model_stability_score < 85:
        metrics.extend(["model_ranking_stability", "metric_sensitivity_analysis"])

    explanation_score = row.get("explanation_readiness_score", np.nan)
    if pd.isna(explanation_score) or explanation_score < 70:
        metrics.extend(
            [
                "feature_importance_stability",
                "top_k_feature_overlap",
                "explanation_rank_correlation",
                "explanation_variance",
            ]
        )

    return list(dict.fromkeys(metrics))


def recommend_protocol(row: pd.Series) -> str:
    quality_score = row.get("quality_score", np.nan)
    complexity_score = row.get("complexity_score", np.nan)
    stability_score = row.get("stability_score", np.nan)
    leakage_score = row.get("leakage_score", np.nan)

    risks = set(infer_primary_risks(row))
    protocol_parts: List[str] = []

    if quality_score < 60:
        protocol_parts.append("clean_before_modeling")
    elif quality_score < 75:
        protocol_parts.append("report_cleaning_impact")

    if "label_distribution_shift_risk" in risks or "feature_distribution_shift_risk" in risks:
        protocol_parts.append("cross_project_evaluation_with_shift_analysis")
    else:
        protocol_parts.append("cross_project_evaluation")

    if complexity_score < 65:
        protocol_parts.append("repeated_runs_with_project_level_reporting")
    else:
        protocol_parts.append("repeated_runs")

    if "class_imbalance_risk" in risks:
        protocol_parts.append("imbalance_aware_metrics")

    if "minority_class_overlap_risk" in risks or "borderline_instance_risk" in risks:
        protocol_parts.append("avoid_single_threshold_claims")

    if stability_score < 70:
        protocol_parts.append("avoid_strong_generalization_claims")
    elif stability_score < 85:
        protocol_parts.append("report_stability_sensitivity")

    if leakage_score < 90:
        protocol_parts.append("inspect_leakage_warnings")

    if "suspicious_feature_name_risk" in risks:
        protocol_parts.append("audit_suspicious_feature_names")

    if "possible_post_release_feature_risk" in risks:
        protocol_parts.append("verify_feature_availability_time")

    if "possible_cross_project_contamination_risk" in risks:
        protocol_parts.append("deduplicate_across_projects_before_transfer")

    if "feature_redundancy_risk" in risks:
        protocol_parts.append("feature_redundancy_analysis")

    if "data_loss_after_cleaning_risk" in risks:
        protocol_parts.append("raw_vs_cleaned_dataset_comparison")

    if "model_ranking_instability_risk" in risks or "moderate_model_ranking_stability_risk" in risks:
        protocol_parts.append("report_model_ranking_stability")
        protocol_parts.append("avoid_single_metric_model_claims")

    if "low_explanation_readiness_risk" in risks or "limited_explanation_readiness_risk" in risks:
        protocol_parts.append("report_explanation_stability")
        protocol_parts.append("avoid_strong_global_explanation_claims")
        protocol_parts.append("treat_explanations_as_dataset_conditional")

    return ";".join(list(dict.fromkeys(protocol_parts)))


def prediction_readiness_label(row: pd.Series) -> str:
    score = row["overall_readiness_score"]
    risks = set(infer_primary_risks(row))

    severe_risks = {
        "leakage_risk",
        "data_quality_risk",
        "dataset_stability_risk",
        "model_ranking_instability_risk",
    }

    if score >= 85 and not risks:
        return "ready_for_standard_prediction"

    if score >= 75 and not severe_risks.intersection(risks):
        return "ready_with_reporting_controls"

    if score >= 60:
        return "usable_with_caution"

    if score >= 45:
        return "limited_readiness_requires_mitigation"

    return "not_ready_without_substantial_cleaning"


def combine_warning_columns(row: pd.Series) -> str:
    warnings: List[str] = []

    for column in [
        "quality_warnings",
        "complexity_warnings",
        "stability_warnings",
        "leakage_warnings",
        "model_stability_warnings",
        "explanation_readiness_warnings",
    ]:
        warnings.extend(split_warnings(row.get(column)))

    warnings.extend(infer_primary_risks(row))
    return unique_join(warnings)


def load_summary_csv(path: str, kind: str) -> pd.DataFrame:
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"{kind} summary file not found: {path}")

    df = pd.read_csv(csv_path)

    if "dataset" not in df.columns:
        raise ValueError(f"{kind} summary must contain a 'dataset' column: {path}")

    return df


def prepare_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "dataset",
        "quality_score",
        "row_retention_rate",
        "missing_value_rate",
        "duplicate_rate",
        "conflicting_duplicate_rate",
        "constant_feature_rate",
        "near_constant_feature_rate",
        "high_correlation_feature_rate",
        "outlier_instance_rate",
        "minority_class_percentage",
        "class_imbalance_ratio",
        "label_entropy",
    ]

    out = df[[col for col in keep_cols if col in df.columns]].copy()

    if "quality_score" not in out.columns:
        raise ValueError("Quality summary is missing 'quality_score'.")

    out["quality_level"] = out["quality_score"].apply(score_level)
    out["quality_warnings"] = df["warnings"] if "warnings" in df.columns else "none"
    return out


def prepare_complexity_summary(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "dataset",
        "complexity_score",
        "nn_same_class_ratio",
        "minority_nn_same_class_ratio",
        "borderline_instance_rate",
        "minority_borderline_rate",
        "mean_feature_overlap",
        "high_overlap_feature_rate",
        "mean_mutual_information",
        "zero_mi_feature_rate",
        "pca_components_95_ratio",
    ]

    out = df[[col for col in keep_cols if col in df.columns]].copy()

    if "complexity_score" not in out.columns:
        raise ValueError("Complexity summary is missing 'complexity_score'.")

    out["complexity_level"] = out["complexity_score"].apply(normalize_complexity_level)
    out["complexity_warnings"] = df["warnings"] if "warnings" in df.columns else "none"
    return out


def prepare_stability_summary(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "dataset",
        "stability_score",
        "project_size_cv",
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
    ]

    out = df[[col for col in keep_cols if col in df.columns]].copy()

    if "stability_score" not in out.columns:
        raise ValueError("Stability summary is missing 'stability_score'.")

    out["stability_level"] = out["stability_score"].apply(score_level)
    out["stability_warnings"] = df["warnings"] if "warnings" in df.columns else "none"
    return out


def prepare_leakage_summary(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["dataset", "leakage_score", "leakage_level", "leakage_warnings"]
        )

    keep_cols = [
        "dataset",
        "mean_suspicious_feature_rate",
        "mean_temporal_feature_count",
        "mean_post_release_feature_count",
        "mean_high_label_correlation_count",
        "mean_duplicate_instance_rate",
        "mean_cross_project_duplicate_rate",
        "mean_leakage_score",
    ]

    out = df[[col for col in keep_cols if col in df.columns]].copy()

    if "mean_leakage_score" not in out.columns:
        raise ValueError("Leakage summary is missing 'mean_leakage_score'.")

    out = out.rename(columns={"mean_leakage_score": "leakage_score"})
    out["leakage_level"] = out["leakage_score"].apply(leakage_level_from_score)
    out["leakage_warnings"] = df["warnings"] if "warnings" in df.columns else "none"
    return out


def prepare_model_stability_summary(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model_stability_score",
                "model_stability_level",
                "model_stability_warnings",
            ]
        )

    keep_cols = [
        "dataset",
        "mean_pairwise_spearman_auc",
        "mean_pairwise_kendall_auc",
        "mean_pairwise_spearman_mcc",
        "mean_pairwise_kendall_mcc",
        "model_ranking_stability_score",
    ]

    out = df[[col for col in keep_cols if col in df.columns]].copy()

    if "model_ranking_stability_score" not in out.columns:
        raise ValueError("Model stability summary is missing 'model_ranking_stability_score'.")

    out = out.rename(columns={"model_ranking_stability_score": "model_stability_score"})
    out["model_stability_level"] = out["model_stability_score"].apply(
        model_stability_level_from_score
    )
    out["model_stability_warnings"] = df["warnings"] if "warnings" in df.columns else "none"
    return out


def prepare_explanation_readiness_summary(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "explanation_readiness_score",
                "explanation_readiness_level",
                "explanation_readiness_warnings",
            ]
        )

    keep_cols = [
        "dataset",
        "mean_project_feature_importance_stability",
        "std_project_feature_importance_stability",
        "mean_pairwise_project_top10_jaccard",
        "mean_pairwise_project_spearman",
        "dataset_feature_importance_stability_score",
        "explanation_readiness_score",
    ]

    out = df[[col for col in keep_cols if col in df.columns]].copy()

    if "explanation_readiness_score" not in out.columns:
        raise ValueError("Explanation readiness summary is missing 'explanation_readiness_score'.")

    out["explanation_readiness_level"] = out["explanation_readiness_score"].apply(
        explanation_readiness_level_from_score
    )
    out["explanation_readiness_warnings"] = df["warnings"] if "warnings" in df.columns else "none"
    return out


def compute_readiness(
    quality_summary: pd.DataFrame,
    complexity_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    leakage_summary: Optional[pd.DataFrame] = None,
    model_stability_summary: Optional[pd.DataFrame] = None,
    explanation_readiness_summary: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    quality = prepare_quality_summary(quality_summary)
    complexity = prepare_complexity_summary(complexity_summary)
    stability = prepare_stability_summary(stability_summary)
    leakage = prepare_leakage_summary(leakage_summary)
    model_stability = prepare_model_stability_summary(model_stability_summary)
    explanation_readiness = prepare_explanation_readiness_summary(
        explanation_readiness_summary
    )

    merged = quality.merge(complexity, on="dataset", how="outer")
    merged = merged.merge(stability, on="dataset", how="outer")

    if not leakage.empty:
        merged = merged.merge(leakage, on="dataset", how="outer")
    else:
        merged["leakage_score"] = 100.0
        merged["leakage_level"] = "not_evaluated"
        merged["leakage_warnings"] = "none"

    if not model_stability.empty:
        merged = merged.merge(model_stability, on="dataset", how="outer")
    else:
        merged["model_stability_score"] = 100.0
        merged["model_stability_level"] = "not_evaluated"
        merged["model_stability_warnings"] = "none"

    if not explanation_readiness.empty:
        merged = merged.merge(explanation_readiness, on="dataset", how="outer")
    else:
        merged["explanation_readiness_score"] = 100.0
        merged["explanation_readiness_level"] = "not_evaluated"
        merged["explanation_readiness_warnings"] = "none"

    merged["quality_score"] = merged["quality_score"].fillna(0.0)
    merged["complexity_score"] = merged["complexity_score"].fillna(0.0)
    merged["stability_score"] = merged["stability_score"].fillna(0.0)
    merged["leakage_score"] = merged["leakage_score"].fillna(100.0)
    merged["model_stability_score"] = merged["model_stability_score"].fillna(100.0)
    merged["explanation_readiness_score"] = merged["explanation_readiness_score"].fillna(100.0)

    merged["overall_readiness_score"] = merged.apply(
        lambda row: compute_overall_readiness_score(
            row["quality_score"],
            row["complexity_score"],
            row["stability_score"],
            row["leakage_score"],
            row["model_stability_score"],
            row["explanation_readiness_score"],
        ),
        axis=1,
    )

    merged["readiness_level"] = merged["overall_readiness_score"].apply(score_level)
    merged["quality_level"] = merged["quality_score"].apply(score_level)
    merged["complexity_level"] = merged["complexity_score"].apply(normalize_complexity_level)
    merged["stability_level"] = merged["stability_score"].apply(score_level)
    merged["leakage_level"] = merged["leakage_score"].apply(leakage_level_from_score)
    merged["model_stability_level"] = merged["model_stability_score"].apply(
        model_stability_level_from_score
    )
    merged["explanation_readiness_level"] = merged["explanation_readiness_score"].apply(
        explanation_readiness_level_from_score
    )

    merged["prediction_readiness"] = merged.apply(prediction_readiness_label, axis=1)
    merged["recommended_protocol"] = merged.apply(recommend_protocol, axis=1)
    merged["recommended_metrics"] = merged.apply(
        lambda row: ";".join(recommend_metrics(row)),
        axis=1,
    )
    merged["primary_risks"] = merged.apply(
        lambda row: unique_join(infer_primary_risks(row)),
        axis=1,
    )
    merged["warnings"] = merged.apply(combine_warning_columns, axis=1)

    for column in READINESS_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan

    return merged[READINESS_COLUMNS].sort_values(
        by="overall_readiness_score",
        ascending=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quality",
        type=str,
        default="outputs/profiles/DAQUA_quality_summary_by_dataset.csv",
    )
    parser.add_argument(
        "--complexity",
        type=str,
        default="outputs/profiles/DAQUA_complexity_summary_by_dataset.csv",
    )
    parser.add_argument(
        "--stability",
        type=str,
        default="outputs/profiles/DAQUA_stability_summary_by_dataset.csv",
    )
    parser.add_argument(
        "--leakage",
        type=str,
        default="outputs/profiles/DAQUA_leakage_summary_by_dataset.csv",
    )
    parser.add_argument(
        "--model-stability",
        type=str,
        default="outputs/profiles/DAQUA_model_ranking_stability.csv",
    )
    parser.add_argument(
        "--explanation-readiness",
        type=str,
        default="outputs/profiles/DAQUA_explanation_readiness.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/profiles/DAQUA_readiness_scores.csv",
    )

    args = parser.parse_args()

    quality = load_summary_csv(args.quality, "Quality")
    complexity = load_summary_csv(args.complexity, "Complexity")
    stability = load_summary_csv(args.stability, "Stability")

    leakage = None
    leakage_path = Path(args.leakage)
    if leakage_path.exists():
        leakage = load_summary_csv(args.leakage, "Leakage")
    else:
        logger.warning("Leakage summary not found. Computing readiness without leakage: %s", args.leakage)

    model_stability = None
    model_stability_path = Path(args.model_stability)
    if model_stability_path.exists():
        model_stability = load_summary_csv(args.model_stability, "Model stability")
    else:
        logger.warning(
            "Model stability summary not found. Computing readiness without model stability: %s",
            args.model_stability,
        )

    explanation_readiness = None
    explanation_readiness_path = Path(args.explanation_readiness)
    if explanation_readiness_path.exists():
        explanation_readiness = load_summary_csv(
            args.explanation_readiness,
            "Explanation readiness",
        )
    else:
        logger.warning(
            "Explanation readiness summary not found. Computing readiness without explanation readiness: %s",
            args.explanation_readiness,
        )

    readiness = compute_readiness(
        quality_summary=quality,
        complexity_summary=complexity,
        stability_summary=stability,
        leakage_summary=leakage,
        model_stability_summary=model_stability,
        explanation_readiness_summary=explanation_readiness,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    readiness.to_csv(args.out, index=False)

    print(f"Saved DAQUA readiness scores to: {args.out}")
    print(
        readiness[
            [
                "dataset",
                "quality_score",
                "complexity_score",
                "stability_score",
                "leakage_score",
                "model_stability_score",
                "explanation_readiness_score",
                "overall_readiness_score",
                "readiness_level",
                "prediction_readiness",
                "primary_risks",
            ]
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    main()