# daqua/reports/report_generator.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd


logger = logging.getLogger(__name__)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    return pd.read_csv(path)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Optional file not found: %s", path)
        return pd.DataFrame()

    return pd.read_csv(path)


def fmt(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"

    if isinstance(value, float):
        return f"{value:.{digits}f}"

    return str(value)


def existing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [col for col in columns if col in df.columns]


def markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    if df.empty:
        return "_No data available._\n"

    selected_columns = existing_columns(df, columns)

    if not selected_columns:
        return "_No requested columns available._\n"

    lines = []
    lines.append("| " + " | ".join(selected_columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(selected_columns)) + " |")

    for _, row in df[selected_columns].iterrows():
        values = [fmt(row[col]) for col in selected_columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def risk_bullets(readiness: pd.DataFrame) -> str:
    lines: List[str] = []

    for _, row in readiness.iterrows():
        dataset = row["dataset"]

        lines.append(f"### {dataset}")
        lines.append("")
        lines.append(f"- **Primary risks:** `{row.get('primary_risks', 'none')}`")
        lines.append(f"- **Warnings:** `{row.get('warnings', 'none')}`")
        lines.append(f"- **Recommended protocol:** `{row.get('recommended_protocol', 'NA')}`")
        lines.append(f"- **Recommended metrics:** `{row.get('recommended_metrics', 'NA')}`")
        lines.append("")

    return "\n".join(lines)


def family_phrase(count: int) -> str:
    if count == 1:
        return "1 dataset family"
    return f"{count} dataset families"


def interpretation_paragraph(readiness: pd.DataFrame) -> str:
    if readiness.empty:
        return "No readiness results were available for interpretation."

    best = readiness.sort_values("overall_readiness_score", ascending=False).iloc[0]
    worst = readiness.sort_values("overall_readiness_score", ascending=True).iloc[0]
    mean_score = readiness["overall_readiness_score"].mean()

    limited_low_count = int(
        readiness["readiness_level"].isin(["limited", "low"]).sum()
    )

    leakage_risk_count = 0
    if "leakage_score" in readiness.columns:
        leakage_risk_count = int((readiness["leakage_score"] < 90).sum())

    model_instability_count = 0
    if "model_stability_score" in readiness.columns:
        model_instability_count = int((readiness["model_stability_score"] < 85).sum())

    explanation_risk_count = 0
    if "explanation_readiness_score" in readiness.columns:
        explanation_risk_count = int(
            (readiness["explanation_readiness_score"] < 70).sum()
        )

    text = (
        f"Across the analysed defect dataset families, the mean DAQUA readiness score was "
        f"{mean_score:.2f}. The highest-ranked dataset was {best['dataset']} "
        f"with a readiness score of {best['overall_readiness_score']:.2f}, while the "
        f"lowest-ranked dataset was {worst['dataset']} with a score of "
        f"{worst['overall_readiness_score']:.2f}. "
    )

    if limited_low_count:
        text += (
            f"{family_phrase(limited_low_count)} showed limited or low readiness, indicating "
            f"that model-comparison results on these datasets should be interpreted with "
            f"explicit controls for data quality, complexity, stability, leakage, model-ranking "
            f"stability, and explanation reliability. "
        )

    if leakage_risk_count:
        text += (
            f"{family_phrase(leakage_risk_count)} showed nontrivial leakage warnings, which "
            f"should be treated as audit signals rather than definitive evidence of leakage. "
        )

    if model_instability_count:
        text += (
            f"{family_phrase(model_instability_count)} showed less than high model-ranking "
            f"stability, suggesting that claims about the best model should be reported with "
            f"metric sensitivity and project-level stability analysis. "
        )

    if explanation_risk_count:
        text += (
            f"{family_phrase(explanation_risk_count)} showed limited or low explanation "
            f"readiness, indicating that feature-importance explanations should be treated as "
            f"dataset-conditional rather than universally stable explanations. "
        )

    text += (
        "These findings support the DAQUA argument that software defect prediction datasets "
        "should be audited before algorithmic comparison, because predictive performance, "
        "model ranking, explanation reliability, leakage sensitivity, and cross-project "
        "generalization are conditioned by measurable dataset properties."
    )

    return text


def build_report(profiles_dir: Path, out_path: Path) -> None:
    readiness_path = profiles_dir / "DAQUA_readiness_scores.csv"
    quality_path = profiles_dir / "DAQUA_quality_summary_by_dataset.csv"
    complexity_path = profiles_dir / "DAQUA_complexity_summary_by_dataset.csv"
    stability_path = profiles_dir / "DAQUA_stability_summary_by_dataset.csv"
    leakage_path = profiles_dir / "DAQUA_leakage_summary_by_dataset.csv"
    model_stability_path = profiles_dir / "DAQUA_model_ranking_stability.csv"
    explanation_path = profiles_dir / "DAQUA_explanation_readiness.csv"
    loaded_path = profiles_dir / "DAQUA_loaded_projects.csv"

    readiness = read_csv(readiness_path)
    quality = read_csv(quality_path)
    complexity = read_csv(complexity_path)
    stability = read_csv(stability_path)
    leakage = read_optional_csv(leakage_path)
    model_stability = read_optional_csv(model_stability_path)
    explanation = read_optional_csv(explanation_path)
    loaded = read_csv(loaded_path)

    readiness = readiness.sort_values("overall_readiness_score", ascending=False)

    lines: List[str] = []

    lines.append("# DAQUA Summary Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "DAQUA evaluates whether software defect datasets are reliable, stable, "
        "leakage-aware, model-ranking-stable, explanation-stability-aware, and suitable for "
        "trustworthy prediction before model comparison."
    )
    lines.append("")
    lines.append(f"- Number of dataset families: **{readiness['dataset'].nunique()}**")
    lines.append(f"- Number of projects: **{loaded['project'].nunique()}**")
    lines.append("")

    lines.append("## Readiness Ranking")
    lines.append("")
    lines.append(
        markdown_table(
            readiness,
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
            ],
        )
    )

    lines.append("")
    lines.append("## Quality Summary")
    lines.append("")
    lines.append(
        markdown_table(
            quality,
            [
                "dataset",
                "n_projects",
                "row_retention_rate",
                "minority_class_percentage",
                "class_imbalance_ratio",
                "high_correlation_feature_rate",
                "outlier_instance_rate",
                "quality_score",
            ],
        )
    )

    lines.append("")
    lines.append("## Complexity Summary")
    lines.append("")
    lines.append(
        markdown_table(
            complexity,
            [
                "dataset",
                "n_projects",
                "minority_class_percentage",
                "nn_same_class_ratio",
                "minority_nn_same_class_ratio",
                "borderline_instance_rate",
                "mean_feature_overlap",
                "complexity_score",
            ],
        )
    )

    lines.append("")
    lines.append("## Stability Summary")
    lines.append("")
    lines.append(
        markdown_table(
            stability,
            [
                "dataset",
                "n_projects",
                "project_size_cv",
                "feature_set_consistency",
                "label_prevalence_range",
                "mean_pairwise_ks",
                "mean_pairwise_stability_score",
                "stability_score",
            ],
        )
    )

    lines.append("")
    lines.append("## Leakage Summary")
    lines.append("")
    lines.append(
        markdown_table(
            leakage,
            [
                "dataset",
                "n_projects",
                "mean_suspicious_feature_rate",
                "mean_temporal_feature_count",
                "mean_post_release_feature_count",
                "mean_high_label_correlation_count",
                "mean_cross_project_duplicate_rate",
                "mean_leakage_score",
                "leakage_level",
                "warnings",
            ],
        )
    )

    lines.append("")
    lines.append("## Model-Ranking Stability Summary")
    lines.append("")
    lines.append(
        markdown_table(
            model_stability,
            [
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
            ],
        )
    )

    lines.append("")
    lines.append("## Explanation Readiness Summary")
    lines.append("")
    lines.append(
        markdown_table(
            explanation,
            [
                "dataset",
                "n_projects",
                "mean_project_feature_importance_stability",
                "mean_pairwise_project_top10_jaccard",
                "mean_pairwise_project_spearman",
                "dataset_feature_importance_stability_score",
                "explanation_readiness_score",
                "explanation_readiness_level",
                "primary_risks",
                "warnings",
            ],
        )
    )

    lines.append("")
    lines.append("## Dataset Risks and Recommendations")
    lines.append("")
    lines.append(risk_bullets(readiness))

    lines.append("")
    lines.append("## Research Interpretation")
    lines.append("")
    lines.append(interpretation_paragraph(readiness))
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Saved DAQUA report to: %s", out_path)
    print(f"Saved DAQUA report to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profiles-dir",
        type=str,
        default="outputs/profiles",
        help="Directory containing DAQUA profile CSV files.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="outputs/reports/DAQUA_summary_report.md",
        help="Output Markdown report path.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    build_report(
        profiles_dir=Path(args.profiles_dir),
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()