#daqua/main.py
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict

import pandas as pd

from daqua.explainability.explanation_stability import compute_explanation_readiness
from daqua.explainability.feature_importance import (
    compute_all_feature_importances,
    summarize_dataset_feature_importance_stability,
    summarize_project_feature_importance_stability,
)
from daqua.loaders.defect_loader import load_all_projects, projects_to_metadata_frame
from daqua.models.baseline_models import (
    evaluate_all_baselines,
    summarize_model_performance,
    summarize_ranking_stability,
)
from daqua.profiling.complexity import (
    profile_complexity,
    summarize_complexity_by_dataset,
)
from daqua.profiling.leakage import (
    profile_leakage,
    summarize_leakage_by_dataset,
)
from daqua.profiling.quality import (
    profile_quality,
    summarize_quality_by_dataset,
)
from daqua.profiling.readiness import compute_readiness
from daqua.profiling.stability import (
    profile_pairwise_stability,
    summarize_stability_by_dataset,
)


logger = logging.getLogger(__name__)


def ensure_output_dirs(out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = out_dir.parent / "reports"
    logs_dir = out_dir.parent / "logs"

    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "profiles": out_dir,
        "reports": reports_dir,
        "logs": logs_dir,
    }


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved: %s", path)


def run_daqua_pipeline(
    root: str,
    out_dir: str,
    include_model_stability: bool = True,
    include_explanation_readiness: bool = True,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> Dict[str, Path]:
    start_time = time.time()

    paths = ensure_output_dirs(Path(out_dir))
    profile_dir = paths["profiles"]

    logger.info("Starting DAQUA pipeline")
    logger.info("Dataset root: %s", root)
    logger.info("Output directory: %s", profile_dir)

    projects = load_all_projects(root)

    if not projects:
        raise RuntimeError(f"No projects loaded from root: {root}")

    metadata = projects_to_metadata_frame(projects)
    metadata_path = profile_dir / "DAQUA_loaded_projects.csv"
    save_csv(metadata, metadata_path)

    logger.info("Running data quality profiling")
    quality_profile = profile_quality(projects)
    quality_summary = summarize_quality_by_dataset(quality_profile)

    quality_profile_path = profile_dir / "DAQUA_quality_profile.csv"
    quality_summary_path = profile_dir / "DAQUA_quality_summary_by_dataset.csv"

    save_csv(quality_profile, quality_profile_path)
    save_csv(quality_summary, quality_summary_path)

    logger.info("Running data complexity profiling")
    complexity_profile = profile_complexity(projects)
    complexity_summary = summarize_complexity_by_dataset(complexity_profile)

    complexity_profile_path = profile_dir / "DAQUA_complexity_profile.csv"
    complexity_summary_path = profile_dir / "DAQUA_complexity_summary_by_dataset.csv"

    save_csv(complexity_profile, complexity_profile_path)
    save_csv(complexity_summary, complexity_summary_path)

    logger.info("Running data stability profiling")
    stability_pairwise = profile_pairwise_stability(projects)
    stability_summary = summarize_stability_by_dataset(projects, stability_pairwise)

    stability_pairwise_path = profile_dir / "DAQUA_stability_pairwise_profile.csv"
    stability_summary_path = profile_dir / "DAQUA_stability_summary_by_dataset.csv"

    save_csv(stability_pairwise, stability_pairwise_path)
    save_csv(stability_summary, stability_summary_path)

    logger.info("Running data leakage profiling")
    leakage_profile = profile_leakage(projects)
    leakage_summary = summarize_leakage_by_dataset(leakage_profile)

    leakage_profile_path = profile_dir / "DAQUA_leakage_profile.csv"
    leakage_summary_path = profile_dir / "DAQUA_leakage_summary_by_dataset.csv"

    save_csv(leakage_profile, leakage_profile_path)
    save_csv(leakage_summary, leakage_summary_path)

    model_results_path = profile_dir / "DAQUA_baseline_model_results.csv"
    model_summary_path = profile_dir / "DAQUA_baseline_model_summary.csv"
    model_ranking_path = profile_dir / "DAQUA_model_ranking_stability.csv"
    model_ranking = None

    if include_model_stability:
        logger.info(
            "Running baseline model-ranking stability profiling | folds=%d repeats=%d",
            n_splits,
            n_repeats,
        )

        model_results = evaluate_all_baselines(
            projects=projects,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        model_summary = summarize_model_performance(model_results)
        model_ranking = summarize_ranking_stability(model_summary)

        save_csv(model_results, model_results_path)
        save_csv(model_summary, model_summary_path)
        save_csv(model_ranking, model_ranking_path)
    else:
        logger.info("Skipping baseline model-ranking stability profiling")

    feature_importance_path = profile_dir / "DAQUA_feature_importance_profile.csv"
    feature_project_stability_path = profile_dir / "DAQUA_feature_importance_project_stability.csv"
    feature_dataset_stability_path = profile_dir / "DAQUA_feature_importance_dataset_stability.csv"
    explanation_readiness_path = profile_dir / "DAQUA_explanation_readiness.csv"
    explanation_readiness = None

    if include_explanation_readiness:
        logger.info(
            "Running feature-importance and explanation-readiness profiling | folds=%d repeats=%d",
            n_splits,
            n_repeats,
        )

        feature_importance = compute_all_feature_importances(
            projects=projects,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        feature_project_stability = summarize_project_feature_importance_stability(
            feature_importance
        )

        feature_dataset_stability = summarize_dataset_feature_importance_stability(
            importances=feature_importance,
            project_stability=feature_project_stability,
        )

        explanation_readiness = compute_explanation_readiness(
            feature_dataset_stability
        )

        save_csv(feature_importance, feature_importance_path)
        save_csv(feature_project_stability, feature_project_stability_path)
        save_csv(feature_dataset_stability, feature_dataset_stability_path)
        save_csv(explanation_readiness, explanation_readiness_path)
    else:
        logger.info("Skipping explanation-readiness profiling")

    logger.info("Computing DAQUA readiness scores")
    readiness = compute_readiness(
        quality_summary=quality_summary,
        complexity_summary=complexity_summary,
        stability_summary=stability_summary,
        leakage_summary=leakage_summary,
        model_stability_summary=model_ranking,
        explanation_readiness_summary=explanation_readiness,
    )

    readiness_path = profile_dir / "DAQUA_readiness_scores.csv"
    save_csv(readiness, readiness_path)

    elapsed = time.time() - start_time
    logger.info("DAQUA pipeline completed in %.2f seconds", elapsed)

    print("\nDAQUA readiness summary")
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
            ]
        ].round(4).to_string(index=False)
    )

    generated_paths = [
        metadata_path,
        quality_profile_path,
        quality_summary_path,
        complexity_profile_path,
        complexity_summary_path,
        stability_pairwise_path,
        stability_summary_path,
        leakage_profile_path,
        leakage_summary_path,
    ]

    if include_model_stability:
        generated_paths.extend(
            [
                model_results_path,
                model_summary_path,
                model_ranking_path,
            ]
        )

    if include_explanation_readiness:
        generated_paths.extend(
            [
                feature_importance_path,
                feature_project_stability_path,
                feature_dataset_stability_path,
                explanation_readiness_path,
            ]
        )

    generated_paths.append(readiness_path)

    print("\nGenerated files")
    for path in generated_paths:
        print(f"- {path}")

    outputs = {
        "metadata": metadata_path,
        "quality_profile": quality_profile_path,
        "quality_summary": quality_summary_path,
        "complexity_profile": complexity_profile_path,
        "complexity_summary": complexity_summary_path,
        "stability_pairwise": stability_pairwise_path,
        "stability_summary": stability_summary_path,
        "leakage_profile": leakage_profile_path,
        "leakage_summary": leakage_summary_path,
        "readiness": readiness_path,
    }

    if include_model_stability:
        outputs.update(
            {
                "baseline_model_results": model_results_path,
                "baseline_model_summary": model_summary_path,
                "model_ranking_stability": model_ranking_path,
            }
        )

    if include_explanation_readiness:
        outputs.update(
            {
                "feature_importance": feature_importance_path,
                "feature_importance_project_stability": feature_project_stability_path,
                "feature_importance_dataset_stability": feature_dataset_stability_path,
                "explanation_readiness": explanation_readiness_path,
            }
        )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DAQUA: Data Quality and Stability Readiness Framework for Software Defect Prediction"
    )

    parser.add_argument(
        "--root",
        type=str,
        default="Data-set",
        help="Root folder containing defect dataset folders.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/profiles",
        help="Directory where DAQUA output CSV files will be saved.",
    )

    parser.add_argument(
        "--skip-model-stability",
        action="store_true",
        help="Skip baseline model-ranking stability profiling.",
    )

    parser.add_argument(
        "--skip-explanation-readiness",
        action="store_true",
        help="Skip feature-importance and explanation-readiness profiling.",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified CV folds.",
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Number of repeated CV runs.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    run_daqua_pipeline(
        root=args.root,
        out_dir=args.out_dir,
        include_model_stability=not args.skip_model_stability,
        include_explanation_readiness=not args.skip_explanation_readiness,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()