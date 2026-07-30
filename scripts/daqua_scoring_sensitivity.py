from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROFILE_DIR = Path("outputs/profiles_9datasets_full")
OUT_DIR = Path("outputs/robustness_9datasets/scoring_sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIMENSIONS = [
    "quality",
    "complexity",
    "stability",
    "leakage",
    "model_stability",
    "explanation_readiness",
]

DEFAULT_DIMENSION_WEIGHTS = {
    "quality": 0.20,
    "complexity": 0.18,
    "stability": 0.18,
    "leakage": 0.17,
    "model_stability": 0.12,
    "explanation_readiness": 0.15,
}

DEFAULT_OPERATIONAL_BOUNDARIES = {
    "ready": 85.0,
    "ready_with_reporting_controls": 75.0,
    "usable_with_caution": 60.0,
    "limited_readiness": 45.0,
}

# Each tuple:
# dimension, indicator, raw-column, orientation, saturation-threshold, default-weight
# orientation: higher_is_better or lower_is_better
INDICATOR_SPECS = [
    # Data quality
    ("quality", "row_retention_rate", "row_retention_rate", "higher_is_better", 1.00, 0.25),
    ("quality", "minority_class_percentage", "minority_class_percentage", "higher_is_better", 0.50, 0.20),
    ("quality", "class_imbalance_ratio", "class_imbalance_ratio", "lower_is_better", 20.00, 0.20),
    ("quality", "high_correlation_feature_rate", "high_correlation_feature_rate", "lower_is_better", 1.00, 0.20),
    ("quality", "outlier_instance_rate", "outlier_instance_rate", "lower_is_better", 1.00, 0.15),

    # Data complexity
    ("complexity", "minority_class_percentage", "minority_class_percentage", "higher_is_better", 0.50, 0.20),
    ("complexity", "nn_same_class_ratio", "nn_same_class_ratio", "higher_is_better", 1.00, 0.25),
    ("complexity", "minority_nn_same_class_ratio", "minority_nn_same_class_ratio", "higher_is_better", 1.00, 0.25),
    ("complexity", "borderline_instance_rate", "borderline_instance_rate", "lower_is_better", 1.00, 0.20),
    ("complexity", "mean_feature_overlap", "mean_feature_overlap", "lower_is_better", 1.00, 0.10),

    # Data stability
    ("stability", "project_size_cv", "project_size_cv", "lower_is_better", 2.00, 0.25),
    ("stability", "feature_set_consistency", "feature_set_consistency", "higher_is_better", 1.00, 0.25),
    ("stability", "label_prevalence_range", "label_prevalence_range", "lower_is_better", 1.00, 0.25),
    ("stability", "mean_pairwise_ks", "mean_pairwise_ks", "lower_is_better", 1.00, 0.15),
    ("stability", "mean_pairwise_stability_score", "mean_pairwise_stability_score", "higher_is_better", 100.00, 0.10),

    # Leakage readiness
    ("leakage", "mean_suspicious_feature_rate", "mean_suspicious_feature_rate", "lower_is_better", 1.00, 0.25),
    ("leakage", "mean_temporal_feature_count", "mean_temporal_feature_count", "lower_is_better", 5.00, 0.20),
    ("leakage", "mean_post_release_feature_count", "mean_post_release_feature_count", "lower_is_better", 3.00, 0.25),
    ("leakage", "mean_high_label_correlation_count", "mean_high_label_correlation_count", "lower_is_better", 3.00, 0.20),
    ("leakage", "mean_cross_project_duplicate_rate", "mean_cross_project_duplicate_rate", "lower_is_better", 0.10, 0.10),

    # Model-ranking stability
    ("model_stability", "mean_pairwise_spearman_auc", "mean_pairwise_spearman_auc", "higher_is_better_corr", 1.00, 0.30),
    ("model_stability", "mean_pairwise_kendall_auc", "mean_pairwise_kendall_auc", "higher_is_better_corr", 1.00, 0.25),
    ("model_stability", "mean_pairwise_spearman_mcc", "mean_pairwise_spearman_mcc", "higher_is_better_corr", 1.00, 0.25),
    ("model_stability", "mean_pairwise_kendall_mcc", "mean_pairwise_kendall_mcc", "higher_is_better_corr", 1.00, 0.20),

    # Explanation readiness
    ("explanation_readiness", "mean_project_feature_importance_stability", "mean_project_feature_importance_stability", "higher_is_better", 100.00, 0.35),
    ("explanation_readiness", "mean_pairwise_project_top10_jaccard", "mean_pairwise_project_top10_jaccard", "higher_is_better", 1.00, 0.25),
    ("explanation_readiness", "mean_pairwise_project_spearman", "mean_pairwise_project_spearman", "higher_is_better_corr", 1.00, 0.20),
    ("explanation_readiness", "dataset_feature_importance_stability_score", "dataset_feature_importance_stability_score", "higher_is_better", 100.00, 0.20),
]


def read_csv(name: str) -> pd.DataFrame:
    path = PROFILE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_inputs() -> Dict[str, pd.DataFrame]:
    return {
        "readiness": read_csv("DAQUA_readiness_scores.csv"),
        "quality": read_csv("DAQUA_quality_summary_by_dataset.csv"),
        "complexity": read_csv("DAQUA_complexity_summary_by_dataset.csv"),
        "stability": read_csv("DAQUA_stability_summary_by_dataset.csv"),
        "leakage": read_csv("DAQUA_leakage_summary_by_dataset.csv"),
        "model_stability": read_csv("DAQUA_model_ranking_stability.csv"),
        "explanation_readiness": read_csv("DAQUA_explanation_readiness.csv"),
    }


def canonical_dataset_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "dataset" not in df.columns:
        raise ValueError("Expected column 'dataset'")
    df["dataset"] = df["dataset"].astype(str)
    return df


def build_indicator_table(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for dim, indicator, raw_col, orientation, tau, weight in INDICATOR_SPECS:
        df = canonical_dataset_names(inputs[dim])

        if raw_col not in df.columns:
            continue

        for _, r in df.iterrows():
            raw_value = r[raw_col]
            rows.append({
                "dataset": r["dataset"],
                "dimension": dim,
                "indicator": indicator,
                "raw_value": raw_value,
                "orientation": orientation,
                "saturation_threshold": tau,
                "indicator_weight": weight,
            })

    out = pd.DataFrame(rows)
    return out


def clipped_linear(raw: float, orientation: str, tau: float) -> float:
    if pd.isna(raw):
        return np.nan

    raw = float(raw)

    if orientation == "higher_is_better":
        return max(0.0, min(1.0, raw / tau))

    if orientation == "lower_is_better":
        return max(0.0, min(1.0, 1.0 - raw / tau))

    if orientation == "higher_is_better_corr":
        # correlation values are in [-1, 1]. Map to [0, 1].
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))

    raise ValueError(f"Unknown orientation: {orientation}")


def logistic_norm(raw: float, orientation: str, tau: float) -> float:
    if pd.isna(raw):
        return np.nan

    raw = float(raw)

    if orientation == "higher_is_better_corr":
        x = (raw + 1.0) / 2.0
        return max(0.0, min(1.0, x))

    if tau == 0:
        return np.nan

    x = raw / tau

    if orientation == "higher_is_better":
        z = 8.0 * (x - 0.5)
        return 1.0 / (1.0 + math.exp(-z))

    if orientation == "lower_is_better":
        z = 8.0 * (x - 0.5)
        return 1.0 - (1.0 / (1.0 + math.exp(-z)))

    raise ValueError(f"Unknown orientation: {orientation}")


def add_quantile_scores(indicators: pd.DataFrame) -> pd.DataFrame:
    df = indicators.copy()
    df["normalized"] = np.nan

    for indicator, sub in df.groupby("indicator"):
        values = pd.to_numeric(sub["raw_value"], errors="coerce")
        ranks = values.rank(method="average", pct=True)

        for idx, rank_value in ranks.items():
            orientation = df.loc[idx, "orientation"]
            if pd.isna(rank_value):
                score = np.nan
            elif orientation in {"higher_is_better", "higher_is_better_corr"}:
                score = float(rank_value)
            elif orientation == "lower_is_better":
                score = float(1.0 - rank_value)
            else:
                raise ValueError(f"Unknown orientation: {orientation}")
            df.loc[idx, "normalized"] = score

    return df


def normalize_indicators(
    indicators: pd.DataFrame,
    normalization: str = "clipped",
    threshold_factor: float = 1.0,
) -> pd.DataFrame:
    df = indicators.copy()

    if normalization == "quantile":
        return add_quantile_scores(df)

    vals = []
    for _, r in df.iterrows():
        tau = float(r["saturation_threshold"]) * threshold_factor
        raw = r["raw_value"]
        orientation = r["orientation"]

        if normalization == "clipped":
            vals.append(clipped_linear(raw, orientation, tau))
        elif normalization == "logistic":
            vals.append(logistic_norm(raw, orientation, tau))
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

    df["normalized"] = vals
    return df


def renormalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(weights.values())
    if s == 0:
        raise ValueError("Cannot renormalize zero-sum weights.")
    return {k: v / s for k, v in weights.items()}


def compute_dimension_scores(
    indicators: pd.DataFrame,
    equal_indicator_weights: bool = False,
    perturb_indicator: Tuple[str, str, float] | None = None,
) -> pd.DataFrame:
    df = indicators.copy()

    rows = []
    for (dataset, dimension), sub in df.groupby(["dataset", "dimension"]):
        sub = sub.copy()

        weights = {}
        for _, r in sub.iterrows():
            key = r["indicator"]
            weights[key] = float(r["indicator_weight"])

        if equal_indicator_weights:
            weights = {k: 1.0 for k in weights}

        if perturb_indicator is not None:
            p_dim, p_indicator, factor = perturb_indicator
            if p_dim == dimension and p_indicator in weights:
                weights[p_indicator] *= factor

        weights = renormalize_weights(weights)

        score = 0.0
        used = 0.0

        for _, r in sub.iterrows():
            val = r["normalized"]
            if pd.isna(val):
                continue
            w = weights[r["indicator"]]
            score += w * float(val)
            used += w

        if used > 0:
            score = 100.0 * score / used
        else:
            score = np.nan

        rows.append({
            "dataset": dataset,
            "dimension": dimension,
            "score": score,
        })

    long = pd.DataFrame(rows)
    wide = long.pivot(index="dataset", columns="dimension", values="score").reset_index()

    for dim in DIMENSIONS:
        if dim not in wide.columns:
            wide[dim] = np.nan

    return wide[["dataset"] + DIMENSIONS]


def operational_label(score: float, boundaries: Dict[str, float]) -> str:
    if pd.isna(score):
        return "not_available"
    if score >= boundaries["ready"]:
        return "ready"
    if score >= boundaries["ready_with_reporting_controls"]:
        return "ready_with_reporting_controls"
    if score >= boundaries["usable_with_caution"]:
        return "usable_with_caution"
    if score >= boundaries["limited_readiness"]:
        return "limited_readiness"
    return "not_ready"


def compute_overall(
    dimension_scores: pd.DataFrame,
    dim_weights: Dict[str, float],
    boundaries: Dict[str, float],
    config_name: str,
    config_group: str,
) -> pd.DataFrame:
    weights = renormalize_weights(dim_weights)
    df = dimension_scores.copy()

    for dim in DIMENSIONS:
        if dim not in df.columns:
            df[dim] = np.nan

    scores = []
    for _, r in df.iterrows():
        numerator = 0.0
        denominator = 0.0

        for dim in DIMENSIONS:
            val = r[dim]
            if pd.isna(val):
                continue
            numerator += weights[dim] * float(val)
            denominator += weights[dim]

        overall = numerator / denominator if denominator > 0 else np.nan
        scores.append(overall)

    df["overall_score"] = scores
    df["readiness_label"] = df["overall_score"].apply(lambda x: operational_label(x, boundaries))
    df["lowest_dimension"] = df[DIMENSIONS].idxmin(axis=1)

    df["configuration"] = config_name
    df["configuration_group"] = config_group

    return df[
        [
            "configuration",
            "configuration_group",
            "dataset",
            "quality",
            "complexity",
            "stability",
            "leakage",
            "model_stability",
            "explanation_readiness",
            "overall_score",
            "readiness_label",
            "lowest_dimension",
        ]
    ]


def anchor_dimension_scores(
    official_default: pd.DataFrame,
    recomputed_default: pd.DataFrame,
    recomputed_alt: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert an alternative indicator-level scoring run into a DAQUA-consistent
    perturbation by applying only the alternative-minus-baseline delta to the
    official full-run DAQUA dimension scores.

    This avoids comparing approximate reimplementations of the DAQUA indicator
    formulas directly against the official DAQUA output.
    """
    base = official_default[["dataset"] + DIMENSIONS].copy()
    d0 = recomputed_default[["dataset"] + DIMENSIONS].copy()
    d1 = recomputed_alt[["dataset"] + DIMENSIONS].copy()

    merged = base.merge(d0, on="dataset", suffixes=("_official", "_recomputed_default"))
    merged = merged.merge(d1, on="dataset", suffixes=("", "_recomputed_alt"))

    out = pd.DataFrame()
    out["dataset"] = merged["dataset"]

    for dim in DIMENSIONS:
        official_col = f"{dim}_official"
        recomputed_default_col = f"{dim}_recomputed_default"
        recomputed_alt_col = dim

        official = pd.to_numeric(merged[official_col], errors="coerce")
        r0 = pd.to_numeric(merged[recomputed_default_col], errors="coerce")
        r1 = pd.to_numeric(merged[recomputed_alt_col], errors="coerce")

        delta = r1 - r0

        # If the recomputed indicator model cannot estimate a dimension
        # such as model stability for single-project families, retain the
        # official DAQUA dimension score.
        anchored = official.where(delta.isna(), official + delta)
        out[dim] = anchored.clip(lower=0.0, upper=100.0)

    return out


def default_scores_from_daqua(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = canonical_dataset_names(inputs["readiness"]).copy()

    rename = {
        "quality_score": "quality",
        "complexity_score": "complexity",
        "stability_score": "stability",
        "leakage_score": "leakage",
        "model_stability_score": "model_stability",
        "explanation_readiness_score": "explanation_readiness",
        "overall_readiness_score": "overall_score",
        "prediction_readiness": "readiness_label",
    }

    df = df.rename(columns=rename)

    for dim in DIMENSIONS:
        if dim not in df.columns:
            df[dim] = np.nan

    # Single-project families may have NA in model-ranking output, but final
    # DAQUA readiness file contains the operational value used by the pipeline.
    df["lowest_dimension"] = df[DIMENSIONS].idxmin(axis=1)
    df["configuration"] = "Default"
    df["configuration_group"] = "Default"

    return df[
        [
            "configuration",
            "configuration_group",
            "dataset",
            "quality",
            "complexity",
            "stability",
            "leakage",
            "model_stability",
            "explanation_readiness",
            "overall_score",
            "readiness_label",
            "lowest_dimension",
        ]
    ]


def compare_to_default(default_df: pd.DataFrame, alt_df: pd.DataFrame) -> Dict[str, object]:
    d = default_df[["dataset", "overall_score", "readiness_label", "lowest_dimension"]].copy()
    a = alt_df[["dataset", "overall_score", "readiness_label", "lowest_dimension"]].copy()

    m = d.merge(a, on="dataset", suffixes=("_default", "_alt"))

    if len(m) < 2:
        rank_corr = np.nan
    else:
        rank_corr = spearmanr(m["overall_score_default"], m["overall_score_alt"]).correlation

    m["abs_change"] = (m["overall_score_alt"] - m["overall_score_default"]).abs()
    changed = m.loc[m["readiness_label_default"] != m["readiness_label_alt"], "dataset"].tolist()

    return {
        "rank_correlation": rank_corr,
        "median_abs_score_change": float(m["abs_change"].median()),
        "max_abs_score_change": float(m["abs_change"].max()),
        "label_agreement": float((m["readiness_label_default"] == m["readiness_label_alt"]).mean() * 100.0),
        "changed_families": "; ".join(changed) if changed else "None",
        "n_changed_families": len(changed),
        "lowest_dimension_agreement": float((m["lowest_dimension_default"] == m["lowest_dimension_alt"]).mean() * 100.0),
    }


def summarize_config_groups(default_df: pd.DataFrame, all_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for group, sub in all_scores.groupby("configuration_group"):
        if group == "Default":
            continue

        per_config = []
        for config, cdf in sub.groupby("configuration"):
            metrics = compare_to_default(default_df, cdf)
            metrics["configuration"] = config
            per_config.append(metrics)

        tmp = pd.DataFrame(per_config)

        rows.append({
            "configuration": group,
            "rank_correlation": tmp["rank_correlation"].min(),
            "median_abs_score_change": tmp["median_abs_score_change"].max(),
            "max_abs_score_change": tmp["max_abs_score_change"].max(),
            "label_agreement": tmp["label_agreement"].min(),
            "changed_families": "; ".join(sorted(set(
                fam
                for x in tmp["changed_families"].tolist()
                for fam in str(x).split("; ")
                if fam and fam != "None"
            ))) or "None",
            "n_changed_families": int(tmp["n_changed_families"].max()),
            "lowest_dimension_agreement": tmp["lowest_dimension_agreement"].min(),
        })

    return pd.DataFrame(rows)


def make_latex_table(summary: pd.DataFrame, out_path: Path) -> None:
    order = [
        "Equal indicator weights",
        "Equal dimension weights",
        "Indicator weights $\\pm20\\%$",
        "Dimension weights $\\pm20\\%$",
        "Thresholds $-20\\%$",
        "Thresholds $+20\\%$",
        "Logistic normalization",
        "Quantile normalization",
        "Decision boundaries $-5$",
        "Decision boundaries $+5$",
    ]

    display = summary.copy()
    display["sort_key"] = display["configuration"].apply(lambda x: order.index(x) if x in order else 999)
    display = display.sort_values("sort_key")

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{Sensitivity of DAQUA conclusions to alternative audit configurations.}")
    lines.append(r"\label{tab:sensitivity}")
    lines.append(r"\begin{tabular}{lrrrrl}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Configuration} &")
    lines.append(r"\shortstack{\textbf{Rank}\\\textbf{correlation}} &")
    lines.append(r"\shortstack{\textbf{Median absolute}\\\textbf{score change}} &")
    lines.append(r"\shortstack{\textbf{Maximum absolute}\\\textbf{score change}} &")
    lines.append(r"\shortstack{\textbf{Label}\\\textbf{agreement}} &")
    lines.append(r"\textbf{Changed families} \\")
    lines.append(r"\midrule")

    for _, r in display.iterrows():
        config = r["configuration"]
        rank_corr = "--" if pd.isna(r["rank_correlation"]) else f"{r['rank_correlation']:.2f}"
        med = f"{r['median_abs_score_change']:.2f}"
        mx = f"{r['max_abs_score_change']:.2f}"
        agree = f"{r['label_agreement']:.1f}\\%"
        changed = str(r["changed_families"]).replace("_", r"\_")
        lines.append(f"{config} & {rank_corr} & {med} & {mx} & {agree} & {changed} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out_path.write_text("\n".join(lines))


def main() -> None:
    inputs = load_inputs()

    indicator_table = build_indicator_table(inputs)
    indicator_table.to_csv(OUT_DIR / "indicator_specification_table.csv", index=False)

    pd.DataFrame([
        {"dimension": k, "dimension_weight": v}
        for k, v in DEFAULT_DIMENSION_WEIGHTS.items()
    ]).to_csv(OUT_DIR / "dimension_weights.csv", index=False)

    pd.DataFrame([
        {"boundary": k, "score": v}
        for k, v in DEFAULT_OPERATIONAL_BOUNDARIES.items()
    ]).to_csv(OUT_DIR / "readiness_boundaries.csv", index=False)

    pd.DataFrame([
        {"warning_group": "severe", "threshold_rule": "dataset-specific severe warning logic from DAQUA readiness module"},
        {"warning_group": "moderate", "threshold_rule": "dataset-specific moderate warning logic from DAQUA readiness module"},
        {"warning_group": "low", "threshold_rule": "dataset-specific low warning logic from DAQUA readiness module"},
    ]).to_csv(OUT_DIR / "warning_thresholds_and_severity.csv", index=False)

    configs = []

    default_df = default_scores_from_daqua(inputs)
    configs.append(default_df)

    # Equal dimension weights
    eq_dim = DEFAULT_DIMENSION_WEIGHTS.copy()
    eq_dim = {k: 1.0 / len(DIMENSIONS) for k in DIMENSIONS}
    configs.append(compute_overall(
        default_df[["dataset"] + DIMENSIONS],
        eq_dim,
        DEFAULT_OPERATIONAL_BOUNDARIES,
        "Equal dimension weights",
        "Equal dimension weights",
    ))

    # Boundary perturbation: score unchanged, labels changed
    for shift in [-5.0, 5.0]:
        b = {k: v + shift for k, v in DEFAULT_OPERATIONAL_BOUNDARIES.items()}
        name = f"Decision boundaries {'-5' if shift < 0 else '+5'}"
        configs.append(compute_overall(
            default_df[["dataset"] + DIMENSIONS],
            DEFAULT_DIMENSION_WEIGHTS,
            b,
            name,
            name,
        ))

    # Dimension weight perturbation
    for dim in DIMENSIONS:
        for factor in [0.8, 1.2]:
            w = DEFAULT_DIMENSION_WEIGHTS.copy()
            w[dim] *= factor
            name = f"Dimension weight {dim} x{factor}"
            configs.append(compute_overall(
                default_df[["dataset"] + DIMENSIONS],
                w,
                DEFAULT_OPERATIONAL_BOUNDARIES,
                name,
                "Dimension weights $\\pm20\\%$",
            ))

    # Recomputed baseline for indicator-level perturbations.
    # Alternative indicator configurations are anchored to official DAQUA
    # dimension scores through alternative-minus-baseline deltas.
    norm_default = normalize_indicators(
        indicator_table,
        normalization="clipped",
        threshold_factor=1.0,
    )

    dim_scores_recomputed_default = compute_dimension_scores(
        norm_default,
        equal_indicator_weights=False,
    )

    # Indicator-level configurations
    for normalization, group_name, threshold_factor, equal_indicator in [
        ("clipped", "Equal indicator weights", 1.0, True),
        ("clipped", "Thresholds $-20\\%$", 0.8, False),
        ("clipped", "Thresholds $+20\\%$", 1.2, False),
        ("logistic", "Logistic normalization", 1.0, False),
        ("quantile", "Quantile normalization", 1.0, False),
    ]:
        norm = normalize_indicators(
            indicator_table,
            normalization=normalization,
            threshold_factor=threshold_factor,
        )

        dim_scores_raw_alt = compute_dimension_scores(
            norm,
            equal_indicator_weights=equal_indicator,
        )

        dim_scores = anchor_dimension_scores(
            official_default=default_df,
            recomputed_default=dim_scores_recomputed_default,
            recomputed_alt=dim_scores_raw_alt,
        )

        configs.append(compute_overall(
            dim_scores,
            DEFAULT_DIMENSION_WEIGHTS,
            DEFAULT_OPERATIONAL_BOUNDARIES,
            group_name,
            group_name,
        ))

    # Indicator weight perturbations
    unique_indicators = indicator_table[["dimension", "indicator"]].drop_duplicates()

    for _, row in unique_indicators.iterrows():
        dim = row["dimension"]
        ind = row["indicator"]

        for factor in [0.8, 1.2]:
            dim_scores_raw_alt = compute_dimension_scores(
                norm_default,
                equal_indicator_weights=False,
                perturb_indicator=(dim, ind, factor),
            )

            dim_scores = anchor_dimension_scores(
                official_default=default_df,
                recomputed_default=dim_scores_recomputed_default,
                recomputed_alt=dim_scores_raw_alt,
            )

            configs.append(compute_overall(
                dim_scores,
                DEFAULT_DIMENSION_WEIGHTS,
                DEFAULT_OPERATIONAL_BOUNDARIES,
                f"Indicator weight {dim}:{ind} x{factor}",
                "Indicator weights $\\pm20\\%$",
            ))

    all_scores = pd.concat(configs, ignore_index=True)
    all_scores.to_csv(OUT_DIR / "all_configuration_scores.csv", index=False)

    # Per-configuration comparison
    per_config_rows = []
    for config, cdf in all_scores.groupby("configuration"):
        if config == "Default":
            continue
        m = compare_to_default(default_df, cdf)
        m["configuration"] = config
        m["configuration_group"] = cdf["configuration_group"].iloc[0]
        per_config_rows.append(m)

    per_config = pd.DataFrame(per_config_rows)
    per_config.to_csv(OUT_DIR / "per_configuration_comparison.csv", index=False)

    summary = summarize_config_groups(default_df, all_scores)

    order = [
        "Equal indicator weights",
        "Equal dimension weights",
        "Indicator weights $\\pm20\\%$",
        "Dimension weights $\\pm20\\%$",
        "Thresholds $-20\\%$",
        "Thresholds $+20\\%$",
        "Logistic normalization",
        "Quantile normalization",
        "Decision boundaries $-5$",
        "Decision boundaries $+5$",
    ]

    summary["sort_key"] = summary["configuration"].apply(lambda x: order.index(x) if x in order else 999)
    summary = summary.sort_values("sort_key").drop(columns=["sort_key"])

    summary.to_csv(OUT_DIR / "sensitivity_summary.csv", index=False)
    make_latex_table(summary, OUT_DIR / "tab_sensitivity.tex")

    print("\nScoring sensitivity analysis completed.")
    print(f"Output directory: {OUT_DIR}")
    print("\nMain summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
