from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


PROFILE_DIR = Path("outputs/profiles_9datasets_full")
OUT_DIR = Path("outputs/robustness_9datasets/model_ranking")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = PROFILE_DIR / "DAQUA_baseline_model_results.csv"
DEFAULT_STABILITY_PATH = PROFILE_DIR / "DAQUA_model_ranking_stability.csv"


DATASET_COL_CANDIDATES = ["dataset", "dataset_family"]
PROJECT_COL_CANDIDATES = ["project", "project_name", "system", "release"]
MODEL_COL_CANDIDATES = ["model", "classifier", "classifier_name", "algorithm", "learner"]

METRIC_CANDIDATES = {
    "AUC": ["auc", "AUC", "roc_auc", "ROC_AUC", "test_auc", "mean_auc"],
    "MCC": ["mcc", "MCC", "test_mcc", "mean_mcc"],
    "PR_AUC": [
        "pr_auc",
        "PR_AUC",
        "prauc",
        "PR-AUC",
        "average_precision",
        "AveragePrecision",
        "average_precision_score",
        "ap",
        "AP",
        "test_pr_auc",
        "mean_pr_auc",
    ],
}


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    lookup = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    if required:
        raise ValueError(f"Could not find any of columns: {candidates}\nAvailable columns: {list(df.columns)}")
    return None


def find_metric_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    out = {}
    for metric, candidates in METRIC_CANDIDATES.items():
        out[metric] = find_col(df, candidates, required=False)
    return out


def safe_corr(x: List[float], y: List[float], method: str) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan

    if method == "spearman":
        return spearmanr(x, y).correlation

    if method == "kendall":
        return kendalltau(x, y).correlation

    raise ValueError(method)


def mapped_corr_score(rho: float, tau: float) -> float:
    vals = []
    if pd.notna(rho):
        vals.append((rho + 1.0) / 2.0)
    if pd.notna(tau):
        vals.append((tau + 1.0) / 2.0)

    if not vals:
        return np.nan

    return float(np.mean(vals) * 100.0)


def prepare_project_model_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    dataset_col = find_col(df, DATASET_COL_CANDIDATES)
    project_col = find_col(df, PROJECT_COL_CANDIDATES)
    model_col = find_col(df, MODEL_COL_CANDIDATES)
    metric_cols = find_metric_cols(df)

    keep = [dataset_col, project_col, model_col] + [c for c in metric_cols.values() if c is not None]
    tmp = df[keep].copy()

    rename = {
        dataset_col: "dataset",
        project_col: "project",
        model_col: "model",
    }

    for metric, col in metric_cols.items():
        if col is not None:
            rename[col] = metric

    tmp = tmp.rename(columns=rename)

    for metric in METRIC_CANDIDATES:
        if metric not in tmp.columns:
            tmp[metric] = np.nan
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")

    # If the file contains fold/repeat rows, aggregate to one value per
    # dataset-project-model.
    agg = (
        tmp.groupby(["dataset", "project", "model"], as_index=False)
        [["AUC", "MCC", "PR_AUC"]]
        .mean()
    )

    return agg, metric_cols


def rank_models_within_project(metrics: pd.DataFrame, metric: str, allowed_models: Optional[List[str]] = None) -> pd.DataFrame:
    df = metrics.copy()

    if allowed_models is not None:
        df = df[df["model"].isin(allowed_models)].copy()

    df = df[pd.notna(df[metric])].copy()

    if df.empty:
        return df.assign(rank=np.nan)

    df["rank"] = (
        df.groupby(["dataset", "project"])[metric]
        .rank(method="average", ascending=False)
    )

    return df


def compute_metric_stability(metrics: pd.DataFrame, metric: str, allowed_models: Optional[List[str]] = None) -> pd.DataFrame:
    ranked = rank_models_within_project(metrics, metric, allowed_models=allowed_models)

    rows = []

    for dataset, ds in ranked.groupby("dataset"):
        projects = sorted(ds["project"].unique())
        model_count = ds["model"].nunique()

        pair_rhos = []
        pair_taus = []
        pair_rows = []

        for p1, p2 in combinations(projects, 2):
            a = ds[ds["project"] == p1][["model", "rank"]]
            b = ds[ds["project"] == p2][["model", "rank"]]

            m = a.merge(b, on="model", suffixes=("_p1", "_p2"))

            if len(m) < 2:
                rho = np.nan
                tau = np.nan
            else:
                rho = safe_corr(m["rank_p1"].tolist(), m["rank_p2"].tolist(), "spearman")
                tau = safe_corr(m["rank_p1"].tolist(), m["rank_p2"].tolist(), "kendall")

            pair_rhos.append(rho)
            pair_taus.append(tau)

            pair_rows.append({
                "dataset": dataset,
                "metric": metric,
                "project_1": p1,
                "project_2": p2,
                "n_common_models": len(m),
                "spearman": rho,
                "kendall": tau,
            })

        mean_rho = np.nanmean(pair_rhos) if pair_rhos else np.nan
        mean_tau = np.nanmean(pair_taus) if pair_taus else np.nan
        stability = mapped_corr_score(mean_rho, mean_tau)

        rows.append({
            "dataset": dataset,
            "metric": metric,
            "n_projects": len(projects),
            "n_models": model_count,
            "mean_pairwise_spearman": mean_rho,
            "mean_pairwise_kendall": mean_tau,
            "metric_ranking_stability_score": stability,
            "n_project_pairs": len(pair_rows),
        })

    return pd.DataFrame(rows)


def compute_combined_auc_mcc_stability(metrics: pd.DataFrame, allowed_models: Optional[List[str]] = None) -> pd.DataFrame:
    auc = compute_metric_stability(metrics, "AUC", allowed_models=allowed_models)
    mcc = compute_metric_stability(metrics, "MCC", allowed_models=allowed_models)

    a = auc.rename(columns={
        "mean_pairwise_spearman": "spearman_auc",
        "mean_pairwise_kendall": "kendall_auc",
        "metric_ranking_stability_score": "auc_stability",
    })

    b = mcc.rename(columns={
        "mean_pairwise_spearman": "spearman_mcc",
        "mean_pairwise_kendall": "kendall_mcc",
        "metric_ranking_stability_score": "mcc_stability",
    })

    merged = a[["dataset", "n_projects", "n_models", "spearman_auc", "kendall_auc", "auc_stability"]].merge(
        b[["dataset", "spearman_mcc", "kendall_mcc", "mcc_stability"]],
        on="dataset",
        how="outer",
    )

    vals = []
    for _, r in merged.iterrows():
        metric_vals = [
            r.get("spearman_auc"),
            r.get("kendall_auc"),
            r.get("spearman_mcc"),
            r.get("kendall_mcc"),
        ]
        mapped = [
            (v + 1.0) / 2.0
            for v in metric_vals
            if pd.notna(v)
        ]
        vals.append(np.mean(mapped) * 100.0 if mapped else np.nan)

    merged["combined_auc_mcc_stability_score"] = vals

    return merged


def leave_one_classifier_out(metrics: pd.DataFrame) -> pd.DataFrame:
    all_models = sorted(metrics["model"].dropna().unique())

    rows = []

    for excluded in all_models:
        allowed = [m for m in all_models if m != excluded]
        combined = compute_combined_auc_mcc_stability(metrics, allowed_models=allowed)

        for _, r in combined.iterrows():
            rows.append({
                "excluded_classifier": excluded,
                "dataset": r["dataset"],
                "n_projects": r["n_projects"],
                "n_models_remaining": len(allowed),
                "spearman_auc": r["spearman_auc"],
                "kendall_auc": r["kendall_auc"],
                "spearman_mcc": r["spearman_mcc"],
                "kendall_mcc": r["kendall_mcc"],
                "combined_auc_mcc_stability_score": r["combined_auc_mcc_stability_score"],
            })

    return pd.DataFrame(rows)


def summarize_leave_one_out(loo: pd.DataFrame, default_combined: pd.DataFrame) -> pd.DataFrame:
    default = default_combined[["dataset", "combined_auc_mcc_stability_score"]].rename(
        columns={"combined_auc_mcc_stability_score": "default_combined_stability"}
    )

    merged = loo.merge(default, on="dataset", how="left")
    merged["abs_change"] = (
        merged["combined_auc_mcc_stability_score"] - merged["default_combined_stability"]
    ).abs()

    rows = []

    for dataset, sub in merged.groupby("dataset"):
        default_value = sub["default_combined_stability"].iloc[0]

        valid_stability = sub["combined_auc_mcc_stability_score"].dropna()
        valid_change = sub["abs_change"].dropna()

        if valid_stability.empty:
            min_loo = np.nan
            max_loo = np.nan
            median_abs_change = np.nan
            max_abs_change = np.nan
            most_sensitive_exclusion = "not_available"
        else:
            min_loo = valid_stability.min()
            max_loo = valid_stability.max()
            median_abs_change = valid_change.median()
            max_abs_change = valid_change.max()

            if valid_change.empty:
                most_sensitive_exclusion = "not_available"
            else:
                idx = sub["abs_change"].idxmax()
                most_sensitive_exclusion = sub.loc[idx, "excluded_classifier"]

        rows.append({
            "dataset": dataset,
            "default_combined_stability": default_value,
            "min_leave_one_out_stability": min_loo,
            "max_leave_one_out_stability": max_loo,
            "median_abs_change": median_abs_change,
            "max_abs_change": max_abs_change,
            "most_sensitive_exclusion": most_sensitive_exclusion,
        })

    return pd.DataFrame(rows)


def metric_conclusion_comparison(metric_stability: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset, sub in metric_stability.groupby("dataset"):
        vals = {
            r["metric"]: r["metric_ranking_stability_score"]
            for _, r in sub.iterrows()
        }

        available = {k: v for k, v in vals.items() if pd.notna(v)}

        if available:
            strongest_metric = max(available, key=available.get)
            weakest_metric = min(available, key=available.get)
            spread = max(available.values()) - min(available.values())
        else:
            strongest_metric = "not_available"
            weakest_metric = "not_available"
            spread = np.nan

        rows.append({
            "dataset": dataset,
            "auc_stability": vals.get("AUC", np.nan),
            "mcc_stability": vals.get("MCC", np.nan),
            "pr_auc_stability": vals.get("PR_AUC", np.nan),
            "metric_stability_spread": spread,
            "strongest_metric": strongest_metric,
            "weakest_metric": weakest_metric,
        })

    return pd.DataFrame(rows)


def make_latex_metric_table(comparison: pd.DataFrame, out_path: Path) -> None:
    df = comparison.copy()
    df = df.sort_values("dataset")

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{Model-ranking robustness across evaluation metrics.}")
    lines.append(r"\label{tab:model_metric_robustness}")
    lines.append(r"\begin{tabular}{lrrrrll}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{AUC stab.} & \textbf{MCC stab.} & \textbf{PR-AUC stab.} & \textbf{Spread} & \textbf{Strongest} & \textbf{Weakest} \\")
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        def fmt(x):
            return "N/A" if pd.isna(x) else f"{x:.2f}"

        dataset = str(r["dataset"]).replace("_", r"\_")
        lines.append(
            f"{dataset} & {fmt(r['auc_stability'])} & {fmt(r['mcc_stability'])} & "
            f"{fmt(r['pr_auc_stability'])} & {fmt(r['metric_stability_spread'])} & "
            f"{r['strongest_metric']} & {r['weakest_metric']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out_path.write_text("\n".join(lines))


def make_latex_loo_table(summary: pd.DataFrame, out_path: Path) -> None:
    df = summary.copy()
    df = df.sort_values("dataset")

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{Leave-one-classifier-out robustness of model-ranking stability.}")
    lines.append(r"\label{tab:model_loo_robustness}")
    lines.append(r"\begin{tabular}{lrrrrl}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Default} & \textbf{Min LOO} & \textbf{Max LOO} & \textbf{Max change} & \textbf{Most sensitive exclusion} \\")
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        def fmt(x):
            return "N/A" if pd.isna(x) else f"{x:.2f}"

        dataset = str(r["dataset"]).replace("_", r"\_")
        excluded = str(r["most_sensitive_exclusion"]).replace("_", r"\_")
        lines.append(
            f"{dataset} & {fmt(r['default_combined_stability'])} & "
            f"{fmt(r['min_leave_one_out_stability'])} & "
            f"{fmt(r['max_leave_one_out_stability'])} & "
            f"{fmt(r['max_abs_change'])} & {excluded} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out_path.write_text("\n".join(lines))


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing baseline results: {RESULTS_PATH}")

    raw = pd.read_csv(RESULTS_PATH)
    metrics, metric_cols = prepare_project_model_metrics(raw)

    metrics.to_csv(OUT_DIR / "project_model_metric_values.csv", index=False)

    print("Detected metric columns:")
    for metric, col in metric_cols.items():
        print(f"  {metric}: {col if col is not None else 'NOT FOUND'}")

    metric_frames = []
    for metric in ["AUC", "MCC", "PR_AUC"]:
        metric_frames.append(compute_metric_stability(metrics, metric))

    metric_stability = pd.concat(metric_frames, ignore_index=True)
    metric_stability.to_csv(OUT_DIR / "metric_ranking_stability.csv", index=False)

    comparison = metric_conclusion_comparison(metric_stability)
    comparison.to_csv(OUT_DIR / "metric_conclusion_comparison.csv", index=False)
    make_latex_metric_table(comparison, OUT_DIR / "tab_model_metric_robustness.tex")

    default_combined = compute_combined_auc_mcc_stability(metrics)
    default_combined.to_csv(OUT_DIR / "combined_auc_mcc_stability.csv", index=False)

    loo = leave_one_classifier_out(metrics)
    loo.to_csv(OUT_DIR / "leave_one_classifier_out.csv", index=False)

    loo_summary = summarize_leave_one_out(loo, default_combined)
    loo_summary.to_csv(OUT_DIR / "leave_one_classifier_out_summary.csv", index=False)
    make_latex_loo_table(loo_summary, OUT_DIR / "tab_model_loo_robustness.tex")

    print("\nModel-ranking robustness completed.")
    print(f"Output directory: {OUT_DIR}")

    print("\nMetric robustness summary:")
    print(comparison.to_string(index=False))

    print("\nLeave-one-classifier-out summary:")
    print(loo_summary.to_string(index=False))


if __name__ == "__main__":
    main()
