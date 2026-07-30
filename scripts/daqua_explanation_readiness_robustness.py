from __future__ import annotations

import argparse
import math
import warnings
from itertools import combinations
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold


METHODS = ["native", "permutation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default="Data-set")
    parser.add_argument("--profiles-dir", default="outputs/profiles_9datasets_full")
    parser.add_argument("--out-dir", default="outputs/robustness_9datasets/explanation_readiness_smoke")

    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--n-permutation-repeats", type=int, default=3)
    parser.add_argument("--max-permutation-rows", type=int, default=3000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    return parser.parse_args()


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]

    if len(a) < 2:
        return np.nan

    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan

    return float(spearmanr(a, b).correlation)


def mapped_corr_score(x: float) -> float:
    if pd.isna(x):
        return np.nan
    return float(((x + 1.0) / 2.0) * 100.0)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return np.nan
    union = a | b
    if not union:
        return np.nan
    return len(a & b) / len(union)


def readiness_level(score: float) -> str:
    if pd.isna(score):
        return "not_available"
    if score >= 85:
        return "high"
    if score >= 70:
        return "moderate"
    if score >= 50:
        return "limited"
    return "low"


def _norm_col_name(x: Any) -> str:
    return "".join(ch.lower() for ch in str(x) if ch.isalnum())


def _read_project_table_from_path(path_value: Any) -> pd.DataFrame:
    p = Path(path_value)

    if not p.exists():
        raise FileNotFoundError(f"Project path does not exist: {p}")

    suffix = p.suffix.lower()

    if suffix == ".arff":
        from scipy.io import arff
        data, _ = arff.loadarff(str(p))
        df = pd.DataFrame(data)

        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].apply(
                    lambda v: v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else v
                )
        return df

    return pd.read_csv(p)


def _find_label_column(df: pd.DataFrame, preferred: Any = None) -> Optional[str]:
    label_candidates = [
        preferred,
        "bugs",
        "bug",
        "#bugs",
        "number of bugs",
        "numberofbugs",
        "bug_count",
        "defective",
        "Defective",
        "isDefective",
        "is_defective",
        "defects",
        "defect",
        "RealBug",
        "realbug",
        "buggy",
        "label",
        "class",
        "target",
    ]

    norm_to_col = {_norm_col_name(c): c for c in df.columns}

    for cand in label_candidates:
        if cand is None:
            continue

        key = _norm_col_name(cand)
        if key in norm_to_col:
            return norm_to_col[key]

        # Common singular/plural fallback.
        if key.endswith("s") and key[:-1] in norm_to_col:
            return norm_to_col[key[:-1]]

        if (key + "s") in norm_to_col:
            return norm_to_col[key + "s"]

    # Last fallback: search by label-like names.
    label_tokens = ["bug", "defect", "fault", "label", "class", "target"]
    matches = []
    for c in df.columns:
        key = _norm_col_name(c)
        if any(tok in key for tok in label_tokens):
            matches.append(c)

    # Prefer short/count-like columns over long textual fields.
    if matches:
        numeric_like = []
        for c in matches:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().mean() > 0.80:
                numeric_like.append(c)

        if numeric_like:
            return numeric_like[0]

        return matches[0]

    return None


def extract_project_object(project_obj: Any) -> Tuple[str, str, pd.DataFrame, pd.Series]:
    dataset = getattr(project_obj, "dataset", None)
    project = getattr(project_obj, "project", None)
    preferred_label = getattr(project_obj, "label_column", None)

    table_source = "loader_df"

    if hasattr(project_obj, "df"):
        df = getattr(project_obj, "df").copy()
    else:
        df = None

    label_column = None

    if df is not None:
        label_column = _find_label_column(df, preferred_label)

    # Some DAQUA loader objects expose a feature-only df but retain the
    # original project path and label metadata. In that case, reload the raw
    # table and recover the label there.
    if label_column is None and hasattr(project_obj, "path"):
        df = _read_project_table_from_path(getattr(project_obj, "path"))
        table_source = "raw_path"
        label_column = _find_label_column(df, preferred_label)

    if df is None or label_column is None:
        available = []
        if df is not None:
            available = list(df.columns)
        raise ValueError(
            f"Could not identify label column for {dataset}/{project}. "
            f"preferred_label={preferred_label}; available_columns={available}"
        )

    y = df[label_column].copy()

    drop_cols = [label_column]

    if hasattr(project_obj, "dropped_columns"):
        dropped = getattr(project_obj, "dropped_columns")
        if dropped is not None:
            drop_cols.extend([c for c in dropped if c in df.columns])

    # Remove obvious identifier/text/meta columns. Numeric conversion below
    # will also remove nonnumeric columns, but dropping these avoids accidental
    # numeric IDs becoming features.
    meta_like = [
        "id",
        "name",
        "path",
        "file",
        "filename",
        "class",
        "package",
        "project",
        "version",
        "release",
        "commit",
        "date",
        "time",
    ]

    for c in df.columns:
        key = _norm_col_name(c)
        if any(tok == key or key.endswith(tok) for tok in meta_like):
            if c != label_column:
                drop_cols.append(c)

    X = df.drop(columns=list(dict.fromkeys(drop_cols)), errors="ignore").copy()

    if dataset is None:
        dataset = getattr(project_obj, "dataset_name", "unknown_dataset")

    if project is None:
        project = getattr(project_obj, "project_name", "unknown_project")

    X = pd.DataFrame(X).copy()
    y = pd.Series(y).copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    y = pd.to_numeric(y, errors="coerce")
    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    # Standardize count labels into binary defect labels.
    unique_vals = set(pd.Series(y.dropna().unique()).astype(float).tolist())
    if not unique_vals.issubset({0.0, 1.0}):
        y = (y > 0).astype(int)
    else:
        y = y.astype(int)

    if X.shape[1] == 0:
        raise ValueError(
            f"No numeric feature columns remain for {dataset}/{project} "
            f"after extracting from {table_source}."
        )

    return str(dataset), str(project), X, y


def load_projects(root: str) -> List[Tuple[str, str, pd.DataFrame, pd.Series]]:
    from daqua.loaders.defect_loader import load_all_projects

    loaded = load_all_projects(Path(root))

    projects = []
    for obj in loaded:
        dataset, project, X, y = extract_project_object(obj)

        if len(X) == 0:
            continue

        if y.nunique() < 2:
            continue

        if X.shape[1] < 2:
            continue

        projects.append((dataset, project, X, y))

    return projects


def effective_splits(y: pd.Series, requested_splits: int) -> int:
    counts = y.value_counts()
    if len(counts) < 2:
        return 0
    min_class = int(counts.min())
    return max(0, min(requested_splits, min_class))


def top_k_features(feature_names: List[str], importances: np.ndarray, k: int = 10) -> List[str]:
    order = np.argsort(-np.asarray(importances, dtype=float))
    return [feature_names[i] for i in order[: min(k, len(order))]]


def rank_importances(importances: np.ndarray) -> np.ndarray:
    # Higher importance gets rank 1.
    return rankdata(-np.asarray(importances, dtype=float), method="average")


def compute_project_importance(
    dataset: str,
    project: str,
    X: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows = []

    n_splits = effective_splits(y, args.n_splits)

    if n_splits < 2:
        print(f"SKIP {dataset}/{project}: insufficient class count for CV.")
        return rows

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )

    feature_names = list(X.columns)

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        seed = args.random_state + fold_id

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=args.n_jobs,
        )

        model.fit(X_train, y_train)

        native_importances = np.asarray(model.feature_importances_, dtype=float)
        native_ranks = rank_importances(native_importances)
        native_top10 = set(top_k_features(feature_names, native_importances, k=10))

        for feature, importance, rank in zip(feature_names, native_importances, native_ranks):
            rows.append({
                "dataset": dataset,
                "project": project,
                "method": "native",
                "fold_id": fold_id,
                "feature": feature,
                "importance": float(importance),
                "rank": float(rank),
                "is_top10": feature in native_top10,
            })

        if args.max_permutation_rows and len(X_test) > args.max_permutation_rows:
            rng = np.random.default_rng(seed)
            sample_idx = rng.choice(len(X_test), size=args.max_permutation_rows, replace=False)
            X_perm = X_test.iloc[sample_idx]
            y_perm = y_test.iloc[sample_idx]
        else:
            X_perm = X_test
            y_perm = y_test

        scoring = "roc_auc" if y_perm.nunique() == 2 else "balanced_accuracy"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perm = permutation_importance(
                model,
                X_perm,
                y_perm,
                scoring=scoring,
                n_repeats=args.n_permutation_repeats,
                random_state=seed,
                n_jobs=1,
            )

        perm_importances = np.asarray(perm.importances_mean, dtype=float)
        perm_importances = np.maximum(perm_importances, 0.0)
        perm_ranks = rank_importances(perm_importances)
        perm_top10 = set(top_k_features(feature_names, perm_importances, k=10))

        for feature, importance, rank in zip(feature_names, perm_importances, perm_ranks):
            rows.append({
                "dataset": dataset,
                "project": project,
                "method": "permutation",
                "fold_id": fold_id,
                "feature": feature,
                "importance": float(importance),
                "rank": float(rank),
                "is_top10": feature in perm_top10,
            })

    return rows


def compute_within_project_stability(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dataset, project, method), sub in profile.groupby(["dataset", "project", "method"]):
        piv_rank = sub.pivot_table(
            index="fold_id",
            columns="feature",
            values="rank",
            aggfunc="mean",
        )

        piv_imp = sub.pivot_table(
            index="fold_id",
            columns="feature",
            values="importance",
            aggfunc="mean",
        )

        fold_ids = list(piv_rank.index)

        spearmans = []
        jaccards = []

        top_sets = {}
        for fold_id, fold_sub in sub.groupby("fold_id"):
            top_sets[fold_id] = set(fold_sub.loc[fold_sub["is_top10"], "feature"].astype(str))

        for a, b in combinations(fold_ids, 2):
            common = piv_rank.columns[piv_rank.loc[a].notna() & piv_rank.loc[b].notna()]
            if len(common) >= 2:
                s = safe_spearman(
                    piv_rank.loc[a, common].values,
                    piv_rank.loc[b, common].values,
                )
                spearmans.append(s)

            j = jaccard(top_sets.get(a, set()), top_sets.get(b, set()))
            if pd.notna(j):
                jaccards.append(j)

        mean_spearman = np.nanmean(spearmans) if spearmans else np.nan
        mean_jaccard = np.nanmean(jaccards) if jaccards else np.nan

        rows.append({
            "dataset": dataset,
            "project": project,
            "method": method,
            "n_folds": len(fold_ids),
            "within_project_spearman": mean_spearman,
            "within_project_rank_stability": mapped_corr_score(mean_spearman),
            "within_project_top10_jaccard": mean_jaccard,
            "within_project_top10_jaccard_score": mean_jaccard * 100.0 if pd.notna(mean_jaccard) else np.nan,
        })

    return pd.DataFrame(rows)


def aggregate_project_profiles(profile: pd.DataFrame) -> pd.DataFrame:
    agg = (
        profile.groupby(["dataset", "project", "method", "feature"], as_index=False)
        .agg(mean_importance=("importance", "mean"))
    )

    rows = []
    for (dataset, project, method), sub in agg.groupby(["dataset", "project", "method"]):
        sub = sub.copy()
        sub["rank"] = rank_importances(sub["mean_importance"].values)
        top10 = set(top_k_features(sub["feature"].tolist(), sub["mean_importance"].values, k=10))
        sub["is_top10"] = sub["feature"].isin(top10)
        rows.append(sub)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_cross_project_stability(project_profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dataset, method), sub in project_profiles.groupby(["dataset", "method"]):
        projects = sorted(sub["project"].unique())

        spearmans = []
        jaccards = []

        for p1, p2 in combinations(projects, 2):
            a = sub[sub["project"] == p1][["feature", "rank", "is_top10"]]
            b = sub[sub["project"] == p2][["feature", "rank", "is_top10"]]

            merged = a.merge(b, on="feature", suffixes=("_p1", "_p2"))

            if len(merged) >= 2:
                s = safe_spearman(
                    merged["rank_p1"].values,
                    merged["rank_p2"].values,
                )
                spearmans.append(s)

            top_a = set(a.loc[a["is_top10"], "feature"].astype(str))
            top_b = set(b.loc[b["is_top10"], "feature"].astype(str))
            j = jaccard(top_a, top_b)
            if pd.notna(j):
                jaccards.append(j)

        mean_spearman = np.nanmean(spearmans) if spearmans else np.nan
        mean_jaccard = np.nanmean(jaccards) if jaccards else np.nan

        rows.append({
            "dataset": dataset,
            "method": method,
            "n_projects": len(projects),
            "mean_pairwise_project_spearman": mean_spearman,
            "mean_pairwise_project_spearman_score": mapped_corr_score(mean_spearman),
            "mean_pairwise_project_top10_jaccard": mean_jaccard,
            "mean_pairwise_project_top10_jaccard_score": mean_jaccard * 100.0 if pd.notna(mean_jaccard) else np.nan,
        })

    return pd.DataFrame(rows)


def compute_explanation_readiness(
    project_stability: pd.DataFrame,
    cross_project: pd.DataFrame,
) -> pd.DataFrame:
    within = (
        project_stability.groupby(["dataset", "method"], as_index=False)
        .agg(
            mean_project_feature_importance_stability=("within_project_rank_stability", "mean"),
            mean_within_project_top10_jaccard=("within_project_top10_jaccard", "mean"),
        )
    )

    df = within.merge(cross_project, on=["dataset", "method"], how="left")

    scores = []

    for _, r in df.iterrows():
        W = r["mean_project_feature_importance_stability"]
        J = r["mean_pairwise_project_top10_jaccard_score"]
        R = r["mean_pairwise_project_spearman_score"]

        # Single-project families cannot estimate cross-project explanation stability.
        # We use a neutral value of 50 for unavailable cross-project terms and let
        # the output table expose the N/A indicators.
        J_eff = 50.0 if pd.isna(J) else J
        R_eff = 50.0 if pd.isna(R) else R

        dataset_fi_score = np.nanmean([W, J_eff, R_eff])

        E = (
            0.40 * W
            + 0.25 * J_eff
            + 0.20 * R_eff
            + 0.15 * dataset_fi_score
        )

        scores.append({
            "dataset_feature_importance_stability_score": dataset_fi_score,
            "explanation_readiness_score": E,
            "explanation_readiness_level": readiness_level(E),
        })

    score_df = pd.DataFrame(scores)
    out = pd.concat([df.reset_index(drop=True), score_df], axis=1)

    return out


def compare_methods(readiness: pd.DataFrame, project_profiles: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    native = readiness[readiness["method"] == "native"].copy()
    perm = readiness[readiness["method"] == "permutation"].copy()

    comparison = native.merge(
        perm,
        on="dataset",
        suffixes=("_native", "_permutation"),
    )

    comparison["abs_score_change"] = (
        comparison["explanation_readiness_score_permutation"]
        - comparison["explanation_readiness_score_native"]
    ).abs()

    comparison["level_agreement"] = (
        comparison["explanation_readiness_level_native"]
        == comparison["explanation_readiness_level_permutation"]
    )

    # Cross-method project-level top-10 agreement
    method_overlap_rows = []

    for (dataset, project), sub in project_profiles.groupby(["dataset", "project"]):
        native_top = set(
            sub.loc[
                (sub["method"] == "native") & (sub["is_top10"]),
                "feature",
            ].astype(str)
        )
        perm_top = set(
            sub.loc[
                (sub["method"] == "permutation") & (sub["is_top10"]),
                "feature",
            ].astype(str)
        )

        method_overlap_rows.append({
            "dataset": dataset,
            "project": project,
            "native_permutation_top10_jaccard": jaccard(native_top, perm_top),
        })

    method_overlap = pd.DataFrame(method_overlap_rows)

    overlap_summary = (
        method_overlap.groupby("dataset", as_index=False)
        .agg(mean_native_permutation_top10_jaccard=("native_permutation_top10_jaccard", "mean"))
    )

    comparison = comparison.merge(overlap_summary, on="dataset", how="left")

    return comparison, method_overlap


def make_latex_table(comparison: pd.DataFrame, out_path: Path) -> None:
    df = comparison.copy()
    df = df.sort_values("explanation_readiness_score_native", ascending=False)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{Explanation-readiness robustness across feature-importance methods.}")
    lines.append(r"\label{tab:explanation_method_robustness}")
    lines.append(r"\begin{tabular}{lrrrrrl}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Dataset} & \textbf{Native} & \textbf{Permutation} & "
        r"\textbf{Abs. change} & \textbf{Top-10 agreement} & "
        r"\textbf{Level agreement} & \textbf{Permutation level} \\"
    )
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        dataset = str(r["dataset"]).replace("_", r"\_")
        native = r["explanation_readiness_score_native"]
        perm = r["explanation_readiness_score_permutation"]
        change = r["abs_score_change"]
        top10 = r["mean_native_permutation_top10_jaccard"]
        agree = "Yes" if bool(r["level_agreement"]) else "No"
        level = str(r["explanation_readiness_level_permutation"]).replace("_", r"\_")

        def fmt(x):
            return "N/A" if pd.isna(x) else f"{x:.2f}"

        lines.append(
            f"{dataset} & {fmt(native)} & {fmt(perm)} & {fmt(change)} & "
            f"{fmt(top10)} & {agree} & {level} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out_path.write_text("\n".join(lines))


def make_global_summary(comparison: pd.DataFrame) -> Dict[str, Any]:
    valid = comparison.dropna(
        subset=[
            "explanation_readiness_score_native",
            "explanation_readiness_score_permutation",
        ]
    ).copy()

    if len(valid) >= 2:
        rank_corr = safe_spearman(
            valid["explanation_readiness_score_native"].values,
            valid["explanation_readiness_score_permutation"].values,
        )
    else:
        rank_corr = np.nan

    return {
        "n_datasets": len(valid),
        "rank_correlation": rank_corr,
        "median_abs_score_change": valid["abs_score_change"].median(),
        "max_abs_score_change": valid["abs_score_change"].max(),
        "level_agreement_percentage": valid["level_agreement"].mean() * 100.0,
        "mean_top10_method_agreement": valid["mean_native_permutation_top10_jaccard"].mean(),
    }


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Starting DAQUA explanation-readiness robustness.")
    print(f"Root: {args.root}")
    print(f"Output directory: {out_dir}")
    print(f"Splits: {args.n_splits}")
    print(f"Repeats: {args.n_repeats}")
    print(f"Permutation repeats: {args.n_permutation_repeats}")
    print(f"Max permutation rows: {args.max_permutation_rows}")

    projects = load_projects(args.root)

    print(f"Loaded projects for explanation robustness: {len(projects)}")

    all_rows = []

    for i, (dataset, project, X, y) in enumerate(projects, start=1):
        print(f"[{i}/{len(projects)}] {dataset}/{project} | rows={len(X)} features={X.shape[1]}")
        rows = compute_project_importance(dataset, project, X, y, args)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No feature-importance rows were generated.")

    profile = pd.DataFrame(all_rows)
    profile.to_csv(out_dir / "feature_importance_native_vs_permutation.csv", index=False)

    project_stability = compute_within_project_stability(profile)
    project_stability.to_csv(out_dir / "project_explanation_stability_by_method.csv", index=False)

    project_profiles = aggregate_project_profiles(profile)
    project_profiles.to_csv(out_dir / "project_aggregated_feature_profiles_by_method.csv", index=False)

    cross_project = compute_cross_project_stability(project_profiles)
    cross_project.to_csv(out_dir / "cross_project_explanation_stability_by_method.csv", index=False)

    readiness = compute_explanation_readiness(project_stability, cross_project)
    readiness.to_csv(out_dir / "explanation_readiness_by_method.csv", index=False)

    comparison, method_overlap = compare_methods(readiness, project_profiles)
    comparison.to_csv(out_dir / "explanation_method_comparison.csv", index=False)
    method_overlap.to_csv(out_dir / "project_native_permutation_top10_overlap.csv", index=False)

    make_latex_table(comparison, out_dir / "tab_explanation_method_robustness.tex")

    global_summary = make_global_summary(comparison)
    pd.DataFrame([global_summary]).to_csv(out_dir / "explanation_method_global_summary.csv", index=False)

    print("\nExplanation-readiness robustness completed.")
    print(f"Output directory: {out_dir}")

    print("\nGlobal summary:")
    for k, v in global_summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\nMethod comparison:")
    cols = [
        "dataset",
        "explanation_readiness_score_native",
        "explanation_readiness_score_permutation",
        "abs_score_change",
        "mean_native_permutation_top10_jaccard",
        "explanation_readiness_level_native",
        "explanation_readiness_level_permutation",
        "level_agreement",
    ]
    print(comparison[cols].sort_values("explanation_readiness_score_native", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
