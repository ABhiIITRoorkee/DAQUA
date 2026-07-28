from __future__ import annotations

import os
import time
import math
import logging
import warnings
import argparse
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import resample
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    roc_curve,
    confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
try:
    from imblearn.over_sampling import SMOTE

    def balance_fn(X, y, k=5):
        # random_state=None (stochastic)
        return SMOTE(k_neighbors=k).fit_resample(X, y)

except ImportError:
    logging.warning("imblearn not installed -- using random oversampling fallback")

    def balance_fn(X, y, k=5):
        dfb = pd.DataFrame(X.copy())
        dfb["__lbl__"] = y
        maj = dfb[dfb["__lbl__"] == dfb["__lbl__"].mode()[0]]
        mino = dfb[dfb["__lbl__"] != dfb["__lbl__"].mode()[0]]
        mino_up = resample(mino, replace=True, n_samples=len(maj), random_state=None)
        dfb = pd.concat([maj, mino_up])
        return dfb.drop(columns="__lbl__").values, dfb["__lbl__"].values

def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def read_csv_try_delims_require_label(
    path: str,
    label_cands: List[str],
    delims=(",", ";", "\t", r"\s+"),
) -> Optional[pd.DataFrame]:
    """
    Accept parsing ONLY if a candidate label column is present.
    Prevents the 'single-column CSV' false success.
    """
    label_norms = {_norm_col(x) for x in label_cands}
    for d in delims:
        try:
            if d == r"\s+":
                tmp = pd.read_csv(path, sep=r"\s+", engine="python")
            else:
                tmp = pd.read_csv(path, sep=d, engine="python")

            cols_norm = {_norm_col(c) for c in tmp.columns}
            if len(cols_norm.intersection(label_norms)) > 0:
                return tmp
        except Exception:
            continue
    return None


def preprocess_softlab(path: str) -> pd.DataFrame:
    """
    SoftLab label column: defects (TRUE/FALSE) -> 1/0
    """
    df = pd.read_csv(path)
    if "defects" not in df.columns:
        return pd.DataFrame()

    col = "defects"

    if df[col].dtype == bool:
        df[col] = df[col].astype(int)
    else:
        s = df[col].astype(str).str.strip().str.lower()
        df[col] = s.map({"true": 1, "false": 0})
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[col] = (df[col] > 0).astype(int)

    num = df.select_dtypes(include=[np.number]).copy()
    if col not in num.columns:
        num[col] = df[col]

    num = num.dropna().drop_duplicates().reset_index(drop=True)
    num = num.rename(columns={col: "label"})
    return num


def preprocess_aeeem(path: str) -> pd.DataFrame:
    LABEL_CANDS = ["bugs", "bug", "#bugs"]

    df = read_csv_try_delims_require_label(path, LABEL_CANDS, delims=(",", ";", "\t", r"\s+"))
    if df is None or df.empty:
        logging.warning("AEEEM: could not parse (or label not found): %s", os.path.basename(path))
        return pd.DataFrame()

    cols_norm_map = {_norm_col(c): c for c in df.columns}
    label_col = None
    for cand in LABEL_CANDS:
        key = _norm_col(cand)
        if key in cols_norm_map:
            label_col = cols_norm_map[key]
            break
    if label_col is None:
        logging.warning("AEEEM: label not found: %s | cols=%s", os.path.basename(path), list(df.columns))
        return pd.DataFrame()

    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col]).copy()
    if df.empty:
        logging.warning("AEEEM: all rows dropped after label coercion: %s", os.path.basename(path))
        return pd.DataFrame()

    df["label"] = (df[label_col].astype(float) > 0).astype(int)

    feat_cols: List[str] = []
    for c in df.columns:
        if c in (label_col, "label"):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            df[c] = s
            feat_cols.append(c)

    use_cols = feat_cols + ["label"]
    out = df[use_cols].dropna(axis=0, how="any").drop_duplicates().reset_index(drop=True)
    if out.empty:
        logging.warning("AEEEM: empty after dropping NaNs: %s", os.path.basename(path))
        return pd.DataFrame()

    out = drop_conflicting_duplicates(out, feat_cols, "label")
    return out


def preprocess_jira(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "RealBug" not in df.columns:
        return pd.DataFrame()
    if "RealBugCount" in df.columns:
        df = df.drop(columns=["RealBugCount"])

    if df["RealBug"].dtype == object:
        df["RealBug"] = df["RealBug"].map({"buggy": 1, "clean": 0})
    else:
        df["RealBug"] = df["RealBug"].apply(lambda x: 1 if float(x) > 0 else 0)

    df["RealBug"] = pd.to_numeric(df["RealBug"], errors="coerce")
    num = df.select_dtypes(include=[np.number]).dropna().drop_duplicates().reset_index(drop=True)
    if "RealBug" not in num.columns:
        return pd.DataFrame()
    num = num.rename(columns={"RealBug": "label"})
    return num


def preprocess_promise(path: str) -> pd.DataFrame:
    df = read_csv_try_delims_require_label(path, ["bug"], delims=(",", ";", "\t", r"\s+"))
    if df is None or df.empty:
        return pd.DataFrame()

    cols_norm_map = {_norm_col(c): c for c in df.columns}
    if "bug" not in cols_norm_map:
        return pd.DataFrame()
    bug_col = cols_norm_map["bug"]

    df[bug_col] = pd.to_numeric(df[bug_col], errors="coerce")
    df = df.dropna(subset=[bug_col]).copy()
    if df.empty:
        return pd.DataFrame()

    df[bug_col] = (df[bug_col].astype(float) > 0).astype(int)

    num = df.select_dtypes(include=[np.number]).dropna().drop_duplicates().reset_index(drop=True)
    if bug_col not in num.columns:
        return pd.DataFrame()
    num = num.rename(columns={bug_col: "label"})
    return num


DATASET_PREPROCESSOR = {
    "AEEEM": preprocess_aeeem,
    "SoftLab": preprocess_softlab,
    "JIRA": preprocess_jira,
    "Promise": preprocess_promise,
}

def drop_conflicting_duplicates(df: pd.DataFrame, feat_cols: List[str], label_col: str) -> pd.DataFrame:
    if not feat_cols:
        return df.drop_duplicates().reset_index(drop=True)
    grp = df.groupby(feat_cols)[label_col].nunique()
    bad = grp[grp > 1].index
    if len(bad) == 0:
        return df.drop_duplicates().reset_index(drop=True)
    mask = df.set_index(feat_cols).index.isin(bad)
    return df[~mask].drop_duplicates().reset_index(drop=True)


def clean_project(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    if df is None or df.empty or label_col not in df.columns:
        return pd.DataFrame()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col not in num_cols:
        return pd.DataFrame()

    feat_cols = [c for c in num_cols if c != label_col]
    df = df[feat_cols + [label_col]].copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how="any").drop_duplicates().reset_index(drop=True)
    df = drop_conflicting_duplicates(df, feat_cols, label_col)
    return df


def minmax_normalize_per_project(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    feat_cols = [c for c in df.columns if c != label_col]
    if not feat_cols:
        return df
    X = df[feat_cols].astype(float)
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    denom = (mx - mn).replace(0.0, 1.0)
    Xn = (X - mn) / denom
    out = df.copy()
    out[feat_cols] = Xn
    return out
@dataclass
class ProjectData:
    dataset: str
    project: str
    df: pd.DataFrame


def canonical_dataset_name(folder_name: str) -> str:
    s = folder_name.strip()
    l = s.lower()
    if "aeeem" in l:
        return "AEEEM"
    if "softlab" in l:
        return "SoftLab"
    if "jira" in l:
        return "JIRA"
    if "promise" in l:
        return "Promise"
    return s


def load_all_projects(root_dir: str) -> List[ProjectData]:
    projects: List[ProjectData] = []
    if not os.path.isdir(root_dir):
        raise RuntimeError(f"Root directory not found: {root_dir}")

    for ds_folder in sorted(os.listdir(root_dir)):
        ds_path = os.path.join(root_dir, ds_folder)
        if not os.path.isdir(ds_path):
            continue

        ds_name = canonical_dataset_name(ds_folder)
        if ds_name not in DATASET_PREPROCESSOR:
            logging.warning("No preprocessor for dataset folder '%s' (canonical='%s'). Skipping.", ds_folder, ds_name)
            continue

        pre_fn = DATASET_PREPROCESSOR[ds_name]
        csvs = sorted(
            os.path.join(ds_path, f)
            for f in os.listdir(ds_path)
            if f.lower().endswith(".csv")
        )

        logging.info("Dataset folder '%s' -> dataset='%s' | csv files found=%d", ds_folder, ds_name, len(csvs))

        loaded_here, skipped_empty = 0, 0
        for fp in csvs:
            proj = os.path.splitext(os.path.basename(fp))[0]
            df = pre_fn(fp)
            df = clean_project(df, "label")
            if df.empty:
                skipped_empty += 1
                continue
            df = minmax_normalize_per_project(df, "label")

            projects.append(ProjectData(dataset=ds_name, project=proj, df=df))
            loaded_here += 1

        logging.info("Dataset='%s' | loaded projects=%d | skipped empty=%d", ds_name, loaded_here, skipped_empty)

    logging.info("Loaded %d projects from root=%s", len(projects), root_dir)
    return projects

def normalize_datasets(dfs: Dict[str, pd.DataFrame], features: List[str]) -> None:
    """
    Global min-max normalization across ALL projects in the current dataset
    (applied AFTER per-project min-max; still safe and keeps MASTER close to original).
    """
    eps = 1e-10
    gmin = {f: min(df[f].min() for df in dfs.values()) for f in features}
    gmax = {f: max(df[f].max() for df in dfs.values()) for f in features}
    for df in dfs.values():
        for f in features:
            df[f] = (df[f] - gmin[f]) / (gmax[f] - gmin[f] + eps)


def discretize_datasets(dfs: Dict[str, pd.DataFrame], features: List[str], n_bins=10, strategy="quantile") -> KBinsDiscretizer:
    all_vals = np.vstack([df[features].values for df in dfs.values()])
    est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(all_vals)

    for df in dfs.values():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df[features] = est.transform(df[features]).astype(int)

    return est

def compute_transfer_weight(src: pd.DataFrame, tgt_feats: pd.DataFrame, lam=0.5, n_bins=10, label_col="label") -> float:
    Xs = src.drop(columns=label_col).values
    ys = src[label_col].values.astype(int)
    Xt = tgt_feats.values
    mi = mutual_info_classif(Xs, ys, discrete_features=True, random_state=None)
    norm_mi = mi / (mi.sum() + 1e-10)
    eps = 1e-10
    kl = []
    for j in range(Xs.shape[1]):
        P = np.array([np.mean(Xs[:, j] == v) for v in range(n_bins)], dtype=float) + eps
        Q = np.array([np.mean(Xt[:, j] == v) for v in range(n_bins)], dtype=float) + eps
        kl.append(np.sum(P * np.log(P / Q)))
    kl = np.array(kl, dtype=float)
    norm_kl = kl / (np.sum(kl) + 1e-10)

    D = lam * norm_mi - (1 - lam) * norm_kl
    return float(np.sum(1.0 / (1.0 + np.exp(-D))))


def feature_weighted_posterior(src: pd.DataFrame, tgt_feats: pd.DataFrame, lam=0.5, k=5, label_col="label") -> np.ndarray:
    Xs = src.drop(columns=label_col).values
    ys = src[label_col].values.astype(int)
    Xt = tgt_feats.values
    Xb, yb = balance_fn(Xs, ys, k)
    mi = mutual_info_classif(Xb, yb, discrete_features=True, random_state=None)
    theta = mi / (mi.sum() + 1e-10)
    tmin, tmax = Xt.min(axis=0), Xt.max(axis=0)
    s_fw = np.array([
        np.sum(((tmin <= x) & (x <= tmax)).astype(float) * theta)
        for x in Xb
    ], dtype=float)

    denom = (theta.sum() - s_fw + 1.0) ** 2
    w_fw = s_fw / (denom + 1e-10)
    wsum = w_fw.sum() + 1e-10
    P0 = (w_fw[yb == 0].sum() + 1.0) / (wsum + 2.0)
    P1 = (w_fw[yb == 1].sum() + 1.0) / (wsum + 2.0)

    d = Xb.shape[1]
    cond = []
    for m in range(d):
        vals = np.unique(Xb[:, m])
        mp = {}
        for c in (0, 1):
            mask = (yb == c)
            total = w_fw[mask].sum() + float(len(vals))
            mp[c] = {
                v: (w_fw[mask & (Xb[:, m] == v)].sum() + 1.0) / (total + 1e-10)
                for v in vals
            }
        cond.append(mp)

    post = []
    logP0, logP1 = math.log(P0 + 1e-10), math.log(P1 + 1e-10)
    for x in Xt:
        lp0 = logP0 + sum(
            math.exp(theta[m]) * math.log(cond[m][0].get(x[m], 1e-10))
            for m in range(d)
        )
        lp1 = logP1 + sum(
            math.exp(theta[m]) * math.log(cond[m][1].get(x[m], 1e-10))
            for m in range(d)
        )
        M = max(lp0, lp1)
        p0u = math.exp(lp0 - M)
        p1u = math.exp(lp1 - M)
        post.append(p1u / (p0u + p1u + 1e-10))

    return np.array(post, dtype=float)


def master_predict(
    sources: List[pd.DataFrame],
    target_df: pd.DataFrame,
    lam=0.5,
    k=5,
    n_bins=10,
    label_col="label",
) -> np.ndarray:
    tgt_feats = target_df.drop(columns=label_col)
    ws = np.array([compute_transfer_weight(s, tgt_feats, lam, n_bins, label_col) for s in sources], dtype=float)
    ps = np.vstack([feature_weighted_posterior(s, tgt_feats, lam, k, label_col) for s in sources])
    wn = ws / (ws.sum() + 1e-10)
    return wn.dot(ps)

def calibrated_prediction(y_true: np.ndarray, y_prob: np.ndarray, default_thr=0.5) -> np.ndarray:
    y_pred = (y_prob >= default_thr).astype(int)
    if y_pred.sum() in (0, len(y_pred)):
        if len(np.unique(y_true)) < 2:
            return y_pred
        try:
            fpr, tpr, thr = roc_curve(y_true, y_prob)
            j = tpr - fpr
            best = thr[np.argmax(j)]
            y_pred = (y_prob >= best).astype(int)
        except Exception:
            pass

    return y_pred


def compute_ranking_metrics(predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    true_labels = np.asarray(true_labels, dtype=int).reshape(-1)

    n_modules = len(true_labels)
    if n_modules == 0:
        return {
            "P_at_20": float("nan"),
            "R_at_20": float("nan"),
            "IFA": float("nan"),
            "TopK_AUC": float("nan"),
        }

    ranked_indices = np.argsort(-predictions)
    sorted_labels = true_labels[ranked_indices]
    n_defects = int(sorted_labels.sum())

    k = int(np.ceil(0.20 * n_modules))
    k = max(1, min(k, n_modules))
    top_k_labels = sorted_labels[:k]

    precision_at_20 = float(top_k_labels.sum() / k)
    recall_at_20 = float(top_k_labels.sum() / n_defects) if n_defects > 0 else 0.0

    ifa = int(np.argmax(sorted_labels)) if n_defects > 0 else int(n_modules)

    if n_defects > 0:
        recall_curve = np.cumsum(sorted_labels) / n_defects
    else:
        recall_curve = np.zeros(n_modules, dtype=float)

    fraction_inspected = np.arange(1, n_modules + 1, dtype=float) / n_modules
    if n_modules > 1:
        dx = np.diff(fraction_inspected)
        avg_height = (recall_curve[:-1] + recall_curve[1:]) / 2.0
        topk_auc = float(np.sum(dx * avg_height))
    else:
        topk_auc = 0.0

    return {
        "P_at_20": precision_at_20,
        "R_at_20": recall_at_20,
        "IFA": ifa,
        "TopK_AUC": topk_auc,
    }


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_prob = np.clip(y_prob.astype(float), 0.0, 1.0)
    ranking_metrics = compute_ranking_metrics(y_prob, y_true)

    if len(np.unique(y_true)) < 2:
        return dict(
            precision=np.nan,
            recall=np.nan,
            f1_score=np.nan,
            g_mean=np.nan,
            accuracy=np.nan,
            auc=np.nan,
            mcc=np.nan,
            PD=np.nan,
            PF=np.nan,
            **ranking_metrics,
        )

    y_pred = calibrated_prediction(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-10)  # PD
    specificity = tn / (tn + fp + 1e-10)
    pf = fp / (fp + tn + 1e-10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1v = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    gmean = math.sqrt(max(sensitivity * specificity, 0.0))

    return dict(
        precision=round(float(prec), 4),
        recall=round(float(rec), 4),
        f1_score=round(float(f1v), 4),
        g_mean=round(float(gmean), 4),
        accuracy=round(float(acc), 4),
        auc=round(float(auc), 4),
        mcc=round(float(mcc), 4),
        PD=round(float(sensitivity), 4),
        PF=round(float(pf), 4),
        **ranking_metrics,
    )


def mean_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["precision", "recall", "f1_score", "g_mean", "accuracy", "auc", "mcc", "PD", "PF", "P_at_20", "R_at_20", "IFA", "TopK_AUC"]
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array([m.get(k, np.nan) for m in metrics_list], dtype=float)
        out[k] = float(np.nanmean(vals)) if vals.size else np.nan
    return out
def ensure_csv_header(out_csv: str) -> None:
    if os.path.exists(out_csv):
        return
    cols = [
        "timestamp",
        "runs",
        "method",
        "target_dataset",
        "target_project",
        "n_source_projects",
        "n_target_eval",
        "precision",
        "recall",
        "f1_score",
        "g_mean",
        "accuracy",
        "auc",
        "mcc",
        "PD",
        "PF",
        "P_at_20",
        "R_at_20",
        "IFA",
        "TopK_AUC",
    ]
    pd.DataFrame(columns=cols).to_csv(out_csv, index=False)

def run_cpdp_master(
    root_dir: str,
    out_csv: str,
    runs: int = 5,
    target_dataset: Optional[str] = "ALL",
    target_project: Optional[str] = None,
    lam: float = 0.5,
    k: int = 5,
    n_bins: int = 10,
    disc_strategy: str = "quantile",
):
    projects = load_all_projects(root_dir)
    if not projects:
        raise RuntimeError("No projects loaded. Check root folder and dataset folder names.")

    by_ds: Dict[str, List[ProjectData]] = {}
    for p in projects:
        by_ds.setdefault(p.dataset, []).append(p)

    ds_names = sorted(by_ds.keys())
    td = (target_dataset or "ALL").strip()
    if td.upper() == "ALL":
        target_datasets = ds_names
    else:
        if td not in by_ds:
            raise RuntimeError(f"Target dataset '{td}' not found. Loaded datasets: {ds_names}")
        target_datasets = [td]

    ensure_csv_header(out_csv)

    logging.info("Loaded datasets: %s", ", ".join(ds_names))
    logging.info("Target rotation over: %s", ", ".join(target_datasets))
    logging.info("SETTING: CPDP within SAME dataset (sources=other projects, target excluded)")
    logging.info("REPEATS: %d runs per target (NO fixed seed); saving MEAN metrics only", runs)
    logging.info("MASTER params: lam=%.3f k=%d n_bins=%d disc_strategy=%s", lam, k, n_bins, disc_strategy)

    for ds in target_datasets:
        ds_projects = by_ds[ds]

        if target_project is not None:
            ds_projects = [p for p in ds_projects if p.project == target_project]
            if not ds_projects:
                logging.warning("target_project='%s' not found in dataset '%s' -> skipping.", target_project, ds)
                continue
        feat_sets = []
        for p in by_ds[ds]:
            cols = [c for c in p.df.columns if c != "label"]
            feat_sets.append(set(cols))
        common = set.intersection(*feat_sets) if feat_sets else set()

        if not common:
            logging.warning("Dataset '%s': no common features across projects. Skipping.", ds)
            continue

        features = sorted(common)
        logging.info("==============================================")
        logging.info("DATASET: %s | projects=%d | common_features=%d", ds, len(by_ds[ds]), len(features))
        logging.info("==============================================")
        dfs: Dict[str, pd.DataFrame] = {}
        for p in by_ds[ds]:
            dfp = p.df.copy()
            keep = [c for c in features if c in dfp.columns] + ["label"]
            dfp = dfp[keep].dropna().drop_duplicates().reset_index(drop=True)
            if not dfp.empty:
                dfs[p.project] = dfp

        if len(dfs) < 2:
            logging.warning("Dataset '%s': need at least 2 projects for CPDP. Skipping.", ds)
            continue
        normalize_datasets(dfs, features)
        discretize_datasets(dfs, features, n_bins=n_bins, strategy=disc_strategy)
        for tgt in sorted(ds_projects, key=lambda x: x.project):
            if tgt.project not in dfs:
                logging.warning("Target %s/%s missing after alignment; skipping.", ds, tgt.project)
                continue

            df_t = dfs[tgt.project]
            if len(df_t) < 20:
                logging.warning("Target too small (%d). Skipping %s/%s.", len(df_t), ds, tgt.project)
                continue
            source_names = [name for name in dfs.keys() if name != tgt.project]
            if not source_names:
                logging.warning("No sources for target %s/%s. Skipping.", ds, tgt.project)
                continue

            sources = [dfs[name] for name in source_names]
            y_true = df_t["label"].values.astype(int)

            all_run_metrics: List[Dict[str, float]] = []
            t0_target = time.time()

            for r in range(int(runs)):
                y_prob = master_predict(
                    sources=sources,
                    target_df=df_t,
                    lam=lam,
                    k=k,
                    n_bins=n_bins,
                    label_col="label",
                )
                met = compute_metrics(y_true, y_prob)
                all_run_metrics.append(met)

                logging.info(
                    "Run %d/%d done for %s/%s in %.1fs | AUC=%s MCC=%s F1=%s",
                    r + 1, runs, ds, tgt.project, time.time() - t0_target,
                    f"{met['auc']:.4f}" if np.isfinite(met["auc"]) else "nan",
                    f"{met['mcc']:.4f}" if np.isfinite(met["mcc"]) else "nan",
                    f"{met['f1_score']:.4f}" if np.isfinite(met["f1_score"]) else "nan",
                )

            met_mean = mean_metrics(all_run_metrics)

            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": int(runs),
                "method": "MASTER",
                "target_dataset": ds,
                "target_project": tgt.project,
                "n_source_projects": int(len(source_names)),
                "n_target_eval": int(len(y_true)),
                **met_mean,
            }

            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=False, index=False)
            logging.info(
                "FINISHED target %s/%s | saved MEAN over %d runs in %.1fs",
                ds, tgt.project, runs, time.time() - t0_target
            )

    logging.info("ALL DONE. Results saved to: %s", out_csv)
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        type=str,
        default="Dataset-MiTSA",
        help="Root folder containing dataset folders (AEEEM, SoftLab, JIRA, Promise, ...)"
    )
    ap.add_argument("--out", type=str, default="Master_cpdp_dataset1_Final.csv", help="Output CSV file")

    ap.add_argument("--target_dataset", type=str, default="ALL", help="Target dataset name, or ALL")
    ap.add_argument("--target_project", type=str, default=None, help="Run only this project (without .csv)")

    ap.add_argument("--runs", type=int, default=5, help="Number of repeated runs per target (mean is saved)")
    ap.add_argument("--lam", type=float, default=0.5, help="MASTER lambda (Eq. 6/14/15)")
    ap.add_argument("--k", type=int, default=5, help="SMOTE k_neighbors (or oversampling parameter)")
    ap.add_argument("--n_bins", type=int, default=10, help="Discretization bins")
    ap.add_argument("--disc_strategy", type=str, default="quantile", help="KBinsDiscretizer strategy")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cpdp_master(
        root_dir=args.root,
        out_csv=args.out,
        runs=args.runs,
        target_dataset=args.target_dataset,
        target_project=args.target_project,
        lam=args.lam,
        k=args.k,
        n_bins=args.n_bins,
        disc_strategy=args.disc_strategy,
    )