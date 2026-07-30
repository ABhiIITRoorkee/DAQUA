# daqua/loaders/defect_loader.py` with this full updated version.

# It fixes the main issue by adding:

# * recursive loading with `rglob`
# * support for `apachejit`, `Kamei`, `GHPR`, `ContinuousDefect`, `UnifiedBugDataSet_File`, `BugHunter`, and future generic datasets
# * ARFF loading support for BugHunter-style datasets
# * generic label detection
# * combined dataset splitting by project/repository/system columns
# * skip rules for irrelevant ApacheJIT files such as `commit_links`, `keys`, yearly files, and train/test splits

# ```python
# daqua/loaders/defect_loader.py



from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class ProjectData:
    dataset: str
    project: str
    path: str
    df: pd.DataFrame
    raw_shape: Tuple[int, int]
    label_column: str
    dropped_columns: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# Label candidates
# ---------------------------------------------------------------------

GENERIC_LABEL_CANDIDATES: List[str] = [
    "label",
    "target",
    "bug",
    "bugs",
    "#bugs",
    "Bug",
    "Bugs",
    "defect",
    "defects",
    "Defect",
    "Defects",
    "defective",
    "Defective",
    "isDefective",
    "is_defective",
    "buggy",
    "Buggy",
    "isBuggy",
    "is_buggy",
    "contains_bug",
    "containsbug",
    "has_bug",
    "hasbug",
    "faulty",
    "Faulty",
    "problems",
    "class",
    "Class",
    "RealBug",
    "Number of Bugs",
]

LABEL_CANDIDATES: Dict[str, List[str]] = {
    "AEEEM": ["bugs", "bug", "#bugs", "label"],
    "JIRA": ["RealBug", "bug", "bugs", "defects", "label"],
    "NASA": ["defects", "defective", "bug", "bugs", "label", "class", "problems"],
    "Promise": ["bug", "bugs", "defects", "defective", "label"],
    "ReLink": ["bug", "bugs", "defect", "defects", "defective", "isDefective", "label"],

    # New datasets
    "Kamei": ["bug", "buggy", "defect", "defective", "label", "contains_bug"],
    "ApacheJIT": ["bug", "buggy", "defect", "defective", "label", "contains_bug"],
    "GHPR": ["bug", "bugs", "buggy", "defect", "defective", "label", "contains_bug"],
    "ContinuousDefect": ["bug", "bugs", "buggy", "defect", "defective", "label", "contains_bug"],
    "UnifiedBugDataSet_File": ["bug", "bugs", "defect", "defects", "defective", "label"],
    "UnifiedBugDataSet_Class": ["bug", "bugs", "defect", "defects", "defective", "label"],
    "BugHunter": ["Number of Bugs", "number_of_bugs", "NumberOfBugs", "number bugs", "bug_count", "bugCount", "BugCount", "bugs", "bug", "defect", "defects", "defective", "label", "class"],
    "Jureczko": ["bug", "bugs", "defect", "defects", "defective", "label"],
}


IDENTIFIER_COLUMNS = {
    "id",
    "ids",
    "name",
    "file",
    "filename",
    "file_name",
    "filepath",
    "file_path",
    "fullpath",
    "full_path",
    "class",
    "classname",
    "class_name",
    "package",
    "project",
    "projectname",
    "project_name",
    "repo",
    "repository",
    "repositoryname",
    "repository_name",
    "version",
    "release",
    "module",
    "path",
    "commit",
    "commitid",
    "commit_id",
    "commit_hash",
    "hash",
    "sha",
    "revision",
    "rev",
    "date",
    "time",
    "timestamp",
    "author",
    "committer",
    "developer",
    "email",
    "url",
    "link",
}


KNOWN_LEAKAGE_COLUMNS = {
    "realbugcount",
    "bugcount",
    "bugcounts",
    "bugs",
    "bug",
    "defects",
    "defect",
    "defective",
    "isdefective",
    "label",
    "target",
    "problems",
    "containsbug",
    "contains_bug",
    "buggy",
    "isbuggy",
    "is_buggy",
}


COMBINED_DATASETS = {
    "GHPR",
    "ContinuousDefect",
    "UnifiedBugDataSet_File",
    "UnifiedBugDataSet_Class",
    "ApacheJIT",
}


PROJECT_SPLIT_CANDIDATES: Dict[str, List[str]] = {
    "GHPR": [
        "project",
        "project_name",
        "projectname",
        "repo",
        "repository",
        "repository_name",
        "full_name",
        "owner_repo",
        "system",
    ],
    "ContinuousDefect": [
        "project",
        "project_name",
        "projectname",
        "repo",
        "repository",
        "system",
        "product",
        "release",
        "version",
    ],
    "UnifiedBugDataSet_File": [
        "project",
        "project_name",
        "projectname",
        "system",
        "software",
        "product",
        "dataset",
        "repository",
        "repo",
    ],
    "UnifiedBugDataSet_Class": [
        "project",
        "project_name",
        "projectname",
        "system",
        "software",
        "product",
        "dataset",
        "repository",
        "repo",
    ],
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _norm_col(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _safe_name(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^\w.\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "unknown"


def canonical_dataset_name(folder_name: str) -> str:
    name = folder_name.strip()
    lower = name.lower()

    if "aeeem" in lower:
        return "AEEEM"
    if "jira" in lower:
        return "JIRA"
    if "nasa" in lower or "mdp" in lower:
        return "NASA"
    if "promise" in lower:
        return "Promise"
    if "relink" in lower:
        return "ReLink"

    if "kamei" in lower:
        return "Kamei"
    if "apachejit" in lower or "apache-jit" in lower:
        return "ApacheJIT"
    if "ghpr" in lower:
        return "GHPR"
    if "continuousdefect" in lower or "continuous-defect" in lower:
        return "ContinuousDefect"
    if "unifiedbugdataset" in lower or "unifiedbug" in lower:
        if "class" in lower:
            return "UnifiedBugDataSet_Class"
        return "UnifiedBugDataSet_File"
    if "bughunter" in lower or "bug-hunter" in lower:
        return "BugHunter"
    if "jureczko" in lower:
        return "Jureczko"

    return name


def label_candidates_for_dataset(dataset: str) -> List[str]:
    specific = LABEL_CANDIDATES.get(dataset, [])
    ordered: List[str] = []

    for item in specific + GENERIC_LABEL_CANDIDATES:
        if item not in ordered:
            ordered.append(item)

    return ordered


def candidate_label_columns(df: pd.DataFrame, dataset: str) -> List[str]:
    candidates = label_candidates_for_dataset(dataset)
    norm_to_original = {_norm_col(col): col for col in df.columns}

    found: List[str] = []

    for candidate in candidates:
        key = _norm_col(candidate)
        if key in norm_to_original:
            original = norm_to_original[key]
            if original not in found:
                found.append(original)

    return found


# ---------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------

def read_csv_auto(path: str, dataset: str) -> Optional[pd.DataFrame]:
    delimiters = [",", ";", "\t", r"\s+"]

    best_df: Optional[pd.DataFrame] = None

    for delimiter in delimiters:
        try:
            if delimiter == r"\s+":
                df = pd.read_csv(
                    path,
                    sep=r"\s+",
                    engine="python",
                    on_bad_lines="skip",
                )
            else:
                df = pd.read_csv(
                    path,
                    sep=delimiter,
                    engine="python",
                    on_bad_lines="skip",
                )

            if df is None or df.empty or df.shape[1] <= 1:
                continue

            if best_df is None:
                best_df = df

            if candidate_label_columns(df, dataset):
                return df

        except Exception:
            continue

    if best_df is not None:
        return best_df

    try:
        df = pd.read_csv(path, on_bad_lines="skip")
        if df is not None and not df.empty:
            return df
    except Exception as exc:
        logger.warning("Could not read CSV: %s | %s", path, exc)

    return None


def read_arff_auto(path: str) -> Optional[pd.DataFrame]:
    try:
        from scipy.io import arff  # type: ignore

        data, _ = arff.loadarff(path)
        df = pd.DataFrame(data)

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x
                )

        return df

    except Exception:
        pass

    # Lightweight fallback ARFF parser for dense ARFF files.
    try:
        attributes: List[str] = []
        rows: List[List[str]] = []
        in_data = False

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()

                if not line or line.startswith("%"):
                    continue

                lower = line.lower()

                if lower.startswith("@attribute"):
                    match = re.match(
                        r"@attribute\s+('.*?'|\".*?\"|\S+)\s+(.+)",
                        line,
                        flags=re.IGNORECASE,
                    )
                    if match:
                        attr = match.group(1).strip("'\"")
                        attributes.append(attr)
                    continue

                if lower.startswith("@data"):
                    in_data = True
                    continue

                if in_data:
                    if line.startswith("{"):
                        # Sparse ARFF is not handled by this fallback parser.
                        continue
                    parsed = next(csv.reader([line]))
                    rows.append(parsed)

        if not attributes or not rows:
            return None

        width = len(attributes)
        normalized_rows = []

        for row in rows:
            if len(row) == width:
                normalized_rows.append(row)

        if not normalized_rows:
            return None

        return pd.DataFrame(normalized_rows, columns=attributes)

    except Exception as exc:
        logger.warning("Could not read ARFF: %s | %s", path, exc)
        return None



def sanitize_dataframe(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Clean malformed CSV/ARFF column names and quoted numeric values.
    This is needed for datasets such as BugHunter, where headers may
    appear as Bugs" and numeric cells may be surrounded by quotes.
    """
    if df is None:
        return None

    df = df.copy()

    cleaned_columns = []
    seen = {}

    for col in df.columns:
        name = str(col).strip()
        name = name.replace("\ufeff", "")
        name = name.strip()
        name = name.strip('"').strip("'").strip()

        if not name:
            name = "unnamed"

        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0

        cleaned_columns.append(name)

    df.columns = cleaned_columns

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.strip('"')
                .str.strip("'")
                .str.strip()
            )

    return df


def read_table_auto(path: str, dataset: str) -> Optional[pd.DataFrame]:
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        return sanitize_dataframe(read_csv_auto(path, dataset))

    if suffix == ".arff":
        return sanitize_dataframe(read_arff_auto(path))

    return None


# ---------------------------------------------------------------------
# Cleaning and standardization
# ---------------------------------------------------------------------

def normalize_label(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype("Int64")

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = numeric.notna().mean()

    if numeric_ratio >= 0.80:
        return (numeric > 0).astype("Int64")

    mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "bug": 1,
        "buggy": 1,
        "clean": 0,
        "defective": 1,
        "nondefective": 0,
        "non-defective": 0,
        "non_defective": 0,
        "faulty": 1,
        "notfaulty": 0,
        "not-faulty": 0,
        "not_faulty": 0,
        "positive": 1,
        "negative": 0,
        "1": 1,
        "0": 0,
    }

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.strip('"')
        .str.strip("'")
        .str.strip()
        .str.lower()
        .str.replace("_", "-", regex=False)
    )

    labels = cleaned.map(mapping)

    unresolved = labels.isna()
    if unresolved.any():
        fallback_numeric = pd.to_numeric(cleaned[unresolved], errors="coerce")
        labels.loc[unresolved] = (fallback_numeric > 0).astype(float)

    return labels.astype("Int64")


def should_drop_column(column: str, label_column: str, dataset: str) -> bool:
    norm = _norm_col(column)

    if column == label_column:
        return True

    if norm.startswith("unnamed"):
        return True

    if norm in {_norm_col(c) for c in label_candidates_for_dataset(dataset)}:
        return True

    if norm in {_norm_col(c) for c in KNOWN_LEAKAGE_COLUMNS}:
        return True

    if norm in {_norm_col(c) for c in IDENTIFIER_COLUMNS}:
        return True

    return False


def numeric_feature_frame(
    df: pd.DataFrame,
    dataset: str,
    label_column: str,
) -> Tuple[pd.DataFrame, List[str]]:
    feature_data: Dict[str, pd.Series] = {}
    dropped_columns: List[str] = []

    for column in df.columns:
        if should_drop_column(column, label_column, dataset):
            dropped_columns.append(column)
            continue

        series = df[column]

        if series.dtype == object:
            series = (
                series.astype(str)
                .str.strip()
                .str.strip('"')
                .str.strip("'")
                .str.replace(",", ".", regex=False)
            )

        numeric = pd.to_numeric(series, errors="coerce")

        if numeric.notna().sum() == 0:
            dropped_columns.append(column)
            continue

        feature_data[column] = numeric

    features = pd.DataFrame(feature_data)
    return features, dropped_columns


def clean_project_dataframe(
    df: pd.DataFrame,
    dataset: str,
    label_column: str,
) -> Tuple[pd.DataFrame, List[str]]:
    # BugHunter stores bug information as strings/lists in the Bugs column.
    # Empty values mean clean; non-empty bug references mean defective.
    if dataset == "BugHunter" and _norm_col(label_column) in {"bugs", "bug"}:
        raw_labels = (
            df[label_column]
            .astype(str)
            .str.strip()
            .str.strip('"')
            .str.strip("'")
            .str.strip()
            .str.lower()
        )

        empty_tokens = {
            "",
            "0",
            "0.0",
            "false",
            "no",
            "none",
            "nan",
            "null",
            "[]",
            "[ ]",
            "{}",
            "{ }",
        }

        labels = (~raw_labels.isin(empty_tokens)).astype("Int64")
    else:
        labels = normalize_label(df[label_column])

    features, dropped_columns = numeric_feature_frame(
        df=df,
        dataset=dataset,
        label_column=label_column,
    )

    if features.empty:
        return pd.DataFrame(), dropped_columns

    out = features.copy()
    out["label"] = labels

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(axis=0, how="any")
    out = out.drop_duplicates()
    out = out.reset_index(drop=True)

    if "label" not in out.columns:
        return pd.DataFrame(), dropped_columns

    unique_labels = set(out["label"].dropna().astype(int).unique().tolist())

    if not unique_labels.issubset({0, 1}):
        logger.warning("Unexpected labels found in dataset=%s: %s", dataset, sorted(unique_labels))
        return pd.DataFrame(), dropped_columns

    if len(unique_labels) < 2:
        return pd.DataFrame(), dropped_columns

    feature_cols = [col for col in out.columns if col != "label"]

    if not feature_cols:
        return pd.DataFrame(), dropped_columns

    out["label"] = out["label"].astype(int)
    return out, dropped_columns


# ---------------------------------------------------------------------
# Combined dataset splitting
# ---------------------------------------------------------------------

def find_split_column(
    df: pd.DataFrame,
    dataset: str,
    label_column: str,
) -> Optional[str]:
    if dataset not in COMBINED_DATASETS:
        return None

    candidates = PROJECT_SPLIT_CANDIDATES.get(dataset, [])
    norm_to_original = {_norm_col(col): col for col in df.columns}

    for candidate in candidates:
        key = _norm_col(candidate)

        if key not in norm_to_original:
            continue

        col = norm_to_original[key]

        if col == label_column:
            continue

        values = df[col].dropna().astype(str).str.strip()
        nunique = values.nunique()

        if 2 <= nunique <= 300:
            counts = values.value_counts()
            valid_groups = (counts >= 20).sum()

            if valid_groups >= 2:
                return col

    return None


def split_combined_dataframe(
    df: pd.DataFrame,
    dataset: str,
    path: str,
    label_column: str,
    base_project_name: str,
) -> List[Tuple[str, pd.DataFrame]]:
    split_col = find_split_column(df, dataset, label_column)

    if split_col is None:
        return [(base_project_name, df)]

    pieces: List[Tuple[str, pd.DataFrame]] = []

    for value, group in df.groupby(split_col, dropna=True):
        group_name = _safe_name(str(value))

        if group.shape[0] < 20:
            continue

        project_name = f"{base_project_name}__{group_name}"
        pieces.append((project_name, group.copy()))

    if not pieces:
        return [(base_project_name, df)]

    logger.info(
        "Split combined file '%s' using column '%s' into %d project groups",
        path,
        split_col,
        len(pieces),
    )

    return pieces


# ---------------------------------------------------------------------
# File skipping rules
# ---------------------------------------------------------------------

def should_skip_file(path: Path, dataset: str) -> bool:
    name = path.name.lower()
    stem = path.stem.lower()
    parts = [part.lower() for part in path.parts]

    if name.startswith("."):
        return True

    if "__macosx" in parts:
        return True

    if path.suffix.lower() not in {".csv", ".arff"}:
        return True

    # BugHunter contains many granularities and feature-set variants.
    # For DAQUA's main file/module-level analysis, keep only:
    # BugHunterDataset-1.0/full/<project>/file.csv
    # and skip method/class variants and aggregate "all".
    if dataset == "BugHunter":
        if "full" not in parts:
            return True
        if name != "file.csv":
            return True
        if "all" in parts:
            return True

    # ApacheJIT contains helper/link files in data/.
    # Keep the combined metric-label file if available.
    if dataset == "ApacheJIT":
        if "dataset" in parts and stem == "apachejit_total":
            return False
        if "dataset" in parts and stem.startswith("apachejit_"):
            return True
        if "data" in parts:
            return True
        if name.startswith("commit_links"):
            return True
        if name.startswith("keys_"):
            return True
        if stem in {"clean_filtered", "apache_metrics_kamei"}:
            return True
        if re.fullmatch(r"20\d{2}", stem):
            return True
        if stem.startswith("apachejit_train"):
            return True
        if stem.startswith("apachejit_test"):
            return True

    # GHPR baseline file is usually experimental baseline results, not a project dataset.
    if dataset == "GHPR":
        if stem == "baseline":
            return True

    # Keep only the intended unified file/class file depending on folder mapping.
    if dataset == "UnifiedBugDataSet_File":
        if "unified-class" in name:
            return True

    if dataset == "UnifiedBugDataSet_Class":
        if "unified-file" in name:
            return True

    return False


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def make_project_data(
    raw_df: pd.DataFrame,
    dataset: str,
    path: str,
    project_name: str,
    label_column: str,
) -> Optional[ProjectData]:
    cleaned_df, dropped_columns = clean_project_dataframe(
        df=raw_df,
        dataset=dataset,
        label_column=label_column,
    )

    if cleaned_df.empty:
        return None

    return ProjectData(
        dataset=dataset,
        project=project_name,
        path=str(path),
        df=cleaned_df,
        raw_shape=raw_df.shape,
        label_column=label_column,
        dropped_columns=dropped_columns,
    )


def load_project_file(
    path: str,
    dataset: str,
    base_project_name: Optional[str] = None,
) -> List[ProjectData]:
    raw_df = read_table_auto(path, dataset)

    if raw_df is None or raw_df.empty:
        logger.warning("Empty or unreadable file skipped: %s", path)
        return []

    label_columns = candidate_label_columns(raw_df, dataset)

    if not label_columns:
        logger.warning(
            "No label column found. Skipping %s | dataset=%s | columns=%s",
            path,
            dataset,
            list(raw_df.columns),
        )
        return []

    if base_project_name is None:
        base_project_name = _safe_name(Path(path).stem)

    for label_column in label_columns:
        pieces = split_combined_dataframe(
            df=raw_df,
            dataset=dataset,
            path=path,
            label_column=label_column,
            base_project_name=base_project_name,
        )

        loaded: List[ProjectData] = []

        for project_name, piece_df in pieces:
            project = make_project_data(
                raw_df=piece_df,
                dataset=dataset,
                path=path,
                project_name=project_name,
                label_column=label_column,
            )

            if project is not None:
                loaded.append(project)

        if loaded:
            return loaded

    logger.warning(
        "No usable rows/features after trying label candidates. Skipping %s | dataset=%s | candidates=%s",
        path,
        dataset,
        label_columns,
    )

    return []


def load_all_projects(root_dir: str) -> List[ProjectData]:
    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root_dir}")

    projects: List[ProjectData] = []

    for dataset_folder in sorted(root.iterdir()):
        if not dataset_folder.is_dir():
            continue

        dataset = canonical_dataset_name(dataset_folder.name)

        candidate_files = sorted(
            [
                path
                for path in dataset_folder.rglob("*")
                if path.is_file() and path.suffix.lower() in {".csv", ".arff"}
            ]
        )

        candidate_files = [
            path for path in candidate_files if not should_skip_file(path, dataset)
        ]

        logger.info(
            "Loading dataset folder '%s' as '%s' | candidate files=%d",
            dataset_folder.name,
            dataset,
            len(candidate_files),
        )

        loaded_count = 0
        skipped_count = 0

        for file_path in candidate_files:
            try:
                rel = file_path.relative_to(dataset_folder).with_suffix("")
                base_project_name = _safe_name("__".join(rel.parts))
            except Exception:
                base_project_name = _safe_name(file_path.stem)

            loaded_projects = load_project_file(
                path=str(file_path),
                dataset=dataset,
                base_project_name=base_project_name,
            )

            if not loaded_projects:
                skipped_count += 1
                continue

            projects.extend(loaded_projects)
            loaded_count += len(loaded_projects)

        logger.info(
            "Dataset '%s' loaded projects=%d | skipped files=%d",
            dataset,
            loaded_count,
            skipped_count,
        )

    logger.info("Total loaded projects=%d from root=%s", len(projects), root_dir)
    return projects


def projects_to_metadata_frame(projects: Sequence[ProjectData]) -> pd.DataFrame:
    rows = []

    for project in projects:
        rows.append(
            {
                "dataset": project.dataset,
                "project": project.project,
                "path": project.path,
                "raw_rows": project.raw_shape[0],
                "raw_columns": project.raw_shape[1],
                "clean_rows": project.df.shape[0],
                "clean_features": project.df.shape[1] - 1,
                "label_column": project.label_column,
                "dropped_columns_count": len(project.dropped_columns),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    loaded_projects = load_all_projects("Data-set")
    metadata = projects_to_metadata_frame(loaded_projects)

    if metadata.empty:
        print("No projects loaded.")
    else:
        print(metadata.to_string(index=False))
        print()
        print("Datasets loaded:", metadata["dataset"].nunique())
        print("Projects loaded:", len(metadata))
        print()
        print(metadata.groupby("dataset")["project"].count().sort_values(ascending=False))
# ```

# After replacing the file, run only the loader test first:

# ```bash
# cd ~/Home/SPARC

# python3 daqua/loaders/defect_loader.py | tee outputs/logs/loader_test_11datasets.log
# ```

# Then check the dataset counts:

# ```bash
# grep -E "Dataset '|Total loaded projects|Datasets loaded|Projects loaded" outputs/logs/loader_test_11datasets.log
# ```

# You should now see more than the original 5 datasets. If one dataset is still skipped, the log will show the exact label-column or cleaning issue.
