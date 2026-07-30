from pathlib import Path
import re
import pandas as pd
import numpy as np


RAW_ROOT = Path("Data-set/BugHunterDataset-1.0/full")
OUT_ROOT = Path("Data-set/BugHunterClean/full")
MANIFEST = Path("outputs/preprocessing/bughunter_clean_manifest.csv")


def safe_name(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"[^\w.\-]+", "_", x)
    return x.strip("_") or "unknown"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = []
    seen = {}

    for c in df.columns:
        name = str(c).replace("\ufeff", "").strip().strip('"').strip("'").strip()
        if not name:
            name = "unnamed"

        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0

        cols.append(name)

    df.columns = cols
    return df


def find_bugs_col(df: pd.DataFrame):
    bug_label_names = {
        "bug",
        "bugs",
        "numberofbugs",
        "numberbugs",
        "bugcount",
        "numberofdefects",
        "defectcount",
    }

    for c in df.columns:
        norm = re.sub(r"[^a-z0-9]+", "", str(c).lower())
        if norm in bug_label_names:
            return c

    return None


def make_label(s: pd.Series) -> pd.Series:
    raw = (
        s.astype(str)
        .str.strip()
        .str.strip('"')
        .str.strip("'")
        .str.strip()
    )

    numeric = pd.to_numeric(raw, errors="coerce")

    if numeric.notna().mean() >= 0.80:
        return (numeric.fillna(0) > 0).astype(int)

    empty = {
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

    cleaned = raw.str.lower()
    return (~cleaned.isin(empty)).astype(int)


def numeric_features(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    drop_like = {
        "id", "name", "file", "filename", "path", "filepath",
        "class", "classname", "package", "project", "version",
        "release", "module", "method", "signature", "bugs", "bug"
    }

    out = {}

    for c in df.columns:
        norm = re.sub(r"[^a-z0-9]+", "", str(c).lower())

        if c == label_col or norm in drop_like:
            continue

        s = df[c]

        if s.dtype == object:
            s = (
                s.astype(str)
                .str.strip()
                .str.strip('"')
                .str.strip("'")
                .str.strip()
                .str.replace(",", ".", regex=False)
            )

        num = pd.to_numeric(s, errors="coerce")

        if num.notna().sum() == 0:
            continue

        out[c] = num

    return pd.DataFrame(out)


def main():
    rows = []

    if not RAW_ROOT.exists():
        raise SystemExit(f"Raw BugHunter folder not found: {RAW_ROOT}")

    files = sorted(RAW_ROOT.glob("*/file.csv"))

    for f in files:
        project = f.parent.name

        try:
            df = pd.read_csv(f, engine="python", on_bad_lines="skip")
        except Exception as e:
            print("READ_FAIL", f, e)
            continue

        df = clean_columns(df)
        label_col = find_bugs_col(df)

        if label_col is None:
            print("NO_BUGS_COL", f, df.columns.tolist())
            continue

        y = make_label(df[label_col])
        X = numeric_features(df, label_col)

        clean = X.copy()
        clean["label"] = y

        clean = clean.replace([np.inf, -np.inf], np.nan)
        clean = clean.dropna(axis=0, how="any")
        clean = clean.drop_duplicates()
        clean = clean.reset_index(drop=True)

        if clean.empty:
            print("EMPTY_AFTER_CLEAN", f)
            continue

        if clean["label"].nunique() < 2:
            print("ONE_CLASS_SKIP", f, clean["label"].value_counts().to_dict())
            continue

        if clean.shape[1] <= 1:
            print("NO_FEATURES_SKIP", f)
            continue

        out_dir = OUT_ROOT / safe_name(project)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "file.csv"
        clean.to_csv(out_file, index=False)

        rows.append({
            "project": project,
            "raw_path": str(f),
            "clean_path": str(out_file),
            "raw_rows": df.shape[0],
            "raw_cols": df.shape[1],
            "clean_rows": clean.shape[0],
            "clean_features": clean.shape[1] - 1,
            "positive_rate": float(clean["label"].mean()),
            "label_col": label_col,
        })

        print("SAVED", out_file, clean.shape)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(MANIFEST, index=False)

    print()
    print("BugHunter cleaned projects:", len(rows))
    print("Manifest:", MANIFEST)


if __name__ == "__main__":
    main()
