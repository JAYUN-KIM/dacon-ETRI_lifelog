import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUB_DIR = ROOT / "data" / "raw" / "data" / "submissions"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
REQUIRED_COLUMNS = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = DEFAULT_SUB_DIR / path
    return path


def validate(path):
    df = pd.read_csv(path)
    print("path:", path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("head:")
    print(df.head())
    print("nulls:")
    print(df.isnull().sum())
    print("describe:")
    print(df.describe())

    errors = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    if df.shape != (250, 10):
        errors.append(f"unexpected shape: {df.shape}, expected (250, 10)")
    if df.isnull().sum().sum() > 0:
        errors.append("submission has null values")
    if not missing:
        in_range = ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all()
        print("all targets in [0,1]:", bool(in_range))
        print("target min:")
        print(df[TARGETS].min())
        print("target max:")
        print(df[TARGETS].max())
        if not in_range:
            errors.append("target probabilities are outside [0, 1]")

    if errors:
        print("[FAIL]", " | ".join(errors))
        raise SystemExit(1)
    print("[OK] submission validation passed")
    return df


def compare(current_path, compare_path):
    current = pd.read_csv(current_path)
    baseline = pd.read_csv(compare_path)
    diff = (current[TARGETS] - baseline[TARGETS]).abs()
    print("compare_to:", compare_path)
    print("mean_abs_diff_all:", float(diff.values.mean()))
    print("max_abs_diff_all:", float(diff.values.max()))
    print("per_target_mean_abs_diff:")
    print(diff.mean())
    print("new_minus_compare_mean:")
    print(current[TARGETS].mean() - baseline[TARGETS].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Submission csv path or file name under data/raw/data/submissions")
    parser.add_argument("--compare", help="Optional baseline submission path or file name")
    args = parser.parse_args()

    path = resolve_path(args.path)
    validate(path)

    if args.compare:
        compare_path = resolve_path(args.compare)
        compare(path, compare_path)


if __name__ == "__main__":
    main()
