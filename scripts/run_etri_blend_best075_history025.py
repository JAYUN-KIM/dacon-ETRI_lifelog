from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path("/mnt/c/etri-lifelog/data/raw/data")
SUB_DIR = BASE_DIR / "submissions"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

# Current best model submission
BEST_PATH = SUB_DIR / "sub_seed3_routing_q5050_s2080_alpha098.csv"

# Different-family model: target history prior + sensor/routing model
# This was worse alone but close enough and structurally different.
HISTORY_PATH = SUB_DIR / "sub_target_history_seed3_routing_q5050_s2080_alpha098.csv"

# First experimental submission-level ensemble
# Keep best dominant and add only a small amount of history-prior signal.
BEST_W = 0.75
HISTORY_W = 0.25

OUT_PATH = SUB_DIR / "sub_blend_best075_history025.csv"


def check_submission(df, name):
    required = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

    if df[TARGETS].isnull().sum().sum() > 0:
        raise ValueError(f"{name} has null values")

    if not ((df[TARGETS] >= 0).all().all() and (df[TARGETS] <= 1).all().all()):
        raise ValueError(f"{name} target values are outside [0, 1]")


def main():
    if not BEST_PATH.exists():
        raise FileNotFoundError(f"Best submission not found: {BEST_PATH}")
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"History submission not found: {HISTORY_PATH}")

    best = pd.read_csv(BEST_PATH)
    hist = pd.read_csv(HISTORY_PATH)

    check_submission(best, "best")
    check_submission(hist, "history")

    if len(best) != len(hist):
        raise ValueError(f"Row count mismatch: best={len(best)}, history={len(hist)}")

    key_cols = ["subject_id", "sleep_date", "lifelog_date"]
    for c in key_cols:
        if not best[c].astype(str).equals(hist[c].astype(str)):
            raise ValueError(f"Key column mismatch: {c}")

    out = best.copy()

    # Weighted arithmetic blend in submission space.
    for t in TARGETS:
        out[t] = BEST_W * best[t].astype(float) + HISTORY_W * hist[t].astype(float)

    # Conservative clipping.
    out[TARGETS] = out[TARGETS].clip(0.03, 0.97)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print("saved:", OUT_PATH)
    print("blend:", f"best={BEST_W}, history={HISTORY_W}")
    print()
    print(out.head())
    print()
    print("null counts:")
    print(out.isnull().sum())
    print()
    print("describe:")
    print(out[TARGETS].describe())
    print()
    print("all targets in [0,1]:", ((out[TARGETS] >= 0) & (out[TARGETS] <= 1)).all().all())

    # Difference check
    diff_best = (out[TARGETS] - best[TARGETS]).abs()
    diff_hist = (out[TARGETS] - hist[TARGETS]).abs()
    print()
    print("changed cells vs best:", int((diff_best > 1e-12).sum().sum()))
    print("max abs diff vs best:", float(diff_best.to_numpy().max()))
    print("mean abs diff vs best:", float(diff_best.to_numpy().mean()))
    print("mean abs diff vs history:", float(diff_hist.to_numpy().mean()))


if __name__ == "__main__":
    main()
