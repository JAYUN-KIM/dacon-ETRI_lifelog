from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUB_DIR = ROOT / "data" / "raw" / "data" / "submissions"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

BASE_ANCHOR = SUB_DIR / "sub_0510_patternproj_w015.csv"
BEST_SLEEP = SUB_DIR / "sub_0511_sleepwin_sleep_targets.csv"
SLEEP_MODEL = SUB_DIR / "sub_0511_sleep_window_model.csv"


def load_submission(path):
    df = pd.read_csv(path)
    for col in TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def blend(anchor, prior, weights):
    out = anchor.copy()
    for target in TARGETS:
        weight = float(weights.get(target, 0.0))
        out[target] = np.clip(
            (1.0 - weight) * anchor[target].to_numpy(float) + weight * prior[target].to_numpy(float),
            0.04,
            0.96,
        )
    return out


def summarize(name, candidate, compare):
    diff = (candidate[TARGETS] - compare[TARGETS]).abs()
    return {
        "candidate": name,
        "diff_vs_best": float(diff.to_numpy().mean()),
        "max_diff_vs_best": float(diff.to_numpy().max()),
        "mean_q": float(candidate[["Q1", "Q2", "Q3"]].to_numpy().mean()),
        "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].to_numpy().mean()),
    }


def save_candidate(name, anchor, prior, weights, compare):
    candidate = blend(anchor, prior, weights)
    candidate.to_csv(SUB_DIR / name, index=False)
    return summarize(name, candidate, compare)


def main():
    base = load_submission(BASE_ANCHOR)
    best = load_submission(BEST_SLEEP)
    sleep_model = load_submission(SLEEP_MODEL)

    # 2026-05-11 public results:
    # all w006 = 0.5881257172, target-wise sleep = 0.5878818802, all w012 = 0.588077498.
    # Interpretation: sleep-window signal is real, but Q2/Q3 should stay tiny.
    candidates = [
        (
            "sub_0512_sleepwin_plus_soft.csv",
            {"Q1": 0.018, "Q2": 0.003, "Q3": 0.003, "S1": 0.020, "S2": 0.017, "S3": 0.020, "S4": 0.017},
        ),
        (
            "sub_0512_sleepwin_s_focus.csv",
            {"Q1": 0.014, "Q2": 0.001, "Q3": 0.001, "S1": 0.022, "S2": 0.018, "S3": 0.020, "S4": 0.018},
        ),
        (
            "sub_0512_sleepwin_q1_s1s3.csv",
            {"Q1": 0.022, "Q2": 0.002, "Q3": 0.000, "S1": 0.022, "S2": 0.014, "S3": 0.022, "S4": 0.014},
        ),
        (
            "sub_0512_sleepwin_guarded.csv",
            {"Q1": 0.016, "Q2": 0.003, "Q3": 0.002, "S1": 0.018, "S2": 0.015, "S3": 0.018, "S4": 0.015},
        ),
    ]

    rows = []
    for name, weights in candidates:
        rows.append(save_candidate(name, base, sleep_model, weights, best))

    summary = pd.DataFrame(rows).sort_values("diff_vs_best").reset_index(drop=True)
    print("0512 sleep-window refine candidates vs current best:")
    print(summary.to_string(index=False))
    print("\nSuggested submit order:")
    print("1) sub_0512_sleepwin_guarded.csv")
    print("2) sub_0512_sleepwin_plus_soft.csv")
    print("3) sub_0512_sleepwin_s_focus.csv")


if __name__ == "__main__":
    main()
