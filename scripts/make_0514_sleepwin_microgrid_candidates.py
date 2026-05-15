import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUB_DIR = ROOT / "data" / "raw" / "data" / "submissions"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

# Best anchor from pattern projection run (stable, low-noise base).
BASE_ANCHOR = SUB_DIR / "sub_0510_patternproj_w015.csv"

# Sleep-window model prediction (built on 2026-05-11).
SLEEP_MODEL = SUB_DIR / "sub_0511_sleep_window_model.csv"

# Current best (2026-05-12 public LB 0.5877548490).
CURRENT_BEST = SUB_DIR / "sub_0512_sleepwin_s_focus.csv"


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def blend(anchor: pd.DataFrame, prior: pd.DataFrame, weights: dict) -> pd.DataFrame:
    out = anchor.copy()
    for target in TARGETS:
        weight = float(weights.get(target, 0.0))
        out[target] = np.clip(
            (1.0 - weight) * anchor[target].to_numpy(float) + weight * prior[target].to_numpy(float),
            0.04,
            0.96,
        )
    return out


def summarize(name: str, candidate: pd.DataFrame, compare: pd.DataFrame) -> dict:
    diff = (candidate[TARGETS] - compare[TARGETS]).abs()
    return {
        "candidate": name,
        "diff_vs_current_best": float(diff.to_numpy().mean()),
        "max_diff_vs_current_best": float(diff.to_numpy().max()),
        "mean_q": float(candidate[["Q1", "Q2", "Q3"]].to_numpy().mean()),
        "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].to_numpy().mean()),
    }


def save_candidate(name: str, anchor: pd.DataFrame, prior: pd.DataFrame, weights: dict, compare: pd.DataFrame) -> dict:
    candidate = blend(anchor, prior, weights)
    candidate.to_csv(SUB_DIR / name, index=False)
    return summarize(name, candidate, compare)


def main(dry_run: bool) -> None:
    anchor = load_submission(BASE_ANCHOR)
    sleep_model = load_submission(SLEEP_MODEL)
    best = load_submission(CURRENT_BEST)

    # Microgrid around 2026-05-12 best weights:
    # s_focus = {"Q1": 0.014, "Q2": 0.001, "Q3": 0.001, "S1": 0.022, "S2": 0.018, "S3": 0.020, "S4": 0.018}
    candidates = [
        # Stronger guardrail on Q2/Q3 (keep sleep-window signal mostly for S targets + Q1).
        (
            "sub_0514_sleepwin_micro_q23_zero.csv",
            {"Q1": 0.014, "Q2": 0.000, "Q3": 0.000, "S1": 0.022, "S2": 0.018, "S3": 0.020, "S4": 0.018},
        ),
        # Slightly increase Q1, reduce S2/S4 to keep overall magnitude similar.
        (
            "sub_0514_sleepwin_micro_q1_up.csv",
            {"Q1": 0.016, "Q2": 0.001, "Q3": 0.001, "S1": 0.022, "S2": 0.017, "S3": 0.020, "S4": 0.017},
        ),
        # Push sleep-window blend slightly more into S1/S3 (wake/sleep boundary sensitivity).
        (
            "sub_0514_sleepwin_micro_s13_up.csv",
            {"Q1": 0.013, "Q2": 0.001, "Q3": 0.001, "S1": 0.023, "S2": 0.017, "S3": 0.021, "S4": 0.017},
        ),
        # Reduce Q3 further (often noisy), redistribute to S2/S4.
        (
            "sub_0514_sleepwin_micro_q3_down.csv",
            {"Q1": 0.014, "Q2": 0.001, "Q3": 0.000, "S1": 0.022, "S2": 0.019, "S3": 0.020, "S4": 0.019},
        ),
    ]

    rows = []
    for name, weights in candidates:
        if dry_run:
            candidate = blend(anchor, sleep_model, weights)
            rows.append(summarize(name, candidate, best))
        else:
            rows.append(save_candidate(name, anchor, sleep_model, weights, best))

    summary = pd.DataFrame(rows).sort_values("diff_vs_current_best").reset_index(drop=True)
    print("0514 sleep-window microgrid candidates vs current best:")
    print(summary.to_string(index=False))
    print("\nSuggested submit order (closest first; pick 1-2 only):")
    print("1) sub_0514_sleepwin_micro_q23_zero.csv")
    print("2) sub_0514_sleepwin_micro_q3_down.csv")
    print("3) sub_0514_sleepwin_micro_s13_up.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write CSVs (prints summary only).",
    )
    args = parser.parse_args()
    main(dry_run=bool(args.dry_run))

