from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "data"
TRAIN_PATH = DATA_DIR / "ch2026_metrics_train.csv"
SAMPLE_PATH = DATA_DIR / "ch2026_submission_sample.csv"
SUB_DIR = DATA_DIR / "submissions"
CURRENT_BEST = SUB_DIR / "sub_0515_metricproxy_s_only.csv"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def recency_weighted_mean(values, half_life=10.0):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan
    age = np.arange(len(values))[::-1]
    weights = 0.5 ** (age / half_life)
    return float(np.sum(values * weights) / np.sum(weights))


def trend_slope(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 5:
        return 0.0
    x = np.linspace(-1, 1, len(values))
    try:
        return float(np.polyfit(x, values, 1)[0])
    except Exception:
        return 0.0


def build_subject_target_prior(train, sample):
    rows = []
    global_means = train[TARGETS].mean()
    for sid, test_g in sample.groupby("subject_id"):
        train_g = train[train["subject_id"] == sid].sort_values("lifelog_date")
        test_len = len(test_g)
        for target in TARGETS:
            y = train_g[target].astype(float).to_numpy()
            if len(y) == 0:
                pred = float(global_means[target])
            else:
                overall = float(np.mean(y))
                recent_n = min(max(8, test_len // 2), len(y))
                recent = float(np.mean(y[-recent_n:]))
                ewma = recency_weighted_mean(y, half_life=max(6.0, len(y) / 5.0))
                slope = trend_slope(y[-min(len(y), 24):])
                # Extrapolate only weakly; binary labels are noisy.
                projected = np.clip(ewma + 0.18 * slope, 0.02, 0.98)
                stability = 1.0 / (1.0 + float(np.std(y[-recent_n:])) if recent_n > 1 else 1.0)
                pred = (
                    0.28 * overall
                    + 0.34 * recent
                    + 0.28 * ewma
                    + 0.10 * projected
                )
                # If the recent window is unstable, pull back toward the subject overall.
                pred = stability * pred + (1 - stability) * (0.65 * overall + 0.35 * float(global_means[target]))
            rows.append({"subject_id": sid, "target": target, "prior_mean": float(np.clip(pred, 0.05, 0.95))})
    return pd.DataFrame(rows)


def apply_subject_mean_shift(best, prior_long, strengths):
    out = best.copy()
    best_means = best.groupby("subject_id")[TARGETS].mean()
    prior = prior_long.pivot(index="subject_id", columns="target", values="prior_mean")
    for sid, idx in best.groupby("subject_id").groups.items():
        for target in TARGETS:
            strength = float(strengths.get(target, 0.0))
            if strength <= 0 or sid not in prior.index:
                continue
            target_mean = float(prior.loc[sid, target])
            current_mean = float(best_means.loc[sid, target])
            shift = np.clip(target_mean - current_mean, -0.08, 0.08)
            out.loc[idx, target] = np.clip(best.loc[idx, target].to_numpy(float) + strength * shift, 0.04, 0.96)
    return out


def save_candidate(name, best, prior_long, strengths):
    cand = apply_subject_mean_shift(best, prior_long, strengths)
    cand.to_csv(SUB_DIR / name, index=False)
    diff = (cand[TARGETS] - best[TARGETS]).abs()
    return {
        "candidate": name,
        "mean_abs_diff": float(diff.to_numpy().mean()),
        "max_abs_diff": float(diff.to_numpy().max()),
        "mean_q": float(cand[["Q1", "Q2", "Q3"]].to_numpy().mean()),
        "mean_s": float(cand[["S1", "S2", "S3", "S4"]].to_numpy().mean()),
    }


def main():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["lifelog_date"])
    sample = pd.read_csv(SAMPLE_PATH, parse_dates=["lifelog_date"])
    best = pd.read_csv(CURRENT_BEST)
    prior_long = build_subject_target_prior(train, sample)

    print("Subject target prior preview:")
    print(prior_long.head(20).to_string(index=False))

    rows = []
    rows.append(
        save_candidate(
            "sub_0516_subject_mean_s_only.csv",
            best,
            prior_long,
            {"Q1": 0.000, "Q2": 0.000, "Q3": 0.000, "S1": 0.035, "S2": 0.035, "S3": 0.030, "S4": 0.030},
        )
    )
    rows.append(
        save_candidate(
            "sub_0516_subject_mean_qs_soft.csv",
            best,
            prior_long,
            {"Q1": 0.020, "Q2": 0.018, "Q3": 0.018, "S1": 0.035, "S2": 0.035, "S3": 0.030, "S4": 0.030},
        )
    )
    rows.append(
        save_candidate(
            "sub_0516_subject_mean_q_focus.csv",
            best,
            prior_long,
            {"Q1": 0.030, "Q2": 0.028, "Q3": 0.026, "S1": 0.015, "S2": 0.015, "S3": 0.012, "S4": 0.012},
        )
    )

    summary = pd.DataFrame(rows).sort_values("mean_abs_diff").reset_index(drop=True)
    print("\n0516 subject-mean forecast candidates vs current best:")
    print(summary.to_string(index=False))
    print("\nSuggested submit order:")
    print("1) sub_0516_subject_mean_s_only.csv")
    print("2) sub_0516_subject_mean_qs_soft.csv")
    print("3) sub_0516_subject_mean_q_focus.csv")


if __name__ == "__main__":
    main()
