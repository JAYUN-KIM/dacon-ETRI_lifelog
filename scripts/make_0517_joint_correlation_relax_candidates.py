from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "data"
TRAIN_PATH = DATA_DIR / "ch2026_metrics_train.csv"
SUB_DIR = DATA_DIR / "submissions"
CURRENT_BEST = SUB_DIR / "sub_0515_metricproxy_s_only.csv"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def logit(x):
    x = np.clip(x, 1e-5, 1 - 1e-5)
    return np.log(x / (1 - x))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def centered_rank(series):
    return series.rank(pct=True, method="average").to_numpy(float) - 0.5


def relax_q1_s_tail(best, strength=0.020, use_logit=False):
    out = best.copy()
    q1_rank = centered_rank(best["Q1"])
    # Current best has very strong negative Q1-S3/S4 correlation. Nudge S3/S4
    # upward for high-Q1 rows and downward for low-Q1 rows to relax that link.
    for target, scale in [("S3", 1.00), ("S4", 0.80), ("S2", 0.30)]:
        delta = strength * scale * q1_rank
        if use_logit:
            out[target] = sigmoid(logit(best[target].to_numpy(float)) + delta)
        else:
            out[target] = best[target].to_numpy(float) + delta
        out[target] = np.clip(out[target], 0.04, 0.96)
    return out


def relax_s_block(best, strength=0.018):
    out = best.copy()
    s_mean_rank = centered_rank(best[["S1", "S2", "S3", "S4"]].mean(axis=1))
    # S block correlations are much stronger than train. Pull each S target a
    # little away from the shared S-block factor while preserving marginal means.
    for target, scale in [("S1", 0.70), ("S2", 0.85), ("S3", 0.85), ("S4", 0.70)]:
        own = centered_rank(best[target])
        residual = own - s_mean_rank
        delta = strength * scale * residual
        out[target] = np.clip(best[target].to_numpy(float) + delta, 0.04, 0.96)
    return out


def relax_q2_q3(best, strength=0.010):
    out = best.copy()
    # Q2/Q3 train correlation is real. This candidate keeps that pair but
    # reduces their coupling to S1/S2 a little.
    stress_rank = centered_rank(best[["Q2", "Q3"]].mean(axis=1))
    for target, scale in [("S1", -0.50), ("S2", -0.35), ("Q2", 0.15), ("Q3", 0.15)]:
        out[target] = np.clip(best[target].to_numpy(float) + strength * scale * stress_rank, 0.04, 0.96)
    return out


def summarize(name, cand, best):
    diff = (cand[TARGETS] - best[TARGETS]).abs()
    corr = cand[TARGETS].corr()
    return {
        "candidate": name,
        "mean_abs_diff": float(diff.to_numpy().mean()),
        "max_abs_diff": float(diff.to_numpy().max()),
        "q1_s3_corr": float(corr.loc["Q1", "S3"]),
        "q1_s4_corr": float(corr.loc["Q1", "S4"]),
        "s2_s3_corr": float(corr.loc["S2", "S3"]),
        "mean_q": float(cand[["Q1", "Q2", "Q3"]].to_numpy().mean()),
        "mean_s": float(cand[["S1", "S2", "S3", "S4"]].to_numpy().mean()),
    }


def save(name, cand, best):
    cand.to_csv(SUB_DIR / name, index=False)
    return summarize(name, cand, best)


def main():
    train = pd.read_csv(TRAIN_PATH)
    best = pd.read_csv(CURRENT_BEST)

    print("Train corr snapshot:")
    print(train[TARGETS].corr().loc[["Q1", "S2"], ["S3", "S4"]].round(3).to_string())
    print("\nCurrent best corr snapshot:")
    print(best[TARGETS].corr().loc[["Q1", "S2"], ["S3", "S4"]].round(3).to_string())

    candidates = [
        ("sub_0517_corr_relax_q1_s34_prob.csv", relax_q1_s_tail(best, strength=0.014, use_logit=False)),
        ("sub_0517_corr_relax_q1_s34_logit.csv", relax_q1_s_tail(best, strength=0.020, use_logit=True)),
        ("sub_0517_corr_relax_s_block.csv", relax_s_block(best, strength=0.014)),
        ("sub_0517_corr_relax_qstress.csv", relax_q2_q3(best, strength=0.012)),
    ]

    rows = [save(name, cand, best) for name, cand in candidates]
    summary = pd.DataFrame(rows).sort_values("mean_abs_diff").reset_index(drop=True)
    print("\n0517 joint-correlation relaxation candidates vs current best:")
    print(summary.to_string(index=False))
    print("\nSuggested submit order:")
    print("1) sub_0517_corr_relax_q1_s34_logit.csv")
    print("2) sub_0517_corr_relax_q1_s34_prob.csv")
    print("3) sub_0517_corr_relax_s_block.csv")


if __name__ == "__main__":
    main()
