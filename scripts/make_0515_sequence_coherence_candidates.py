from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUB_DIR = ROOT / "data" / "raw" / "data" / "submissions"
TRAIN_PATH = ROOT / "data" / "raw" / "data" / "ch2026_metrics_train.csv"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

# Current public best as of 2026-05-12.
CURRENT_BEST = SUB_DIR / "sub_0512_sleepwin_s_focus.csv"


def logit(x):
    x = np.clip(x, 1e-5, 1 - 1e-5)
    return np.log(x / (1 - x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_submission(path):
    df = pd.read_csv(path)
    df["lifelog_date"] = pd.to_datetime(df["lifelog_date"])
    for t in TARGETS:
        df[t] = pd.to_numeric(df[t], errors="coerce")
    return df


def train_persistence_summary():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["lifelog_date"])
    rows = []
    for target in TARGETS:
        corrs = []
        flips = []
        for _, group in train.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id"):
            values = group[target].astype(float).to_numpy()
            if len(values) > 2:
                corr = pd.Series(values).autocorr(lag=1)
                if np.isfinite(corr):
                    corrs.append(corr)
                flips.extend((values[1:] != values[:-1]).astype(float).tolist())
        rows.append(
            {
                "target": target,
                "lag1_corr": float(np.mean(corrs)) if corrs else 0.0,
                "flip_rate": float(np.mean(flips)) if flips else np.nan,
            }
        )
    return pd.DataFrame(rows)


def triangular_smooth(values):
    s = pd.Series(values, dtype=float)
    prev = s.shift(1)
    nxt = s.shift(-1)
    out = 0.50 * s + 0.25 * prev.fillna(s) + 0.25 * nxt.fillna(s)
    return out.to_numpy(float)


def ema_bidirectional(values, alpha=0.42):
    values = np.asarray(values, dtype=float)
    fwd = values.copy()
    for i in range(1, len(values)):
        fwd[i] = alpha * values[i] + (1 - alpha) * fwd[i - 1]
    bwd = values.copy()
    for i in range(len(values) - 2, -1, -1):
        bwd[i] = alpha * values[i] + (1 - alpha) * bwd[i + 1]
    return 0.5 * fwd + 0.5 * bwd


def build_smoothed_prior(best, mode):
    prior = best.copy()
    for sid, index in best.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id").groups.items():
        ordered_idx = list(index)
        for target in TARGETS:
            values = best.loc[ordered_idx, target].to_numpy(float)
            if mode == "tri_prob":
                smoothed = triangular_smooth(values)
            elif mode == "tri_logit":
                smoothed = sigmoid(triangular_smooth(logit(values)))
            elif mode == "ema_prob":
                smoothed = ema_bidirectional(values, alpha=0.42)
            else:
                raise ValueError(mode)
            prior.loc[ordered_idx, target] = np.clip(smoothed, 0.04, 0.96)
    return prior


def blend(best, prior, weights, adaptive=False):
    out = best.copy()
    for target in TARGETS:
        base_w = float(weights.get(target, 0.0))
        delta = prior[target].to_numpy(float) - best[target].to_numpy(float)
        if adaptive:
            # Only nudge cells whose prediction is locally inconsistent enough.
            scale = np.quantile(np.abs(delta), 0.70) + 1e-9
            cell_w = base_w * np.clip(np.abs(delta) / scale, 0.25, 1.25)
        else:
            cell_w = base_w
        out[target] = np.clip(best[target].to_numpy(float) + cell_w * delta, 0.04, 0.96)
    return out


def save(name, best, prior, weights, adaptive=False):
    cand = blend(best, prior, weights, adaptive=adaptive)
    path = SUB_DIR / name
    cand.to_csv(path, index=False)
    diff = (cand[TARGETS] - best[TARGETS]).abs()
    return {
        "candidate": name,
        "mean_abs_diff": float(diff.to_numpy().mean()),
        "max_abs_diff": float(diff.to_numpy().max()),
        "mean_q": float(cand[["Q1", "Q2", "Q3"]].to_numpy().mean()),
        "mean_s": float(cand[["S1", "S2", "S3", "S4"]].to_numpy().mean()),
    }


def main():
    best = load_submission(CURRENT_BEST)

    print("Train target persistence summary:")
    print(train_persistence_summary().to_string(index=False))

    tri_prob = build_smoothed_prior(best, "tri_prob")
    tri_logit = build_smoothed_prior(best, "tri_logit")
    ema_prob = build_smoothed_prior(best, "ema_prob")

    rows = []
    rows.append(
        save(
            "sub_0515_seqcoh_q23_tri_prob.csv",
            best,
            tri_prob,
            {"Q1": 0.004, "Q2": 0.020, "Q3": 0.018, "S1": 0.002, "S2": 0.000, "S3": 0.000, "S4": 0.000},
        )
    )
    rows.append(
        save(
            "sub_0515_seqcoh_q23_tri_logit.csv",
            best,
            tri_logit,
            {"Q1": 0.004, "Q2": 0.018, "Q3": 0.016, "S1": 0.002, "S2": 0.000, "S3": 0.000, "S4": 0.000},
        )
    )
    rows.append(
        save(
            "sub_0515_seqcoh_adaptive_q.csv",
            best,
            ema_prob,
            {"Q1": 0.006, "Q2": 0.024, "Q3": 0.020, "S1": 0.000, "S2": 0.000, "S3": 0.000, "S4": 0.000},
            adaptive=True,
        )
    )
    rows.append(
        save(
            "sub_0515_seqcoh_all_tiny.csv",
            best,
            tri_prob,
            {"Q1": 0.004, "Q2": 0.012, "Q3": 0.012, "S1": 0.003, "S2": 0.002, "S3": 0.002, "S4": 0.002},
        )
    )

    summary = pd.DataFrame(rows).sort_values("mean_abs_diff").reset_index(drop=True)
    print("\n0515 sequence-coherence candidates vs current best:")
    print(summary.to_string(index=False))
    print("\nSuggested submit order:")
    print("1) sub_0515_seqcoh_q23_tri_prob.csv")
    print("2) sub_0515_seqcoh_q23_tri_logit.csv")
    print("3) sub_0515_seqcoh_adaptive_q.csv")


if __name__ == "__main__":
    main()
