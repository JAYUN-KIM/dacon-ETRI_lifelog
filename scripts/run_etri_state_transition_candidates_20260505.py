import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("/mnt/c/etri-lifelog/data/raw/data")
TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
SUB_PATH = BASE_DIR / "ch2026_submission_sample.csv"
SUB_DIR = BASE_DIR / "submissions"
ANCHOR_PATH = SUB_DIR / "sub_seed3_routing_q6040_s2080_alpha098.csv"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
Q_TARGETS = ["Q1", "Q2", "Q3"]
S_TARGETS = ["S1", "S2", "S3", "S4"]


def infer_test_frame_from_submission(train_df: pd.DataFrame, sub_df: pd.DataFrame) -> pd.DataFrame:
    if {"subject_id", "lifelog_date"}.issubset(sub_df.columns):
        test_df = sub_df[["subject_id", "lifelog_date"]].copy()
        test_df["subject_id"] = test_df["subject_id"].astype(str)
        test_df["lifelog_date"] = pd.to_datetime(test_df["lifelog_date"])
        return test_df

    n_test = len(sub_df)
    last_dates = train_df.groupby("subject_id")["lifelog_date"].max().sort_values()
    subjects = list(last_dates.index)
    reps = math.ceil(n_test / len(subjects))
    subject_seq = (subjects * reps)[:n_test]
    next_dates = {
        sid: train_df.loc[train_df["subject_id"] == sid, "lifelog_date"].max() + pd.Timedelta(days=1)
        for sid in subjects
    }

    rows = []
    for sid in subject_seq:
        rows.append((sid, next_dates[sid]))
        next_dates[sid] += pd.Timedelta(days=1)
    return pd.DataFrame(rows, columns=["subject_id", "lifelog_date"])


def logloss_binary(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y_true = np.asarray(y_true, dtype=float)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())


def mean_or_nan(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def ewma_tail(values: np.ndarray, alpha: float) -> float:
    if len(values) == 0:
        return np.nan
    out = float(values[0])
    for value in values[1:]:
        out = alpha * float(value) + (1 - alpha) * out
    return out


def trend_adjustment(values: np.ndarray, horizon_days: int) -> float:
    if len(values) < 5:
        return 0.0
    tail = np.asarray(values[-14:], dtype=float)
    if len(np.unique(tail)) <= 1:
        return 0.0
    x = np.arange(len(tail), dtype=float)
    slope = float(np.polyfit(x, tail, 1)[0])
    # Binary targets are noisy; trend can help only as a tiny directional prior.
    return float(np.clip(slope * min(max(horizon_days, 0), 14) * 0.12, -0.06, 0.06))


def subject_state_probability(
    history: np.ndarray,
    global_mean: float,
    horizon_days: int,
    dow_adjustment: float = 0.0,
) -> float:
    history = np.asarray(history, dtype=float)
    if len(history) == 0:
        base = global_mean
    else:
        n = len(history)
        all_mean = mean_or_nan(history)
        last3 = mean_or_nan(history[-3:])
        last7 = mean_or_nan(history[-7:])
        last14 = mean_or_nan(history[-14:])
        ewma_fast = ewma_tail(history, 0.45)
        ewma_slow = ewma_tail(history, 0.22)

        # Strong recency, but with enough all-history/global mass to avoid
        # overreacting to one noisy sleep survey answer.
        recent_mix = (
            0.20 * last3
            + 0.28 * last7
            + 0.20 * last14
            + 0.20 * ewma_fast
            + 0.12 * ewma_slow
        )
        subject_mix = 0.72 * recent_mix + 0.20 * all_mean + 0.08 * global_mean

        # Small-sample smoothing is still useful for early test dates.
        smooth = min(n / 18.0, 1.0)
        base = smooth * subject_mix + (1 - smooth) * (0.60 * all_mean + 0.40 * global_mean)
        base += trend_adjustment(history, horizon_days)

    base += dow_adjustment
    return float(np.clip(base, 0.04, 0.96))


def build_dow_adjustments(train: pd.DataFrame, target: str) -> dict[int, float]:
    global_mean = float(train[target].mean())
    dow_mean = train.groupby("dow")[target].mean()
    dow_count = train.groupby("dow")[target].count()
    out = {}
    for dow in range(7):
        if dow not in dow_mean.index:
            out[dow] = 0.0
            continue
        # Very conservative target-level calendar effect.
        reliability = min(float(dow_count.loc[dow]) / 80.0, 1.0)
        out[dow] = float(np.clip((float(dow_mean.loc[dow]) - global_mean) * 0.10 * reliability, -0.025, 0.025))
    return out


def build_state_prior(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sub_template: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    out = sub_template.copy()
    train_sorted = train.sort_values(["subject_id", "lifelog_date"]).copy()
    test_sorted = test.reset_index().sort_values(["subject_id", "lifelog_date"]).copy()

    train_sorted["dow"] = train_sorted["lifelog_date"].dt.dayofweek
    test_sorted["dow"] = test_sorted["lifelog_date"].dt.dayofweek

    for target in TARGETS:
        global_mean = float(train_sorted[target].mean())
        dow_adjust = build_dow_adjustments(train_sorted, target)
        preds = np.zeros(len(test), dtype=float)

        subject_groups = {
            sid: g[["lifelog_date", target]].sort_values("lifelog_date").copy()
            for sid, g in train_sorted.groupby("subject_id")
        }

        for row in test_sorted.itertuples(index=False):
            sid = str(row.subject_id)
            test_date = pd.Timestamp(row.lifelog_date)
            dow = int(row.dow)
            group = subject_groups.get(sid)

            if group is None or len(group) == 0:
                history = np.array([], dtype=float)
                horizon = 0
            elif mode == "past_only":
                hist_df = group[group["lifelog_date"] < test_date]
                history = hist_df[target].astype(float).to_numpy()
                last_date = hist_df["lifelog_date"].max() if len(hist_df) else group["lifelog_date"].min()
                horizon = int((test_date - last_date).days) if pd.notna(last_date) else 0
            elif mode == "full_subject":
                history = group[target].astype(float).to_numpy()
                last_date = group["lifelog_date"].max()
                horizon = int((test_date - last_date).days)
            else:
                raise ValueError(f"unknown mode: {mode}")

            preds[int(row.index)] = subject_state_probability(
                history=history,
                global_mean=global_mean,
                horizon_days=horizon,
                dow_adjustment=dow_adjust.get(dow, 0.0),
            )

        out[target] = preds

    return out


def forward_validate_state_prior(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train_sorted = train.sort_values(["subject_id", "lifelog_date"]).copy()
    train_sorted["dow"] = train_sorted["lifelog_date"].dt.dayofweek

    for target in TARGETS:
        global_mean = float(train_sorted[target].mean())
        dow_adjust = build_dow_adjustments(train_sorted, target)
        y_true = []
        pred_subject_mean = []
        pred_last7 = []
        pred_state = []

        for _, g in train_sorted.groupby("subject_id"):
            values = g[target].astype(float).to_numpy()
            dates = pd.to_datetime(g["lifelog_date"]).to_numpy()
            dows = g["dow"].astype(int).to_numpy()
            for i in range(3, len(g)):
                hist = values[:i]
                y_true.append(values[i])
                pred_subject_mean.append(float(np.clip(np.mean(hist), 0.04, 0.96)))
                pred_last7.append(float(np.clip(np.mean(hist[-7:]), 0.04, 0.96)))
                horizon = int((pd.Timestamp(dates[i]) - pd.Timestamp(dates[i - 1])).days)
                pred_state.append(
                    subject_state_probability(
                        hist,
                        global_mean,
                        horizon,
                        dow_adjustment=dow_adjust.get(int(dows[i]), 0.0),
                    )
                )

        rows.append(
            {
                "target": target,
                "n_eval": len(y_true),
                "subject_mean_logloss": logloss_binary(np.array(y_true), np.array(pred_subject_mean)),
                "last7_logloss": logloss_binary(np.array(y_true), np.array(pred_last7)),
                "state_prior_logloss": logloss_binary(np.array(y_true), np.array(pred_state)),
            }
        )

    result = pd.DataFrame(rows)
    avg = {
        "target": "AVG",
        "n_eval": int(result["n_eval"].sum()),
        "subject_mean_logloss": float(result["subject_mean_logloss"].mean()),
        "last7_logloss": float(result["last7_logloss"].mean()),
        "state_prior_logloss": float(result["state_prior_logloss"].mean()),
    }
    return pd.concat([result, pd.DataFrame([avg])], ignore_index=True)


def blend_with_anchor(anchor: pd.DataFrame, state_prior: pd.DataFrame, q_weight: float, s_weight: float) -> pd.DataFrame:
    out = anchor.copy()
    for target in Q_TARGETS:
        out[target] = (1 - q_weight) * anchor[target].astype(float) + q_weight * state_prior[target].astype(float)
    for target in S_TARGETS:
        out[target] = (1 - s_weight) * anchor[target].astype(float) + s_weight * state_prior[target].astype(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def save_submission(df: pd.DataFrame, name: str) -> Path:
    path = SUB_DIR / name
    df.to_csv(path, index=False)
    print(f"[SAVE] {path}")
    return path


def summarize_candidate(name: str, candidate: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, float | str]:
    diff = (candidate[TARGETS].astype(float) - anchor[TARGETS].astype(float)).abs()
    return {
        "candidate": name,
        "mean_abs_diff": float(diff.values.mean()),
        "max_abs_diff": float(diff.values.max()),
        "q_mean_abs_diff": float(diff[Q_TARGETS].values.mean()),
        "s_mean_abs_diff": float(diff[S_TARGETS].values.mean()),
        "mean_q": float(candidate[Q_TARGETS].values.mean()),
        "mean_s": float(candidate[S_TARGETS].values.mean()),
    }


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    sub = pd.read_csv(SUB_PATH)
    anchor = pd.read_csv(ANCHOR_PATH)

    train["subject_id"] = train["subject_id"].astype(str)
    train["lifelog_date"] = pd.to_datetime(train["lifelog_date"])
    if "subject_id" in sub.columns:
        sub["subject_id"] = sub["subject_id"].astype(str)
    if "lifelog_date" in sub.columns:
        sub["lifelog_date"] = pd.to_datetime(sub["lifelog_date"])

    test = infer_test_frame_from_submission(train, sub)

    print("=" * 100)
    print("ETRI STATE TRANSITION PRIOR CANDIDATES")
    print("New axis: subject-level target state / recency dynamics, no sensor model retraining.")
    print("Anchor:", ANCHOR_PATH.name)
    print("=" * 100)

    cv = forward_validate_state_prior(train)
    print("\n[Forward validation on train target history only]")
    print(cv.to_string(index=False))

    past_prior = build_state_prior(train, test, anchor, mode="past_only")
    full_prior = build_state_prior(train, test, anchor, mode="full_subject")

    candidates = {
        "sub_stateprior_pastonly_pure_20260505.csv": past_prior,
        "sub_stateprior_fullsubject_pure_20260505.csv": full_prior,
        "sub_anchor_q6040_statepast_q10_s05_20260505.csv": blend_with_anchor(anchor, past_prior, q_weight=0.10, s_weight=0.05),
        "sub_anchor_q6040_statepast_q14_s07_20260505.csv": blend_with_anchor(anchor, past_prior, q_weight=0.14, s_weight=0.07),
        "sub_anchor_q6040_statepast_q18_s09_20260505.csv": blend_with_anchor(anchor, past_prior, q_weight=0.18, s_weight=0.09),
        "sub_anchor_q6040_statefull_q08_s04_20260505.csv": blend_with_anchor(anchor, full_prior, q_weight=0.08, s_weight=0.04),
        "sub_anchor_q6040_statefull_q12_s06_20260505.csv": blend_with_anchor(anchor, full_prior, q_weight=0.12, s_weight=0.06),
    }

    summaries = []
    for name, df in candidates.items():
        save_submission(df, name)
        summaries.append(summarize_candidate(name, df, anchor))

    summary_df = pd.DataFrame(summaries)
    print("\n[Candidate drift vs anchor]")
    print(summary_df.to_string(index=False))

    print("\n[Recommendation]")
    print("1) First aggressive-but-controlled candidate: sub_anchor_q6040_statepast_q14_s07_20260505.csv")
    print("2) Safer candidate: sub_anchor_q6040_statepast_q10_s05_20260505.csv")
    print("3) Pure state priors are diagnostic only; submit only if you want a real gamble.")


if __name__ == "__main__":
    main()
