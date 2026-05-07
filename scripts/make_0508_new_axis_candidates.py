from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "raw" / "data"
SUB_DIR = BASE_DIR / "submissions"
TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"

CURRENT_BEST = SUB_DIR / "sub_adaptive_dateconf_dateprior_b03_s10_20260507.csv"
DATE_PRIOR = SUB_DIR / "sub_dateinterp_smooth_tau10_pure_20260506.csv"
BRACKET_PRIOR = SUB_DIR / "sub_bracket_bracket_mid_pure_20260507.csv"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def validate(df: pd.DataFrame, name: str) -> None:
    required = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    if df.shape != (250, 10):
        raise ValueError(f"{name} unexpected shape: {df.shape}")
    if df.isnull().sum().sum() != 0:
        raise ValueError(f"{name} has null values")
    if not ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all():
        raise ValueError(f"{name} has probabilities outside [0,1]")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logloss(y_true: np.ndarray, pred: np.ndarray) -> float:
    pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
    y_true = np.asarray(y_true, dtype=float)
    return float(-(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred)).mean())


def shift_to_mean(probs: np.ndarray, target_mean: float) -> np.ndarray:
    logits = logit(probs)
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        mean = sigmoid(logits + mid).mean()
        if mean < target_mean:
            lo = mid
        else:
            hi = mid
    return sigmoid(logits + (lo + hi) / 2.0)


def mean_align(anchor: pd.DataFrame, train: pd.DataFrame, gamma: float) -> pd.DataFrame:
    out = anchor.copy()
    anchor_mean = anchor[TARGETS].mean()
    train_mean = train[TARGETS].mean()
    desired = (1 - gamma) * anchor_mean + gamma * train_mean
    for target in TARGETS:
        out[target] = shift_to_mean(anchor[target].to_numpy(float), float(desired[target]))
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def build_date_confidence(train: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    train_dates = {
        str(sid): pd.to_datetime(group["lifelog_date"]).sort_values().to_numpy()
        for sid, group in train.groupby("subject_id")
    }
    rows = []
    for row in sample.itertuples(index=False):
        sid = str(row.subject_id)
        date = pd.Timestamp(row.lifelog_date)
        dates = train_dates.get(sid)
        if dates is None or len(dates) == 0:
            before_dist, after_dist, min_dist, bracketed = np.nan, np.nan, 99.0, 0.0
        else:
            deltas = np.array([(pd.Timestamp(d) - date).days for d in dates], dtype=float)
            before = np.abs(deltas[deltas < 0])
            after = np.abs(deltas[deltas > 0])
            before_dist = float(before.min()) if len(before) else np.nan
            after_dist = float(after.min()) if len(after) else np.nan
            min_dist = float(np.nanmin([before_dist, after_dist]))
            bracketed = float(np.isfinite(before_dist) and np.isfinite(after_dist))
        near = float(np.exp(-min_dist / 10.0))
        both = bracketed * float(
            np.exp(-(np.nan_to_num(before_dist, nan=30.0) + np.nan_to_num(after_dist, nan=30.0)) / 24.0)
        )
        confidence = float(np.clip(0.65 * near + 0.35 * both, 0.0, 1.0))
        rows.append(
            {
                "subject_id": sid,
                "lifelog_date": date,
                "min_dist": min_dist,
                "bracketed": bracketed,
                "confidence": confidence,
            }
        )
    return pd.DataFrame(rows)


def subject_reliability(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, group in train.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < 12:
            reliability = 0.75
        else:
            errs = []
            base_errs = []
            for i in range(4, n):
                hist = group.iloc[:i]
                curr = group.iloc[i]
                prev = hist[TARGETS].tail(7).mean()
                subj = hist[TARGETS].mean()
                globalish = train[TARGETS].mean()
                pred = (0.55 * prev + 0.30 * subj + 0.15 * globalish).to_numpy(float)
                base = subj.to_numpy(float)
                y = curr[TARGETS].to_numpy(float)
                errs.append(logloss(y, pred))
                base_errs.append(logloss(y, base))
            gain = np.mean(base_errs) - np.mean(errs)
            reliability = float(np.clip(0.85 + gain * 4.0, 0.55, 1.20))
        rows.append({"subject_id": str(sid), "subject_reliability": reliability})
    return pd.DataFrame(rows)


def blend_with_weight_matrix(anchor: pd.DataFrame, prior: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    out = anchor.copy()
    for target in TARGETS:
        w = weights[target].to_numpy(float)
        out[target] = (1 - w) * anchor[target].to_numpy(float) + w * prior[target].to_numpy(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def subject_reliable_adaptive(anchor: pd.DataFrame, prior: pd.DataFrame, train: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    conf = build_date_confidence(train, sample)
    rel = subject_reliability(train)
    meta = sample[["subject_id"]].merge(rel, on="subject_id", how="left")
    reliability = meta["subject_reliability"].fillna(0.8).to_numpy(float)
    row_conf = conf["confidence"].to_numpy(float)
    base_weight = np.clip((0.020 + 0.095 * row_conf) * reliability, 0.0, 0.145)
    mult = {"Q1": 1.02, "Q2": 1.00, "Q3": 1.02, "S1": 0.78, "S2": 0.88, "S3": 0.68, "S4": 0.85}
    weights = pd.DataFrame({target: np.clip(base_weight * mult[target], 0.0, 0.16) for target in TARGETS})
    return blend_with_weight_matrix(anchor, prior, weights)


def current_streak(values: np.ndarray) -> tuple[float, int]:
    if len(values) == 0:
        return 0.5, 0
    last = float(values[-1])
    length = 0
    for value in values[::-1]:
        if float(value) == last:
            length += 1
        else:
            break
    return last, length


def fit_streak_tables(train: pd.DataFrame) -> dict[str, dict[tuple[int, int], float]]:
    tables: dict[str, dict[tuple[int, int], float]] = {}
    for target in TARGETS:
        buckets: dict[tuple[int, int], list[float]] = {}
        for _, group in train.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id"):
            vals = group[target].to_numpy(float)
            for i in range(3, len(vals)):
                last, length = current_streak(vals[:i])
                key = (int(last), int(min(length, 5)))
                buckets.setdefault(key, []).append(float(vals[i]))
        global_mean = float(train[target].mean())
        table = {}
        for key, ys in buckets.items():
            table[key] = float((np.sum(ys) + 8.0 * global_mean) / (len(ys) + 8.0))
        tables[target] = table
    return tables


def streak_prior(train: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    tables = fit_streak_tables(train)
    global_mean = train[TARGETS].mean()
    histories = {
        str(sid): group.sort_values("lifelog_date")[["lifelog_date"] + TARGETS].copy()
        for sid, group in train.groupby("subject_id")
    }
    rows = []
    for row in sample.sort_values(["subject_id", "lifelog_date"]).itertuples(index=False):
        sid = str(row.subject_id)
        hist = histories.get(sid, pd.DataFrame(columns=["lifelog_date"] + TARGETS))
        pred = {}
        for target in TARGETS:
            vals = hist[target].to_numpy(float) if len(hist) else np.array([], dtype=float)
            last, length = current_streak(vals)
            key = (int(last), int(min(length, 5)))
            table_p = tables[target].get(key, float(global_mean[target]))
            subj_mean = float(hist[target].mean()) if len(hist) else float(global_mean[target])
            pred[target] = float(np.clip(0.58 * table_p + 0.27 * subj_mean + 0.15 * global_mean[target], 0.045, 0.955))
        rows.append({"subject_id": sid, "lifelog_date": pd.Timestamp(row.lifelog_date), **pred})
        histories[sid] = pd.concat(
            [hist, pd.DataFrame([{"lifelog_date": pd.Timestamp(row.lifelog_date), **pred}])],
            ignore_index=True,
        )
    pred_df = pd.DataFrame(rows)
    # restore sample order
    ordered = sample[["subject_id", "lifelog_date"]].copy()
    ordered["lifelog_date"] = pd.to_datetime(ordered["lifelog_date"])
    merged = ordered.merge(pred_df, on=["subject_id", "lifelog_date"], how="left")
    return merged[TARGETS]


def simple_blend(anchor: pd.DataFrame, prior: pd.DataFrame, weight: float, mult: dict[str, float]) -> pd.DataFrame:
    out = anchor.copy()
    for target in TARGETS:
        w = weight * mult.get(target, 1.0)
        out[target] = (1 - w) * anchor[target].to_numpy(float) + w * prior[target].to_numpy(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def summarize(name: str, candidate: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, float | str]:
    diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
    return {
        "candidate": name,
        "diff_vs_best": float(diff.values.mean()),
        "max_diff": float(diff.values.max()),
        "q_diff": float(diff[["Q1", "Q2", "Q3"]].values.mean()),
        "s_diff": float(diff[["S1", "S2", "S3", "S4"]].values.mean()),
        "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
        "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
    }


def save(name: str, df: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, float | str]:
    validate(df, name)
    df.to_csv(SUB_DIR / name, index=False)
    return summarize(name, df, anchor)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
    sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
    anchor = pd.read_csv(CURRENT_BEST)
    date_prior = pd.read_csv(DATE_PRIOR)
    bracket_prior = pd.read_csv(BRACKET_PRIOR)
    validate(anchor, CURRENT_BEST.name)
    validate(date_prior, DATE_PRIOR.name)
    validate(bracket_prior, BRACKET_PRIOR.name)

    rows = []

    # Axis 1: low-risk calibration on top of the current best.
    for gamma in [0.035, 0.055]:
        name = f"sub_0508_currentbest_meanalign_g{int(gamma*1000):03d}.csv"
        rows.append(save(name, mean_align(anchor, train, gamma), anchor))

    # Axis 2: subject-reliability gated date prior.
    rel_date = subject_reliable_adaptive(anchor, date_prior, train, sample)
    rows.append(save("sub_0508_subjrel_dateprior_adaptive.csv", rel_date, anchor))

    rel_bracket = subject_reliable_adaptive(anchor, bracket_prior, train, sample)
    rows.append(save("sub_0508_subjrel_bracket_adaptive.csv", rel_bracket, anchor))

    # Axis 3: streak/reversion prior. Kept weak because it recursively feeds predicted probabilities.
    streak = sample.copy()
    streak[TARGETS] = streak_prior(train, sample).to_numpy(float)
    validate(streak, "streak_prior")
    streak_mult = {"Q1": 0.85, "Q2": 0.95, "Q3": 0.95, "S1": 0.75, "S2": 0.85, "S3": 0.60, "S4": 0.80}
    rows.append(save("sub_0508_streak_reversion_w055.csv", simple_blend(anchor, streak, 0.055, streak_mult), anchor))
    rows.append(save("sub_0508_streak_reversion_w080.csv", simple_blend(anchor, streak, 0.080, streak_mult), anchor))

    summary = pd.DataFrame(rows).sort_values(["diff_vs_best", "candidate"]).reset_index(drop=True)
    print("0508 new-axis candidate summary vs current best 0.5886910305:")
    print(summary.to_string(index=False))
    print("\nSuggested submit order:")
    print("1) sub_0508_currentbest_meanalign_g035.csv")
    print("2) sub_0508_subjrel_dateprior_adaptive.csv")
    print("3) sub_0508_streak_reversion_w055.csv")


if __name__ == "__main__":
    main()
