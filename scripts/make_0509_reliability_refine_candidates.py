from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "raw" / "data"
SUB_DIR = BASE_DIR / "submissions"
TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"

CURRENT_BEST = SUB_DIR / "sub_0508_subjrel_dateprior_adaptive.csv"
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
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def logloss(y_true: np.ndarray, pred: np.ndarray) -> float:
    pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
    y_true = np.asarray(y_true, dtype=float)
    return float(-(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred)).mean())


def shift_to_mean(probs: np.ndarray, target_mean: float) -> np.ndarray:
    logits = logit(probs)
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if sigmoid(logits + mid).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return sigmoid(logits + (lo + hi) / 2)


def mean_align(anchor: pd.DataFrame, train: pd.DataFrame, gamma: float) -> pd.DataFrame:
    out = anchor.copy()
    desired = (1 - gamma) * anchor[TARGETS].mean() + gamma * train[TARGETS].mean()
    for target in TARGETS:
        out[target] = shift_to_mean(anchor[target].to_numpy(float), float(desired[target]))
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def build_date_confidence(train: pd.DataFrame, sample: pd.DataFrame, near_tau: float, both_tau: float, mix: float) -> pd.Series:
    train_dates = {
        str(sid): pd.to_datetime(group["lifelog_date"]).sort_values().to_numpy()
        for sid, group in train.groupby("subject_id")
    }
    values = []
    for row in sample.itertuples(index=False):
        sid = str(row.subject_id)
        date = pd.Timestamp(row.lifelog_date)
        dates = train_dates.get(sid)
        if dates is None or len(dates) == 0:
            min_dist = 99.0
            bracketed = 0.0
            before_dist = 30.0
            after_dist = 30.0
        else:
            deltas = np.array([(pd.Timestamp(d) - date).days for d in dates], dtype=float)
            before = np.abs(deltas[deltas < 0])
            after = np.abs(deltas[deltas > 0])
            before_dist = float(before.min()) if len(before) else 30.0
            after_dist = float(after.min()) if len(after) else 30.0
            min_dist = float(min(before_dist, after_dist))
            bracketed = float(len(before) > 0 and len(after) > 0)
        near = float(np.exp(-min_dist / near_tau))
        both = bracketed * float(np.exp(-(before_dist + after_dist) / both_tau))
        values.append(float(np.clip(mix * near + (1 - mix) * both, 0.0, 1.0)))
    return pd.Series(values)


def subject_reliability(train: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    global_mean = train[TARGETS].mean()
    for sid, group in train.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < 12:
            reliability = 0.78
        else:
            gains = []
            for target in TARGETS:
                errs = []
                base_errs = []
                for i in range(4, n):
                    hist = group.iloc[:i]
                    curr = group.iloc[i]
                    recent = float(hist[target].tail(7).mean())
                    subj = float(hist[target].mean())
                    pred = 0.55 * recent + 0.30 * subj + 0.15 * float(global_mean[target])
                    base = subj
                    y = float(curr[target])
                    errs.append(logloss(np.array([y]), np.array([pred])))
                    base_errs.append(logloss(np.array([y]), np.array([base])))
                gains.append(float(np.mean(base_errs) - np.mean(errs)))

            if mode == "overall":
                gain = float(np.mean(gains))
                reliability = float(np.clip(0.85 + gain * 4.0, 0.55, 1.20))
                row = {"subject_id": str(sid), **{target: reliability for target in TARGETS}}
            elif mode == "target":
                row = {"subject_id": str(sid)}
                for target, gain in zip(TARGETS, gains):
                    row[target] = float(np.clip(0.84 + gain * 3.2, 0.58, 1.18))
            else:
                raise ValueError(mode)
            rows.append(row)
            continue

        rows.append({"subject_id": str(sid), **{target: reliability for target in TARGETS}})
    return pd.DataFrame(rows)


def adaptive_prior(
    anchor: pd.DataFrame,
    prior: pd.DataFrame,
    train: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    rel_mode: str,
    near_tau: float,
    both_tau: float,
    conf_mix: float,
    base: float,
    scale: float,
    cap: float,
    mult: dict[str, float],
) -> pd.DataFrame:
    conf = build_date_confidence(train, sample, near_tau=near_tau, both_tau=both_tau, mix=conf_mix).to_numpy(float)
    rel = subject_reliability(train, rel_mode)
    meta = sample[["subject_id"]].merge(rel, on="subject_id", how="left")
    out = anchor.copy()
    row_base = base + scale * conf
    for target in TARGETS:
        reliability = meta[target].fillna(0.85).to_numpy(float)
        weight = np.clip(row_base * reliability * mult.get(target, 1.0), 0.0, cap)
        out[target] = (1 - weight) * anchor[target].to_numpy(float) + weight * prior[target].to_numpy(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def blend_two_priors(
    anchor: pd.DataFrame,
    prior_a: pd.DataFrame,
    prior_b: pd.DataFrame,
    train: pd.DataFrame,
    sample: pd.DataFrame,
) -> pd.DataFrame:
    # Date prior is known good; bracket prior only enters when a date is bracketed closely.
    conf = build_date_confidence(train, sample, near_tau=10.0, both_tau=22.0, mix=0.62).to_numpy(float)
    bracket_gate = np.clip((conf - 0.45) / 0.45, 0.0, 1.0)
    mixed = prior_a.copy()
    for target in TARGETS:
        mixed[target] = (1 - 0.22 * bracket_gate) * prior_a[target].to_numpy(float) + (0.22 * bracket_gate) * prior_b[target].to_numpy(float)
    mult = {"Q1": 1.00, "Q2": 0.98, "Q3": 1.02, "S1": 0.72, "S2": 0.84, "S3": 0.55, "S4": 0.82}
    return adaptive_prior(
        anchor,
        mixed,
        train,
        sample,
        rel_mode="overall",
        near_tau=10.0,
        both_tau=22.0,
        conf_mix=0.62,
        base=0.012,
        scale=0.070,
        cap=0.120,
        mult=mult,
    )


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

    rows.append(save("sub_0509_best_meanalign_g030.csv", mean_align(anchor, train, 0.030), anchor))

    base_mult = {"Q1": 1.00, "Q2": 0.98, "Q3": 1.02, "S1": 0.72, "S2": 0.84, "S3": 0.55, "S4": 0.82}
    rows.append(
        save(
            "sub_0509_rel_targetscale_soft.csv",
            adaptive_prior(
                anchor,
                date_prior,
                train,
                sample,
                rel_mode="target",
                near_tau=11.0,
                both_tau=26.0,
                conf_mix=0.70,
                base=0.010,
                scale=0.065,
                cap=0.115,
                mult=base_mult,
            ),
            anchor,
        )
    )
    rows.append(
        save(
            "sub_0509_rel_overall_midcap.csv",
            adaptive_prior(
                anchor,
                date_prior,
                train,
                sample,
                rel_mode="overall",
                near_tau=9.0,
                both_tau=22.0,
                conf_mix=0.62,
                base=0.015,
                scale=0.075,
                cap=0.125,
                mult=base_mult,
            ),
            anchor,
        )
    )
    rows.append(save("sub_0509_rel_mixed_date_bracket.csv", blend_two_priors(anchor, date_prior, bracket_prior, train, sample), anchor))

    # More experimental: Q-led variant. Keep S very conservative.
    q_mult = {"Q1": 1.10, "Q2": 1.05, "Q3": 1.12, "S1": 0.55, "S2": 0.68, "S3": 0.40, "S4": 0.62}
    rows.append(
        save(
            "sub_0509_rel_qled_ssoft.csv",
            adaptive_prior(
                anchor,
                date_prior,
                train,
                sample,
                rel_mode="target",
                near_tau=10.0,
                both_tau=24.0,
                conf_mix=0.68,
                base=0.012,
                scale=0.078,
                cap=0.130,
                mult=q_mult,
            ),
            anchor,
        )
    )

    summary = pd.DataFrame(rows).sort_values("diff_vs_best").reset_index(drop=True)
    print("0509 reliability refine candidates vs current best 0.5884463893:")
    print(summary.to_string(index=False))
    print("\nSuggested submit order:")
    print("1) sub_0509_best_meanalign_g030.csv")
    print("2) sub_0509_rel_targetscale_soft.csv")
    print("3) sub_0509_rel_mixed_date_bracket.csv")


if __name__ == "__main__":
    main()
