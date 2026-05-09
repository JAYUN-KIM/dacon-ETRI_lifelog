from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "raw" / "data"
SUB_DIR = BASE_DIR / "submissions"
TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"

ANCHOR_PATH = SUB_DIR / "sub_0509_best_meanalign_g030.csv"
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


def logloss(y_true, pred) -> float:
    pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
    y_true = np.asarray(y_true, dtype=float)
    return float(-(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred)).mean())


def date_confidence(train: pd.DataFrame, sample: pd.DataFrame, near_tau: float, both_tau: float, mix: float) -> np.ndarray:
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
            min_dist, before_dist, after_dist, bracketed = 99.0, 30.0, 30.0, 0.0
        else:
            deltas = np.array([(pd.Timestamp(d) - date).days for d in dates], dtype=float)
            before = np.abs(deltas[deltas < 0])
            after = np.abs(deltas[deltas > 0])
            before_dist = float(before.min()) if len(before) else 30.0
            after_dist = float(after.min()) if len(after) else 30.0
            min_dist = min(before_dist, after_dist)
            bracketed = float(len(before) > 0 and len(after) > 0)
        near = float(np.exp(-min_dist / near_tau))
        both = bracketed * float(np.exp(-(before_dist + after_dist) / both_tau))
        values.append(float(np.clip(mix * near + (1 - mix) * both, 0.0, 1.0)))
    return np.array(values, dtype=float)


def subject_rel(train: pd.DataFrame, targetwise: bool) -> pd.DataFrame:
    rows = []
    global_mean = train[TARGETS].mean()
    for sid, group in train.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id"):
        group = group.reset_index(drop=True)
        row = {"subject_id": str(sid)}
        gains = {}
        for target in TARGETS:
            errs = []
            base_errs = []
            for i in range(4, len(group)):
                hist = group.iloc[:i]
                curr = group.iloc[i]
                pred = 0.55 * float(hist[target].tail(7).mean()) + 0.30 * float(hist[target].mean()) + 0.15 * float(global_mean[target])
                base = float(hist[target].mean())
                y = float(curr[target])
                errs.append(logloss([y], [pred]))
                base_errs.append(logloss([y], [base]))
            gains[target] = float(np.mean(base_errs) - np.mean(errs)) if errs else 0.0
        if targetwise:
            for target in TARGETS:
                row[target] = float(np.clip(0.84 + gains[target] * 3.2, 0.58, 1.18))
        else:
            rel = float(np.clip(0.85 + np.mean(list(gains.values())) * 4.0, 0.55, 1.20))
            for target in TARGETS:
                row[target] = rel
        rows.append(row)
    return pd.DataFrame(rows)


def adaptive(anchor, prior, train, sample, *, targetwise, base, scale, cap, mult, near_tau=10.0, both_tau=24.0, mix=0.66):
    conf = date_confidence(train, sample, near_tau, both_tau, mix)
    rel = subject_rel(train, targetwise)
    meta = sample[["subject_id"]].merge(rel, on="subject_id", how="left")
    out = anchor.copy()
    row_base = base + scale * conf
    for target in TARGETS:
        reliability = meta[target].fillna(0.85).to_numpy(float)
        w = np.clip(row_base * reliability * mult[target], 0.0, cap)
        out[target] = (1 - w) * anchor[target].to_numpy(float) + w * prior[target].to_numpy(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def mixed_prior(date_prior, bracket_prior, train, sample):
    conf = date_confidence(train, sample, near_tau=10.0, both_tau=22.0, mix=0.62)
    gate = np.clip((conf - 0.45) / 0.45, 0.0, 1.0)
    out = date_prior.copy()
    for target in TARGETS:
        out[target] = (1 - 0.20 * gate) * date_prior[target].to_numpy(float) + (0.20 * gate) * bracket_prior[target].to_numpy(float)
    return out


def summarize(name, candidate, anchor):
    diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
    return {
        "candidate": name,
        "diff_vs_g030": float(diff.values.mean()),
        "max_diff": float(diff.values.max()),
        "q_diff": float(diff[["Q1", "Q2", "Q3"]].values.mean()),
        "s_diff": float(diff[["S1", "S2", "S3", "S4"]].values.mean()),
        "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
        "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
    }


def save(name, candidate, anchor):
    validate(candidate, name)
    candidate.to_csv(SUB_DIR / name, index=False)
    return summarize(name, candidate, anchor)


def main():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
    sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
    anchor = pd.read_csv(ANCHOR_PATH)
    date_prior = pd.read_csv(DATE_PRIOR)
    bracket_prior = pd.read_csv(BRACKET_PRIOR)
    validate(anchor, ANCHOR_PATH.name)
    validate(date_prior, DATE_PRIOR.name)
    validate(bracket_prior, BRACKET_PRIOR.name)

    rows = []
    mult_safe = {"Q1": 1.00, "Q2": 0.98, "Q3": 1.02, "S1": 0.70, "S2": 0.82, "S3": 0.50, "S4": 0.78}
    mult_q = {"Q1": 1.10, "Q2": 1.04, "Q3": 1.12, "S1": 0.50, "S2": 0.62, "S3": 0.35, "S4": 0.58}

    rows.append(
        save(
            "sub_0509_afterg030_targetscale_soft.csv",
            adaptive(anchor, date_prior, train, sample, targetwise=True, base=0.006, scale=0.050, cap=0.090, mult=mult_safe),
            anchor,
        )
    )
    rows.append(
        save(
            "sub_0509_afterg030_targetscale_mid.csv",
            adaptive(anchor, date_prior, train, sample, targetwise=True, base=0.009, scale=0.060, cap=0.105, mult=mult_safe),
            anchor,
        )
    )
    rows.append(
        save(
            "sub_0509_afterg030_qled_ssoft.csv",
            adaptive(anchor, date_prior, train, sample, targetwise=True, base=0.008, scale=0.058, cap=0.105, mult=mult_q),
            anchor,
        )
    )
    mix = mixed_prior(date_prior, bracket_prior, train, sample)
    rows.append(
        save(
            "sub_0509_afterg030_mixed_date_bracket.csv",
            adaptive(anchor, mix, train, sample, targetwise=False, base=0.007, scale=0.052, cap=0.095, mult=mult_safe),
            anchor,
        )
    )

    summary = pd.DataFrame(rows).sort_values("diff_vs_g030").reset_index(drop=True)
    print("Candidates after g030 anchor:")
    print(summary.to_string(index=False))
    print("\nSuggested remaining submits:")
    print("1) sub_0509_afterg030_targetscale_soft.csv")
    print("2) sub_0509_afterg030_mixed_date_bracket.csv")


if __name__ == "__main__":
    main()
