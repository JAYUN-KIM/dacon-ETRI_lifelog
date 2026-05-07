from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "raw" / "data"
SUB_DIR = BASE_DIR / "submissions"
TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"

ANCHOR_PATH = SUB_DIR / "sub_currentbest_meanalign_g060_20260507.csv"
DATE_PRIOR_PATH = SUB_DIR / "sub_dateinterp_smooth_tau10_pure_20260506.csv"
BRACKET_PRIOR_PATH = SUB_DIR / "sub_bracket_bracket_mid_pure_20260507.csv"
HYBRID_PRIOR_PATH = SUB_DIR / "sub_hybrid_bracket_calendar_pure_20260507.csv"

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


def build_date_confidence(train: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train_dates = {
        str(sid): pd.to_datetime(group["lifelog_date"]).sort_values().to_numpy()
        for sid, group in train.groupby("subject_id")
    }
    for row in sample.itertuples(index=False):
        sid = str(row.subject_id)
        pred_date = pd.Timestamp(row.lifelog_date)
        dates = train_dates.get(sid)
        if dates is None or len(dates) == 0:
            before_dist = np.nan
            after_dist = np.nan
            min_dist = 99.0
            bracketed = 0.0
        else:
            deltas = np.array([(pd.Timestamp(d) - pred_date).days for d in dates], dtype=float)
            before = np.abs(deltas[deltas < 0])
            after = np.abs(deltas[deltas > 0])
            before_dist = float(before.min()) if len(before) else np.nan
            after_dist = float(after.min()) if len(after) else np.nan
            min_dist = float(np.nanmin([before_dist, after_dist]))
            bracketed = float(np.isfinite(before_dist) and np.isfinite(after_dist))

        near_score = float(np.exp(-min_dist / 10.0))
        both_score = bracketed * float(np.exp(-(np.nan_to_num(before_dist, nan=30.0) + np.nan_to_num(after_dist, nan=30.0)) / 24.0))
        rows.append(
            {
                "subject_id": sid,
                "lifelog_date": pred_date,
                "before_dist": before_dist,
                "after_dist": after_dist,
                "min_dist": min_dist,
                "bracketed": bracketed,
                "near_score": near_score,
                "both_score": both_score,
                "confidence": float(np.clip(0.65 * near_score + 0.35 * both_score, 0.0, 1.0)),
            }
        )
    return pd.DataFrame(rows)


def adaptive_blend(
    anchor: pd.DataFrame,
    prior: pd.DataFrame,
    conf: pd.Series,
    base: float,
    scale: float,
    target_mult: dict[str, float] | None = None,
    tag: str = "",
) -> pd.DataFrame:
    out = anchor.copy()
    row_weight = np.clip(base + scale * conf.to_numpy(dtype=float), 0.0, 0.22)
    target_mult = target_mult or {target: 1.0 for target in TARGETS}
    for target in TARGETS:
        weight = np.clip(row_weight * float(target_mult.get(target, 1.0)), 0.0, 0.24)
        out[target] = (1.0 - weight) * anchor[target].astype(float) + weight * prior[target].astype(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    out.attrs["tag"] = tag
    out.attrs["avg_weight"] = float(row_weight.mean())
    out.attrs["max_weight"] = float(row_weight.max())
    return out


def summarize(name: str, candidate: pd.DataFrame, anchor: pd.DataFrame, weight_info: str) -> dict[str, float | str]:
    diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
    return {
        "candidate": name,
        "weight_info": weight_info,
        "diff_vs_g060": float(diff.values.mean()),
        "max_diff": float(diff.values.max()),
        "q_diff": float(diff[["Q1", "Q2", "Q3"]].values.mean()),
        "s_diff": float(diff[["S1", "S2", "S3", "S4"]].values.mean()),
        "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
        "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
    }


def save_candidate(name: str, candidate: pd.DataFrame, anchor: pd.DataFrame, weight_info: str) -> dict[str, float | str]:
    validate(candidate, name)
    candidate.to_csv(SUB_DIR / name, index=False)
    return summarize(name, candidate, anchor, weight_info)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
    sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
    anchor = pd.read_csv(ANCHOR_PATH)
    date_prior = pd.read_csv(DATE_PRIOR_PATH)
    bracket_prior = pd.read_csv(BRACKET_PRIOR_PATH)
    hybrid_prior = pd.read_csv(HYBRID_PRIOR_PATH)

    for name, df in [
        (ANCHOR_PATH.name, anchor),
        (DATE_PRIOR_PATH.name, date_prior),
        (BRACKET_PRIOR_PATH.name, bracket_prior),
        (HYBRID_PRIOR_PATH.name, hybrid_prior),
    ]:
        validate(df, name)

    conf = build_date_confidence(train, sample)
    print("Date confidence summary:")
    print(conf[["before_dist", "after_dist", "min_dist", "bracketed", "confidence"]].describe().to_string())

    candidates = []
    configs = [
        (
            "dateconf_dateprior_b03_s10",
            date_prior,
            0.03,
            0.10,
            {"Q1": 1.00, "Q2": 1.00, "Q3": 1.00, "S1": 0.85, "S2": 0.90, "S3": 0.75, "S4": 0.90},
        ),
        (
            "dateconf_dateprior_b04_s12",
            date_prior,
            0.04,
            0.12,
            {"Q1": 1.00, "Q2": 1.00, "Q3": 1.00, "S1": 0.85, "S2": 0.90, "S3": 0.75, "S4": 0.90},
        ),
        (
            "dateconf_bracket_b03_s10",
            bracket_prior,
            0.03,
            0.10,
            {"Q1": 0.95, "Q2": 1.00, "Q3": 1.00, "S1": 0.85, "S2": 0.90, "S3": 0.75, "S4": 0.90},
        ),
        (
            "dateconf_hybrid_b03_s10",
            hybrid_prior,
            0.03,
            0.10,
            {"Q1": 0.95, "Q2": 1.00, "Q3": 1.00, "S1": 0.85, "S2": 0.90, "S3": 0.75, "S4": 0.90},
        ),
    ]

    for tag, prior, base, scale, target_mult in configs:
        out = adaptive_blend(anchor, prior, conf["confidence"], base, scale, target_mult, tag)
        weights = np.clip(base + scale * conf["confidence"].to_numpy(dtype=float), 0.0, 0.22)
        info = f"avg={weights.mean():.4f}, max={weights.max():.4f}, base={base:.2f}, scale={scale:.2f}"
        candidates.append(save_candidate(f"sub_adaptive_{tag}_20260507.csv", out, anchor, info))

    summary = pd.DataFrame(candidates).sort_values("diff_vs_g060").reset_index(drop=True)
    print("\nAdaptive prior candidate summary vs current best g060:")
    print(summary.to_string(index=False))
    print("\nSuggested if taking a new-axis shot:")
    print("sub_adaptive_dateconf_dateprior_b03_s10_20260507.csv")


if __name__ == "__main__":
    main()
