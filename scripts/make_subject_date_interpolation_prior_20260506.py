from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "raw" / "data"
TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"
SUB_DIR = BASE_DIR / "submissions"

CURRENT_BEST_PATH = SUB_DIR / "sub_reset_targetdyn_revert_anchor_w14_grid_20260506.csv"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


@dataclass(frozen=True)
class InterpConfig:
    tag: str
    tau: float
    k: int
    prior_strength: float
    future_weight: float
    past_weight: float
    clip_low: float = 0.045
    clip_high: float = 0.955


CONFIGS = [
    InterpConfig("near_tau3", tau=3.0, k=7, prior_strength=1.8, future_weight=1.00, past_weight=1.00),
    InterpConfig("near_tau5", tau=5.0, k=9, prior_strength=2.4, future_weight=1.00, past_weight=1.00),
    InterpConfig("near_tau7", tau=7.0, k=11, prior_strength=3.0, future_weight=1.00, past_weight=1.00),
    InterpConfig("future_soft", tau=5.0, k=9, prior_strength=2.4, future_weight=0.85, past_weight=1.05),
    InterpConfig("smooth_tau10", tau=10.0, k=13, prior_strength=4.0, future_weight=0.95, past_weight=1.00),
]


def binary_logloss(y_true: np.ndarray, pred: np.ndarray) -> float:
    pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
    y_true = np.asarray(y_true, dtype=float)
    return float(-(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred)).mean())


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


def fit_subject_tables(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(sid): group.sort_values("lifelog_date")[["lifelog_date"] + TARGETS].copy()
        for sid, group in train.groupby("subject_id")
    }


def predict_target(
    subject_table: pd.DataFrame,
    pred_date: pd.Timestamp,
    target: str,
    global_mean: float,
    config: InterpConfig,
) -> float:
    if len(subject_table) == 0:
        return float(np.clip(global_mean, config.clip_low, config.clip_high))

    dates = pd.to_datetime(subject_table["lifelog_date"])
    delta = (dates - pred_date).dt.days.to_numpy(dtype=float)
    abs_delta = np.abs(delta)
    order = np.argsort(abs_delta)[: config.k]

    chosen_delta = delta[order]
    chosen_abs = abs_delta[order]
    values = subject_table[target].to_numpy(dtype=float)[order]
    side_weight = np.where(chosen_delta >= 0, config.future_weight, config.past_weight)
    weights = np.exp(-chosen_abs / config.tau) * side_weight

    subject_mean = float(subject_table[target].mean())
    smooth_prior = 0.75 * subject_mean + 0.25 * global_mean

    if float(weights.sum()) <= 1e-12:
        pred = smooth_prior
    else:
        local = float(np.sum(weights * values) / np.sum(weights))
        reliability = float(np.sum(weights) / (np.sum(weights) + config.prior_strength))
        pred = reliability * local + (1.0 - reliability) * smooth_prior

    return float(np.clip(pred, config.clip_low, config.clip_high))


def build_prior(train_fit: pd.DataFrame, test_frame: pd.DataFrame, config: InterpConfig) -> pd.DataFrame:
    tables = fit_subject_tables(train_fit)
    global_mean = {target: float(train_fit[target].mean()) for target in TARGETS}
    rows = []

    for row in test_frame.itertuples(index=False):
        sid = str(row.subject_id)
        pred_date = pd.Timestamp(row.lifelog_date)
        table = tables.get(sid, pd.DataFrame(columns=["lifelog_date"] + TARGETS))
        pred_row = {}
        for target in TARGETS:
            pred_row[target] = predict_target(table, pred_date, target, global_mean[target], config)
        rows.append(pred_row)

    return pd.DataFrame(rows)


def interpolation_cv(train: pd.DataFrame, config: InterpConfig) -> dict[str, float]:
    y_all = []
    p_all = []
    per_target = {target: {"y": [], "p": []} for target in TARGETS}

    for _, group in train.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id"):
        group = group.reset_index(drop=True)
        if len(group) < 15:
            continue
        # Hold out interleaved rows rather than only the tail. This mirrors
        # the actual test layout, where many dates sit between known train dates.
        holdout_mask = ((np.arange(len(group)) + 2) % 5 == 0)
        holdout = group[holdout_mask].copy()
        fit = pd.concat(
            [
                train[train["subject_id"] != group["subject_id"].iloc[0]],
                group[~holdout_mask],
            ],
            ignore_index=True,
        )
        preds = build_prior(fit, holdout[["subject_id", "sleep_date", "lifelog_date"]], config)
        y = holdout[TARGETS].to_numpy(dtype=float)
        p = preds[TARGETS].to_numpy(dtype=float)
        y_all.append(y)
        p_all.append(p)
        for j, target in enumerate(TARGETS):
            per_target[target]["y"].extend(y[:, j].tolist())
            per_target[target]["p"].extend(p[:, j].tolist())

    y_true = np.vstack(y_all)
    pred = np.vstack(p_all)
    out = {"overall": binary_logloss(y_true, pred)}
    for target in TARGETS:
        out[target] = binary_logloss(np.array(per_target[target]["y"]), np.array(per_target[target]["p"]))
    return out


def blend(anchor: pd.DataFrame, prior: pd.DataFrame, weight: float) -> pd.DataFrame:
    out = anchor.copy()
    out[TARGETS] = (1.0 - weight) * anchor[TARGETS].astype(float) + weight * prior[TARGETS].astype(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def targetwise_blend(anchor: pd.DataFrame, prior: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = anchor.copy()
    for target in TARGETS:
        weight = float(weights.get(target, 0.0))
        out[target] = (1.0 - weight) * anchor[target].astype(float) + weight * prior[target].astype(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def summarize(name: str, candidate: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, float | str]:
    diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
    return {
        "candidate": name,
        "diff_vs_current_best": float(diff.values.mean()),
        "max_diff": float(diff.values.max()),
        "q_diff": float(diff[["Q1", "Q2", "Q3"]].values.mean()),
        "s_diff": float(diff[["S1", "S2", "S3", "S4"]].values.mean()),
        "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
        "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
    }


def main() -> None:
    train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
    sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
    anchor = pd.read_csv(CURRENT_BEST_PATH)
    validate(anchor, CURRENT_BEST_PATH.name)

    cv_rows = []
    for config in CONFIGS:
        cv_rows.append({"config": config.tag, **interpolation_cv(train, config)})
    cv = pd.DataFrame(cv_rows).sort_values("overall").reset_index(drop=True)
    print("Interleaved subject-date interpolation CV:")
    print(cv.to_string(index=False))

    best_tag = str(cv.loc[0, "config"])
    best_config = next(config for config in CONFIGS if config.tag == best_tag)
    print(f"\nBest interpolation config: {best_config.tag}")

    prior = build_prior(train, sample[["subject_id", "sleep_date", "lifelog_date"]], best_config)
    pure = sample.copy()
    pure[TARGETS] = prior[TARGETS].to_numpy(dtype=float)
    pure[TARGETS] = pure[TARGETS].clip(1e-6, 1 - 1e-6)
    pure_name = f"sub_dateinterp_{best_config.tag}_pure_20260506.csv"
    validate(pure, pure_name)
    pure.to_csv(SUB_DIR / pure_name, index=False)

    candidates = [summarize(pure_name, pure, anchor)]
    for weight in [0.04, 0.07, 0.10, 0.13, 0.16]:
        name = f"sub_dateinterp_{best_config.tag}_anchor_w{int(weight * 100):02d}_20260506.csv"
        out = blend(anchor, pure, weight)
        validate(out, name)
        out.to_csv(SUB_DIR / name, index=False)
        candidates.append(summarize(name, out, anchor))

    tw_configs = {
        "tw_q10_s06": {"Q1": 0.10, "Q2": 0.10, "Q3": 0.10, "S1": 0.06, "S2": 0.06, "S3": 0.04, "S4": 0.06},
        "tw_q12_s08": {"Q1": 0.12, "Q2": 0.12, "Q3": 0.12, "S1": 0.08, "S2": 0.08, "S3": 0.05, "S4": 0.08},
        "tw_q08_s10": {"Q1": 0.08, "Q2": 0.08, "Q3": 0.08, "S1": 0.10, "S2": 0.10, "S3": 0.06, "S4": 0.10},
    }
    for tag, weights in tw_configs.items():
        name = f"sub_dateinterp_{best_config.tag}_anchor_{tag}_20260506.csv"
        out = targetwise_blend(anchor, pure, weights)
        validate(out, name)
        out.to_csv(SUB_DIR / name, index=False)
        candidates.append(summarize(name, out, anchor))

    summary = pd.DataFrame(candidates).sort_values("diff_vs_current_best").reset_index(drop=True)
    print("\nSaved candidate summary vs current best w14:")
    print(summary.to_string(index=False))
    print("\nSuggested submit order if current best = 0.5898630289:")
    print(f"1) sub_dateinterp_{best_config.tag}_anchor_w07_20260506.csv")
    print(f"2) sub_dateinterp_{best_config.tag}_anchor_w10_20260506.csv")
    print(f"3) sub_dateinterp_{best_config.tag}_anchor_tw_q12_s08_20260506.csv")


if __name__ == "__main__":
    main()
