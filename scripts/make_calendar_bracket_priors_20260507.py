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

CURRENT_BEST_PATH = SUB_DIR / "sub_dateinterp_smooth_tau10_anchor_w07_20260506.csv"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


@dataclass(frozen=True)
class BracketConfig:
    tag: str
    nearest_tau: float
    subject_strength: float
    bracket_scale: float
    clip_low: float = 0.045
    clip_high: float = 0.955


@dataclass(frozen=True)
class CalendarConfig:
    tag: str
    tau: float
    k: int
    subject_weight: float
    calendar_weight: float
    dow_weight: float
    prior_strength: float
    clip_low: float = 0.045
    clip_high: float = 0.955


BRACKET_CONFIGS = [
    BracketConfig("bracket_soft", nearest_tau=6.0, subject_strength=5.0, bracket_scale=0.82),
    BracketConfig("bracket_mid", nearest_tau=9.0, subject_strength=7.0, bracket_scale=0.72),
    BracketConfig("bracket_smooth", nearest_tau=13.0, subject_strength=9.0, bracket_scale=0.62),
]

CALENDAR_CONFIGS = [
    CalendarConfig("cal_tau10", tau=10.0, k=40, subject_weight=0.60, calendar_weight=0.30, dow_weight=0.10, prior_strength=5.0),
    CalendarConfig("cal_tau16", tau=16.0, k=55, subject_weight=0.56, calendar_weight=0.34, dow_weight=0.10, prior_strength=7.0),
    CalendarConfig("cal_dow", tau=14.0, k=50, subject_weight=0.52, calendar_weight=0.32, dow_weight=0.16, prior_strength=7.0),
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


def smooth_mean(pos: float, total: float, prior: float, strength: float) -> float:
    return float((pos + strength * prior) / (total + strength))


def predict_bracket_target(
    table: pd.DataFrame,
    pred_date: pd.Timestamp,
    target: str,
    global_mean: float,
    config: BracketConfig,
) -> float:
    if len(table) == 0:
        return float(np.clip(global_mean, config.clip_low, config.clip_high))

    dates = pd.to_datetime(table["lifelog_date"])
    values = table[target].astype(float).to_numpy()
    deltas = (dates - pred_date).dt.days.to_numpy(dtype=float)
    subject_mean = float(np.mean(values))
    subject_prior = 0.82 * subject_mean + 0.18 * global_mean

    before_idx = np.where(deltas < 0)[0]
    after_idx = np.where(deltas > 0)[0]
    exact_idx = np.where(deltas == 0)[0]

    pieces = []
    weights = []
    if len(exact_idx):
        pieces.append(float(values[exact_idx[-1]]))
        weights.append(1.8)
    if len(before_idx):
        idx = before_idx[np.argmin(np.abs(deltas[before_idx]))]
        dist = abs(float(deltas[idx]))
        pieces.append(float(values[idx]))
        weights.append(np.exp(-dist / config.nearest_tau))
    if len(after_idx):
        idx = after_idx[np.argmin(np.abs(deltas[after_idx]))]
        dist = abs(float(deltas[idx]))
        pieces.append(float(values[idx]))
        weights.append(np.exp(-dist / config.nearest_tau))

    if not pieces or sum(weights) <= 1e-12:
        bracket = subject_prior
        confidence = 0.0
    else:
        bracket = float(np.average(pieces, weights=weights))
        confidence = float(sum(weights) / (sum(weights) + config.subject_strength))

    pred = config.bracket_scale * (confidence * bracket + (1 - confidence) * subject_prior)
    pred += (1 - config.bracket_scale) * subject_prior
    return float(np.clip(pred, config.clip_low, config.clip_high))


def build_bracket_prior(train_fit: pd.DataFrame, frame: pd.DataFrame, config: BracketConfig) -> pd.DataFrame:
    tables = fit_subject_tables(train_fit)
    global_mean = {target: float(train_fit[target].mean()) for target in TARGETS}
    rows = []
    for row in frame.itertuples(index=False):
        sid = str(row.subject_id)
        pred_date = pd.Timestamp(row.lifelog_date)
        table = tables.get(sid, pd.DataFrame(columns=["lifelog_date"] + TARGETS))
        rows.append(
            {
                target: predict_bracket_target(table, pred_date, target, global_mean[target], config)
                for target in TARGETS
            }
        )
    return pd.DataFrame(rows)


def predict_calendar_target(
    train_fit: pd.DataFrame,
    sid: str,
    pred_date: pd.Timestamp,
    target: str,
    config: CalendarConfig,
) -> float:
    global_mean = float(train_fit[target].mean())
    subject_values = train_fit.loc[train_fit["subject_id"] == sid, target].astype(float)
    subject_mean = smooth_mean(float(subject_values.sum()), len(subject_values), global_mean, 6.0)

    others = train_fit[train_fit["subject_id"] != sid].copy()
    if len(others) == 0:
        calendar_mean = global_mean
    else:
        delta = (pd.to_datetime(others["lifelog_date"]) - pred_date).dt.days.abs().to_numpy(dtype=float)
        order = np.argsort(delta)[: config.k]
        chosen = others.iloc[order]
        dist = delta[order]
        weights = np.exp(-dist / config.tau)
        if float(weights.sum()) <= 1e-12:
            calendar_mean = global_mean
        else:
            local = float(np.average(chosen[target].astype(float), weights=weights))
            reliability = float(weights.sum() / (weights.sum() + config.prior_strength))
            calendar_mean = reliability * local + (1 - reliability) * global_mean

    dow = int(pred_date.dayofweek)
    dow_df = train_fit[train_fit["lifelog_date"].dt.dayofweek == dow]
    dow_mean = smooth_mean(float(dow_df[target].sum()), len(dow_df), global_mean, 12.0)

    pred = (
        config.subject_weight * subject_mean
        + config.calendar_weight * calendar_mean
        + config.dow_weight * dow_mean
    )
    return float(np.clip(pred, config.clip_low, config.clip_high))


def build_calendar_prior(train_fit: pd.DataFrame, frame: pd.DataFrame, config: CalendarConfig) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        sid = str(row.subject_id)
        pred_date = pd.Timestamp(row.lifelog_date)
        rows.append(
            {
                target: predict_calendar_target(train_fit, sid, pred_date, target, config)
                for target in TARGETS
            }
        )
    return pd.DataFrame(rows)


def interleaved_cv(train: pd.DataFrame, prior_builder, config) -> dict[str, float]:
    y_all = []
    p_all = []
    per_target = {target: {"y": [], "p": []} for target in TARGETS}

    for _, group in train.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id"):
        group = group.reset_index(drop=True)
        if len(group) < 15:
            continue
        holdout_mask = ((np.arange(len(group)) + 2) % 5 == 0)
        holdout = group[holdout_mask].copy()
        fit = pd.concat(
            [
                train[train["subject_id"] != group["subject_id"].iloc[0]],
                group[~holdout_mask],
            ],
            ignore_index=True,
        )
        preds = prior_builder(fit, holdout[["subject_id", "sleep_date", "lifelog_date"]], config)
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
    out[TARGETS] = (1 - weight) * anchor[TARGETS].astype(float) + weight * prior[TARGETS].astype(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def targetwise_blend(anchor: pd.DataFrame, prior: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = anchor.copy()
    for target in TARGETS:
        weight = float(weights.get(target, 0.0))
        out[target] = (1 - weight) * anchor[target].astype(float) + weight * prior[target].astype(float)
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


def save_candidate(name: str, candidate: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, float | str]:
    validate(candidate, name)
    candidate.to_csv(SUB_DIR / name, index=False)
    return summarize(name, candidate, anchor)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
    sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
    anchor = pd.read_csv(CURRENT_BEST_PATH)
    validate(anchor, CURRENT_BEST_PATH.name)

    cv_rows = []
    for config in BRACKET_CONFIGS:
        cv_rows.append({"family": "bracket", "config": config.tag, **interleaved_cv(train, build_bracket_prior, config)})
    for config in CALENDAR_CONFIGS:
        cv_rows.append({"family": "calendar", "config": config.tag, **interleaved_cv(train, build_calendar_prior, config)})

    cv = pd.DataFrame(cv_rows).sort_values("overall").reset_index(drop=True)
    print("Interleaved CV for new axes:")
    print(cv.to_string(index=False))

    best_bracket_tag = cv[cv["family"] == "bracket"].iloc[0]["config"]
    best_calendar_tag = cv[cv["family"] == "calendar"].iloc[0]["config"]
    bracket_config = next(config for config in BRACKET_CONFIGS if config.tag == best_bracket_tag)
    calendar_config = next(config for config in CALENDAR_CONFIGS if config.tag == best_calendar_tag)

    bracket_prior = build_bracket_prior(train, sample[["subject_id", "sleep_date", "lifelog_date"]], bracket_config)
    calendar_prior = build_calendar_prior(train, sample[["subject_id", "sleep_date", "lifelog_date"]], calendar_config)
    hybrid_prior = 0.55 * bracket_prior[TARGETS] + 0.45 * calendar_prior[TARGETS]
    hybrid_prior = pd.DataFrame(hybrid_prior, columns=TARGETS).clip(0.045, 0.955)

    candidates = []
    for tag, prior in [
        (f"bracket_{bracket_config.tag}", bracket_prior),
        (f"calendar_{calendar_config.tag}", calendar_prior),
        ("hybrid_bracket_calendar", hybrid_prior),
    ]:
        pure = sample.copy()
        pure[TARGETS] = prior[TARGETS].to_numpy(dtype=float)
        pure[TARGETS] = pure[TARGETS].clip(1e-6, 1 - 1e-6)
        candidates.append(save_candidate(f"sub_{tag}_pure_20260507.csv", pure, anchor))

        for weight in [0.04, 0.07, 0.10]:
            name = f"sub_{tag}_anchor_w{int(weight * 100):02d}_20260507.csv"
            candidates.append(save_candidate(name, blend(anchor, pure, weight), anchor))

    tw_weights = {
        "Q1": 0.08,
        "Q2": 0.08,
        "Q3": 0.08,
        "S1": 0.06,
        "S2": 0.06,
        "S3": 0.04,
        "S4": 0.06,
    }
    candidates.append(
        save_candidate(
            "sub_hybrid_bracket_calendar_anchor_tw_q08_s06_20260507.csv",
            targetwise_blend(anchor, pd.concat([sample[["subject_id", "sleep_date", "lifelog_date"]], hybrid_prior], axis=1), tw_weights),
            anchor,
        )
    )

    summary = pd.DataFrame(candidates).sort_values("diff_vs_best").reset_index(drop=True)
    print("\nSaved candidate summary vs current best 0.5892681038:")
    print(summary.to_string(index=False))

    print("\nSuggested 3-submit order:")
    print(f"1) sub_hybrid_bracket_calendar_anchor_w07_20260507.csv")
    print(f"2) sub_calendar_{calendar_config.tag}_anchor_w07_20260507.csv")
    print(f"3) sub_bracket_{bracket_config.tag}_anchor_w07_20260507.csv")


if __name__ == "__main__":
    main()
