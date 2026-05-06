from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "raw" / "data"
TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
SUB_SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"
SUB_DIR = BASE_DIR / "submissions"
ANCHOR_PATH = SUB_DIR / "sub_anchor_q6040_statepast_tw_b_20260505.csv"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
Q_TARGETS = {"Q1", "Q2", "Q3"}


@dataclass(frozen=True)
class DynamicsConfig:
    tag: str
    q_weights: tuple[float, float, float, float, float]
    s_weights: tuple[float, float, float, float, float]
    cross_scale: float
    trend_scale: float
    transition_decay: float
    clip_low: float = 0.045
    clip_high: float = 0.955

    def weights_for(self, target: str) -> tuple[float, float, float, float, float]:
        return self.q_weights if target in Q_TARGETS else self.s_weights


CONFIGS = [
    DynamicsConfig(
        tag="conservative",
        q_weights=(0.42, 0.18, 0.12, 0.22, 0.06),
        s_weights=(0.38, 0.20, 0.16, 0.18, 0.08),
        cross_scale=0.25,
        trend_scale=0.08,
        transition_decay=5.0,
    ),
    DynamicsConfig(
        tag="transition",
        q_weights=(0.34, 0.14, 0.10, 0.34, 0.08),
        s_weights=(0.32, 0.18, 0.14, 0.26, 0.10),
        cross_scale=0.35,
        trend_scale=0.08,
        transition_decay=4.0,
    ),
    DynamicsConfig(
        tag="pattern",
        q_weights=(0.32, 0.14, 0.10, 0.26, 0.18),
        s_weights=(0.30, 0.18, 0.14, 0.22, 0.16),
        cross_scale=0.32,
        trend_scale=0.06,
        transition_decay=4.5,
    ),
    DynamicsConfig(
        tag="revert",
        q_weights=(0.34, 0.26, 0.24, 0.12, 0.04),
        s_weights=(0.32, 0.28, 0.26, 0.10, 0.04),
        cross_scale=0.12,
        trend_scale=0.04,
        transition_decay=3.0,
    ),
]


def clip_prob(p: float, low: float = 0.045, high: float = 0.955) -> float:
    return float(np.clip(p, low, high))


def binary_logloss(y_true: np.ndarray, pred: np.ndarray) -> float:
    pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
    y_true = np.asarray(y_true, dtype=float)
    return float(-(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred)).mean())


def mean_or_global(values: np.ndarray, global_mean: float) -> float:
    if len(values) == 0:
        return float(global_mean)
    return float(np.mean(values))


def ewma_tail(values: np.ndarray, alpha: float, global_mean: float) -> float:
    if len(values) == 0:
        return float(global_mean)
    out = float(values[0])
    for value in values[1:]:
        out = alpha * float(value) + (1 - alpha) * out
    return out


def pattern_code(values: np.ndarray) -> int:
    bits = (np.asarray(values, dtype=float) >= 0.5).astype(int)
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return int(out)


def safe_recent_mix(values: np.ndarray, global_mean: float) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float(global_mean)
    last1 = float(values[-1])
    last3 = mean_or_global(values[-3:], global_mean)
    last7 = mean_or_global(values[-7:], global_mean)
    last14 = mean_or_global(values[-14:], global_mean)
    ewma_fast = ewma_tail(values, 0.48, global_mean)
    ewma_slow = ewma_tail(values, 0.22, global_mean)
    return float(
        0.14 * last1
        + 0.22 * last3
        + 0.24 * last7
        + 0.14 * last14
        + 0.16 * ewma_fast
        + 0.10 * ewma_slow
    )


def trend_adjust(values: np.ndarray, horizon_days: int, scale: float) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 6 or np.nanstd(values[-10:]) < 1e-8:
        return 0.0
    tail = values[-12:]
    x = np.arange(len(tail), dtype=float)
    slope = float(np.polyfit(x, tail, 1)[0])
    horizon = min(max(int(horizon_days), 1), 10)
    return float(np.clip(slope * math.sqrt(horizon) * scale, -0.055, 0.055))


def build_dow_adjustments(train: pd.DataFrame) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for target in TARGETS:
        global_mean = float(train[target].mean())
        mean_by_dow = train.groupby("dow")[target].mean()
        count_by_dow = train.groupby("dow")[target].count()
        out[target] = {}
        for dow in range(7):
            if dow not in mean_by_dow.index:
                out[target][dow] = 0.0
                continue
            reliability = min(float(count_by_dow.loc[dow]) / 80.0, 1.0)
            raw = (float(mean_by_dow.loc[dow]) - global_mean) * 0.08 * reliability
            out[target][dow] = float(np.clip(raw, -0.018, 0.018))
    return out


def consecutive_pairs(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, group in train.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id"):
        group = group.reset_index(drop=True)
        for i in range(1, len(group)):
            prev = group.loc[i - 1, TARGETS].astype(float).to_numpy()
            curr = group.loc[i, TARGETS].astype(float).to_numpy()
            rows.append((sid, pattern_code(prev), *prev, *curr))
    cols = ["subject_id", "pattern"] + [f"prev_{t}" for t in TARGETS] + [f"curr_{t}" for t in TARGETS]
    return pd.DataFrame(rows, columns=cols)


def smooth_rate(pos: float, total: float, prior: float, strength: float) -> float:
    return float((pos + strength * prior) / (total + strength))


def fit_stats(train_fit: pd.DataFrame) -> dict:
    train_fit = train_fit.sort_values(["subject_id", "lifelog_date"]).copy()
    train_fit["dow"] = train_fit["lifelog_date"].dt.dayofweek
    pairs = consecutive_pairs(train_fit)

    global_mean = {target: float(train_fit[target].mean()) for target in TARGETS}
    dow_adjust = build_dow_adjustments(train_fit)

    global_self: dict[str, tuple[float, float]] = {}
    subject_self: dict[tuple[str, str], tuple[float, float]] = {}
    for target in TARGETS:
        prev_col = f"prev_{target}"
        curr_col = f"curr_{target}"
        if len(pairs) == 0:
            global_self[target] = (global_mean[target], global_mean[target])
            continue
        p0_df = pairs[pairs[prev_col] < 0.5]
        p1_df = pairs[pairs[prev_col] >= 0.5]
        p0 = smooth_rate(float(p0_df[curr_col].sum()), len(p0_df), global_mean[target], 8.0)
        p1 = smooth_rate(float(p1_df[curr_col].sum()), len(p1_df), global_mean[target], 8.0)
        global_self[target] = (p0, p1)

        for sid, sid_pairs in pairs.groupby("subject_id"):
            s0 = sid_pairs[sid_pairs[prev_col] < 0.5]
            s1 = sid_pairs[sid_pairs[prev_col] >= 0.5]
            sp0 = smooth_rate(float(s0[curr_col].sum()), len(s0), p0, 5.0)
            sp1 = smooth_rate(float(s1[curr_col].sum()), len(s1), p1, 5.0)
            subject_self[(str(sid), target)] = (sp0, sp1)

    pattern_stats: dict[str, dict[int, tuple[float, int]]] = {target: {} for target in TARGETS}
    if len(pairs):
        for target in TARGETS:
            curr_col = f"curr_{target}"
            for pattern, group in pairs.groupby("pattern"):
                n = len(group)
                p = smooth_rate(float(group[curr_col].sum()), n, global_mean[target], 10.0)
                pattern_stats[target][int(pattern)] = (p, n)

    cross_coef: dict[str, dict[str, float]] = {target: {} for target in TARGETS}
    if len(pairs):
        for target in TARGETS:
            curr_col = f"curr_{target}"
            for source in TARGETS:
                if source == target:
                    cross_coef[target][source] = 0.0
                    continue
                prev_col = f"prev_{source}"
                high = pairs[pairs[prev_col] >= 0.5]
                low = pairs[pairs[prev_col] < 0.5]
                if len(high) < 8 or len(low) < 8:
                    coef = 0.0
                else:
                    reliability = min(min(len(high), len(low)) / 50.0, 1.0)
                    coef = (float(high[curr_col].mean()) - float(low[curr_col].mean())) * reliability
                cross_coef[target][source] = float(np.clip(coef, -0.16, 0.16))

    return {
        "global_mean": global_mean,
        "dow_adjust": dow_adjust,
        "global_self": global_self,
        "subject_self": subject_self,
        "pattern_stats": pattern_stats,
        "cross_coef": cross_coef,
    }


def predict_one(
    subject_id: str,
    history: pd.DataFrame,
    pred_date: pd.Timestamp,
    stats: dict,
    config: DynamicsConfig,
) -> dict[str, float]:
    if len(history):
        history = history.sort_values("lifelog_date").copy()
        last_date = pd.Timestamp(history["lifelog_date"].iloc[-1])
        horizon_days = max(int((pred_date - last_date).days), 1)
        prev_vec = history[TARGETS].iloc[-1].astype(float).to_numpy()
    else:
        horizon_days = 1
        prev_vec = np.array([stats["global_mean"][t] for t in TARGETS], dtype=float)

    transition_strength = float(np.exp(-(max(horizon_days, 1) - 1) / config.transition_decay))
    dow = int(pred_date.dayofweek)
    prev_code = pattern_code(prev_vec)
    out = {}

    for target_idx, target in enumerate(TARGETS):
        global_base = stats["global_mean"][target] + stats["dow_adjust"][target].get(dow, 0.0)
        values = history[target].astype(float).to_numpy() if len(history) else np.array([], dtype=float)

        recent = safe_recent_mix(values, stats["global_mean"][target])
        subject_mean = mean_or_global(values, stats["global_mean"][target])

        p0, p1 = stats["subject_self"].get(
            (subject_id, target),
            stats["global_self"][target],
        )
        prev_prob = float(np.clip(prev_vec[target_idx], 0.0, 1.0))
        self_transition = (1.0 - prev_prob) * p0 + prev_prob * p1
        self_transition = transition_strength * self_transition + (1 - transition_strength) * subject_mean

        pattern_p = stats["pattern_stats"][target].get(prev_code, (global_base, 0))[0]
        pattern_p = transition_strength * pattern_p + (1 - transition_strength) * subject_mean

        cross_adj = 0.0
        for source_idx, source in enumerate(TARGETS):
            coef = stats["cross_coef"][target].get(source, 0.0)
            cross_adj += coef * (float(prev_vec[source_idx]) - stats["global_mean"][source])
        cross_adj *= config.cross_scale * transition_strength / max(len(TARGETS) - 1, 1)

        recent_w, subject_w, global_w, self_w, pattern_w = config.weights_for(target)
        pred = (
            recent_w * recent
            + subject_w * subject_mean
            + global_w * global_base
            + self_w * self_transition
            + pattern_w * pattern_p
        )
        pred += cross_adj
        pred += trend_adjust(values, horizon_days, config.trend_scale)
        out[target] = clip_prob(pred, config.clip_low, config.clip_high)

    return out


def recursive_predict(
    train_fit: pd.DataFrame,
    test_frame: pd.DataFrame,
    config: DynamicsConfig,
) -> pd.DataFrame:
    stats = fit_stats(train_fit)
    output_rows = []

    histories = {
        str(sid): group[["subject_id", "lifelog_date"] + TARGETS].copy()
        for sid, group in train_fit.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id")
    }

    test_sorted = (
        test_frame.reset_index(drop=True)
        .reset_index(names="row_pos")
        .sort_values(["subject_id", "lifelog_date"])
    )
    pred_store = {}
    for row in test_sorted.itertuples(index=False):
        sid = str(row.subject_id)
        pred_date = pd.Timestamp(row.lifelog_date)
        history = histories.get(sid, pd.DataFrame(columns=["subject_id", "lifelog_date"] + TARGETS))
        pred = predict_one(sid, history, pred_date, stats, config)
        pred_store[int(row.row_pos)] = pred

        append_row = {"subject_id": sid, "lifelog_date": pred_date, **pred}
        histories[sid] = pd.concat([history, pd.DataFrame([append_row])], ignore_index=True)

    for idx in range(len(test_frame)):
        output_rows.append(pred_store[idx])
    return pd.DataFrame(output_rows)


def rolling_tail_validation(train: pd.DataFrame, config: DynamicsConfig) -> dict[str, float]:
    y_true_all = []
    pred_all = []
    per_target = {target: {"y": [], "p": []} for target in TARGETS}

    for _, group in train.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id"):
        n = len(group)
        holdout = min(max(6, int(round(n * 0.24))), 14)
        if n - holdout < 10:
            continue
        fit = train.drop(group.tail(holdout).index).copy()
        valid = group.tail(holdout)[["subject_id", "sleep_date", "lifelog_date"]].copy()
        preds = recursive_predict(fit, valid, config)

        y_true = group.tail(holdout)[TARGETS].to_numpy(dtype=float)
        pred = preds[TARGETS].to_numpy(dtype=float)
        y_true_all.append(y_true)
        pred_all.append(pred)
        for j, target in enumerate(TARGETS):
            per_target[target]["y"].extend(y_true[:, j].tolist())
            per_target[target]["p"].extend(pred[:, j].tolist())

    y = np.vstack(y_true_all)
    p = np.vstack(pred_all)
    out = {"overall": binary_logloss(y, p)}
    for target in TARGETS:
        out[target] = binary_logloss(np.array(per_target[target]["y"]), np.array(per_target[target]["p"]))
    return out


def validate_submission(df: pd.DataFrame, name: str) -> None:
    required = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    if df.shape != (250, 10):
        raise ValueError(f"{name} unexpected shape: {df.shape}")
    if df.isnull().sum().sum() != 0:
        raise ValueError(f"{name} has null values")
    if not ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all():
        raise ValueError(f"{name} has out-of-range probabilities")


def save_submission(name: str, sample: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    out = sample.copy()
    out[TARGETS] = preds[TARGETS].to_numpy(dtype=float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    validate_submission(out, name)
    out.to_csv(SUB_DIR / name, index=False)
    return out


def blend_with_anchor(anchor: pd.DataFrame, dyn: pd.DataFrame, weight: float) -> pd.DataFrame:
    out = anchor.copy()
    out[TARGETS] = (1.0 - weight) * anchor[TARGETS].astype(float) + weight * dyn[TARGETS].astype(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def targetwise_blend_with_anchor(anchor: pd.DataFrame, dyn: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = anchor.copy()
    for target in TARGETS:
        w = weights.get(target, 0.0)
        out[target] = (1.0 - w) * anchor[target].astype(float) + w * dyn[target].astype(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def summarize_candidate(name: str, candidate: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, float | str]:
    diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
    return {
        "candidate": name,
        "diff_vs_best": float(diff.values.mean()),
        "max_diff_vs_best": float(diff.values.max()),
        "q_diff": float(diff[["Q1", "Q2", "Q3"]].values.mean()),
        "s_diff": float(diff[["S1", "S2", "S3", "S4"]].values.mean()),
        "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
        "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
    }


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    sample = pd.read_csv(SUB_SAMPLE_PATH)
    anchor = pd.read_csv(ANCHOR_PATH)

    train["subject_id"] = train["subject_id"].astype(str)
    sample["subject_id"] = sample["subject_id"].astype(str)
    anchor["subject_id"] = anchor["subject_id"].astype(str)
    train["lifelog_date"] = pd.to_datetime(train["lifelog_date"])
    sample["lifelog_date"] = pd.to_datetime(sample["lifelog_date"])
    train["sleep_date"] = pd.to_datetime(train["sleep_date"])
    sample["sleep_date"] = pd.to_datetime(sample["sleep_date"])

    validation_rows = []
    for config in CONFIGS:
        result = rolling_tail_validation(train, config)
        validation_rows.append({"config": config.tag, **result})

    validation = pd.DataFrame(validation_rows).sort_values("overall").reset_index(drop=True)
    print("Rolling recursive tail validation:")
    print(validation.to_string(index=False))

    best_tag = str(validation.loc[0, "config"])
    best_config = next(config for config in CONFIGS if config.tag == best_tag)
    print(f"\nBest reset-dynamics config by tail validation: {best_config.tag}")

    dyn_preds = recursive_predict(train, sample[["subject_id", "sleep_date", "lifelog_date"]].copy(), best_config)

    pure_name = f"sub_reset_targetdyn_{best_config.tag}_pure_20260506.csv"
    pure_sub = save_submission(pure_name, sample, dyn_preds)

    candidates = [summarize_candidate(pure_name, pure_sub, anchor)]

    for weight in [0.06, 0.10, 0.14, 0.18]:
        name = f"sub_reset_targetdyn_{best_config.tag}_anchor_w{int(weight*100):02d}_20260506.csv"
        blended = blend_with_anchor(anchor, pure_sub, weight)
        validate_submission(blended, name)
        blended.to_csv(SUB_DIR / name, index=False)
        candidates.append(summarize_candidate(name, blended, anchor))

    tw_weights = {
        "Q1": 0.12,
        "Q2": 0.18,
        "Q3": 0.18,
        "S1": 0.04,
        "S2": 0.08,
        "S3": 0.02,
        "S4": 0.08,
    }
    tw_name = f"sub_reset_targetdyn_{best_config.tag}_anchor_targetwise_20260506.csv"
    tw_sub = targetwise_blend_with_anchor(anchor, pure_sub, tw_weights)
    validate_submission(tw_sub, tw_name)
    tw_sub.to_csv(SUB_DIR / tw_name, index=False)
    candidates.append(summarize_candidate(tw_name, tw_sub, anchor))

    summary = pd.DataFrame(candidates).sort_values("diff_vs_best").reset_index(drop=True)
    print("\nSaved candidate summary vs current best anchor:")
    print(summary.to_string(index=False))

    print("\nSuggested submit order:")
    print(f"1) sub_reset_targetdyn_{best_config.tag}_anchor_w10_20260506.csv")
    print(f"2) sub_reset_targetdyn_{best_config.tag}_anchor_targetwise_20260506.csv")
    print(f"3) {pure_name} only if we decide to take a high-risk reset shot")


if __name__ == "__main__":
    main()
