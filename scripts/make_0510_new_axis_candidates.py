from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "raw" / "data"
SUB_DIR = BASE_DIR / "submissions"

TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"
FEATURE_PATH = BASE_DIR / "artifacts_day_feature_table.csv"

CURRENT_BEST = SUB_DIR / "sub_0509_afterg030_targetscale_soft.csv"
DATE_PRIOR = SUB_DIR / "sub_dateinterp_smooth_tau10_pure_20260506.csv"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEYS = ["subject_id", "lifelog_date"]


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


def logloss(y_true: np.ndarray, pred: np.ndarray) -> float:
    pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
    y_true = np.asarray(y_true, dtype=float)
    return float(-(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred)).mean())


def blend(anchor: pd.DataFrame, prior: pd.DataFrame, weights: dict[str, float] | float) -> pd.DataFrame:
    out = anchor.copy()
    if isinstance(weights, dict):
        for target in TARGETS:
            w = float(weights.get(target, 0.0))
            out[target] = (1 - w) * anchor[target].to_numpy(float) + w * prior[target].to_numpy(float)
    else:
        w = float(weights)
        out[TARGETS] = (1 - w) * anchor[TARGETS].to_numpy(float) + w * prior[TARGETS].to_numpy(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def target_pattern_code(values: np.ndarray) -> int:
    bits = (np.asarray(values, dtype=float) >= 0.5).astype(int)
    code = 0
    for bit in bits:
        code = (code << 1) | int(bit)
    return int(code)


def pattern_projection_prior(train: pd.DataFrame, anchor: pd.DataFrame, prior_strength: float = 0.45) -> pd.DataFrame:
    # Nonlinear joint-target prior: posterior over observed 7-bit patterns.
    patterns = []
    for _, row in train[TARGETS].iterrows():
        bits = row.to_numpy(dtype=int)
        patterns.append(tuple(int(x) for x in bits))
    pattern_df = pd.DataFrame(patterns, columns=TARGETS)
    counts = pattern_df.value_counts().reset_index(name="count")
    pattern_mat = counts[TARGETS].to_numpy(float)
    pattern_prior = (counts["count"].to_numpy(float) + prior_strength) / (len(train) + prior_strength * len(counts))

    probs = anchor[TARGETS].to_numpy(float)
    out_probs = np.zeros_like(probs)
    log_prior = np.log(pattern_prior)
    for i, p in enumerate(np.clip(probs, 1e-5, 1 - 1e-5)):
        logp = log_prior + (pattern_mat * np.log(p) + (1 - pattern_mat) * np.log(1 - p)).sum(axis=1)
        logp -= logp.max()
        weights = np.exp(logp)
        weights /= weights.sum()
        out_probs[i] = weights @ pattern_mat

    out = anchor.copy()
    out[TARGETS] = np.clip(out_probs, 1e-6, 1 - 1e-6)
    return out


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = set(KEYS + TARGETS)
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # Keep stable daily aggregates and prior deviations; avoid pure identifiers.
            cols.append(col)
    return cols


def prepare_feature_matrix(train_feat: pd.DataFrame, test_feat: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    train_x = train_feat[cols].replace([np.inf, -np.inf], np.nan)
    test_x = test_feat[cols].replace([np.inf, -np.inf], np.nan)
    med = train_x.median()
    train_x = train_x.fillna(med)
    test_x = test_x.fillna(med)
    q25 = train_x.quantile(0.25)
    q75 = train_x.quantile(0.75)
    scale = (q75 - q25).replace(0, np.nan).fillna(train_x.std().replace(0, 1)).fillna(1.0)
    train_z = ((train_x - med) / scale).clip(-6, 6).to_numpy(float)
    test_z = ((test_x - med) / scale).clip(-6, 6).to_numpy(float)
    return train_z, test_z


def sensor_knn_prior(train: pd.DataFrame, sample: pd.DataFrame, feature_df: pd.DataFrame, k: int, tau: float) -> pd.DataFrame:
    train_keys = train[KEYS].copy()
    sample_keys = sample[KEYS].copy()
    feature_df = feature_df.copy()
    for frame in [train_keys, sample_keys, feature_df]:
        frame["subject_id"] = frame["subject_id"].astype(str)
        frame["lifelog_date"] = pd.to_datetime(frame["lifelog_date"])

    train_feat = train_keys.merge(feature_df, on=KEYS, how="left")
    test_feat = sample_keys.merge(feature_df, on=KEYS, how="left")
    cols = select_feature_columns(feature_df)
    train_x, test_x = prepare_feature_matrix(train_feat, test_feat, cols)
    y = train[TARGETS].to_numpy(float)

    # Downweight same-subject nearest neighbors to avoid just recreating target prior.
    train_subjects = train["subject_id"].astype(str).to_numpy()
    test_subjects = sample["subject_id"].astype(str).to_numpy()
    out = np.zeros((len(test_x), len(TARGETS)), dtype=float)
    global_mean = train[TARGETS].mean().to_numpy(float)

    for i, x in enumerate(test_x):
        dist = np.sqrt(((train_x - x) ** 2).mean(axis=1))
        order = np.argsort(dist)[: max(k * 3, k)]
        chosen = order[:k]
        d = dist[chosen]
        weights = np.exp(-d / tau)
        same = train_subjects[chosen] == test_subjects[i]
        weights = weights * np.where(same, 0.65, 1.0)
        if weights.sum() <= 1e-12:
            pred = global_mean
        else:
            pred = (weights[:, None] * y[chosen]).sum(axis=0) / weights.sum()
            reliability = weights.sum() / (weights.sum() + 2.5)
            pred = reliability * pred + (1 - reliability) * global_mean
        out[i] = np.clip(pred, 0.045, 0.955)

    prior = sample.copy()
    prior[TARGETS] = out
    return prior


def interleaved_sensor_cv(train: pd.DataFrame, feature_df: pd.DataFrame, k: int, tau: float) -> float:
    preds = []
    ys = []
    for _, group in train.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id"):
        group = group.reset_index(drop=True)
        if len(group) < 15:
            continue
        mask = ((np.arange(len(group)) + 2) % 5 == 0)
        hold = group[mask].copy()
        fit = pd.concat([train[train["subject_id"] != group["subject_id"].iloc[0]], group[~mask]], ignore_index=True)
        pred = sensor_knn_prior(fit, hold[["subject_id", "sleep_date", "lifelog_date"]].copy(), feature_df, k=k, tau=tau)
        preds.append(pred[TARGETS].to_numpy(float))
        ys.append(hold[TARGETS].to_numpy(float))
    return logloss(np.vstack(ys), np.vstack(preds))


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


def save(name: str, candidate: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, float | str]:
    validate(candidate, name)
    candidate.to_csv(SUB_DIR / name, index=False)
    return summarize(name, candidate, anchor)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH, parse_dates=["sleep_date", "lifelog_date"])
    sample = pd.read_csv(SAMPLE_PATH, parse_dates=["sleep_date", "lifelog_date"])
    feature_df = pd.read_csv(FEATURE_PATH, parse_dates=["lifelog_date"])
    anchor = pd.read_csv(CURRENT_BEST)
    date_prior = pd.read_csv(DATE_PRIOR)
    validate(anchor, CURRENT_BEST.name)
    validate(date_prior, DATE_PRIOR.name)

    rows = []

    # Axis 1: nonlinear joint target pattern projection.
    pattern_prior = pattern_projection_prior(train, anchor)
    rows.append(save("sub_0510_patternproj_w015.csv", blend(anchor, pattern_prior, 0.015), anchor))
    rows.append(save("sub_0510_patternproj_w025.csv", blend(anchor, pattern_prior, 0.025), anchor))
    rows.append(save("sub_0510_patternproj_w040.csv", blend(anchor, pattern_prior, 0.040), anchor))

    # Axis 2: sensor feature nearest-neighbor prior. Keep weak because prior sensor experiments overfit.
    cv_rows = []
    for k, tau in [(16, 2.8), (24, 3.4), (32, 4.2)]:
        cv_rows.append({"k": k, "tau": tau, "cv": interleaved_sensor_cv(train, feature_df, k=k, tau=tau)})
    cv = pd.DataFrame(cv_rows).sort_values("cv").reset_index(drop=True)
    print("Sensor KNN interleaved CV:")
    print(cv.to_string(index=False))
    best_k = int(cv.loc[0, "k"])
    best_tau = float(cv.loc[0, "tau"])
    sensor_prior = sensor_knn_prior(train, sample, feature_df, k=best_k, tau=best_tau)
    rows.append(save(f"sub_0510_sensorknn_k{best_k}_w010.csv", blend(anchor, sensor_prior, 0.010), anchor))
    rows.append(save(f"sub_0510_sensorknn_k{best_k}_w020.csv", blend(anchor, sensor_prior, 0.020), anchor))
    rows.append(save(f"sub_0510_sensorknn_k{best_k}_w035.csv", blend(anchor, sensor_prior, 0.035), anchor))

    # Axis 3: combine known date prior with sensor prior only as a residual nudge.
    hybrid_prior = date_prior.copy()
    hybrid_prior[TARGETS] = (0.82 * date_prior[TARGETS].to_numpy(float) + 0.18 * sensor_prior[TARGETS].to_numpy(float)).clip(0.045, 0.955)
    target_weights = {"Q1": 0.018, "Q2": 0.020, "Q3": 0.020, "S1": 0.012, "S2": 0.015, "S3": 0.008, "S4": 0.014}
    rows.append(save("sub_0510_date_sensor_hybrid_tw.csv", blend(anchor, hybrid_prior, target_weights), anchor))

    summary = pd.DataFrame(rows).sort_values("diff_vs_best").reset_index(drop=True)
    print("\n0510 new-axis candidate summary vs current best 0.5883722159:")
    print(summary.to_string(index=False))
    print("\nSuggested submit order:")
    print("1) sub_0510_date_sensor_hybrid_tw.csv")
    print("2) sub_0510_patternproj_w015.csv")
    print("3) sub_0510_patternproj_w025.csv")
    print(f"Fallback risky sensor axis: sub_0510_sensorknn_k{best_k}_w010.csv")


if __name__ == "__main__":
    main()
