from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "raw" / "data"
SUB_DIR = BASE_DIR / "submissions"
TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
ANCHOR_PATH = SUB_DIR / "sub_dateinterp_smooth_tau10_anchor_w07_20260506.csv"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def shift_to_mean(probs: np.ndarray, target_mean: float) -> np.ndarray:
    logits = logit(probs)
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        mean = sigmoid(logits + mid).mean()
        if mean < target_mean:
            lo = mid
        else:
            hi = mid
    return sigmoid(logits + (lo + hi) / 2)


def mean_align(anchor: pd.DataFrame, train_mean: pd.Series, gamma: float) -> pd.DataFrame:
    out = anchor.copy()
    anchor_mean = anchor[TARGETS].mean()
    desired = (1 - gamma) * anchor_mean + gamma * train_mean
    for target in TARGETS:
        out[target] = shift_to_mean(anchor[target].to_numpy(dtype=float), float(desired[target]))
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def temperature(anchor: pd.DataFrame, temp: float) -> pd.DataFrame:
    out = anchor.copy()
    probs = anchor[TARGETS].to_numpy(dtype=float)
    out[TARGETS] = sigmoid(logit(probs) * temp).clip(1e-6, 1 - 1e-6)
    return out


def sym_sqrt(mat: np.ndarray, inverse: bool = False, eps: float = 1e-5) -> np.ndarray:
    vals, vecs = np.linalg.eigh((mat + mat.T) / 2)
    vals = np.clip(vals, eps, None)
    vals = 1.0 / np.sqrt(vals) if inverse else np.sqrt(vals)
    return (vecs * vals) @ vecs.T


def corr_align(anchor: pd.DataFrame, train_corr: np.ndarray, gamma: float) -> pd.DataFrame:
    out = anchor.copy()
    probs = anchor[TARGETS].to_numpy(dtype=float)
    logits = logit(probs)
    mean = logits.mean(axis=0)
    std = logits.std(axis=0) + 1e-6
    z = (logits - mean) / std
    pred_corr = np.corrcoef(z, rowvar=False)
    desired = (1 - gamma) * pred_corr + gamma * train_corr
    desired = (desired + desired.T) / 2
    np.fill_diagonal(desired, 1.0)
    transform = sym_sqrt(pred_corr, inverse=True) @ sym_sqrt(desired)
    z_new = z @ transform
    out[TARGETS] = sigmoid(z_new * std + mean).clip(1e-6, 1 - 1e-6)
    return out


def summarize(name: str, candidate: pd.DataFrame, anchor: pd.DataFrame, train_mean: pd.Series) -> dict[str, float | str]:
    diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
    mean_gap = (candidate[TARGETS].mean() - train_mean).abs().mean()
    anchor_gap = (anchor[TARGETS].mean() - train_mean).abs().mean()
    return {
        "candidate": name,
        "diff_vs_best": float(diff.values.mean()),
        "max_diff": float(diff.values.max()),
        "mean_gap_delta": float(mean_gap - anchor_gap),
        "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
        "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
    }


def save(name: str, candidate: pd.DataFrame, anchor: pd.DataFrame, train_mean: pd.Series) -> dict[str, float | str]:
    validate(candidate, name)
    candidate.to_csv(SUB_DIR / name, index=False)
    return summarize(name, candidate, anchor, train_mean)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    anchor = pd.read_csv(ANCHOR_PATH)
    validate(anchor, ANCHOR_PATH.name)

    train_mean = train[TARGETS].mean()
    train_corr = train[TARGETS].corr().fillna(0.0).to_numpy()
    rows = []

    for gamma in [0.02, 0.04, 0.06]:
        rows.append(save(f"sub_currentbest_meanalign_g{int(gamma*1000):03d}_20260507.csv", mean_align(anchor, train_mean, gamma), anchor, train_mean))

    for temp in [0.97, 1.03, 1.06]:
        rows.append(save(f"sub_currentbest_temp_t{int(temp*1000):04d}_20260507.csv", temperature(anchor, temp), anchor, train_mean))

    for gamma in [0.04, 0.08]:
        rows.append(save(f"sub_currentbest_corralign_g{int(gamma*1000):03d}_20260507.csv", corr_align(anchor, train_corr, gamma), anchor, train_mean))

    summary = pd.DataFrame(rows).sort_values(["diff_vs_best", "candidate"]).reset_index(drop=True)
    print("Current-best calibration candidates:")
    print(summary.to_string(index=False))
    print("\nSuggested calibration candidate:")
    print("sub_currentbest_meanalign_g040_20260507.csv")


if __name__ == "__main__":
    main()
