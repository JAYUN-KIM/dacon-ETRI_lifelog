from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("/mnt/c/etri-lifelog/data/raw/data")
SUB_DIR = DATA_DIR / "submissions"
TRAIN_PATH = DATA_DIR / "ch2026_metrics_train.csv"
ANCHOR_PATH = SUB_DIR / "sub_seed3_routing_q6040_s2080_alpha098.csv"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sym_sqrt(mat, inverse=False, eps=1e-5):
    vals, vecs = np.linalg.eigh((mat + mat.T) / 2)
    vals = np.clip(vals, eps, None)
    if inverse:
        vals = 1 / np.sqrt(vals)
    else:
        vals = np.sqrt(vals)
    return (vecs * vals) @ vecs.T


def corr_distance(a, b):
    return float(np.abs(a - b).mean())


def make_aligned(anchor, train_corr, gamma):
    out = anchor.copy()
    probs = anchor[TARGETS].astype(float).to_numpy()
    logits = logit(probs)
    mean = logits.mean(axis=0)
    std = logits.std(axis=0) + 1e-6
    z = (logits - mean) / std

    pred_corr = np.corrcoef(z, rowvar=False)
    desired_corr = (1 - gamma) * pred_corr + gamma * train_corr
    desired_corr = (desired_corr + desired_corr.T) / 2
    np.fill_diagonal(desired_corr, 1.0)

    transform = sym_sqrt(pred_corr, inverse=True) @ sym_sqrt(desired_corr, inverse=False)
    z_new = z @ transform
    logits_new = z_new * std + mean
    out[TARGETS] = np.clip(sigmoid(logits_new), 0.03, 0.97)
    return out


def save_report(name, out, anchor, train_corr):
    out_path = SUB_DIR / f"sub_{name}.csv"
    out.to_csv(out_path, index=False)
    diff = (out[TARGETS] - anchor[TARGETS]).abs()
    out_corr = out[TARGETS].corr().to_numpy()
    anchor_corr = anchor[TARGETS].corr().to_numpy()
    print("=" * 90)
    print(name)
    print("saved:", out_path)
    print("shape:", out.shape)
    print("nulls:", int(out.isnull().sum().sum()))
    print("all targets in [0,1]:", bool(((out[TARGETS] >= 0) & (out[TARGETS] <= 1)).all().all()))
    print("mean_abs_diff_from_anchor:", float(diff.values.mean()))
    print("max_abs_diff_from_anchor:", float(diff.values.max()))
    print("anchor_corr_dist:", corr_distance(anchor_corr, train_corr))
    print("out_corr_dist:", corr_distance(out_corr, train_corr))
    print("corr_dist_delta:", corr_distance(out_corr, train_corr) - corr_distance(anchor_corr, train_corr))
    print("per_target_mean_abs_diff:")
    print(diff.mean())
    print("target_mean:")
    print(out[TARGETS].mean())


def main():
    train = pd.read_csv(TRAIN_PATH)
    anchor = pd.read_csv(ANCHOR_PATH)
    train_corr = train[TARGETS].corr().fillna(0.0).to_numpy()

    for gamma in [0.05, 0.10, 0.15, 0.20]:
        out = make_aligned(anchor, train_corr, gamma=gamma)
        save_report(f"copula_corralign_q6040_g{int(gamma * 1000):03d}", out, anchor, train_corr)


if __name__ == "__main__":
    main()
