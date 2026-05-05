from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/mnt/c/etri-lifelog")
DATA_DIR = ROOT / "data" / "raw" / "data"
SUB_DIR = DATA_DIR / "submissions"
TRAIN_PATH = DATA_DIR / "ch2026_metrics_train.csv"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


ANCHORS = [
    ("q6040", "sub_seed3_routing_q6040_s2080_alpha098.csv"),
    ("copula_g050", "sub_copula_corralign_q6040_g050.csv"),
]
GAMMAS = [0.03, 0.06, 0.10]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def shift_to_mean(probs, target_mean):
    """Find a logit intercept shift that preserves ranking and matches a target mean."""
    logits = logit(probs)
    lo, hi = -8.0, 8.0
    for _ in range(70):
        mid = (lo + hi) / 2
        mean = sigmoid(logits + mid).mean()
        if mean < target_mean:
            lo = mid
        else:
            hi = mid
    return sigmoid(logits + (lo + hi) / 2)


def make_candidate(anchor_name, anchor_file, gamma, train_mean):
    anchor = pd.read_csv(SUB_DIR / anchor_file)
    out = anchor.copy()
    before_mean = anchor[TARGETS].mean()
    desired_mean = (1 - gamma) * before_mean + gamma * train_mean

    for target in TARGETS:
        out[target] = shift_to_mean(anchor[target].values, desired_mean[target])

    out[TARGETS] = out[TARGETS].clip(0.03, 0.97)
    out_path = SUB_DIR / f"sub_meanalign_{anchor_name}_g{int(gamma * 1000):03d}.csv"
    out.to_csv(out_path, index=False)

    diff = (out[TARGETS] - anchor[TARGETS]).abs()
    print("=" * 90)
    print(f"{anchor_name} gamma={gamma:.2f}")
    print("saved:", out_path)
    print("shape:", out.shape)
    print("nulls:", int(out.isnull().sum().sum()))
    print("all targets in [0,1]:", bool(((out[TARGETS] >= 0) & (out[TARGETS] <= 1)).all().all()))
    print("mean_abs_diff_from_anchor:", float(diff.values.mean()))
    print("max_abs_diff_from_anchor:", float(diff.values.max()))
    print("anchor_mean:")
    print(before_mean)
    print("desired_mean:")
    print(desired_mean)
    print("out_mean:")
    print(out[TARGETS].mean())
    print("per_target_mean_abs_diff:")
    print(diff.mean())


def main():
    train = pd.read_csv(TRAIN_PATH)
    train_mean = train[TARGETS].mean()
    print("train target mean:")
    print(train_mean)

    for anchor_name, anchor_file in ANCHORS:
        for gamma in GAMMAS:
            make_candidate(anchor_name, anchor_file, gamma, train_mean)


if __name__ == "__main__":
    main()
