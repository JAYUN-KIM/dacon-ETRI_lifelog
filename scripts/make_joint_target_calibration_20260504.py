from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/mnt/c/etri-lifelog")
DATA_DIR = ROOT / "data" / "raw" / "data"
SUB_DIR = DATA_DIR / "submissions"
TRAIN_PATH = DATA_DIR / "ch2026_metrics_train.csv"
ANCHOR_PATH = SUB_DIR / "sub_seed3_routing_q6040_s2080_alpha098.csv"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def save_and_report(name, out, anchor):
    out_path = SUB_DIR / f"sub_{name}.csv"
    out.to_csv(out_path, index=False)
    diff = (out[TARGETS] - anchor[TARGETS]).abs()
    print("=" * 90)
    print(name)
    print("saved:", out_path)
    print("shape:", out.shape)
    print("nulls:", int(out.isnull().sum().sum()))
    print("all targets in [0,1]:", bool(((out[TARGETS] >= 0) & (out[TARGETS] <= 1)).all().all()))
    print("mean_abs_diff_from_anchor:", float(diff.values.mean()))
    print("max_abs_diff_from_anchor:", float(diff.values.max()))
    print("per_target_mean_abs_diff:")
    print(diff.mean())
    print("target_mean:")
    print(out[TARGETS].mean())


def make_joint_corr(anchor, corr, beta, name):
    out = anchor.copy()
    pred = anchor[TARGETS].astype(float)
    z = (pred - pred.mean()) / (pred.std(ddof=0) + 1e-6)
    corr_mat = corr.loc[TARGETS, TARGETS].to_numpy().copy()
    np.fill_diagonal(corr_mat, 0.0)

    # Keep the correction intentionally weak: this is a structural nudge, not a new model.
    adjustment = np.clip(z.to_numpy() @ corr_mat.T, -2.0, 2.0)
    logits = logit(pred.to_numpy())
    calibrated = sigmoid(logits + beta * adjustment)
    out[TARGETS] = np.clip(calibrated, 0.03, 0.97)
    save_and_report(name, out, anchor)


def make_subject_prior(anchor, train, weight, smooth, name):
    out = anchor.copy()
    global_mean = train[TARGETS].mean()
    grouped_sum = train.groupby("subject_id")[TARGETS].sum()
    grouped_count = train.groupby("subject_id")[TARGETS[0]].count()
    prior = grouped_sum.add(smooth * global_mean, axis=1).div(grouped_count + smooth, axis=0)
    prior = prior.reindex(anchor["subject_id"]).reset_index(drop=True)
    out[TARGETS] = ((1 - weight) * anchor[TARGETS].to_numpy() + weight * prior[TARGETS].to_numpy()).clip(0.03, 0.97)
    save_and_report(name, out, anchor)


def make_joint_then_prior(anchor, train, corr, beta, weight, smooth, name):
    temp = anchor.copy()
    pred = temp[TARGETS].astype(float)
    z = (pred - pred.mean()) / (pred.std(ddof=0) + 1e-6)
    corr_mat = corr.loc[TARGETS, TARGETS].to_numpy().copy()
    np.fill_diagonal(corr_mat, 0.0)
    adjustment = np.clip(z.to_numpy() @ corr_mat.T, -2.0, 2.0)
    temp[TARGETS] = np.clip(sigmoid(logit(pred.to_numpy()) + beta * adjustment), 0.03, 0.97)

    global_mean = train[TARGETS].mean()
    grouped_sum = train.groupby("subject_id")[TARGETS].sum()
    grouped_count = train.groupby("subject_id")[TARGETS[0]].count()
    prior = grouped_sum.add(smooth * global_mean, axis=1).div(grouped_count + smooth, axis=0)
    prior = prior.reindex(anchor["subject_id"]).reset_index(drop=True)

    out = temp.copy()
    out[TARGETS] = ((1 - weight) * temp[TARGETS].to_numpy() + weight * prior[TARGETS].to_numpy()).clip(0.03, 0.97)
    save_and_report(name, out, anchor)


def main():
    train = pd.read_csv(TRAIN_PATH)
    anchor = pd.read_csv(ANCHOR_PATH)
    corr = train[TARGETS].corr().fillna(0.0)

    print("anchor:", ANCHOR_PATH)
    print("train target mean:")
    print(train[TARGETS].mean())
    print("train target corr:")
    print(corr)

    for beta in [0.02, 0.04, 0.06]:
        make_joint_corr(anchor, corr, beta, f"jointcorr_q6040_beta{int(beta * 1000):03d}")

    for weight in [0.03, 0.05, 0.08]:
        make_subject_prior(anchor, train, weight, smooth=20, name=f"subprior_q6040_w{int(weight * 1000):03d}")

    make_joint_then_prior(
        anchor,
        train,
        corr,
        beta=0.03,
        weight=0.03,
        smooth=20,
        name="jointcorr_subprior_q6040_beta030_w030",
    )


if __name__ == "__main__":
    main()
