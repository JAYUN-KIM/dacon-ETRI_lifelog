
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path("/mnt/c/etri-lifelog/data/raw/data")
TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
SUB_PATH = BASE_DIR / "ch2026_submission_sample.csv"
OUT_PATH = BASE_DIR / "submissions" / "sub_personal_prior_only_seed3_alpha098.csv"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS = [42, 77, 2024]
FIXED_ALPHA = 0.98


def infer_test_frame_from_submission(train_df, sub_df):
    if {"subject_id", "lifelog_date"}.issubset(sub_df.columns):
        test_df = sub_df[["subject_id", "lifelog_date"]].copy()
        test_df["subject_id"] = test_df["subject_id"].astype(str)
        test_df["lifelog_date"] = pd.to_datetime(test_df["lifelog_date"])
        return test_df

    n_test = len(sub_df)
    last_dates = train_df.groupby("subject_id")["lifelog_date"].max().sort_values()
    subjects = list(last_dates.index)
    reps = math.ceil(n_test / len(subjects))
    subject_seq = (subjects * reps)[:n_test]

    next_dates = {
        sid: train_df.loc[train_df["subject_id"] == sid, "lifelog_date"].max() + pd.Timedelta(days=1)
        for sid in subjects
    }

    rows = []
    for sid in subject_seq:
        rows.append((sid, next_dates[sid]))
        next_dates[sid] += pd.Timedelta(days=1)

    return pd.DataFrame(rows, columns=["subject_id", "lifelog_date"])


def add_date_features(df, train_subjects):
    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["lifelog_date"] = pd.to_datetime(df["lifelog_date"])

    subject_map = {sid: i for i, sid in enumerate(sorted(train_subjects))}
    df["subject_code"] = df["subject_id"].map(subject_map).fillna(-1).astype(int)

    df["dow"] = df["lifelog_date"].dt.dayofweek
    df["month"] = df["lifelog_date"].dt.month
    df["day"] = df["lifelog_date"].dt.day
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    global_start = df["lifelog_date"].min()
    df["days_from_global_start"] = (df["lifelog_date"] - global_start).dt.days

    df = df.sort_values(["subject_id", "lifelog_date"]).copy()
    df["subject_day_index"] = df.groupby("subject_id").cumcount()

    subject_start = df.groupby("subject_id")["lifelog_date"].transform("min")
    df["days_from_subject_start"] = (df["lifelog_date"] - subject_start).dt.days

    return df


def build_target_prior_features(train_df, test_df, target):
    key = ["subject_id", "lifelog_date"]

    tr = train_df[key + [target]].sort_values(key).copy()
    grp = tr.groupby("subject_id")[target]

    tr[f"{target}_hist_mean"] = grp.apply(lambda s: s.expanding().mean().shift(1)).reset_index(level=0, drop=True)
    tr[f"{target}_hist_std"] = grp.apply(lambda s: s.expanding().std().shift(1)).reset_index(level=0, drop=True)
    tr[f"{target}_hist_min"] = grp.apply(lambda s: s.expanding().min().shift(1)).reset_index(level=0, drop=True)
    tr[f"{target}_hist_max"] = grp.apply(lambda s: s.expanding().max().shift(1)).reset_index(level=0, drop=True)
    tr[f"{target}_hist_cnt"] = tr.groupby("subject_id").cumcount()

    for w in [3, 7, 14]:
        tr[f"{target}_roll{w}_mean"] = grp.apply(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
        tr[f"{target}_roll{w}_sum"] = grp.apply(
            lambda s: s.shift(1).rolling(w, min_periods=1).sum()
        ).reset_index(level=0, drop=True)

    train_hist = train_df[key].merge(tr.drop(columns=[target]), on=key, how="left").drop(columns=key)

    stats = train_df.groupby("subject_id")[target].agg(["mean", "std", "min", "max", "count"]).reset_index()
    stats.columns = [
        "subject_id",
        f"{target}_hist_mean",
        f"{target}_hist_std",
        f"{target}_hist_min",
        f"{target}_hist_max",
        f"{target}_hist_cnt",
    ]

    recent_rows = []
    for sid, g in train_df.sort_values(key).groupby("subject_id"):
        row = {"subject_id": sid}
        vals = g[target].astype(float).values
        for w in [3, 7, 14]:
            tail = vals[-w:] if len(vals) else np.array([])
            row[f"{target}_roll{w}_mean"] = float(np.mean(tail)) if len(tail) else np.nan
            row[f"{target}_roll{w}_sum"] = float(np.sum(tail)) if len(tail) else np.nan
        recent_rows.append(row)

    recent = pd.DataFrame(recent_rows)
    test_hist = test_df[["subject_id"]].merge(stats, on="subject_id", how="left")
    test_hist = test_hist.merge(recent, on="subject_id", how="left").drop(columns=["subject_id"])

    global_mean = float(train_df[target].mean())
    global_std = float(train_df[target].std()) if not pd.isna(train_df[target].std()) else 0.0
    global_min = float(train_df[target].min())
    global_max = float(train_df[target].max())

    fill = {}
    for c in train_hist.columns:
        if c.endswith("_hist_mean") or c.endswith("_mean"):
            fill[c] = global_mean
        elif c.endswith("_hist_std"):
            fill[c] = global_std
        elif c.endswith("_hist_min"):
            fill[c] = global_min
        elif c.endswith("_hist_max"):
            fill[c] = global_max
        elif c.endswith("_hist_cnt"):
            fill[c] = 0
        elif c.endswith("_sum"):
            fill[c] = global_mean
        else:
            fill[c] = 0

    train_hist = train_hist.fillna(fill)
    test_hist = test_hist.fillna(fill)

    train_hist[f"{target}_hist_mean_minus_global"] = train_hist[f"{target}_hist_mean"] - global_mean
    test_hist[f"{target}_hist_mean_minus_global"] = test_hist[f"{target}_hist_mean"] - global_mean

    return train_hist.reset_index(drop=True), test_hist.reset_index(drop=True)


def shrink_proba(p, alpha=0.98):
    return np.clip(0.5 + alpha * (p - 0.5), 1e-6, 1 - 1e-6)


def clip_proba(p, lo=0.03, hi=0.97):
    return np.clip(p, lo, hi)


def main():
    train = pd.read_csv(TRAIN_PATH)
    sub = pd.read_csv(SUB_PATH)

    train["subject_id"] = train["subject_id"].astype(str)
    train["lifelog_date"] = pd.to_datetime(train["lifelog_date"])

    if "subject_id" in sub.columns:
        sub["subject_id"] = sub["subject_id"].astype(str)
    if "lifelog_date" in sub.columns:
        sub["lifelog_date"] = pd.to_datetime(sub["lifelog_date"])

    test = infer_test_frame_from_submission(train, sub)

    train_base = train[["subject_id", "lifelog_date"] + TARGETS].copy()
    test_base = test[["subject_id", "lifelog_date"]].copy()

    all_dates = pd.concat(
        [
            train_base[["subject_id", "lifelog_date"]],
            test_base[["subject_id", "lifelog_date"]],
        ],
        ignore_index=True,
    )
    all_dates = add_date_features(all_dates, train_base["subject_id"].unique())

    base_cols = [
        "subject_code",
        "dow",
        "month",
        "day",
        "is_weekend",
        "days_from_global_start",
        "subject_day_index",
        "days_from_subject_start",
    ]

    train_date = train_base[["subject_id", "lifelog_date"]].merge(
        all_dates[["subject_id", "lifelog_date"] + base_cols],
        on=["subject_id", "lifelog_date"],
        how="left",
    )[base_cols].reset_index(drop=True)

    test_date = test_base[["subject_id", "lifelog_date"]].merge(
        all_dates[["subject_id", "lifelog_date"] + base_cols],
        on=["subject_id", "lifelog_date"],
        how="left",
    )[base_cols].reset_index(drop=True)

    print("=" * 90)
    print("PERSONAL PRIOR ONLY MODEL")
    print("No sensor parquet features are used.")
    print("Seeds:", SEEDS)
    print("=" * 90)

    pred_lgb = pd.DataFrame(0.0, index=test_base.index, columns=TARGETS)
    pred_cat = pd.DataFrame(0.0, index=test_base.index, columns=TARGETS)

    for seed in SEEDS:
        print(f"\n[SEED] {seed}")

        for target in TARGETS:
            print(f"  > target={target}")
            hist_train, hist_test = build_target_prior_features(train_base, test_base, target)

            X_train = pd.concat([train_date, hist_train], axis=1)
            X_test = pd.concat([test_date, hist_test], axis=1)
            feature_cols = list(X_train.columns)

            y = train_base[target].astype(int)

            med = X_train[feature_cols].median(numeric_only=True)
            X_train = X_train.fillna(med)
            X_test = X_test.fillna(med)

            lgb_model = lgb.LGBMClassifier(
                n_estimators=180,
                learning_rate=0.03,
                num_leaves=7,
                max_depth=3,
                min_child_samples=15,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=2.0,
                reg_lambda=5.0,
                objective="binary",
                class_weight="balanced",
                random_state=seed,
            )
            lgb_model.fit(X_train[feature_cols], y)
            pred_lgb[target] += lgb_model.predict_proba(X_test[feature_cols])[:, 1] / len(SEEDS)

            cat_model = CatBoostClassifier(
                iterations=180,
                learning_rate=0.03,
                depth=3,
                l2_leaf_reg=6.0,
                loss_function="Logloss",
                eval_metric="Logloss",
                random_seed=seed,
                verbose=False,
            )
            cat_model.fit(X_train[feature_cols], y, verbose=False)
            pred_cat[target] += cat_model.predict_proba(X_test[feature_cols])[:, 1] / len(SEEDS)

    final_pred = pd.DataFrame(index=test_base.index, columns=TARGETS, dtype=float)

    for target in TARGETS:
        p = 0.5 * pred_lgb[target].values + 0.5 * pred_cat[target].values
        p = shrink_proba(p, alpha=FIXED_ALPHA)
        p = clip_proba(p, 0.03, 0.97)
        final_pred[target] = p
        print(
            f"[INFO] {target}: mean={final_pred[target].mean():.4f}, "
            f"std={final_pred[target].std():.4f}, min={final_pred[target].min():.4f}, max={final_pred[target].max():.4f}"
        )

    submission = sub.copy()
    for target in TARGETS:
        submission[target] = final_pred[target].values

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_PATH, index=False)

    print("saved:", OUT_PATH)
    print(submission.head())


if __name__ == "__main__":
    main()
