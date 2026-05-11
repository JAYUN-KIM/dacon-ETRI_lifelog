import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss


warnings.filterwarnings("ignore")

ROOT_CANDIDATES = [Path("/mnt/c/etri-lifelog"), Path("C:/etri-lifelog")]
ROOT = next(p for p in ROOT_CANDIDATES if p.exists())
BASE_DIR = ROOT / "data" / "raw" / "data"
SENSOR_DIR = BASE_DIR / "ch2025_data_items"
SUB_DIR = BASE_DIR / "submissions"
TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
SAMPLE_PATH = BASE_DIR / "ch2026_submission_sample.csv"

# Numeric-best anchor from the 2026-05-10 pattern projection run.
ANCHOR_PATH = SUB_DIR / "sub_0510_patternproj_w015.csv"
FALLBACK_ANCHOR_PATH = SUB_DIR / "sub_0509_afterg030_targetscale_soft.csv"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SLEEP_TARGETS = ["Q1", "S1", "S2", "S3", "S4"]


WINDOWS = {
    # The old daily feature table used calendar days. For sleep labels, the
    # behavior window crossing midnight should be more causally aligned.
    "evening": (18, 24),       # lifelog_date 18:00-24:00
    "pre_sleep": (21, 27),     # lifelog_date 21:00-sleep_date 03:00
    "overnight": (24, 32),     # sleep_date 00:00-08:00
    "morning": (30, 36),       # sleep_date 06:00-12:00
    "full_sleepctx": (18, 36), # lifelog_date 18:00-sleep_date 12:00
}


SENSORS = [
    ("ch2025_mActivity.parquet", ["m_activity"], "mact"),
    ("ch2025_mLight.parquet", ["m_light"], "mlight"),
    ("ch2025_mScreenStatus.parquet", ["m_screen_use"], "screen"),
    ("ch2025_mACStatus.parquet", ["m_charging"], "charge"),
    ("ch2025_wHr.parquet", ["heart_rate"], "hr"),
    ("ch2025_wLight.parquet", ["w_light"], "wlight"),
    (
        "ch2025_wPedo.parquet",
        ["step", "step_frequency", "running_step", "walking_step", "distance", "speed", "burned_calories"],
        "pedo",
    ),
]


def make_meta(train, sample):
    train_meta = train[["subject_id", "sleep_date", "lifelog_date"] + TARGETS].copy()
    test_meta = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        test_meta[t] = np.nan
    meta = pd.concat([train_meta.assign(is_train=1), test_meta.assign(is_train=0)], ignore_index=True)
    meta["row_id"] = np.arange(len(meta))
    meta["subject_id"] = meta["subject_id"].astype(str)
    meta["lifelog_date"] = pd.to_datetime(meta["lifelog_date"])
    meta["sleep_date"] = pd.to_datetime(meta["sleep_date"])
    meta["dow"] = meta["lifelog_date"].dt.dayofweek
    meta["month"] = meta["lifelog_date"].dt.month
    meta["is_weekend"] = (meta["dow"] >= 5).astype(int)
    meta["subject_ord"] = meta.groupby("subject_id")["lifelog_date"].rank(method="first").astype(int)
    meta["subject_days_from_start"] = meta.groupby("subject_id")["lifelog_date"].transform(lambda s: (s - s.min()).dt.days)
    return meta


def aggregate_one_sensor(meta, file_name, value_cols, prefix):
    path = SENSOR_DIR / file_name
    df = pd.read_parquet(path)
    df["subject_id"] = df["subject_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cols = [c for c in value_cols if c in df.columns]
    if not cols:
        return pd.DataFrame({"row_id": meta["row_id"]})

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["subject_id", "timestamp"] + cols].dropna(how="all", subset=cols)

    pieces = []
    for sid, rows in meta.groupby("subject_id", sort=False):
        sdf = df[df["subject_id"] == sid].sort_values("timestamp")
        if sdf.empty:
            continue
        for window_name, (start_h, end_h) in WINDOWS.items():
            tmp_rows = []
            for row in rows.itertuples(index=False):
                start = row.lifelog_date + pd.Timedelta(hours=start_h)
                end = row.lifelog_date + pd.Timedelta(hours=end_h)
                part = sdf[(sdf["timestamp"] >= start) & (sdf["timestamp"] < end)]
                rec = {"row_id": row.row_id}
                rec[f"{prefix}_{window_name}_count"] = len(part)
                for c in cols:
                    vals = part[c].dropna()
                    base = f"{prefix}_{window_name}_{c}"
                    if len(vals):
                        rec[f"{base}_mean"] = float(vals.mean())
                        rec[f"{base}_std"] = float(vals.std(ddof=0))
                        rec[f"{base}_min"] = float(vals.min())
                        rec[f"{base}_max"] = float(vals.max())
                        rec[f"{base}_sum"] = float(vals.sum())
                        rec[f"{base}_q25"] = float(vals.quantile(0.25))
                        rec[f"{base}_q75"] = float(vals.quantile(0.75))
                    else:
                        rec[f"{base}_mean"] = np.nan
                        rec[f"{base}_std"] = np.nan
                        rec[f"{base}_min"] = np.nan
                        rec[f"{base}_max"] = np.nan
                        rec[f"{base}_sum"] = np.nan
                        rec[f"{base}_q25"] = np.nan
                        rec[f"{base}_q75"] = np.nan
                tmp_rows.append(rec)
            pieces.append(pd.DataFrame(tmp_rows))

    if not pieces:
        return pd.DataFrame({"row_id": meta["row_id"]})
    # Each piece contains one subject-window slice. Concatenate first and then
    # collapse by row_id so different window columns land on the same row.
    return pd.concat(pieces, ignore_index=True).groupby("row_id", as_index=False).first()


def add_cross_window_features(feat):
    # Behavioral contrasts are often safer than raw levels across subjects.
    pairs = [
        ("screen", "m_screen_use"),
        ("mlight", "m_light"),
        ("wlight", "w_light"),
        ("mact", "m_activity"),
        ("hr", "heart_rate"),
        ("pedo", "step"),
        ("charge", "m_charging"),
    ]
    for prefix, col in pairs:
        eve = f"{prefix}_evening_{col}_mean"
        night = f"{prefix}_overnight_{col}_mean"
        pre = f"{prefix}_pre_sleep_{col}_mean"
        morn = f"{prefix}_morning_{col}_mean"
        if eve in feat and night in feat:
            feat[f"{prefix}_night_minus_evening"] = feat[night] - feat[eve]
        if pre in feat and morn in feat:
            feat[f"{prefix}_morning_minus_presleep"] = feat[morn] - feat[pre]
        if pre in feat and night in feat:
            feat[f"{prefix}_presleep_minus_overnight"] = feat[pre] - feat[night]
    return feat


def build_sleep_window_feature_table(train, sample):
    meta = make_meta(train, sample)
    feat = meta.copy()
    for file_name, cols, prefix in SENSORS:
        print(f"[feature] {file_name}")
        sfeat = aggregate_one_sensor(meta, file_name, cols, prefix)
        feat = feat.merge(sfeat, on="row_id", how="left")
    feat = add_cross_window_features(feat)
    return feat


def temporal_holdout_mask(train_feat):
    mask = pd.Series(False, index=train_feat.index)
    for _, idx in train_feat.groupby("subject_id").groups.items():
        ordered = train_feat.loc[list(idx)].sort_values("lifelog_date").index.to_list()
        n_valid = max(3, int(round(len(ordered) * 0.22)))
        mask.loc[ordered[-n_valid:]] = True
    return mask


def fit_sleep_window_model(feat):
    train_feat = feat[feat["is_train"] == 1].copy().reset_index(drop=True)
    test_feat = feat[feat["is_train"] == 0].copy().reset_index(drop=True)
    valid_mask = temporal_holdout_mask(train_feat)

    drop_cols = ["row_id", "sleep_date", "lifelog_date", "is_train"] + TARGETS
    feature_cols = [c for c in feat.columns if c not in drop_cols]
    cat_cols = ["subject_id"]
    cat_idx = [feature_cols.index(c) for c in cat_cols]

    X = train_feat[feature_cols].copy()
    X_test = test_feat[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)
    numeric_cols = [c for c in feature_cols if c not in cat_cols]
    med = X[numeric_cols].median(numeric_only=True)
    X[numeric_cols] = X[numeric_cols].fillna(med).fillna(0.0)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(med).fillna(0.0)

    pred = test_feat[["subject_id", "sleep_date", "lifelog_date"]].copy()
    cv_rows = []
    for target in TARGETS:
        y = train_feat[target].astype(int)
        mean = float(y.mean())
        model_params = dict(
            loss_function="Logloss",
            iterations=420,
            depth=2,
            learning_rate=0.025,
            l2_leaf_reg=28,
            random_seed=20260511,
            verbose=False,
            allow_writing_files=False,
        )
        val_model = CatBoostClassifier(**model_params)
        val_model.fit(X.loc[~valid_mask], y.loc[~valid_mask], cat_features=cat_idx)
        vp = val_model.predict_proba(X.loc[valid_mask])[:, 1]
        vp = np.clip(0.82 * vp + 0.18 * mean, 0.04, 0.96)
        cv_rows.append({"target": target, "temporal_logloss": log_loss(y.loc[valid_mask], vp, labels=[0, 1])})

        full_model = CatBoostClassifier(**{**model_params, "iterations": 520})
        full_model.fit(X, y, cat_features=cat_idx)
        tp = full_model.predict_proba(X_test)[:, 1]
        pred[target] = np.clip(0.82 * tp + 0.18 * mean, 0.04, 0.96)

    cv = pd.DataFrame(cv_rows)
    print("\nSleep-window temporal holdout logloss:")
    print(cv.to_string(index=False))
    print("mean:", cv["temporal_logloss"].mean())
    return pred, cv


def save_blend(name, anchor, window_pred, weights):
    out = anchor.copy()
    if isinstance(weights, dict):
        for t in TARGETS:
            w = float(weights.get(t, 0.0))
            out[t] = np.clip((1 - w) * anchor[t].to_numpy(float) + w * window_pred[t].to_numpy(float), 0.04, 0.96)
    else:
        w = float(weights)
        out[TARGETS] = np.clip((1 - w) * anchor[TARGETS].to_numpy(float) + w * window_pred[TARGETS].to_numpy(float), 0.04, 0.96)
    path = SUB_DIR / name
    out.to_csv(path, index=False)
    diff = (out[TARGETS] - anchor[TARGETS]).abs()
    return {
        "candidate": name,
        "mean_abs_diff": float(diff.to_numpy().mean()),
        "max_abs_diff": float(diff.to_numpy().max()),
        "mean_q": float(out[["Q1", "Q2", "Q3"]].to_numpy().mean()),
        "mean_s": float(out[["S1", "S2", "S3", "S4"]].to_numpy().mean()),
    }


if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    anchor_path = ANCHOR_PATH if ANCHOR_PATH.exists() else FALLBACK_ANCHOR_PATH
    anchor = pd.read_csv(anchor_path)
    print("anchor:", anchor_path.name)

    feat = build_sleep_window_feature_table(train, sample)
    feat_path = BASE_DIR / "artifacts" / "sleep_window_features_20260511.csv"
    feat.to_csv(feat_path, index=False)
    print("feature table:", feat.shape, feat_path)

    window_pred, cv = fit_sleep_window_model(feat)
    model_path = SUB_DIR / "sub_0511_sleep_window_model.csv"
    window_pred.to_csv(model_path, index=False)

    rows = []
    rows.append(save_blend("sub_0511_sleepwin_w006.csv", anchor, window_pred, 0.006))
    rows.append(save_blend("sub_0511_sleepwin_w012.csv", anchor, window_pred, 0.012))
    rows.append(save_blend("sub_0511_sleepwin_sleep_targets.csv", anchor, window_pred, {
        "Q1": 0.014, "Q2": 0.004, "Q3": 0.004, "S1": 0.016, "S2": 0.014, "S3": 0.016, "S4": 0.014,
    }))

    summary = pd.DataFrame(rows).sort_values("mean_abs_diff").reset_index(drop=True)
    print("\nSleep-window candidate summary vs anchor:")
    print(summary.to_string(index=False))
    print("\nSuggested submit order:")
    print("1) sub_0511_sleepwin_w006.csv")
    print("2) sub_0511_sleepwin_sleep_targets.csv")
    print("3) sub_0511_sleepwin_w012.csv")
