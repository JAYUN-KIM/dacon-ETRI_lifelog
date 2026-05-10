import ast
import math
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
ANCHOR_PATH = SUB_DIR / "sub_0510_patternproj_w015.csv"
FALLBACK_ANCHOR_PATH = SUB_DIR / "sub_0509_afterg030_targetscale_soft.csv"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def safe_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() == "nan":
            return []
        try:
            return ast.literal_eval(s)
        except Exception:
            return []
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass
    return []


def add_time_cols(df):
    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["lifelog_date"] = df["timestamp"].dt.floor("D")
    df["hour"] = df["timestamp"].dt.hour
    df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] <= 23)).astype(int)
    df["is_late"] = ((df["hour"] >= 21) & (df["hour"] <= 23)).astype(int)
    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 22)).astype(int)
    df["is_day"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)
    return df


def flatten_cols(frame, prefix):
    frame.columns = [
        "_".join([str(x) for x in col if str(x)])
        if isinstance(col, tuple)
        else str(col)
        for col in frame.columns
    ]
    return frame.rename(columns={"subject_id_": "subject_id", "lifelog_date_": "lifelog_date"})


def aggregate_numeric(df, value_cols, prefix):
    keys = ["subject_id", "lifelog_date"]
    out = df.groupby(keys)[value_cols].agg(["mean", "std", "min", "max", "sum"]).reset_index()
    out = flatten_cols(out, prefix)
    rename = {c: f"{prefix}_{c}" for c in out.columns if c not in keys}
    out = out.rename(columns=rename)

    for flag in ["is_evening", "is_late", "is_night"]:
        part = df[df[flag] == 1]
        if len(part) == 0:
            continue
        part_agg = part.groupby(keys)[value_cols].agg(["mean", "sum", "max"]).reset_index()
        part_agg = flatten_cols(part_agg, prefix)
        part_name = flag.replace("is_", "")
        part_agg = part_agg.rename(
            columns={c: f"{prefix}_{part_name}_{c}" for c in part_agg.columns if c not in keys}
        )
        out = out.merge(part_agg, on=keys, how="left")

    cnt = df.groupby(keys).size().reset_index(name=f"{prefix}_row_count")
    out = out.merge(cnt, on=keys, how="left")
    return out


def broad_ambience_scores(x):
    rows = safe_list(x)
    cats = {
        "speech": ["Speech", "Conversation", "Narration", "Child speech"],
        "music": ["Music", "Singing", "Musical"],
        "vehicle": ["Vehicle", "Car", "Bus", "Train", "Truck", "Motor vehicle", "Rail transport"],
        "outside": ["Outside", "Wind", "Rustling leaves"],
        "inside": ["Inside"],
        "animal": ["Animal", "Dog", "Cat", "Bird", "Insect"],
        "sleep_noise": ["Snoring", "Breathing"],
        "water": ["Water", "Liquid", "Steam", "Spray"],
        "door_object": ["Door", "Tap", "Dishes", "Cutlery", "Printer"],
    }
    scores = {f"amb_{k}": 0.0 for k in cats}
    top = 0.0
    entropy_terms = []
    for item in rows:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        label = str(item[0])
        try:
            score = float(item[1])
        except Exception:
            continue
        top = max(top, score)
        if score > 0:
            entropy_terms.append(-score * math.log(score + 1e-12))
        for name, needles in cats.items():
            if any(n.lower() in label.lower() for n in needles):
                scores[f"amb_{name}"] += score
    scores["amb_top_score"] = top
    scores["amb_entropy"] = float(sum(entropy_terms))
    scores["amb_label_count"] = len(rows)
    return scores


def build_ambience_features():
    df = pd.read_parquet(SENSOR_DIR / "ch2025_mAmbience.parquet")
    df = add_time_cols(df)
    score_df = pd.DataFrame(df["m_ambience"].apply(broad_ambience_scores).tolist())
    work = pd.concat([df[["subject_id", "lifelog_date", "hour", "is_evening", "is_late", "is_night"]], score_df], axis=1)
    return aggregate_numeric(work, list(score_df.columns), "amb")


def gps_row_summary(x):
    rows = [r for r in safe_list(x) if isinstance(r, dict)]
    if not rows:
        return {
            "gps_points": 0,
            "gps_speed_mean": np.nan,
            "gps_speed_max": np.nan,
            "gps_lat_range": np.nan,
            "gps_lon_range": np.nan,
            "gps_alt_range": np.nan,
        }
    speeds = pd.to_numeric(pd.Series([r.get("speed", np.nan) for r in rows]), errors="coerce")
    lats = pd.to_numeric(pd.Series([r.get("latitude", np.nan) for r in rows]), errors="coerce")
    lons = pd.to_numeric(pd.Series([r.get("longitude", np.nan) for r in rows]), errors="coerce")
    alts = pd.to_numeric(pd.Series([r.get("altitude", np.nan) for r in rows]), errors="coerce")
    return {
        "gps_points": len(rows),
        "gps_speed_mean": float(speeds.mean()),
        "gps_speed_max": float(speeds.max()),
        "gps_lat_range": float(lats.max() - lats.min()),
        "gps_lon_range": float(lons.max() - lons.min()),
        "gps_alt_range": float(alts.max() - alts.min()),
    }


def build_gps_features():
    df = pd.read_parquet(SENSOR_DIR / "ch2025_mGps.parquet")
    df = add_time_cols(df)
    score_df = pd.DataFrame(df["m_gps"].apply(gps_row_summary).tolist())
    work = pd.concat([df[["subject_id", "lifelog_date", "hour", "is_evening", "is_late", "is_night"]], score_df], axis=1)
    return aggregate_numeric(work, list(score_df.columns), "gps")


def usage_row_summary(x):
    rows = [r for r in safe_list(x) if isinstance(r, dict)]
    total = 0.0
    max_time = 0.0
    comm = media = browser = game = 0.0
    for r in rows:
        name = str(r.get("app_name", ""))
        try:
            t = float(r.get("total_time", 0.0))
        except Exception:
            t = 0.0
        total += t
        max_time = max(max_time, t)
        lname = name.lower()
        if any(k in name for k in ["카카오", "메시지", "통화", "전화"]) or any(k in lname for k in ["message", "call"]):
            comm += t
        if any(k in lname for k in ["youtube", "netflix", "music", "video"]) or any(k in name for k in ["유튜브", "음악"]):
            media += t
        if any(k in lname for k in ["naver", "chrome", "browser", "samsung internet"]) or any(k in name for k in ["네이버"]):
            browser += t
        if any(k in lname for k in ["game"]) or any(k in name for k in ["게임"]):
            game += t
    shares = np.array([comm, media, browser, game], dtype=float)
    shares = shares / (total + 1e-9)
    return {
        "usage_total": total,
        "usage_app_count": len(rows),
        "usage_max_app": max_time,
        "usage_comm_share": shares[0],
        "usage_media_share": shares[1],
        "usage_browser_share": shares[2],
        "usage_game_share": shares[3],
    }


def build_usage_features():
    df = pd.read_parquet(SENSOR_DIR / "ch2025_mUsageStats.parquet")
    df = add_time_cols(df)
    score_df = pd.DataFrame(df["m_usage_stats"].apply(usage_row_summary).tolist())
    work = pd.concat([df[["subject_id", "lifelog_date", "hour", "is_evening", "is_late", "is_night"]], score_df], axis=1)
    return aggregate_numeric(work, list(score_df.columns), "usage")


def radio_row_summary(x):
    rows = [r for r in safe_list(x) if isinstance(r, dict)]
    rssi = pd.to_numeric(pd.Series([r.get("rssi", np.nan) for r in rows]), errors="coerce")
    addresses = [str(r.get("address", "")) for r in rows if r.get("address", "")]
    return {
        "radio_count": len(rows),
        "radio_unique": len(set(addresses)),
        "radio_rssi_mean": float(rssi.mean()) if len(rssi) else np.nan,
        "radio_rssi_max": float(rssi.max()) if len(rssi) else np.nan,
        "radio_strong_count": float((rssi > -70).sum()) if len(rssi) else 0.0,
    }


def build_radio_features(file_name, value_col, prefix):
    df = pd.read_parquet(SENSOR_DIR / file_name)
    df = add_time_cols(df)
    score_df = pd.DataFrame(df[value_col].apply(radio_row_summary).tolist())
    score_df = score_df.rename(columns={c: c.replace("radio_", f"{prefix}_") for c in score_df.columns})
    work = pd.concat([df[["subject_id", "lifelog_date", "hour", "is_evening", "is_late", "is_night"]], score_df], axis=1)
    return aggregate_numeric(work, list(score_df.columns), prefix)


def build_numeric_sensor(file_name, value_cols, prefix):
    df = pd.read_parquet(SENSOR_DIR / file_name)
    df = add_time_cols(df)
    cols = [c for c in value_cols if c in df.columns]
    return aggregate_numeric(df, cols, prefix)


def build_feature_table(train, sample):
    meta = pd.concat(
        [
            train[["subject_id", "sleep_date", "lifelog_date"] + TARGETS].assign(is_train=1),
            sample[["subject_id", "sleep_date", "lifelog_date"]].assign(**{t: np.nan for t in TARGETS}, is_train=0),
        ],
        ignore_index=True,
    )
    meta["subject_id"] = meta["subject_id"].astype(str)
    meta["lifelog_date"] = pd.to_datetime(meta["lifelog_date"])
    meta["sleep_date"] = pd.to_datetime(meta["sleep_date"])
    meta["dow"] = meta["lifelog_date"].dt.dayofweek
    meta["month"] = meta["lifelog_date"].dt.month
    meta["day"] = meta["lifelog_date"].dt.day
    meta["is_weekend"] = (meta["dow"] >= 5).astype(int)
    meta["days_from_subject_start"] = meta.groupby("subject_id")["lifelog_date"].transform(lambda s: (s - s.min()).dt.days)

    features = [
        build_ambience_features(),
        build_gps_features(),
        build_usage_features(),
        build_radio_features("ch2025_mWifi.parquet", "m_wifi", "wifi"),
        build_radio_features("ch2025_mBle.parquet", "m_ble", "ble"),
        build_numeric_sensor("ch2025_wLight.parquet", ["w_light"], "wlight"),
        build_numeric_sensor(
            "ch2025_wPedo.parquet",
            ["step", "step_frequency", "running_step", "walking_step", "distance", "speed", "burned_calories"],
            "pedo2",
        ),
    ]

    out = meta
    for feat in features:
        feat["lifelog_date"] = pd.to_datetime(feat["lifelog_date"])
        out = out.merge(feat, on=["subject_id", "lifelog_date"], how="left")

    return out


def temporal_holdout_mask(train_df):
    mask = pd.Series(False, index=train_df.index)
    for _, idx in train_df.groupby("subject_id").groups.items():
        ordered = train_df.loc[list(idx)].sort_values("lifelog_date").index.to_list()
        n_valid = max(3, int(round(len(ordered) * 0.22)))
        mask.loc[ordered[-n_valid:]] = True
    return mask


def fit_raw_context_model(feat):
    train_feat = feat[feat["is_train"] == 1].copy().reset_index(drop=True)
    test_feat = feat[feat["is_train"] == 0].copy().reset_index(drop=True)
    valid_mask = temporal_holdout_mask(train_feat)

    drop_cols = ["sleep_date", "lifelog_date", "is_train"] + TARGETS
    feature_cols = [c for c in feat.columns if c not in drop_cols]
    cat_cols = ["subject_id"]
    cat_idx = [feature_cols.index(c) for c in cat_cols]

    X_all = train_feat[feature_cols].copy()
    X_test = test_feat[feature_cols].copy()
    for c in cat_cols:
        X_all[c] = X_all[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    numeric_cols = [c for c in feature_cols if c not in cat_cols]
    med = X_all[numeric_cols].median(numeric_only=True)
    X_all[numeric_cols] = X_all[numeric_cols].fillna(med).fillna(0.0)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(med).fillna(0.0)

    preds = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
    cv_rows = []

    for target in TARGETS:
        y = train_feat[target].astype(int)
        mean = float(y.mean())

        val_model = CatBoostClassifier(
            loss_function="Logloss",
            iterations=450,
            depth=2,
            learning_rate=0.025,
            l2_leaf_reg=20,
            random_seed=20260510,
            verbose=False,
            allow_writing_files=False,
        )
        val_model.fit(X_all.loc[~valid_mask], y.loc[~valid_mask], cat_features=cat_idx)
        val_pred = val_model.predict_proba(X_all.loc[valid_mask])[:, 1]
        val_pred = np.clip(0.88 * val_pred + 0.12 * mean, 0.04, 0.96)
        cv_rows.append({"target": target, "temporal_logloss": log_loss(y.loc[valid_mask], val_pred, labels=[0, 1])})

        full_model = CatBoostClassifier(
            loss_function="Logloss",
            iterations=520,
            depth=2,
            learning_rate=0.025,
            l2_leaf_reg=20,
            random_seed=20260510,
            verbose=False,
            allow_writing_files=False,
        )
        full_model.fit(X_all, y, cat_features=cat_idx)
        pred = full_model.predict_proba(X_test)[:, 1]
        preds[target] = np.clip(0.88 * pred + 0.12 * mean, 0.04, 0.96)

    cv = pd.DataFrame(cv_rows)
    print("\nRaw-context temporal holdout logloss:")
    print(cv.to_string(index=False))
    print("mean:", cv["temporal_logloss"].mean())
    return preds, cv


def save_blend(name, anchor, raw_pred, weights):
    out = anchor.copy()
    if isinstance(weights, dict):
        for t in TARGETS:
            w = weights.get(t, 0.0)
            out[t] = np.clip((1 - w) * anchor[t].to_numpy(float) + w * raw_pred[t].to_numpy(float), 0.04, 0.96)
    else:
        w = float(weights)
        out[TARGETS] = np.clip((1 - w) * anchor[TARGETS].to_numpy(float) + w * raw_pred[TARGETS].to_numpy(float), 0.04, 0.96)
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

    feat = build_feature_table(train, sample)
    feat_path = BASE_DIR / "artifacts" / "reset_raw_context_features_20260510.csv"
    feat.to_csv(feat_path, index=False)
    print("feature table:", feat.shape, feat_path)

    raw_pred, cv = fit_raw_context_model(feat)
    raw_path = SUB_DIR / "sub_0510_reset_raw_context_model.csv"
    raw_pred.to_csv(raw_path, index=False)
    print("raw model pred:", raw_path)

    rows = []
    rows.append(save_blend("sub_0510_reset_rawctx_w010.csv", anchor, raw_pred, 0.010))
    rows.append(save_blend("sub_0510_reset_rawctx_w020.csv", anchor, raw_pred, 0.020))
    rows.append(
        save_blend(
            "sub_0510_reset_rawctx_qheavy.csv",
            anchor,
            raw_pred,
            {"Q1": 0.025, "Q2": 0.025, "Q3": 0.025, "S1": 0.010, "S2": 0.010, "S3": 0.010, "S4": 0.010},
        )
    )
    summary = pd.DataFrame(rows).sort_values("mean_abs_diff").reset_index(drop=True)
    print("\nReset raw-context candidate summary vs anchor:")
    print(summary.to_string(index=False))
    print("\nSuggested only-if-last-submit order:")
    print("1) sub_0510_reset_rawctx_w010.csv")
    print("2) sub_0510_reset_rawctx_qheavy.csv")
    print("3) sub_0510_reset_rawctx_w020.csv")
