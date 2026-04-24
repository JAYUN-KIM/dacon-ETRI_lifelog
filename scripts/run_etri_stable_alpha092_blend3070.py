import math, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path("/mnt/c/etri-lifelog/data/raw/data")
SENSOR_DIR = BASE_DIR / "ch2025_data_items"
TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
SUB_PATH = BASE_DIR / "ch2026_submission_sample.csv"
TARGETS = ["Q1","Q2","Q3","S1","S2","S3","S4"]

FIXED_ALPHA = 0.92
LGB_W = 0.30
CAT_W = 0.70
OUT_PATH = BASE_DIR / "submissions" / "sub_stable_alpha092_blend3070.csv"

def infer_test_frame_from_submission(train_df, sub_df):
    if {"subject_id", "lifelog_date"}.issubset(sub_df.columns):
        test_df = sub_df[["subject_id", "lifelog_date"]].copy()
        test_df["subject_id"] = test_df["subject_id"].astype(str)
        test_df["lifelog_date"] = pd.to_datetime(test_df["lifelog_date"])
        return test_df
    n_test = len(sub_df)
    last_dates = train_df.groupby("subject_id")["lifelog_date"].max().sort_values()
    subs = list(last_dates.index)
    reps = math.ceil(n_test / len(subs))
    subject_seq = (subs * reps)[:n_test]
    next_dates = {sid: train_df.loc[train_df["subject_id"] == sid, "lifelog_date"].max() + pd.Timedelta(days=1) for sid in subs}
    rows = []
    for sid in subject_seq:
        rows.append((sid, next_dates[sid]))
        next_dates[sid] += pd.Timedelta(days=1)
    return pd.DataFrame(rows, columns=["subject_id", "lifelog_date"])

def get_sensor_path(keyword):
    files = sorted(SENSOR_DIR.glob("*.parquet"))
    for p in files:
        if keyword.lower() in p.name.lower():
            return p
    return None

def prep_sensor_df(df, value_cols):
    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["lifelog_date"] = df["timestamp"].dt.floor("D")
    df["hour"] = df["timestamp"].dt.hour
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    df["is_day"] = ((df["hour"] >= 9) & (df["hour"] <= 18)).astype(int)
    keep = ["subject_id","lifelog_date","timestamp","hour","is_night","is_day"] + [c for c in value_cols if c in df.columns]
    return df[keep].copy()

def add_simple_agg(df, val_col, prefix):
    g = df.groupby(["subject_id","lifelog_date"])[val_col]
    feat = g.agg(["mean","std","min","max","count"]).reset_index()
    feat.columns = ["subject_id","lifelog_date"] + [f"{prefix}_{c}" for c in ["mean","std","min","max","count"]]
    night = df[df["is_night"] == 1].groupby(["subject_id","lifelog_date"])[val_col].agg(["mean","std"]).reset_index()
    night.columns = ["subject_id","lifelog_date",f"{prefix}_night_mean",f"{prefix}_night_std"]
    day = df[df["is_day"] == 1].groupby(["subject_id","lifelog_date"])[val_col].agg(["mean","std"]).reset_index()
    day.columns = ["subject_id","lifelog_date",f"{prefix}_day_mean",f"{prefix}_day_std"]
    out = feat.merge(night, on=["subject_id","lifelog_date"], how="left")
    out = out.merge(day, on=["subject_id","lifelog_date"], how="left")
    return out

def explode_hr_array(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        if x.size == 0: return []
        return pd.to_numeric(pd.Series(x.ravel()), errors="coerce").dropna().tolist()
    if isinstance(x, (list, tuple)):
        if len(x) == 0: return []
        return pd.to_numeric(pd.Series(list(x)), errors="coerce").dropna().tolist()
    if isinstance(x, str):
        s = x.strip()
        if s == "": return []
        try:
            v = json.loads(s)
            if isinstance(v, np.ndarray):
                return pd.to_numeric(pd.Series(v.ravel()), errors="coerce").dropna().tolist()
            if isinstance(v, (list, tuple)):
                return pd.to_numeric(pd.Series(list(v)), errors="coerce").dropna().tolist()
            return []
        except Exception:
            return []
    try:
        if pd.isna(x): return []
    except Exception:
        pass
    try:
        return [float(x)]
    except Exception:
        return []

def build_activity_features(sensor_map):
    p = sensor_map["activity"]
    if p is None: return None
    df = pd.read_parquet(p)
    df = prep_sensor_df(df, ["m_activity"])
    return add_simple_agg(df, "m_activity", "m_activity")

def build_light_features(sensor_map):
    p = sensor_map["light"]
    if p is None: return None
    df = pd.read_parquet(p)
    df = prep_sensor_df(df, ["m_light"])
    return add_simple_agg(df, "m_light", "m_light")

def build_screen_features(sensor_map):
    p = sensor_map["screen"]
    if p is None: return None
    df = pd.read_parquet(p)
    df = prep_sensor_df(df, ["m_screen_use"])
    return add_simple_agg(df, "m_screen_use", "m_screen_use")

def build_charge_features(sensor_map):
    p = sensor_map["charge"]
    if p is None: return None
    df = pd.read_parquet(p)
    df = prep_sensor_df(df, ["m_charging"])
    return add_simple_agg(df, "m_charging", "m_charging")

def build_hr_features(sensor_map):
    p = sensor_map["hr"]
    if p is None: return None
    df = pd.read_parquet(p)
    candidate_cols = [c for c in df.columns if c not in ["subject_id","timestamp"]]
    preferred = [c for c in candidate_cols if "heart" in c.lower() or "hr" in c.lower()]
    hr_col = preferred[0] if preferred else candidate_cols[0]
    print(f"[INFO] HR selected column: {hr_col}")
    df = prep_sensor_df(df, [hr_col])
    arr = df[hr_col].apply(explode_hr_array)
    df["hr_mean_row"] = arr.apply(lambda v: float(np.mean(v)) if len(v) else np.nan)
    df["hr_std_row"] = arr.apply(lambda v: float(np.std(v)) if len(v) else np.nan)
    df["hr_min_row"] = arr.apply(lambda v: float(np.min(v)) if len(v) else np.nan)
    df["hr_max_row"] = arr.apply(lambda v: float(np.max(v)) if len(v) else np.nan)
    df["hr_median_row"] = arr.apply(lambda v: float(np.median(v)) if len(v) else np.nan)
    df["hr_q75_row"] = arr.apply(lambda v: float(np.quantile(v, 0.75)) if len(v) else np.nan)
    stats = df.groupby(["subject_id","lifelog_date"])[["hr_mean_row","hr_std_row","hr_min_row","hr_max_row","hr_median_row","hr_q75_row"]].mean().reset_index()
    stats.columns = ["subject_id","lifelog_date","heart_rate_mean","heart_rate_std","heart_rate_min","heart_rate_max","heart_rate_median","heart_rate_q75"]
    sleep = df[df["is_night"] == 1].groupby(["subject_id","lifelog_date"])["hr_mean_row"].agg(["mean","std"]).reset_index()
    sleep.columns = ["subject_id","lifelog_date","heart_rate_sleep_mean","heart_rate_sleep_std"]
    active = df[df["is_day"] == 1].groupby(["subject_id","lifelog_date"])["hr_mean_row"].agg(["mean","std"]).reset_index()
    active.columns = ["subject_id","lifelog_date","heart_rate_active_mean","heart_rate_active_std"]
    out = stats.merge(sleep, on=["subject_id","lifelog_date"], how="left")
    out = out.merge(active, on=["subject_id","lifelog_date"], how="left")
    out["heart_rate_sleep_active_diff"] = out["heart_rate_sleep_mean"] - out["heart_rate_active_mean"]
    return out

def build_pedo_features(sensor_map):
    p = sensor_map["pedo"]
    if p is None: return None
    df = pd.read_parquet(p)
    value_cols = [c for c in ["step","distance","speed","calories","running"] if c in df.columns]
    df = prep_sensor_df(df, value_cols)
    agg_dict = {}
    if "step" in df.columns: agg_dict["step"] = ["sum","mean"]
    if "distance" in df.columns: agg_dict["distance"] = ["sum","mean"]
    if "speed" in df.columns: agg_dict["speed"] = ["mean","max"]
    if "calories" in df.columns: agg_dict["calories"] = ["sum","mean"]
    if "running" in df.columns: agg_dict["running"] = ["sum","mean"]
    feat = df.groupby(["subject_id","lifelog_date"]).agg(agg_dict).reset_index()
    feat.columns = ["subject_id","lifelog_date"] + [f"{a}_{b}" for a, b in feat.columns.tolist()[2:]]
    return feat

def add_prior_features(df, cols):
    df = df.sort_values(["subject_id","lifelog_date"]).copy()
    for col in cols:
        grp = df.groupby("subject_id")[col]
        df[f"{col}_prior_mean"] = grp.expanding().mean().shift(1).reset_index(level=0, drop=True)
        df[f"{col}_prior_std"] = grp.expanding().std().shift(1).reset_index(level=0, drop=True)
        df[f"{col}_prior_cnt"] = df.groupby("subject_id").cumcount()
        df[f"{col}_dev"] = df[col] - df[f"{col}_prior_mean"]
    return df

def shrink_proba(p, alpha=0.92):
    return np.clip(0.5 + alpha * (p - 0.5), 1e-6, 1 - 1e-6)

def clip_proba(p, lo=0.03, hi=0.97):
    return np.clip(p, lo, hi)

def main():
    train = pd.read_csv(TRAIN_PATH)
    sub = pd.read_csv(SUB_PATH)
    train["subject_id"] = train["subject_id"].astype(str)
    train["lifelog_date"] = pd.to_datetime(train["lifelog_date"])
    if "subject_id" in sub.columns: sub["subject_id"] = sub["subject_id"].astype(str)
    if "lifelog_date" in sub.columns: sub["lifelog_date"] = pd.to_datetime(sub["lifelog_date"])
    test = infer_test_frame_from_submission(train, sub)

    base_df = pd.concat([train[["subject_id","lifelog_date"] + TARGETS].copy(), test[["subject_id","lifelog_date"]].copy()], axis=0, ignore_index=True).sort_values(["subject_id","lifelog_date"]).reset_index(drop=True)
    base_df["dow"] = base_df["lifelog_date"].dt.dayofweek
    base_df["month"] = base_df["lifelog_date"].dt.month
    base_df["day"] = base_df["lifelog_date"].dt.day
    base_df["is_weekend"] = (base_df["dow"] >= 5).astype(int)
    base_df["days_from_global_start"] = (base_df["lifelog_date"] - base_df["lifelog_date"].min()).dt.days

    sensor_map = {
        "activity": get_sensor_path("mActivity"),
        "light": get_sensor_path("mLight"),
        "screen": get_sensor_path("mScreenStatus"),
        "hr": get_sensor_path("wHr"),
        "pedo": get_sensor_path("wPedo"),
        "charge": get_sensor_path("mACStatus"),
    }

    feat = base_df.copy()
    for ft in [build_activity_features(sensor_map), build_light_features(sensor_map), build_screen_features(sensor_map), build_charge_features(sensor_map), build_hr_features(sensor_map), build_pedo_features(sensor_map)]:
        if ft is not None:
            feat = feat.merge(ft, on=["subject_id","lifelog_date"], how="left")

    stable_candidates = [c for c in feat.columns if c not in ["subject_id","lifelog_date"] + TARGETS]
    core_personal_cols = [c for c in stable_candidates if c in ["m_activity_mean","m_light_mean","m_screen_use_mean","heart_rate_mean","heart_rate_std","heart_rate_sleep_mean","heart_rate_active_mean","step_sum","distance_sum","speed_mean"]]

    feat = add_prior_features(feat, core_personal_cols)
    train_mask = feat[TARGETS[0]].notnull()
    global_means = feat.loc[train_mask, stable_candidates].mean(numeric_only=True)
    global_stds = feat.loc[train_mask, stable_candidates].std(numeric_only=True)

    for col in core_personal_cols:
        feat[f"{col}_prior_mean"] = feat[f"{col}_prior_mean"].fillna(global_means.get(col, 0.0))
        feat[f"{col}_prior_std"] = feat[f"{col}_prior_std"].fillna(global_stds.get(col, 1.0))
        feat[f"{col}_dev"] = feat[f"{col}_dev"].fillna(0.0)

    final_features = stable_candidates + sum([[f"{c}_prior_mean", f"{c}_prior_std", f"{c}_prior_cnt", f"{c}_dev"] for c in core_personal_cols], [])
    final_features = [c for c in final_features if c in feat.columns]

    train_feat = feat[feat[TARGETS[0]].notnull()].copy().reset_index(drop=True)
    test_feat = feat[feat[TARGETS[0]].isnull()].copy().reset_index(drop=True)

    X_train = train_feat[final_features].copy()
    X_test = test_feat[final_features].copy()
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_test = X_test.fillna(med)

    final_pred_lgb = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)
    final_pred_cat = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)

    print("=" * 90)
    print("FULL TRAIN + TEST PREDICTION")
    print("=" * 90)

    for t in TARGETS:
        y = train_feat[t].astype(int)
        print(f"  > target={t}")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=220, learning_rate=0.03, num_leaves=15, max_depth=4,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=3.0, objective="binary",
            class_weight="balanced", random_state=42
        )
        lgb_model.fit(X_train[final_features], y)
        final_pred_lgb[t] = lgb_model.predict_proba(X_test[final_features])[:, 1]

        cat_model = CatBoostClassifier(
            iterations=220, learning_rate=0.03, depth=4,
            loss_function="Logloss", eval_metric="Logloss",
            random_seed=42, verbose=False
        )
        cat_model.fit(X_train[final_features], y, verbose=False)
        final_pred_cat[t] = cat_model.predict_proba(X_test[final_features])[:, 1]

    final_pred = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)
    for t in TARGETS:
        p = LGB_W * final_pred_lgb[t].values + CAT_W * final_pred_cat[t].values
        p = shrink_proba(p, alpha=FIXED_ALPHA)
        p = clip_proba(p, 0.03, 0.97)
        final_pred[t] = p
        print(f"[INFO] {t}: mean={final_pred[t].mean():.4f}, std={final_pred[t].std():.4f}, min={final_pred[t].min():.4f}, max={final_pred[t].max():.4f}")

    submission = sub.copy()
    for t in TARGETS:
        submission[t] = final_pred[t].values

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_PATH, index=False)
    print("saved:", OUT_PATH)
    print(submission.head())

if __name__ == "__main__":
    main()
