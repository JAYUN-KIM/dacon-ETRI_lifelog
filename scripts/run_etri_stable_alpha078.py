import os
import gc
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

BASE_DIR = Path("/mnt/c/etri-lifelog/data/raw/data")
SENSOR_DIR = BASE_DIR / "ch2025_data_items"
TRAIN_PATH = BASE_DIR / "ch2026_metrics_train.csv"
SUB_PATH = BASE_DIR / "ch2026_submission_sample.csv"
OUT_DIR = BASE_DIR / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIXED_ALPHA = 0.78


def log(msg: str):
    print(f"[INFO] {msg}")


def get_sensor_path(keyword: str):
    files = sorted(SENSOR_DIR.glob("*.parquet"))
    for p in files:
        if keyword.lower() in p.name.lower():
            return p
    return None


def infer_test_frame_from_submission(train_df: pd.DataFrame, sub_df: pd.DataFrame) -> pd.DataFrame:
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

    next_dates = {
        sid: train_df.loc[train_df["subject_id"] == sid, "lifelog_date"].max() + pd.Timedelta(days=1)
        for sid in subs
    }
    rows = []
    for sid in subject_seq:
        rows.append((sid, next_dates[sid]))
        next_dates[sid] += pd.Timedelta(days=1)

    return pd.DataFrame(rows, columns=["subject_id", "lifelog_date"])


def prep_sensor_df(df: pd.DataFrame, value_cols):
    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["lifelog_date"] = df["timestamp"].dt.floor("D")
    df["hour"] = df["timestamp"].dt.hour
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    df["is_day"] = ((df["hour"] >= 9) & (df["hour"] <= 18)).astype(int)
    keep = ["subject_id", "lifelog_date", "timestamp", "hour", "is_night", "is_day"] + [c for c in value_cols if c in df.columns]
    return df[keep].copy()


def add_simple_agg(df: pd.DataFrame, val_col: str, prefix: str) -> pd.DataFrame:
    g = df.groupby(["subject_id", "lifelog_date"])[val_col]
    feat = g.agg(["mean", "std", "min", "max", "count"]).reset_index()
    feat.columns = ["subject_id", "lifelog_date"] + [f"{prefix}_{c}" for c in ["mean", "std", "min", "max", "count"]]

    night = df[df["is_night"] == 1].groupby(["subject_id", "lifelog_date"])[val_col].agg(["mean", "std"]).reset_index()
    night.columns = ["subject_id", "lifelog_date", f"{prefix}_night_mean", f"{prefix}_night_std"]

    day = df[df["is_day"] == 1].groupby(["subject_id", "lifelog_date"])[val_col].agg(["mean", "std"]).reset_index()
    day.columns = ["subject_id", "lifelog_date", f"{prefix}_day_mean", f"{prefix}_day_std"]

    out = feat.merge(night, on=["subject_id", "lifelog_date"], how="left")
    out = out.merge(day, on=["subject_id", "lifelog_date"], how="left")
    return out


def explode_hr_array(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return []
        return pd.to_numeric(pd.Series(x.ravel()), errors="coerce").dropna().tolist()
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return []
        return pd.to_numeric(pd.Series(list(x)), errors="coerce").dropna().tolist()
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
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
        if pd.isna(x):
            return []
    except Exception:
        pass
    try:
        return [float(x)]
    except Exception:
        return []


def build_simple_sensor_features(sensor_map):
    tables = []

    mapping = [
        ("activity", "m_activity", "m_activity"),
        ("light", "m_light", "m_light"),
        ("screen", "m_screen_use", "m_screen_use"),
        ("charge", "m_charging", "m_charging"),
    ]
    for key, raw_col, prefix in mapping:
        p = sensor_map.get(key)
        if p is None:
            continue
        log(f"loading {p.name}")
        df = pd.read_parquet(p)
        if raw_col not in df.columns:
            continue
        df = prep_sensor_df(df, [raw_col])
        tables.append(add_simple_agg(df, raw_col, prefix))
        del df
        gc.collect()

    p = sensor_map.get("hr")
    if p is not None:
        log(f"loading {p.name}")
        df = pd.read_parquet(p)
        candidate_cols = [c for c in df.columns if c not in ["subject_id", "timestamp"]]
        if candidate_cols:
            preferred = [c for c in candidate_cols if "heart" in c.lower() or "hr" in c.lower()]
            hr_col = preferred[0] if preferred else candidate_cols[0]
            log(f"selected HR column: {hr_col}")
            df = prep_sensor_df(df, [hr_col])
            arr = df[hr_col].apply(explode_hr_array)
            df["hr_mean_row"] = arr.apply(lambda v: float(np.mean(v)) if len(v) else np.nan)
            df["hr_std_row"] = arr.apply(lambda v: float(np.std(v)) if len(v) else np.nan)
            df["hr_min_row"] = arr.apply(lambda v: float(np.min(v)) if len(v) else np.nan)
            df["hr_max_row"] = arr.apply(lambda v: float(np.max(v)) if len(v) else np.nan)
            df["hr_median_row"] = arr.apply(lambda v: float(np.median(v)) if len(v) else np.nan)
            df["hr_q75_row"] = arr.apply(lambda v: float(np.quantile(v, 0.75)) if len(v) else np.nan)

            stats = df.groupby(["subject_id", "lifelog_date"])[[
                "hr_mean_row", "hr_std_row", "hr_min_row", "hr_max_row", "hr_median_row", "hr_q75_row"
            ]].mean().reset_index()
            stats.columns = [
                "subject_id", "lifelog_date",
                "heart_rate_mean", "heart_rate_std", "heart_rate_min",
                "heart_rate_max", "heart_rate_median", "heart_rate_q75"
            ]

            sleep = df[df["is_night"] == 1].groupby(["subject_id", "lifelog_date"])["hr_mean_row"].agg(["mean", "std"]).reset_index()
            sleep.columns = ["subject_id", "lifelog_date", "heart_rate_sleep_mean", "heart_rate_sleep_std"]

            active = df[df["is_day"] == 1].groupby(["subject_id", "lifelog_date"])["hr_mean_row"].agg(["mean", "std"]).reset_index()
            active.columns = ["subject_id", "lifelog_date", "heart_rate_active_mean", "heart_rate_active_std"]

            out = stats.merge(sleep, on=["subject_id", "lifelog_date"], how="left")
            out = out.merge(active, on=["subject_id", "lifelog_date"], how="left")
            out["heart_rate_sleep_active_diff"] = out["heart_rate_sleep_mean"] - out["heart_rate_active_mean"]
            tables.append(out)
            del df
            gc.collect()

    p = sensor_map.get("pedo")
    if p is not None:
        log(f"loading {p.name}")
        df = pd.read_parquet(p)
        value_cols = [c for c in ["step", "distance", "speed", "calories", "running"] if c in df.columns]
        if value_cols:
            df = prep_sensor_df(df, value_cols)
            agg_dict = {}
            if "step" in df.columns:
                agg_dict["step"] = ["sum", "mean"]
            if "distance" in df.columns:
                agg_dict["distance"] = ["sum", "mean"]
            if "speed" in df.columns:
                agg_dict["speed"] = ["mean", "max"]
            if "calories" in df.columns:
                agg_dict["calories"] = ["sum", "mean"]
            if "running" in df.columns:
                agg_dict["running"] = ["sum", "mean"]

            pedo = df.groupby(["subject_id", "lifelog_date"]).agg(agg_dict).reset_index()
            pedo.columns = ["subject_id", "lifelog_date"] + [f"{a}_{b}" for a, b in pedo.columns.tolist()[2:]]
            tables.append(pedo)
            del df
            gc.collect()

    return tables


def add_prior_features(df: pd.DataFrame, cols):
    df = df.sort_values(["subject_id", "lifelog_date"]).copy()
    for col in cols:
        grp = df.groupby("subject_id")[col]
        prior_mean = grp.expanding().mean().shift(1).reset_index(level=0, drop=True)
        prior_std = grp.expanding().std().shift(1).reset_index(level=0, drop=True)
        prior_cnt = df.groupby("subject_id").cumcount()
        df[f"{col}_prior_mean"] = prior_mean
        df[f"{col}_prior_std"] = prior_std
        df[f"{col}_prior_cnt"] = prior_cnt
        df[f"{col}_dev"] = df[col] - df[f"{col}_prior_mean"]
    return df


def build_global_time_split(df: pd.DataFrame, valid_ratio=0.2):
    uniq_dates = np.sort(df["lifelog_date"].unique())
    n_valid = max(1, int(len(uniq_dates) * valid_ratio))
    valid_dates = set(uniq_dates[-n_valid:])
    tr_idx = df.index[~df["lifelog_date"].isin(valid_dates)].tolist()
    va_idx = df.index[df["lifelog_date"].isin(valid_dates)].tolist()
    return tr_idx, va_idx


def avg_logloss(y_true_df: pd.DataFrame, pred_df: pd.DataFrame):
    scores = []
    for t in TARGETS:
        y_true = y_true_df[t].astype(int).values
        y_pred = np.clip(pred_df[t].values, 1e-6, 1 - 1e-6)
        scores.append(log_loss(y_true, y_pred))
    return float(np.mean(scores)), scores


def shrink_proba(p, alpha=0.8):
    return np.clip(0.5 + alpha * (p - 0.5), 1e-6, 1 - 1e-6)


def clip_proba(p, lo=0.03, hi=0.97):
    return np.clip(p, lo, hi)


def fit_predict_split(X_train: pd.DataFrame, y_df: pd.DataFrame, X_test: pd.DataFrame, tr_idx, va_idx, features):
    oof_lgb = pd.DataFrame(index=va_idx, columns=TARGETS, dtype=float)
    oof_cat = pd.DataFrame(index=va_idx, columns=TARGETS, dtype=float)
    test_lgb = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)
    test_cat = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)

    X_tr = X_train.loc[tr_idx, features]
    X_va = X_train.loc[va_idx, features]
    X_te = X_test[features]

    for t in TARGETS:
        log(f"training split model for {t}")
        y_tr = y_df.loc[tr_idx, t].astype(int)
        y_va = y_df.loc[va_idx, t].astype(int)

        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=3.0,
            objective="binary",
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        cat_model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.03,
            depth=4,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=42,
            verbose=False
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

        oof_lgb.loc[va_idx, t] = lgb_model.predict_proba(X_va)[:, 1]
        oof_cat.loc[va_idx, t] = cat_model.predict_proba(X_va)[:, 1]
        test_lgb[t] = lgb_model.predict_proba(X_te)[:, 1]
        test_cat[t] = cat_model.predict_proba(X_te)[:, 1]

    return oof_lgb, oof_cat, test_lgb, test_cat


def train_full_models(X_train: pd.DataFrame, y_df: pd.DataFrame, X_test: pd.DataFrame, features):
    final_pred_lgb = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)
    final_pred_cat = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)

    for t in TARGETS:
        log(f"training full model for {t}")
        y = y_df[t].astype(int)

        lgb_model = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=3.0,
            objective="binary",
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        lgb_model.fit(X_train[features], y)
        final_pred_lgb[t] = lgb_model.predict_proba(X_test[features])[:, 1]

        cat_model = CatBoostClassifier(
            iterations=220,
            learning_rate=0.03,
            depth=4,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=42,
            verbose=False
        )
        cat_model.fit(X_train[features], y, verbose=False)
        final_pred_cat[t] = cat_model.predict_proba(X_test[features])[:, 1]

    return final_pred_lgb, final_pred_cat


def main():
    log("loading train/submission files")
    train = pd.read_csv(TRAIN_PATH)
    sub = pd.read_csv(SUB_PATH)

    train["subject_id"] = train["subject_id"].astype(str)
    train["lifelog_date"] = pd.to_datetime(train["lifelog_date"])
    if "subject_id" in sub.columns:
        sub["subject_id"] = sub["subject_id"].astype(str)
    if "lifelog_date" in sub.columns:
        sub["lifelog_date"] = pd.to_datetime(sub["lifelog_date"])

    test = infer_test_frame_from_submission(train, sub)

    base_df = pd.concat([
        train[["subject_id", "lifelog_date"] + TARGETS].copy(),
        test[["subject_id", "lifelog_date"]].copy()
    ], axis=0, ignore_index=True).sort_values(["subject_id", "lifelog_date"]).reset_index(drop=True)

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
    log("sensor map:")
    for k, v in sensor_map.items():
        log(f"  {k}: {None if v is None else v.name}")

    feature_tables = build_simple_sensor_features(sensor_map)

    feat = base_df.copy()
    for ft in feature_tables:
        feat = feat.merge(ft, on=["subject_id", "lifelog_date"], how="left")

    stable_candidates = [c for c in feat.columns if c not in ["subject_id", "lifelog_date"] + TARGETS]
    core_personal_cols = [
        c for c in stable_candidates if c in [
            "heart_rate_mean", "heart_rate_std", "heart_rate_sleep_mean", "heart_rate_active_mean",
            "step_sum", "distance_sum", "speed_mean", "m_screen_use_mean", "m_light_mean", "m_activity_mean"
        ]
    ]
    log(f"stable features: {len(stable_candidates)}")
    log(f"core personalization cols: {core_personal_cols}")

    feat = add_prior_features(feat, core_personal_cols)

    train_mask = feat[TARGETS[0]].notnull()
    global_means = feat.loc[train_mask, stable_candidates].mean(numeric_only=True)
    global_stds = feat.loc[train_mask, stable_candidates].std(numeric_only=True)

    for col in core_personal_cols:
        feat[f"{col}_prior_mean"] = feat[f"{col}_prior_mean"].fillna(global_means.get(col, 0.0))
        feat[f"{col}_prior_std"] = feat[f"{col}_prior_std"].fillna(global_stds.get(col, 1.0))
        feat[f"{col}_dev"] = feat[f"{col}_dev"].fillna(0.0)

    final_features = stable_candidates + sum([
        [f"{c}_prior_mean", f"{c}_prior_std", f"{c}_prior_cnt", f"{c}_dev"] for c in core_personal_cols
    ], [])
    final_features = [c for c in final_features if c in feat.columns]

    train_feat = feat[feat[TARGETS[0]].notnull()].copy().reset_index(drop=True)
    test_feat = feat[feat[TARGETS[0]].isnull()].copy().reset_index(drop=True)

    X_train = train_feat[final_features].copy()
    X_test = test_feat[final_features].copy()
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_test = X_test.fillna(med)

    tr_idx, va_idx = build_global_time_split(train_feat, valid_ratio=0.2)
    log(f"train rows: {len(tr_idx)}, valid rows: {len(va_idx)}")

    oof_lgb, oof_cat, test_lgb, test_cat = fit_predict_split(
        X_train, train_feat[TARGETS], X_test, tr_idx, va_idx, final_features
    )
    pred_avg = 0.5 * oof_lgb + 0.5 * oof_cat

    score_lgb, each_lgb = avg_logloss(train_feat.loc[va_idx, TARGETS], oof_lgb)
    score_cat, each_cat = avg_logloss(train_feat.loc[va_idx, TARGETS], oof_cat)
    score_avg, each_avg = avg_logloss(train_feat.loc[va_idx, TARGETS], pred_avg)
    log(f"time split LGB avg_logloss: {score_lgb:.6f} | each: {each_lgb}")
    log(f"time split CAT avg_logloss: {score_cat:.6f} | each: {each_cat}")
    log(f"time split AVG avg_logloss: {score_avg:.6f} | each: {each_avg}")

    fixed_alpha_pred = pred_avg.copy()
    for t in TARGETS:
        fixed_alpha_pred[t] = clip_proba(shrink_proba(fixed_alpha_pred[t].values, alpha=FIXED_ALPHA), 0.03, 0.97)
    fixed_alpha_score, fixed_alpha_each = avg_logloss(train_feat.loc[va_idx, TARGETS], fixed_alpha_pred)
    best_alpha = FIXED_ALPHA
    log(f"fixed alpha mode enabled: alpha={best_alpha}")
    log(f"time split FIXED_ALPHA avg_logloss: {fixed_alpha_score:.6f} | each: {fixed_alpha_each}")

    best_target_model = {}
    for t in TARGETS:
        lgb_score = log_loss(train_feat.loc[va_idx, t].astype(int), np.clip(oof_lgb[t], 1e-6, 1 - 1e-6))
        cat_score = log_loss(train_feat.loc[va_idx, t].astype(int), np.clip(oof_cat[t], 1e-6, 1 - 1e-6))
        best_target_model[t] = "lgb" if lgb_score <= cat_score else "cat"
    log(f"best target model: {best_target_model}")

    final_pred_lgb, final_pred_cat = train_full_models(X_train, train_feat[TARGETS], X_test, final_features)

    final_pred = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)
    for t in TARGETS:
        p = final_pred_lgb[t].values if best_target_model[t] == "lgb" else final_pred_cat[t].values
        p = shrink_proba(p, alpha=best_alpha)
        p = clip_proba(p, 0.03, 0.97)
        final_pred[t] = p

    submission = sub.copy()
    for t in TARGETS:
        submission[t] = final_pred[t].values

    save_path = OUT_DIR / "sub_stable_alpha078.csv"
    submission.to_csv(save_path, index=False)
    log(f"saved submission: {save_path}")

    feature_dump_path = BASE_DIR / "artifacts_day_feature_table.csv"
    feat.to_csv(feature_dump_path, index=False)
    log(f"saved feature table: {feature_dump_path}")

    for t in TARGETS:
        log(
            f"{t}: mean={submission[t].mean():.4f}, std={submission[t].std():.4f}, "
            f"min={submission[t].min():.4f}, max={submission[t].max():.4f}"
        )


if __name__ == "__main__":
    main()
