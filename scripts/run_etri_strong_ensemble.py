import gc
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss

warnings.filterwarnings('ignore')

BASE_DIR = Path('/mnt/c/etri-lifelog/data/raw/data')
SENSOR_DIR = BASE_DIR / 'ch2025_data_items'
TRAIN_PATH = BASE_DIR / 'ch2026_metrics_train.csv'
SUB_PATH = BASE_DIR / 'ch2026_submission_sample.csv'
OUT_DIR = BASE_DIR / 'submissions'
ART_DIR = BASE_DIR / 'artifacts'
OUT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
SEEDS = [13, 42, 77, 101]
LGB_WEIGHTS = [0.55, 0.45]
CAT_WEIGHTS = [0.45, 0.55]
ALPHA_GRID = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
CLIP_GRID = [(0.03, 0.97), (0.04, 0.96), (0.05, 0.95)]


def print_header(msg: str):
    print('\n' + '=' * 90)
    print(msg)
    print('=' * 90)


def clip_proba(p, lo=0.03, hi=0.97):
    return np.clip(np.asarray(p, dtype=float), lo, hi)


def shrink_proba(p, alpha=0.8):
    p = np.asarray(p, dtype=float)
    return clip_proba(0.5 + alpha * (p - 0.5), 1e-6, 1 - 1e-6)


def avg_logloss(y_true_df, pred_df):
    scores = []
    for t in TARGETS:
        y_true = y_true_df[t].astype(int).values
        y_pred = clip_proba(pred_df[t].values, 1e-6, 1 - 1e-6)
        scores.append(log_loss(y_true, y_pred))
    return float(np.mean(scores)), scores


def get_sensor_path(keyword):
    files = sorted(SENSOR_DIR.glob('*.parquet'))
    for p in files:
        if keyword.lower() in p.name.lower():
            return p
    return None


def infer_test_frame_from_submission(train_df, sub_df):
    # if sample already has id/date metadata, use it directly
    required = {'subject_id', 'lifelog_date'}
    if required.issubset(sub_df.columns):
        out = sub_df[['subject_id', 'lifelog_date']].copy()
        out['subject_id'] = out['subject_id'].astype(str)
        out['lifelog_date'] = pd.to_datetime(out['lifelog_date'])
        return out

    # fallback: infer a simple future frame by cycling subjects after their last observed date
    n_test = len(sub_df)
    last_dates = train_df.groupby('subject_id')['lifelog_date'].max().sort_values()
    subs = list(last_dates.index)
    reps = math.ceil(n_test / len(subs))
    subject_seq = (subs * reps)[:n_test]
    next_dates = {sid: last_dates[sid] + pd.Timedelta(days=1) for sid in subs}

    rows = []
    for sid in subject_seq:
        rows.append((sid, next_dates[sid]))
        next_dates[sid] += pd.Timedelta(days=1)

    return pd.DataFrame(rows, columns=['subject_id', 'lifelog_date'])


def prep_sensor_df(df, value_cols):
    df = df.copy()
    df['subject_id'] = df['subject_id'].astype(str)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['lifelog_date'] = df['timestamp'].dt.floor('D')
    df['hour'] = df['timestamp'].dt.hour
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    df['is_day'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)
    keep = ['subject_id', 'lifelog_date', 'timestamp', 'hour', 'is_night', 'is_day'] + [c for c in value_cols if c in df.columns]
    return df[keep].copy()


def add_simple_agg(df, val_col, prefix):
    g = df.groupby(['subject_id', 'lifelog_date'])[val_col]
    feat = g.agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    feat.columns = ['subject_id', 'lifelog_date'] + [f'{prefix}_{c}' for c in ['mean', 'std', 'min', 'max', 'count']]

    night = df[df['is_night'] == 1].groupby(['subject_id', 'lifelog_date'])[val_col].agg(['mean', 'std']).reset_index()
    night.columns = ['subject_id', 'lifelog_date', f'{prefix}_night_mean', f'{prefix}_night_std']

    day = df[df['is_day'] == 1].groupby(['subject_id', 'lifelog_date'])[val_col].agg(['mean', 'std']).reset_index()
    day.columns = ['subject_id', 'lifelog_date', f'{prefix}_day_mean', f'{prefix}_day_std']

    out = feat.merge(night, on=['subject_id', 'lifelog_date'], how='left')
    out = out.merge(day, on=['subject_id', 'lifelog_date'], how='left')
    return out


def explode_hr_array(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return []
        return pd.to_numeric(pd.Series(x.ravel()), errors='coerce').dropna().tolist()
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return []
        return pd.to_numeric(pd.Series(list(x)), errors='coerce').dropna().tolist()
    if isinstance(x, str):
        s = x.strip()
        if s == '':
            return []
        try:
            v = json.loads(s)
            if isinstance(v, np.ndarray):
                return pd.to_numeric(pd.Series(v.ravel()), errors='coerce').dropna().tolist()
            if isinstance(v, (list, tuple)):
                return pd.to_numeric(pd.Series(list(v)), errors='coerce').dropna().tolist()
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


def build_activity_features(sensor_map):
    p = sensor_map['activity']
    if p is None:
        return None
    df = pd.read_parquet(p)
    df = prep_sensor_df(df, ['m_activity'])
    return add_simple_agg(df, 'm_activity', 'm_activity')


def build_light_features(sensor_map):
    p = sensor_map['light']
    if p is None:
        return None
    df = pd.read_parquet(p)
    df = prep_sensor_df(df, ['m_light'])
    return add_simple_agg(df, 'm_light', 'm_light')


def build_screen_features(sensor_map):
    p = sensor_map['screen']
    if p is None:
        return None
    df = pd.read_parquet(p)
    df = prep_sensor_df(df, ['m_screen_use'])
    return add_simple_agg(df, 'm_screen_use', 'm_screen_use')


def build_charge_features(sensor_map):
    p = sensor_map['charge']
    if p is None:
        return None
    df = pd.read_parquet(p)
    df = prep_sensor_df(df, ['m_charging'])
    return add_simple_agg(df, 'm_charging', 'm_charging')


def build_hr_features(sensor_map):
    p = sensor_map['hr']
    if p is None:
        return None

    df = pd.read_parquet(p)
    candidate_cols = [c for c in df.columns if c not in ['subject_id', 'timestamp']]
    if len(candidate_cols) == 0:
        return None
    preferred = [c for c in candidate_cols if 'heart' in c.lower() or 'hr' in c.lower()]
    hr_col = preferred[0] if preferred else candidate_cols[0]
    print(f'[HR] selected column: {hr_col} / dtype: {df[hr_col].dtype}')

    df = prep_sensor_df(df, [hr_col])
    arr = df[hr_col].apply(explode_hr_array)

    df['hr_mean_row'] = arr.apply(lambda v: float(np.mean(v)) if len(v) else np.nan)
    df['hr_std_row'] = arr.apply(lambda v: float(np.std(v)) if len(v) else np.nan)
    df['hr_min_row'] = arr.apply(lambda v: float(np.min(v)) if len(v) else np.nan)
    df['hr_max_row'] = arr.apply(lambda v: float(np.max(v)) if len(v) else np.nan)
    df['hr_median_row'] = arr.apply(lambda v: float(np.median(v)) if len(v) else np.nan)
    df['hr_q75_row'] = arr.apply(lambda v: float(np.quantile(v, 0.75)) if len(v) else np.nan)

    stats = df.groupby(['subject_id', 'lifelog_date'])[['hr_mean_row', 'hr_std_row', 'hr_min_row', 'hr_max_row', 'hr_median_row', 'hr_q75_row']].mean().reset_index()
    stats.columns = [
        'subject_id', 'lifelog_date',
        'heart_rate_mean', 'heart_rate_std', 'heart_rate_min', 'heart_rate_max', 'heart_rate_median', 'heart_rate_q75'
    ]

    sleep = df[df['is_night'] == 1].groupby(['subject_id', 'lifelog_date'])['hr_mean_row'].agg(['mean', 'std']).reset_index()
    sleep.columns = ['subject_id', 'lifelog_date', 'heart_rate_sleep_mean', 'heart_rate_sleep_std']

    active = df[df['is_day'] == 1].groupby(['subject_id', 'lifelog_date'])['hr_mean_row'].agg(['mean', 'std']).reset_index()
    active.columns = ['subject_id', 'lifelog_date', 'heart_rate_active_mean', 'heart_rate_active_std']

    out = stats.merge(sleep, on=['subject_id', 'lifelog_date'], how='left')
    out = out.merge(active, on=['subject_id', 'lifelog_date'], how='left')
    out['heart_rate_sleep_active_diff'] = out['heart_rate_sleep_mean'] - out['heart_rate_active_mean']
    return out


def build_pedo_features(sensor_map):
    p = sensor_map['pedo']
    if p is None:
        return None

    df = pd.read_parquet(p)
    value_cols = [c for c in ['step', 'distance', 'speed', 'calories', 'running'] if c in df.columns]
    if not value_cols:
        return None
    df = prep_sensor_df(df, value_cols)

    agg_dict = {}
    if 'step' in df.columns:
        agg_dict['step'] = ['sum', 'mean', 'max']
    if 'distance' in df.columns:
        agg_dict['distance'] = ['sum', 'mean']
    if 'speed' in df.columns:
        agg_dict['speed'] = ['mean', 'max', 'std']
    if 'calories' in df.columns:
        agg_dict['calories'] = ['sum', 'mean']
    if 'running' in df.columns:
        agg_dict['running'] = ['sum', 'mean']

    feat = df.groupby(['subject_id', 'lifelog_date']).agg(agg_dict).reset_index()
    feat.columns = ['subject_id', 'lifelog_date'] + [f'{a}_{b}' for a, b in feat.columns.tolist()[2:]]
    return feat


def add_prior_features(df, cols):
    df = df.sort_values(['subject_id', 'lifelog_date']).copy()
    for col in cols:
        grp = df.groupby('subject_id')[col]
        prior_mean = grp.expanding().mean().shift(1).reset_index(level=0, drop=True)
        prior_std = grp.expanding().std().shift(1).reset_index(level=0, drop=True)
        prior_cnt = df.groupby('subject_id').cumcount()
        df[f'{col}_prior_mean'] = prior_mean
        df[f'{col}_prior_std'] = prior_std
        df[f'{col}_prior_cnt'] = prior_cnt
        df[f'{col}_dev'] = df[col] - df[f'{col}_prior_mean']
    return df


def add_recent_lag_features(df, cols):
    df = df.sort_values(['subject_id', 'lifelog_date']).copy()
    for col in cols:
        g = df.groupby('subject_id')[col]
        df[f'{col}_lag1'] = g.shift(1)
        df[f'{col}_lag3_mean'] = g.shift(1).rolling(3).mean().reset_index(level=0, drop=True)
        df[f'{col}_lag7_mean'] = g.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
        df[f'{col}_delta_lag1'] = df[col] - df[f'{col}_lag1']
        df[f'{col}_delta_lag3'] = df[col] - df[f'{col}_lag3_mean']
        df[f'{col}_delta_lag7'] = df[col] - df[f'{col}_lag7_mean']
    return df


def build_global_time_split(df, valid_ratio=0.2):
    uniq_dates = np.sort(df['lifelog_date'].unique())
    n_valid = max(1, int(len(uniq_dates) * valid_ratio))
    valid_dates = set(uniq_dates[-n_valid:])
    tr_idx = df.index[~df['lifelog_date'].isin(valid_dates)].tolist()
    va_idx = df.index[df['lifelog_date'].isin(valid_dates)].tolist()
    return tr_idx, va_idx, sorted(valid_dates)


def fit_predict_seeded(train_df, X_train, X_test, feature_cols, tr_idx, va_idx, seeds):
    model_names = []
    oof_map = {}
    test_map = {}

    for seed in seeds:
        model_names.extend([f'lgb_s{seed}', f'cat_s{seed}'])
        oof_map[f'lgb_s{seed}'] = pd.DataFrame(index=va_idx, columns=TARGETS, dtype=float)
        oof_map[f'cat_s{seed}'] = pd.DataFrame(index=va_idx, columns=TARGETS, dtype=float)
        test_map[f'lgb_s{seed}'] = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)
        test_map[f'cat_s{seed}'] = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)

    X_tr = X_train.loc[tr_idx, feature_cols]
    X_va = X_train.loc[va_idx, feature_cols]
    X_te = X_test[feature_cols]

    for t in TARGETS:
        y_tr = train_df.loc[tr_idx, t].astype(int)
        y_va = train_df.loc[va_idx, t].astype(int)
        print(f'  > target={t}')

        for seed in seeds:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=700,
                learning_rate=0.025,
                num_leaves=15,
                max_depth=4,
                min_child_samples=18,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=1.5,
                reg_lambda=4.0,
                objective='binary',
                class_weight='balanced',
                random_state=seed,
                verbosity=-1,
            )
            lgb_model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric='binary_logloss',
                callbacks=[lgb.early_stopping(80, verbose=False)]
            )
            oof_map[f'lgb_s{seed}'].loc[va_idx, t] = lgb_model.predict_proba(X_va)[:, 1]
            test_map[f'lgb_s{seed}'][t] = lgb_model.predict_proba(X_te)[:, 1]

            cat_model = CatBoostClassifier(
                iterations=700,
                learning_rate=0.025,
                depth=4,
                l2_leaf_reg=6.0,
                random_seed=seed,
                loss_function='Logloss',
                eval_metric='Logloss',
                verbose=False,
            )
            cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
            oof_map[f'cat_s{seed}'].loc[va_idx, t] = cat_model.predict_proba(X_va)[:, 1]
            test_map[f'cat_s{seed}'][t] = cat_model.predict_proba(X_te)[:, 1]

            del lgb_model, cat_model
            gc.collect()

    return oof_map, test_map


def mean_pred(frames):
    out = frames[0].copy().astype(float)
    for f in frames[1:]:
        out += f.astype(float)
    out /= len(frames)
    return out


def select_best_per_target(train_true, oof_map):
    all_oof = pd.DataFrame(index=train_true.index)
    return all_oof


def targetwise_search(y_true_df, lgb_oof_mean, cat_oof_mean):
    plan = {}
    val_pred = pd.DataFrame(index=y_true_df.index, columns=TARGETS, dtype=float)

    for t in TARGETS:
        best = None
        y_true = y_true_df[t].astype(int).values
        lgb_p = lgb_oof_mean[t].values.astype(float)
        cat_p = cat_oof_mean[t].values.astype(float)

        for lgb_w, cat_w in zip(LGB_WEIGHTS, CAT_WEIGHTS):
            blend = lgb_w * lgb_p + cat_w * cat_p
            for alpha in ALPHA_GRID:
                shr = shrink_proba(blend, alpha=alpha)
                for lo, hi in CLIP_GRID:
                    fin = clip_proba(shr, lo=lo, hi=hi)
                    score = log_loss(y_true, fin)
                    cand = {
                        'score': float(score),
                        'lgb_w': float(lgb_w),
                        'cat_w': float(cat_w),
                        'alpha': float(alpha),
                        'lo': float(lo),
                        'hi': float(hi),
                    }
                    if best is None or cand['score'] < best['score']:
                        best = cand
        plan[t] = best
        val_pred[t] = clip_proba(
            shrink_proba(best['lgb_w'] * lgb_p + best['cat_w'] * cat_p, alpha=best['alpha']),
            lo=best['lo'], hi=best['hi']
        )
    return plan, val_pred


def train_full_models(X_train, X_test, y_df, feature_cols, seeds):
    lgb_test_map = {}
    cat_test_map = {}

    for seed in seeds:
        lgb_test_map[f'lgb_s{seed}'] = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)
        cat_test_map[f'cat_s{seed}'] = pd.DataFrame(index=X_test.index, columns=TARGETS, dtype=float)

    X_tr_all = X_train[feature_cols]
    X_te = X_test[feature_cols]

    for t in TARGETS:
        y = y_df[t].astype(int)
        print(f'  > full-train target={t}')
        for seed in seeds:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=520,
                learning_rate=0.025,
                num_leaves=15,
                max_depth=4,
                min_child_samples=18,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=1.5,
                reg_lambda=4.0,
                objective='binary',
                class_weight='balanced',
                random_state=seed,
                verbosity=-1,
            )
            lgb_model.fit(X_tr_all, y)
            lgb_test_map[f'lgb_s{seed}'][t] = lgb_model.predict_proba(X_te)[:, 1]

            cat_model = CatBoostClassifier(
                iterations=520,
                learning_rate=0.025,
                depth=4,
                l2_leaf_reg=6.0,
                random_seed=seed,
                loss_function='Logloss',
                eval_metric='Logloss',
                verbose=False,
            )
            cat_model.fit(X_tr_all, y, verbose=False)
            cat_test_map[f'cat_s{seed}'][t] = cat_model.predict_proba(X_te)[:, 1]

            del lgb_model, cat_model
            gc.collect()

    return lgb_test_map, cat_test_map


def main():
    print_header('LOAD TRAIN / SAMPLE SUBMISSION')
    train = pd.read_csv(TRAIN_PATH)
    sample_sub = pd.read_csv(SUB_PATH)
    train['subject_id'] = train['subject_id'].astype(str)
    train['lifelog_date'] = pd.to_datetime(train['lifelog_date'])
    if 'subject_id' in sample_sub.columns:
        sample_sub['subject_id'] = sample_sub['subject_id'].astype(str)
    if 'lifelog_date' in sample_sub.columns:
        sample_sub['lifelog_date'] = pd.to_datetime(sample_sub['lifelog_date'])

    print('train:', train.shape)
    print('sample submission:', sample_sub.shape)

    test = infer_test_frame_from_submission(train, sample_sub)
    print('inferred test frame:', test.shape)

    base_df = pd.concat([
        train[['subject_id', 'lifelog_date'] + TARGETS].copy(),
        test[['subject_id', 'lifelog_date']].copy()
    ], axis=0, ignore_index=True).sort_values(['subject_id', 'lifelog_date']).reset_index(drop=True)

    base_df['dow'] = base_df['lifelog_date'].dt.dayofweek
    base_df['month'] = base_df['lifelog_date'].dt.month
    base_df['day'] = base_df['lifelog_date'].dt.day
    base_df['is_weekend'] = (base_df['dow'] >= 5).astype(int)
    base_df['days_from_global_start'] = (base_df['lifelog_date'] - base_df['lifelog_date'].min()).dt.days
    base_df['subject_day_index'] = base_df.groupby('subject_id').cumcount()

    print_header('DISCOVER SENSOR FILES')
    sensor_map = {
        'activity': get_sensor_path('mActivity'),
        'light': get_sensor_path('mLight'),
        'screen': get_sensor_path('mScreenStatus'),
        'hr': get_sensor_path('wHr'),
        'pedo': get_sensor_path('wPedo'),
        'charge': get_sensor_path('mACStatus'),
    }
    for k, v in sensor_map.items():
        print(k, '->', v)

    print_header('BUILD DAY-LEVEL FEATURES')
    feature_tables = [
        build_activity_features(sensor_map),
        build_light_features(sensor_map),
        build_screen_features(sensor_map),
        build_charge_features(sensor_map),
        build_hr_features(sensor_map),
        build_pedo_features(sensor_map),
    ]

    feat = base_df.copy()
    for ft in feature_tables:
        if ft is not None:
            print('merge feature table:', ft.shape)
            feat = feat.merge(ft, on=['subject_id', 'lifelog_date'], how='left')

    # stable feature set first
    stable_candidates = [c for c in feat.columns if c not in ['subject_id', 'lifelog_date'] + TARGETS]
    core_personal_cols = [
        c for c in stable_candidates if c in [
            'heart_rate_mean', 'heart_rate_std', 'heart_rate_sleep_mean', 'heart_rate_active_mean',
            'step_sum', 'distance_sum', 'speed_mean', 'm_screen_use_mean', 'm_light_mean', 'm_activity_mean'
        ]
    ]

    recent_cols = [
        c for c in [
            'heart_rate_mean', 'heart_rate_sleep_mean', 'heart_rate_active_mean',
            'step_sum', 'distance_sum', 'speed_mean', 'm_screen_use_mean', 'm_light_mean', 'm_activity_mean'
        ] if c in feat.columns
    ]

    print('stable_candidates:', len(stable_candidates))
    print('core_personal_cols:', core_personal_cols)
    print('recent_cols:', recent_cols)

    print_header('ADD PERSONAL / RECENT CHANGE FEATURES')
    feat = add_prior_features(feat, core_personal_cols)
    feat = add_recent_lag_features(feat, recent_cols)

    train_mask = feat[TARGETS[0]].notnull()
    numeric_train = feat.loc[train_mask].select_dtypes(include=[np.number])
    medians = numeric_train.median()
    means = numeric_train.mean()
    stds = numeric_train.std()

    for col in core_personal_cols:
        if f'{col}_prior_mean' in feat.columns:
            feat[f'{col}_prior_mean'] = feat[f'{col}_prior_mean'].fillna(means.get(col, 0.0))
        if f'{col}_prior_std' in feat.columns:
            feat[f'{col}_prior_std'] = feat[f'{col}_prior_std'].fillna(stds.get(col, 1.0))
        if f'{col}_dev' in feat.columns:
            feat[f'{col}_dev'] = feat[f'{col}_dev'].fillna(0.0)

    final_features = [c for c in feat.columns if c not in ['subject_id', 'lifelog_date'] + TARGETS]
    train_feat = feat[feat[TARGETS[0]].notnull()].copy().reset_index(drop=True)
    test_feat = feat[feat[TARGETS[0]].isnull()].copy().reset_index(drop=True)

    X_train = train_feat[final_features].copy().fillna(medians)
    X_test = test_feat[final_features].copy().fillna(medians)

    print('train_feat:', train_feat.shape, 'test_feat:', test_feat.shape)
    print('n_features:', len(final_features))

    dump_path = ART_DIR / 'artifacts_day_feature_table_strong.csv'
    feat.to_csv(dump_path, index=False)
    print('feature dump saved ->', dump_path)

    print_header('VALIDATION SPLIT')
    tr_idx, va_idx, valid_dates = build_global_time_split(train_feat, valid_ratio=0.2)
    print('tr rows:', len(tr_idx), 'va rows:', len(va_idx))
    print('valid date range:', valid_dates[0], '->', valid_dates[-1])

    print_header('SEEDED OOF TRAINING')
    oof_map, _ = fit_predict_seeded(train_feat, X_train, X_test, final_features, tr_idx, va_idx, SEEDS)
    lgb_oof_mean = mean_pred([oof_map[f'lgb_s{s}'] for s in SEEDS])
    cat_oof_mean = mean_pred([oof_map[f'cat_s{s}'] for s in SEEDS])
    avg_oof = 0.5 * lgb_oof_mean + 0.5 * cat_oof_mean

    score_lgb, each_lgb = avg_logloss(train_feat.loc[va_idx, TARGETS], lgb_oof_mean)
    score_cat, each_cat = avg_logloss(train_feat.loc[va_idx, TARGETS], cat_oof_mean)
    score_avg, each_avg = avg_logloss(train_feat.loc[va_idx, TARGETS], avg_oof)
    print('OOF LGB:', score_lgb, each_lgb)
    print('OOF CAT:', score_cat, each_cat)
    print('OOF AVG:', score_avg, each_avg)

    print_header('TARGET-WISE SEARCH: BLEND + SHRINK + CLIP')
    best_plan, tuned_val = targetwise_search(train_feat.loc[va_idx, TARGETS], lgb_oof_mean, cat_oof_mean)
    tuned_score, tuned_each = avg_logloss(train_feat.loc[va_idx, TARGETS], tuned_val)
    print('TUNED OOF:', tuned_score, tuned_each)
    plan_df = pd.DataFrame(best_plan).T
    print(plan_df)
    plan_df.to_csv(ART_DIR / 'targetwise_plan_strong.csv', index=True)

    print_header('FULL TRAIN + TEST PREDICTION')
    lgb_test_map, cat_test_map = train_full_models(X_train, X_test, train_feat[TARGETS], final_features, SEEDS)
    lgb_test_mean = mean_pred([lgb_test_map[f'lgb_s{s}'] for s in SEEDS])
    cat_test_mean = mean_pred([cat_test_map[f'cat_s{s}'] for s in SEEDS])

    final_pred = pd.DataFrame(index=test_feat.index, columns=TARGETS, dtype=float)
    for t in TARGETS:
        plan = best_plan[t]
        raw = plan['lgb_w'] * lgb_test_mean[t].values + plan['cat_w'] * cat_test_mean[t].values
        fin = clip_proba(shrink_proba(raw, alpha=plan['alpha']), lo=plan['lo'], hi=plan['hi'])
        final_pred[t] = fin

    submission = sample_sub.copy()
    for t in TARGETS:
        submission[t] = final_pred[t].values

    save_path = OUT_DIR / 'sub_strong_seeded_targetwise_ensemble.csv'
    submission.to_csv(save_path, index=False)
    print('submission saved ->', save_path)
    print(submission.head())
    print(submission[TARGETS].describe().T)


if __name__ == '__main__':
    main()
