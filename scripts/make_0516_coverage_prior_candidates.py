from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "data"
ARTIFACT_PATH = DATA_DIR / "artifacts" / "sleep_window_features_20260511.csv"
SUB_DIR = DATA_DIR / "submissions"
CURRENT_BEST = SUB_DIR / "sub_0515_metricproxy_s_only.csv"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def robust_z(values, ref_mask):
    s = pd.to_numeric(values, errors="coerce")
    ref = s[ref_mask]
    med = ref.median()
    iqr = ref.quantile(0.75) - ref.quantile(0.25)
    if not np.isfinite(iqr) or iqr < 1e-9:
        iqr = ref.std()
    if not np.isfinite(iqr) or iqr < 1e-9:
        iqr = 1.0
    return ((s.fillna(med) - med) / iqr).clip(-5, 5)


def build_coverage_scores(feat):
    train_mask = feat["is_train"].eq(1)
    count_cols = [c for c in feat.columns if c.endswith("_count")]
    count_z = {c: robust_z(np.log1p(feat[c]), train_mask) for c in count_cols}

    def c(name):
        return count_z.get(name, pd.Series(0.0, index=feat.index))

    out = feat[["subject_id", "sleep_date", "lifelog_date", "is_train"] + TARGETS].copy()
    # Recording density itself can reflect sleep regularity, phone off/on behavior,
    # wearable adherence, or abnormal late-night activity. Keep these as proxy
    # scores; target orientation is calibrated from train labels.
    out["cov_phone_evening"] = (
        0.30 * c("screen_evening_count")
        + 0.24 * c("screen_pre_sleep_count")
        + 0.20 * c("charge_pre_sleep_count")
        + 0.16 * c("mlight_pre_sleep_count")
        + 0.10 * c("mact_pre_sleep_count")
    )
    out["cov_wearable_overnight"] = (
        0.34 * c("wlight_overnight_count")
        + 0.30 * c("pedo_overnight_count")
        + 0.22 * c("mact_overnight_count")
        + 0.14 * c("screen_overnight_count")
    )
    out["cov_full_adherence"] = sum(count_z.values()) / max(len(count_z), 1)
    out["cov_night_vs_day"] = (
        c("screen_overnight_count")
        + c("mact_overnight_count")
        + c("pedo_overnight_count")
        - c("screen_morning_count")
        - c("mact_morning_count")
        - c("pedo_morning_count")
    )
    out["cov_missingness_low"] = -out["cov_full_adherence"]
    return out


def rank_bin_prior(score, y, all_score, bins=6, shrink=24.0):
    global_mean = float(y.mean())
    corr = pd.Series(score).corr(y)
    if not np.isfinite(corr):
        corr = 0.0
    oriented_train = score if corr >= 0 else -score
    oriented_all = all_score if corr >= 0 else -all_score
    train_rank = pd.Series(oriented_train).rank(pct=True, method="average")
    all_rank = pd.Series(oriented_all).rank(pct=True, method="average")
    try:
        qbin = pd.qcut(train_rank, q=bins, labels=False, duplicates="drop")
    except ValueError:
        qbin = pd.Series(np.zeros(len(train_rank), dtype=int), index=train_rank.index)
    stats = pd.DataFrame({"bin": qbin, "y": y}).groupby("bin")["y"].agg(["mean", "count"]).reset_index()
    stats["smooth"] = (stats["mean"] * stats["count"] + global_mean * shrink) / (stats["count"] + shrink)
    mapping = dict(zip(stats["bin"].astype(int), stats["smooth"]))
    if len(mapping) <= 1:
        return pd.Series(global_mean, index=all_score.index), corr
    edges = np.quantile(train_rank, np.linspace(0, 1, len(mapping) + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    bins_all = np.digitize(all_rank.to_numpy(float), edges[1:-1], right=True)
    prior = pd.Series([mapping.get(int(b), global_mean) for b in bins_all], index=all_score.index)
    return (0.76 * prior + 0.24 * global_mean).clip(0.05, 0.95), corr


def make_prior_table(scores):
    train_mask = scores["is_train"].eq(1)
    proxy_map = {
        "Q1": ["cov_wearable_overnight", "cov_full_adherence"],
        "Q2": ["cov_phone_evening", "cov_night_vs_day"],
        "Q3": ["cov_phone_evening", "cov_missingness_low"],
        "S1": ["cov_wearable_overnight", "cov_full_adherence"],
        "S2": ["cov_wearable_overnight", "cov_night_vs_day"],
        "S3": ["cov_phone_evening", "cov_night_vs_day"],
        "S4": ["cov_night_vs_day", "cov_wearable_overnight"],
    }
    prior = scores[["subject_id", "sleep_date", "lifelog_date", "is_train"]].copy()
    diag_rows = []
    for target, proxies in proxy_map.items():
        y = pd.to_numeric(scores.loc[train_mask, target], errors="coerce")
        target_priors = []
        corr_abs = []
        for proxy in proxies:
            p, corr = rank_bin_prior(scores.loc[train_mask, proxy], y, scores[proxy])
            target_priors.append(p)
            corr_abs.append(abs(corr))
            diag_rows.append({"target": target, "proxy": proxy, "corr": corr})
        weights = np.array(corr_abs, dtype=float)
        if weights.sum() <= 1e-9:
            weights = np.ones(len(target_priors)) / len(target_priors)
        else:
            weights = weights / weights.sum()
        combined = sum(w * p for w, p in zip(weights, target_priors))
        prior[target] = combined.clip(0.05, 0.95)
    return prior, pd.DataFrame(diag_rows)


def blend(best, prior_test, weights):
    out = best.copy()
    for target in TARGETS:
        w = float(weights.get(target, 0.0))
        out[target] = np.clip((1 - w) * best[target].to_numpy(float) + w * prior_test[target].to_numpy(float), 0.04, 0.96)
    return out


def save_candidate(name, best, prior_test, weights):
    cand = blend(best, prior_test, weights)
    cand.to_csv(SUB_DIR / name, index=False)
    diff = (cand[TARGETS] - best[TARGETS]).abs()
    return {
        "candidate": name,
        "mean_abs_diff": float(diff.to_numpy().mean()),
        "max_abs_diff": float(diff.to_numpy().max()),
        "mean_q": float(cand[["Q1", "Q2", "Q3"]].to_numpy().mean()),
        "mean_s": float(cand[["S1", "S2", "S3", "S4"]].to_numpy().mean()),
    }


def main():
    feat = pd.read_csv(ARTIFACT_PATH)
    best = pd.read_csv(CURRENT_BEST)
    scores = build_coverage_scores(feat)
    prior_all, diag = make_prior_table(scores)
    prior_test = prior_all[prior_all["is_train"].eq(0)].reset_index(drop=True)

    print("Coverage proxy correlations:")
    print(diag.pivot_table(index="target", columns="proxy", values="corr", aggfunc="first").round(4).fillna("").to_string())

    rows = []
    rows.append(
        save_candidate(
            "sub_0516_coverage_s_tiny.csv",
            best,
            prior_test,
            {"Q1": 0.000, "Q2": 0.000, "Q3": 0.000, "S1": 0.006, "S2": 0.006, "S3": 0.006, "S4": 0.006},
        )
    )
    rows.append(
        save_candidate(
            "sub_0516_coverage_qs_tiny.csv",
            best,
            prior_test,
            {"Q1": 0.004, "Q2": 0.004, "Q3": 0.004, "S1": 0.006, "S2": 0.006, "S3": 0.006, "S4": 0.006},
        )
    )
    rows.append(
        save_candidate(
            "sub_0516_coverage_phone_q.csv",
            best,
            prior_test,
            {"Q1": 0.002, "Q2": 0.008, "Q3": 0.008, "S1": 0.004, "S2": 0.004, "S3": 0.004, "S4": 0.004},
        )
    )

    summary = pd.DataFrame(rows).sort_values("mean_abs_diff").reset_index(drop=True)
    print("\n0516 coverage-prior candidates vs current best:")
    print(summary.to_string(index=False))
    print("\nSuggested submit order:")
    print("1) sub_0516_coverage_s_tiny.csv")
    print("2) sub_0516_coverage_qs_tiny.csv")
    print("3) sub_0516_coverage_phone_q.csv")


if __name__ == "__main__":
    main()
