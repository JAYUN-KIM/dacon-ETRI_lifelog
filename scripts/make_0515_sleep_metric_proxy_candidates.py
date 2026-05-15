from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "data"
ARTIFACT_PATH = DATA_DIR / "artifacts" / "sleep_window_features_20260511.csv"
SUB_DIR = DATA_DIR / "submissions"
CURRENT_BEST = SUB_DIR / "sub_0512_sleepwin_s_focus.csv"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def robust_z(frame, col, ref_mask):
    if col not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    values = pd.to_numeric(frame[col], errors="coerce")
    ref = values[ref_mask]
    med = float(ref.median()) if ref.notna().any() else 0.0
    q75 = float(ref.quantile(0.75)) if ref.notna().any() else 1.0
    q25 = float(ref.quantile(0.25)) if ref.notna().any() else 0.0
    scale = q75 - q25
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(ref.std()) if ref.notna().sum() > 1 else 1.0
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return ((values.fillna(med) - med) / scale).clip(-5, 5)


def add_metric_proxies(feat):
    train_mask = feat["is_train"].eq(1)
    z = lambda c: robust_z(feat, c, train_mask)

    # Low activity / low light / low screen / low step during the sleep context
    # approximates longer and cleaner rest. Signs are intentionally human-readable;
    # target-specific orientation is learned from train correlations below.
    feat["proxy_rest_duration"] = (
        -0.24 * z("mact_full_sleepctx_m_activity_mean")
        -0.20 * z("pedo_full_sleepctx_step_sum")
        -0.18 * z("screen_full_sleepctx_m_screen_use_sum")
        -0.14 * z("mlight_full_sleepctx_m_light_mean")
        -0.10 * z("wlight_full_sleepctx_w_light_mean")
        +0.14 * z("charge_overnight_m_charging_mean")
    )
    feat["proxy_sleep_efficiency"] = (
        -0.28 * z("mact_overnight_m_activity_mean")
        -0.24 * z("screen_overnight_m_screen_use_sum")
        -0.20 * z("pedo_overnight_step_sum")
        -0.14 * z("mlight_overnight_m_light_mean")
        -0.08 * z("wlight_overnight_w_light_mean")
        -0.06 * z("hr_overnight_heart_rate_std")
    )
    feat["proxy_sleep_latency_bad"] = (
        +0.28 * z("screen_pre_sleep_m_screen_use_sum")
        +0.22 * z("mlight_pre_sleep_m_light_mean")
        +0.18 * z("mact_pre_sleep_m_activity_mean")
        +0.12 * z("pedo_pre_sleep_step_sum")
        +0.10 * z("hr_pre_sleep_heart_rate_mean")
        -0.10 * z("charge_pre_sleep_m_charging_mean")
    )
    feat["proxy_wake_after_sleep_bad"] = (
        +0.24 * z("screen_overnight_m_screen_use_sum")
        +0.22 * z("mact_overnight_m_activity_max")
        +0.18 * z("pedo_overnight_step_sum")
        +0.16 * z("mlight_overnight_m_light_max")
        +0.10 * z("hr_overnight_heart_rate_std")
        +0.10 * z("screen_morning_m_screen_use_sum")
    )
    feat["proxy_sleep_quality"] = (
        +0.38 * feat["proxy_sleep_efficiency"]
        +0.28 * feat["proxy_rest_duration"]
        -0.18 * feat["proxy_sleep_latency_bad"]
        -0.16 * feat["proxy_wake_after_sleep_bad"]
    )
    feat["proxy_fatigue_stress"] = (
        +0.24 * z("screen_pre_sleep_m_screen_use_sum")
        +0.20 * z("hr_pre_sleep_heart_rate_mean")
        +0.18 * z("hr_full_sleepctx_heart_rate_std")
        +0.16 * z("mact_morning_m_activity_mean")
        +0.12 * z("pedo_morning_step_sum")
        +0.10 * z("mlight_pre_sleep_m_light_mean")
    )
    return feat


def calibrate_proxy_prior(feat, target_to_proxy, bins=6):
    train_mask = feat["is_train"].eq(1)
    out = feat[["subject_id", "sleep_date", "lifelog_date", "is_train"]].copy()
    diagnostics = []
    for target, proxy_col in target_to_proxy.items():
        score = pd.to_numeric(feat[proxy_col], errors="coerce").fillna(0.0)
        train_score = score[train_mask]
        y = pd.to_numeric(feat.loc[train_mask, target], errors="coerce")
        global_mean = float(y.mean())

        corr = float(pd.Series(train_score).corr(y)) if y.nunique() > 1 else 0.0
        if not np.isfinite(corr):
            corr = 0.0
        oriented = score if corr >= 0 else -score
        train_oriented = oriented[train_mask]

        # Rank-space binning avoids trusting raw proxy scale.
        all_rank = oriented.rank(pct=True, method="average")
        train_rank = all_rank[train_mask]
        try:
            qbin = pd.qcut(train_rank, q=bins, labels=False, duplicates="drop")
        except ValueError:
            qbin = pd.Series(np.zeros(len(train_rank), dtype=int), index=train_rank.index)
        bin_df = pd.DataFrame({"bin": qbin, "y": y})
        stats = bin_df.groupby("bin")["y"].agg(["mean", "count"]).reset_index()
        stats["smooth"] = (stats["mean"] * stats["count"] + global_mean * 18.0) / (stats["count"] + 18.0)
        mapping = dict(zip(stats["bin"].astype(int), stats["smooth"]))

        if len(mapping) <= 1:
            prior = pd.Series(global_mean, index=feat.index)
        else:
            edges = np.quantile(train_rank, np.linspace(0, 1, len(mapping) + 1))
            edges[0] = -np.inf
            edges[-1] = np.inf
            test_bins = np.digitize(all_rank.to_numpy(float), edges[1:-1], right=True)
            prior = pd.Series([mapping.get(int(b), global_mean) for b in test_bins], index=feat.index)

        # Keep the hand-crafted prior conservative; it should be a directional nudge.
        prior = 0.72 * prior + 0.28 * global_mean
        out[target] = np.clip(prior, 0.05, 0.95)
        diagnostics.append(
            {
                "target": target,
                "proxy": proxy_col,
                "train_corr": corr,
                "global_mean": global_mean,
                "prior_train_mean": float(out.loc[train_mask, target].mean()),
                "prior_test_mean": float(out.loc[~train_mask, target].mean()),
            }
        )
    return out, pd.DataFrame(diagnostics)


def blend(best, prior_test, weights):
    out = best.copy()
    for target in TARGETS:
        w = float(weights.get(target, 0.0))
        out[target] = np.clip(
            (1 - w) * best[target].to_numpy(float) + w * prior_test[target].to_numpy(float),
            0.04,
            0.96,
        )
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
    feat = add_metric_proxies(feat)
    best = pd.read_csv(CURRENT_BEST)

    target_to_proxy = {
        "Q1": "proxy_sleep_quality",
        "Q2": "proxy_fatigue_stress",
        "Q3": "proxy_fatigue_stress",
        "S1": "proxy_rest_duration",
        "S2": "proxy_sleep_efficiency",
        "S3": "proxy_sleep_latency_bad",
        "S4": "proxy_wake_after_sleep_bad",
    }
    prior_all, diag = calibrate_proxy_prior(feat, target_to_proxy)
    prior_test = prior_all[prior_all["is_train"].eq(0)].reset_index(drop=True)

    print("Sleep metric proxy diagnostics:")
    print(diag.to_string(index=False))

    rows = []
    rows.append(
        save_candidate(
            "sub_0515_metricproxy_s_only.csv",
            best,
            prior_test,
            {"Q1": 0.000, "Q2": 0.000, "Q3": 0.000, "S1": 0.010, "S2": 0.010, "S3": 0.010, "S4": 0.010},
        )
    )
    rows.append(
        save_candidate(
            "sub_0515_metricproxy_qs_tiny.csv",
            best,
            prior_test,
            {"Q1": 0.006, "Q2": 0.004, "Q3": 0.004, "S1": 0.010, "S2": 0.010, "S3": 0.010, "S4": 0.010},
        )
    )
    rows.append(
        save_candidate(
            "sub_0515_metricproxy_sleep_quality.csv",
            best,
            prior_test,
            {"Q1": 0.012, "Q2": 0.000, "Q3": 0.000, "S1": 0.012, "S2": 0.012, "S3": 0.008, "S4": 0.008},
        )
    )

    summary = pd.DataFrame(rows).sort_values("mean_abs_diff").reset_index(drop=True)
    print("\n0515 sleep-metric proxy candidates vs current best:")
    print(summary.to_string(index=False))
    print("\nSuggested submit order:")
    print("1) sub_0515_metricproxy_s_only.csv")
    print("2) sub_0515_metricproxy_qs_tiny.csv")
    print("3) sub_0515_metricproxy_sleep_quality.csv")


if __name__ == "__main__":
    main()
