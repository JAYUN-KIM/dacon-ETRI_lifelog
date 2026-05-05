from pathlib import Path

import pandas as pd


BASE_DIR = Path("/mnt/c/etri-lifelog/data/raw/data")
SUB_DIR = BASE_DIR / "submissions"

ANCHOR_PATH = SUB_DIR / "sub_seed3_routing_q6040_s2080_alpha098.csv"
PAST_PRIOR_PATH = SUB_DIR / "sub_stateprior_pastonly_pure_20260505.csv"
BEST_TW_B_PATH = SUB_DIR / "sub_anchor_q6040_statepast_tw_b_20260505.csv"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def validate_frame(df: pd.DataFrame, name: str) -> None:
    required = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    if df.shape != (250, 10):
        raise ValueError(f"{name} unexpected shape: {df.shape}")
    if df.isnull().sum().sum() != 0:
        raise ValueError(f"{name} has null values")
    if not ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all():
        raise ValueError(f"{name} has probabilities outside [0,1]")


def blend_targetwise(anchor: pd.DataFrame, prior: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = anchor.copy()
    for target in TARGETS:
        weight = float(weights.get(target, 0.0))
        out[target] = (1 - weight) * anchor[target].astype(float) + weight * prior[target].astype(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def save_and_summarize(
    name: str,
    candidate: pd.DataFrame,
    anchor: pd.DataFrame,
    best_tw_b: pd.DataFrame,
) -> dict[str, float | str]:
    validate_frame(candidate, name)
    path = SUB_DIR / name
    candidate.to_csv(path, index=False)

    diff_anchor = (candidate[TARGETS] - anchor[TARGETS]).abs()
    diff_best = (candidate[TARGETS] - best_tw_b[TARGETS]).abs()
    return {
        "candidate": name,
        "diff_vs_anchor": float(diff_anchor.values.mean()),
        "diff_vs_tw_b": float(diff_best.values.mean()),
        "max_diff_vs_tw_b": float(diff_best.values.max()),
        "q_diff_vs_tw_b": float(diff_best[["Q1", "Q2", "Q3"]].values.mean()),
        "s_diff_vs_tw_b": float(diff_best[["S1", "S2", "S3", "S4"]].values.mean()),
        "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
        "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
    }


def main() -> None:
    anchor = pd.read_csv(ANCHOR_PATH)
    prior = pd.read_csv(PAST_PRIOR_PATH)
    best_tw_b = pd.read_csv(BEST_TW_B_PATH)

    validate_frame(anchor, ANCHOR_PATH.name)
    validate_frame(prior, PAST_PRIOR_PATH.name)
    validate_frame(best_tw_b, BEST_TW_B_PATH.name)

    # Public best so far from this family:
    # tw_b = Q1 .18 / Q2 .26 / Q3 .22 / S1 .00 / S2 .10 / S3 .02 / S4 .08
    configs = {
        # Minimal, clean ablation: keep Q signal, remove S3 which was weak in forward validation.
        "tw_b_s3zero": {"Q1": 0.18, "Q2": 0.26, "Q3": 0.22, "S1": 0.00, "S2": 0.10, "S3": 0.00, "S4": 0.08},
        # Q2/Q3 were the strongest state-transition gains; push them one notch.
        "tw_qboost1": {"Q1": 0.18, "Q2": 0.30, "Q3": 0.26, "S1": 0.00, "S2": 0.10, "S3": 0.00, "S4": 0.08},
        "tw_qboost2": {"Q1": 0.16, "Q2": 0.32, "Q3": 0.28, "S1": 0.00, "S2": 0.10, "S3": 0.00, "S4": 0.08},
        # Slightly raise Q1 too, but avoid touching S1/S3.
        "tw_qbalanced": {"Q1": 0.20, "Q2": 0.28, "Q3": 0.24, "S1": 0.00, "S2": 0.10, "S3": 0.00, "S4": 0.08},
        # If S2/S4 recency is useful but S1/S3 are noisy.
        "tw_s24boost": {"Q1": 0.18, "Q2": 0.26, "Q3": 0.22, "S1": 0.00, "S2": 0.14, "S3": 0.00, "S4": 0.12},
        # More experimental: Q-only state prior. Good if S priors were mostly noise.
        "tw_qonly": {"Q1": 0.18, "Q2": 0.26, "Q3": 0.22, "S1": 0.00, "S2": 0.00, "S3": 0.00, "S4": 0.00},
        # Aggressive Q2/Q3 card. Use only if qboost1 improves.
        "tw_qaggr": {"Q1": 0.20, "Q2": 0.34, "Q3": 0.30, "S1": 0.00, "S2": 0.10, "S3": 0.00, "S4": 0.08},
        # Slight retreat from tw_b, useful if tw_b was near optimum and extra movement hurts.
        "tw_soft": {"Q1": 0.16, "Q2": 0.24, "Q3": 0.20, "S1": 0.00, "S2": 0.08, "S3": 0.00, "S4": 0.06},
    }

    summary = []
    for tag, weights in configs.items():
        name = f"sub_anchor_q6040_statepast_{tag}_20260505.csv"
        candidate = blend_targetwise(anchor, prior, weights)
        summary.append(save_and_summarize(name, candidate, anchor, best_tw_b))

    summary_df = pd.DataFrame(summary).sort_values(["diff_vs_tw_b", "candidate"]).reset_index(drop=True)
    print(summary_df.to_string(index=False))
    print("\nSuggested submit order after tw_b public 0.5919692903:")
    print("1) sub_anchor_q6040_statepast_tw_qboost1_20260505.csv")
    print("2) sub_anchor_q6040_statepast_tw_b_s3zero_20260505.csv")
    print("3) sub_anchor_q6040_statepast_tw_qaggr_20260505.csv only if qboost1 improves")


if __name__ == "__main__":
    main()
