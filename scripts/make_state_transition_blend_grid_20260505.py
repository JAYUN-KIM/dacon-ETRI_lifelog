from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("/mnt/c/etri-lifelog/data/raw/data")
SUB_DIR = BASE_DIR / "submissions"

ANCHOR_PATH = SUB_DIR / "sub_seed3_routing_q6040_s2080_alpha098.csv"
PAST_PRIOR_PATH = SUB_DIR / "sub_stateprior_pastonly_pure_20260505.csv"
FULL_PRIOR_PATH = SUB_DIR / "sub_stateprior_fullsubject_pure_20260505.csv"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
Q_TARGETS = ["Q1", "Q2", "Q3"]
S_TARGETS = ["S1", "S2", "S3", "S4"]


def validate_frame(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in ["subject_id", "sleep_date", "lifelog_date"] + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    if df.shape != (250, 10):
        raise ValueError(f"{name} unexpected shape: {df.shape}")
    if df.isnull().sum().sum() != 0:
        raise ValueError(f"{name} has nulls")
    in_range = ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all()
    if not in_range:
        raise ValueError(f"{name} has probabilities outside [0,1]")


def blend_scalar(anchor: pd.DataFrame, prior: pd.DataFrame, q_weight: float, s_weight: float) -> pd.DataFrame:
    weights = {target: q_weight for target in Q_TARGETS}
    weights.update({target: s_weight for target in S_TARGETS})
    return blend_targetwise(anchor, prior, weights)


def blend_targetwise(anchor: pd.DataFrame, prior: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = anchor.copy()
    for target in TARGETS:
        w = float(weights.get(target, 0.0))
        out[target] = (1 - w) * anchor[target].astype(float) + w * prior[target].astype(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def blend_two_priors(
    anchor: pd.DataFrame,
    past_prior: pd.DataFrame,
    full_prior: pd.DataFrame,
    past_ratio: float,
    q_weight: float,
    s_weight: float,
) -> pd.DataFrame:
    prior = anchor.copy()
    prior[TARGETS] = past_ratio * past_prior[TARGETS].astype(float) + (1 - past_ratio) * full_prior[TARGETS].astype(float)
    return blend_scalar(anchor, prior, q_weight=q_weight, s_weight=s_weight)


def save_and_summarize(name: str, df: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, float | str]:
    path = SUB_DIR / name
    df.to_csv(path, index=False)
    validate_frame(df, name)

    diff = (df[TARGETS].astype(float) - anchor[TARGETS].astype(float)).abs()
    return {
        "candidate": name,
        "mean_abs_diff": float(diff.values.mean()),
        "max_abs_diff": float(diff.values.max()),
        "q_diff": float(diff[Q_TARGETS].values.mean()),
        "s_diff": float(diff[S_TARGETS].values.mean()),
        "q_mean": float(df[Q_TARGETS].values.mean()),
        "s_mean": float(df[S_TARGETS].values.mean()),
    }


def main() -> None:
    anchor = pd.read_csv(ANCHOR_PATH)
    past = pd.read_csv(PAST_PRIOR_PATH)
    full = pd.read_csv(FULL_PRIOR_PATH)

    validate_frame(anchor, ANCHOR_PATH.name)
    validate_frame(past, PAST_PRIOR_PATH.name)
    validate_frame(full, FULL_PRIOR_PATH.name)

    candidates: dict[str, pd.DataFrame] = {}

    # Continuation grid around the public-winning q14/s07 direction.
    for q_weight, s_weight in [
        (0.16, 0.08),
        (0.18, 0.09),
        (0.20, 0.10),
        (0.22, 0.11),
        (0.24, 0.12),
        (0.28, 0.14),
    ]:
        tag = f"q{int(q_weight * 100):02d}_s{int(s_weight * 100):02d}"
        candidates[f"sub_anchor_q6040_statepast_{tag}_20260505.csv"] = blend_scalar(anchor, past, q_weight, s_weight)

    # Target-wise route: forward validation said Q targets, especially Q2/Q3,
    # gain more from state dynamics than S1/S3.
    targetwise_configs = {
        "tw_a": {"Q1": 0.18, "Q2": 0.22, "Q3": 0.20, "S1": 0.02, "S2": 0.10, "S3": 0.03, "S4": 0.08},
        "tw_b": {"Q1": 0.18, "Q2": 0.26, "Q3": 0.22, "S1": 0.00, "S2": 0.10, "S3": 0.02, "S4": 0.08},
        "tw_c": {"Q1": 0.14, "Q2": 0.24, "Q3": 0.20, "S1": 0.00, "S2": 0.08, "S3": 0.00, "S4": 0.06},
        "tw_d": {"Q1": 0.22, "Q2": 0.28, "Q3": 0.24, "S1": 0.02, "S2": 0.12, "S3": 0.03, "S4": 0.10},
    }
    for tag, weights in targetwise_configs.items():
        candidates[f"sub_anchor_q6040_statepast_{tag}_20260505.csv"] = blend_targetwise(anchor, past, weights)

    # Experimental but not primary: combine mostly past-only with a little full-subject prior.
    # This is included as a diagnostic; past-only remains the cleaner future-prediction choice.
    for past_ratio, q_weight, s_weight in [
        (0.85, 0.16, 0.08),
        (0.85, 0.20, 0.10),
        (0.70, 0.16, 0.08),
    ]:
        tag = f"mixp{int(past_ratio * 100)}_q{int(q_weight * 100):02d}_s{int(s_weight * 100):02d}"
        candidates[f"sub_anchor_q6040_stateblend_{tag}_20260505.csv"] = blend_two_priors(
            anchor,
            past,
            full,
            past_ratio=past_ratio,
            q_weight=q_weight,
            s_weight=s_weight,
        )

    summary = []
    for name, df in candidates.items():
        summary.append(save_and_summarize(name, df, anchor))

    summary_df = pd.DataFrame(summary).sort_values(["mean_abs_diff", "candidate"]).reset_index(drop=True)
    print(summary_df.to_string(index=False))
    print("\nSuggested submit order if previous q14_s07 scored 0.592269:")
    print("1) sub_anchor_q6040_statepast_q20_s10_20260505.csv  # same axis, more assertive")
    print("2) sub_anchor_q6040_statepast_tw_b_20260505.csv      # target-wise experimental")
    print("3) sub_anchor_q6040_statepast_q24_s12_20260505.csv  # only if q20/s10 improves")


if __name__ == "__main__":
    main()
