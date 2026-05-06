from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUB_DIR = ROOT / "data" / "raw" / "data" / "submissions"

ANCHOR_PATH = SUB_DIR / "sub_anchor_q6040_statepast_tw_b_20260505.csv"
DYN_PATH = SUB_DIR / "sub_reset_targetdyn_revert_pure_20260506.csv"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def validate(df: pd.DataFrame, name: str) -> None:
    required = ["subject_id", "sleep_date", "lifelog_date"] + TARGETS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    if df.shape != (250, 10):
        raise ValueError(f"{name} unexpected shape: {df.shape}")
    if df.isnull().sum().sum() != 0:
        raise ValueError(f"{name} has nulls")
    if not ((df[TARGETS] >= 0) & (df[TARGETS] <= 1)).all().all():
        raise ValueError(f"{name} out of [0,1]")


def uniform_blend(anchor: pd.DataFrame, dyn: pd.DataFrame, weight: float) -> pd.DataFrame:
    out = anchor.copy()
    out[TARGETS] = (1 - weight) * anchor[TARGETS].astype(float) + weight * dyn[TARGETS].astype(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def targetwise_blend(anchor: pd.DataFrame, dyn: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = anchor.copy()
    for target in TARGETS:
        weight = float(weights.get(target, 0.0))
        out[target] = (1 - weight) * anchor[target].astype(float) + weight * dyn[target].astype(float)
    out[TARGETS] = out[TARGETS].clip(1e-6, 1 - 1e-6)
    return out


def summarize(name: str, candidate: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, float | str]:
    diff = (candidate[TARGETS] - anchor[TARGETS]).abs()
    return {
        "candidate": name,
        "diff_vs_anchor": float(diff.values.mean()),
        "max_diff": float(diff.values.max()),
        "q_diff": float(diff[["Q1", "Q2", "Q3"]].values.mean()),
        "s_diff": float(diff[["S1", "S2", "S3", "S4"]].values.mean()),
        "mean_q": float(candidate[["Q1", "Q2", "Q3"]].values.mean()),
        "mean_s": float(candidate[["S1", "S2", "S3", "S4"]].values.mean()),
    }


def main() -> None:
    anchor = pd.read_csv(ANCHOR_PATH)
    dyn = pd.read_csv(DYN_PATH)
    validate(anchor, ANCHOR_PATH.name)
    validate(dyn, DYN_PATH.name)

    candidates: list[dict[str, float | str]] = []

    for weight in [0.08, 0.12, 0.14, 0.16, 0.20]:
        name = f"sub_reset_targetdyn_revert_anchor_w{int(weight * 100):02d}_grid_20260506.csv"
        out = uniform_blend(anchor, dyn, weight)
        validate(out, name)
        out.to_csv(SUB_DIR / name, index=False)
        candidates.append(summarize(name, out, anchor))

    targetwise_configs = {
        # w10 improved, so test a slightly Q-led shape without moving S3 too much.
        "tw_q12_s08": {"Q1": 0.12, "Q2": 0.12, "Q3": 0.12, "S1": 0.08, "S2": 0.08, "S3": 0.06, "S4": 0.08},
        # More aggressive Q dynamics, conservative S.
        "tw_q14_s08": {"Q1": 0.14, "Q2": 0.14, "Q3": 0.14, "S1": 0.08, "S2": 0.08, "S3": 0.06, "S4": 0.08},
        # Uniform-ish but pull S3 back because it has historically been noisy.
        "tw_w14_s3low": {"Q1": 0.14, "Q2": 0.14, "Q3": 0.14, "S1": 0.14, "S2": 0.14, "S3": 0.06, "S4": 0.14},
        # If the dynamics signal is mostly Q-state driven.
        "tw_q16_s06": {"Q1": 0.16, "Q2": 0.16, "Q3": 0.16, "S1": 0.06, "S2": 0.06, "S3": 0.04, "S4": 0.06},
    }

    for tag, weights in targetwise_configs.items():
        name = f"sub_reset_targetdyn_revert_anchor_{tag}_20260506.csv"
        out = targetwise_blend(anchor, dyn, weights)
        validate(out, name)
        out.to_csv(SUB_DIR / name, index=False)
        candidates.append(summarize(name, out, anchor))

    summary = pd.DataFrame(candidates).sort_values("diff_vs_anchor").reset_index(drop=True)
    print(summary.to_string(index=False))
    print("\nIf w10 public = 0.5902811814, suggested next order:")
    print("1) sub_reset_targetdyn_revert_anchor_w14_grid_20260506.csv")
    print("2) sub_reset_targetdyn_revert_anchor_tw_w14_s3low_20260506.csv")
    print("3) sub_reset_targetdyn_revert_anchor_w16_grid_20260506.csv")


if __name__ == "__main__":
    main()
