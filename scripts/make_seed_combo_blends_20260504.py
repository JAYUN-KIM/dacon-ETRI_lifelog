from pathlib import Path

import pandas as pd


BASE_DIR = Path("/mnt/c/etri-lifelog/data/raw/data/submissions")
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


BLEND_CONFIGS = [
    {
        "name": "q6040_anchor075_seed77_2024_1004_025",
        "anchor": "sub_seed3_routing_q6040_s2080_alpha098.csv",
        "candidate": "sub_seed3_routing_q6040_s2080_alpha098_seeds77_2024_1004.csv",
        "anchor_weight": 0.75,
    },
    {
        "name": "q6040_anchor075_seed42_77_777_025",
        "anchor": "sub_seed3_routing_q6040_s2080_alpha098.csv",
        "candidate": "sub_seed3_routing_q6040_s2080_alpha098_seeds42_77_777.csv",
        "anchor_weight": 0.75,
    },
    {
        "name": "q6040_anchor085_seed77_2024_1004_015",
        "anchor": "sub_seed3_routing_q6040_s2080_alpha098.csv",
        "candidate": "sub_seed3_routing_q6040_s2080_alpha098_seeds77_2024_1004.csv",
        "anchor_weight": 0.85,
    },
]


def main():
    for cfg in BLEND_CONFIGS:
        anchor = pd.read_csv(BASE_DIR / cfg["anchor"])
        candidate = pd.read_csv(BASE_DIR / cfg["candidate"])
        out = anchor.copy()
        aw = cfg["anchor_weight"]
        cw = 1.0 - aw
        out[TARGETS] = (aw * anchor[TARGETS] + cw * candidate[TARGETS]).clip(0.0, 1.0)
        out_path = BASE_DIR / f"sub_blend_{cfg['name']}.csv"
        out.to_csv(out_path, index=False)

        diff = (out[TARGETS] - anchor[TARGETS]).abs()
        print("=" * 80)
        print(cfg["name"], f"anchor={aw:.2f}", f"candidate={cw:.2f}")
        print("saved:", out_path)
        print("shape:", out.shape)
        print("nulls:", int(out.isnull().sum().sum()))
        print("all targets in [0,1]:", bool(((out[TARGETS] >= 0) & (out[TARGETS] <= 1)).all().all()))
        print("mean_abs_diff_from_anchor:", float(diff.values.mean()))
        print("max_abs_diff_from_anchor:", float(diff.values.max()))
        print("per_target_mean_abs_diff:")
        print(diff.mean())


if __name__ == "__main__":
    main()
