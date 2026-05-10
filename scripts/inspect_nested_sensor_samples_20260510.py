from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SENSOR_DIR = ROOT / "data" / "raw" / "data" / "ch2025_data_items"
FILES = [
    "ch2025_mAmbience.parquet",
    "ch2025_mGps.parquet",
    "ch2025_mUsageStats.parquet",
    "ch2025_mWifi.parquet",
    "ch2025_mBle.parquet",
    "ch2025_wPedo.parquet",
]


def main():
    for name in FILES:
        path = SENSOR_DIR / name
        df = pd.read_parquet(path)
        value_col = [c for c in df.columns if c not in ["subject_id", "timestamp"]][0]
        print(f"\n--- {name} value_col={value_col} shape={df.shape} ---")
        print(df[[value_col]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
