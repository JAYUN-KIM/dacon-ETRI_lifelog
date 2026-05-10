from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SENSOR_DIR = ROOT / "data" / "raw" / "data" / "ch2025_data_items"


def main():
    paths = sorted(SENSOR_DIR.glob("*.parquet"))
    print(f"sensor parquet files: {len(paths)}")
    for path in paths:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            print(f"\n--- {path.name} ---")
            print(f"read error: {type(exc).__name__}: {exc}")
            continue

        print(f"\n--- {path.name} {df.shape} ---")
        print("columns:", list(df.columns)[:60])
        for col in ["subject_id", "timestamp", "ts", "datetime", "lifelog_date"]:
            if col in df.columns:
                try:
                    print(
                        col,
                        "min=", df[col].min(),
                        "max=", df[col].max(),
                        "nunique=", df[col].nunique(),
                    )
                except Exception as exc:
                    print(col, "summary error:", exc)
        print("null ratio head:")
        print(df.isna().mean().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()
