import pandas as pd


def get_global_val_test_mmsis(which, path="train_val_test_mmsis.csv"):
    split_df = pd.read_csv(path)
    #split_df["mmsi"] = split_df["mmsi"].astype(int)
    print(split_df["mmsi"].dtype)
    return set(split_df.loc[split_df["split"] == which, "mmsi"])
 
# All vessels in each quarter (no MMSI split -- the split is by TIME).
val_mmsis = get_global_val_test_mmsis(which="validation")
test_mmsis = get_global_val_test_mmsis(which="test")

print(f" Val (2024) vessels: {len(val_mmsis)} | Test (2024) vessels: {len(test_mmsis)}")


df = pd.read_parquet("2023_10_12.parquet", engine="pyarrow")
print(df["mmsi"].dtype)