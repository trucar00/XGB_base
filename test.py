import pandas as pd

df = pd.read_parquet("data/2025_1_3_feats.parquet", engine="pyarrow")
print(df.columns)
print(df.head())

print(df["gear_report"].unique())