from xgboost import XGBClassifier
import pickle
import pandas as pd
import numpy as np

# -----------------------
# Load model
# -----------------------
model = XGBClassifier()
model.load_model("models/.json")

# -----------------------
# Load metadata
# -----------------------
BASE_FEATURES = [
    "cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk",
    "log_dist", "ra_dcog", "log_dt", "dist_to_shore_km"
]
SEASON_FEATURES = ["month_sin", "month_cos"]
FEATURES = BASE_FEATURES + SEASON_FEATURES
threshold = 0.5


# -----------------------
# Read 2025 data
# -----------------------
df_2025 = pd.read_parquet(
    "data/2025_1_3_feats.parquet",
    engine="pyarrow"
)

# Add seasonal features
df_2025["date_time_utc"] = pd.to_datetime(df_2025["date_time_utc"])

month = df_2025["date_time_utc"].dt.month

df_2025["month_sin"] = np.sin(2 * np.pi * month / 12)
df_2025["month_cos"] = np.cos(2 * np.pi * month / 12)


# -----------------------
# Predict
# -----------------------
X_2025 = df_2025[FEATURES]

proba = model.predict_proba(X_2025)[:, 1]

pred = (proba >= threshold).astype(int)

df_2025["pred_fishing"] = pred
df_2025["pred_proba"] = proba


# -----------------------
# Save predictions
# -----------------------
df_2025.to_parquet(
    "predictions_2025.parquet",
    index=False
)

print(df_2025[["pred_fishing", "pred_proba"]].head())