import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from xgboost import XGBClassifier
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
import json

TRAIN_FILES = ["../../LSTM/three_months/feats_all_w_traps_online/2023_1_3_feats.parquet"]  # Q1 2024
VAL_FILES   = ["../../LSTM/three_months/feats_all_w_traps_online/2024_1_3_feats.parquet"]  # Q1 2024

BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk",
                 "log_dist", "ra_dcog", "log_dt"]

SEASON_FEATURES = ["month_sin", "month_cos"]
FEATURES = BASE_FEATURES + SEASON_FEATURES
TARGET = "y_train"

def get_val_test_mmsis(test_or_val, path="../../split_mmsis_val_test.csv"):
    split_df = pd.read_csv(path)
    return set(split_df.loc[split_df["split"] == test_or_val, "mmsi"])

val_mmsi = get_val_test_mmsis(test_or_val="validation")

def add_monthly_and_extract_trainable(df):
    df = df[df["sample_weight"] == 1].copy()          # keep confident-label rows
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    month = df["date_time_utc"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)
    df[BASE_FEATURES] = df[BASE_FEATURES].astype(np.float32)
    df[TARGET] = df[TARGET].astype(np.int8)
    return df


train_df = pd.concat((pd.read_parquet(f, engine="pyarrow") for f in TRAIN_FILES),
                     ignore_index=True)

val_mmsi_list = list(val_mmsi)
dfs = []
for f in VAL_FILES:
    df_part = pd.read_parquet(
        f,
        engine="pyarrow",
        filters=[("mmsi", "in", val_mmsi_list)]
    )
    dfs.append(df_part)

val_df = pd.concat(dfs, ignore_index=True)

train_df = add_monthly_and_extract_trainable(train_df).dropna(subset=[TARGET])
val_df   = add_monthly_and_extract_trainable(val_df).dropna(subset=[TARGET])

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = val_df[FEATURES]
y_val = val_df[TARGET]

# Combine train + val
X_trainval = pd.concat([X_train, X_val], ignore_index=True)
y_trainval = pd.concat([y_train, y_val], ignore_index=True)

# Tell GridSearch which rows are train and which are validation
test_fold = [-1] * len(X_train) + [0] * len(X_val)
ps = PredefinedSplit(test_fold)

cv_params = {
    "max_depth": [4, 6],
    "min_child_weight": [3],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [300],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

scoring = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}

xgb_cv = GridSearchCV(
    estimator=xgb,
    param_grid=cv_params,
    scoring=scoring,
    refit=False,          # or "average_precision" / "recall"
    cv=ps,
    n_jobs=1,
    verbose=2,
)

sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)
sample_weight_all = np.concatenate([
    sample_weight_train,
    np.ones(len(y_val), dtype=np.float64),   # val rows: only scored, never fit
])

xgb_cv.fit(
    X_trainval,
    y_trainval,
    sample_weight=sample_weight_train,
)

print(xgb_cv.best_params_)
print(xgb_cv.best_score_)

result = {
    "best_params": xgb_cv.best_params_,
    "best_f1_val": float(xgb_cv.best_score_),
    "features": FEATURES,
    "train_files": TRAIN_FILES,
    "val_files": VAL_FILES,
}
with open("xgb_best_params.json", "w") as fp:
    json.dump(result, fp, indent=2)