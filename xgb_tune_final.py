import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from xgboost import XGBClassifier
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
import json
import gc

TUNING_FILES = ["three_months/feats_new_rule_bilstm/2023_1_3_feats.parquet", "three_months/feats_new_rule_bilstm/2023_7_9_feats.parquet"]  # Q1/Q3 2023

BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "log_dist", "ra_dcog", "log_dt"]

SEASON_FEATURES = ["month_sin", "month_cos"]

FEATURES = BASE_FEATURES + SEASON_FEATURES

TARGET = "y_train"

needed_cols = ["mmsi", "date_time_utc"] + BASE_FEATURES

def all_mmsis_in(files):
    s = set()
    for f in files:
        mmsis = pd.read_parquet(f, columns=["mmsi"])["mmsi"]
        mmsis = pd.to_numeric(mmsis, errors="coerce").dropna().astype("int64")
        s.update(mmsis.unique())
    return s

def get_global_val_test_mmsis(which, path="../../train_val_test_mmsis.csv"):
    split_df = pd.read_csv(path)
    split_df["mmsi"] = split_df["mmsi"].astype("int64")
    mmsis = set(split_df.loc[split_df["split"] == which,"mmsi"])
    return mmsis
 
# Validation and test mmsis so we dont use them for tuning
GLOB_val_mmsis = get_global_val_test_mmsis(which="validation")
GLOB_test_mmsis = get_global_val_test_mmsis(which="test")
all_mmsis_in_tuning = all_mmsis_in(TUNING_FILES)

tuning_mmsis = all_mmsis_in_tuning - GLOB_val_mmsis - GLOB_test_mmsis
assert tuning_mmsis.isdisjoint(GLOB_val_mmsis), "Tuning MMSIS include val MMSIs!"
assert tuning_mmsis.isdisjoint(GLOB_test_mmsis), "Tuning MMSIS include test MMSIs!"

print(f"MMSIs available for tuning after excluding val/test mmsis training file: {len(tuning_mmsis)}")

# Split tuning mmsis in 80/20 for tuning
tuning_mmsis = np.array(list(tuning_mmsis))
rng = np.random.default_rng(42)
rng.shuffle(tuning_mmsis)

# Split into train test and validation set by mmsi so that no vessel appear in both.
n = len(tuning_mmsis)
train_mmsis = set(tuning_mmsis[:int(0.80*n)])
val_mmsis   = set(tuning_mmsis[int(0.80*n):])

print(f"Train 80% of Q1 Q3 2023 vessels excluding the global val and test mmsis: {len(train_mmsis)} | Val 20% of the available vessels: {len(val_mmsis)}")
print(f"Are there vessels in both train and val?: "
      f"{len(train_mmsis & val_mmsis)}")

def add_monthly_and_extract_trainable(df):
    df = df[df["sample_weight"] == 1].copy()          # keep confident-label rows
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    month = df["date_time_utc"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)
    df[BASE_FEATURES] = df[BASE_FEATURES].astype(np.float32)
    df[TARGET] = df[TARGET].astype(np.int8)
    return df

val_mmsi_list = list(val_mmsis)
needed_cols = BASE_FEATURES + ["date_time_utc", TARGET, "sample_weight", "mmsi"]

train_df = pd.concat(
    [
        pd.read_parquet(
            f,
            engine="pyarrow",
            columns=needed_cols,
            filters=[("sample_weight", "=", 1)]
        )
        for f in TUNING_FILES
    ],
    ignore_index=True
)

val_df = pd.concat(
    [
        pd.read_parquet(
            f,
            engine="pyarrow",
            columns=needed_cols,
            filters=[
                ("mmsi", "in", val_mmsi_list),
                ("sample_weight", "=", 1),
            ]
        )
        for f in TUNING_FILES
    ],
    ignore_index=True
)

train_df = add_monthly_and_extract_trainable(train_df).dropna(subset=[TARGET])
val_df   = add_monthly_and_extract_trainable(val_df).dropna(subset=[TARGET])

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = val_df[FEATURES]
y_val = val_df[TARGET]

del train_df, val_df
gc.collect()

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
    refit="f1",          # or "average_precision" / "recall"
    cv=ps,
    n_jobs=1,
    verbose=2,
)

sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)
sample_weight_all = np.concatenate([
    sample_weight_train,
    np.ones(len(y_val), dtype=np.float32),
])

xgb_cv.fit(
    X_trainval,
    y_trainval,
    sample_weight=sample_weight_all,
)

print(xgb_cv.best_params_)
print(xgb_cv.best_score_)

result = {
    "best_params": xgb_cv.best_params_,
    "best_f1_val": float(xgb_cv.best_score_),
    "features": FEATURES,
    "tuning_files": TUNING_FILES,
}

with open("xgb_best_params_final.json", "w") as fp:
    json.dump(result, fp, indent=2)