from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from xgboost import plot_importance
import pandas as pd
import numpy as np

from pathlib import Path
import pandas as pd
import numpy as np

files = [
    "../../LSTM/three_months/feats_all_gear2/2024_1_3_feats.parquet",
    "../../LSTM/three_months/feats_all_gear2/2024_4_6_feats.parquet",
    "../../LSTM/three_months/feats_all_gear2/2024_7_9_feats.parquet",
    "../../LSTM/three_months/feats_all_gear2/2024_10_12_feats.parquet",
]

BASE_FEATURES = [
    "cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk",
    "log_dist", "ra_dcog", "log_dt", "dist_to_shore_km"
]
SEASON_FEATURES = ["month_sin", "month_cos"]
FEATURES = BASE_FEATURES + SEASON_FEATURES
target = "y_train"

needed_cols = [
    "mmsi",
    "date_time_utc",
    "sample_weight",
    target,
] + BASE_FEATURES


dfs = []

for f in files:
    print("Reading", f)

    tmp = pd.read_parquet(
        f,
        columns=needed_cols,
        engine="pyarrow"
    )

    # Keep only fishing / confident no-fishing
    tmp = tmp[tmp["sample_weight"] != 0].copy()

    tmp["date_time_utc"] = pd.to_datetime(tmp["date_time_utc"])
    month = tmp["date_time_utc"].dt.month

    tmp["month_sin"] = np.sin(2 * np.pi * month / 12)
    tmp["month_cos"] = np.cos(2 * np.pi * month / 12)

    dfs.append(tmp)


df = pd.concat(dfs, ignore_index=True)

print("Total rows:", len(df))
print("Unique vessels:", df["mmsi"].nunique())
print(df[target].value_counts())

# Split into train test and validation set by mmsi so that no vessel appear in both.
rng = np.random.default_rng(42)
mmsis = df["mmsi"].unique().copy()
rng.shuffle(mmsis)
n = len(mmsis)
train_mmsi = set(mmsis[: int(0.70 * n)])
val_mmsi   = set(mmsis[int(0.70 * n) : int(0.85 * n)])
test_mmsi  = set(mmsis[int(0.85 * n) :])


train_df = df[df["mmsi"].isin(train_mmsi)]
val_df = df[df["mmsi"].isin(val_mmsi)]
test_df = df[df["mmsi"].isin(test_mmsi)]


train_df = train_df.dropna(subset=[target]).copy()
val_df   = val_df.dropna(subset=[target]).copy()
test_df  = test_df.dropna(subset=[target]).copy()

X_train = train_df[FEATURES]
y_train = train_df[target].astype(int)
groups_train = train_df["mmsi"].values

X_val = val_df[FEATURES]
y_val = val_df[target].astype(int)

X_test = test_df[FEATURES]
y_test = test_df[target].astype(int)


# -----------------------
# XGBoost base model
# -----------------------
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)


# -----------------------
# Parameter grid
# -----------------------
cv_params = {
    "max_depth":        [4, 6, 8],
    "min_child_weight": [1, 3, 5],
    "learning_rate":    [0.05, 0.1, 0.2],
    "n_estimators":     [200, 400, 600],
    "subsample":        [0.8],
    "colsample_bytree": [0.8],
}


# -----------------------
# Grouped cross-validation
# -----------------------
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
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
    cv=cv,
    n_jobs=-1,
    verbose=2,
)

# -----------------------
# Class imbalance weights
# -----------------------
sample_weight_train = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

# -----------------------
# Fit grid search
# -----------------------
xgb_cv.fit(
    X_train,
    y_train,
    groups=groups_train,
    sample_weight=sample_weight_train,
)



print("Best params:")
print(xgb_cv.best_params_)

print(f"\nBest CV F1: {xgb_cv.best_score_:.4f}")

# -----------------------
# Evaluate on validation set
# -----------------------
best_model = xgb_cv.best_estimator_

val_proba = best_model.predict_proba(X_val)[:, 1]
val_pred = (val_proba >= 0.5).astype(int)

print("\nValidation results")
print(confusion_matrix(y_val, val_pred))
print(classification_report(y_val, val_pred, digits=3))

print("ROC AUC:", roc_auc_score(y_val, val_proba))
print("PR AUC:", average_precision_score(y_val, val_proba))


# -----------------------
# Evaluate on test set
# -----------------------
test_proba = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= 0.5).astype(int)

print("\nTest results")
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred, digits=3))

print("ROC AUC:", roc_auc_score(y_test, test_proba))
print("PR AUC:", average_precision_score(y_test, test_proba))


print("Validation log loss:", log_loss(y_val, val_proba))
print("Test log loss:", log_loss(y_test, test_proba))

# -----------------------
# Feature importance
# -----------------------
""" fig, ax = plt.subplots(figsize=(8, 6))
plot_importance(
    best_model,
    importance_type="gain",
    max_num_features=20,
    ax=ax,
)
plt.tight_layout()
plt.show() """