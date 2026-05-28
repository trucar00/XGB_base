import pandas as pd
from xgboost import XGBClassifier
from xgboost import plot_importance
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt


df = pd.read_parquet("data/2024_1_3_feats.parquet", engine="pyarrow")
df = df[df["sample_weight"] != 0].copy()
df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
month = df["date_time_utc"].dt.month

df["month_sin"] = np.sin(2 * np.pi * month / 12)
df["month_cos"] = np.cos(2 * np.pi * month / 12)

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


BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "log_dist", "ra_dcog", "log_dt", "dist_to_shore_km"]
SEASON_FEATURES = ["month_sin", "month_cos"]
FEATURES = BASE_FEATURES + SEASON_FEATURES

target = "y_train"

train_df = train_df.dropna(subset=[target])
val_df   = val_df.dropna(subset=[target])
test_df  = test_df.dropna(subset=[target])

X_train = train_df[FEATURES]
y_train = train_df[target].astype(int)

X_val = val_df[FEATURES]
y_val = val_df[target].astype(int)

X_test = test_df[FEATURES]
y_test = test_df[target].astype(int)

n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()

scale_pos_weight = n_neg / n_pos

print("Train positives:", n_pos)
print("Train negatives:", n_neg)
print("scale_pos_weight:", scale_pos_weight)


model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",

    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,

    subsample=0.8,
    colsample_bytree=0.8,

    scale_pos_weight=scale_pos_weight,

    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)


# -----------------------
# Evaluate on validation
# -----------------------
val_proba = model.predict_proba(X_val)[:, 1]
val_pred = (val_proba >= 0.5).astype(int)

print("\nValidation results")
print(confusion_matrix(y_val, val_pred))
print(classification_report(y_val, val_pred, digits=3))

print("ROC AUC:", roc_auc_score(y_val, val_proba))
print("PR AUC:", average_precision_score(y_val, val_proba))


# -----------------------
# Evaluate on test
# -----------------------
test_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= 0.5).astype(int)

print("\nTest results")
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred, digits=3))

print("ROC AUC:", roc_auc_score(y_test, test_proba))
print("PR AUC:", average_precision_score(y_test, test_proba))


# -----------------------
# Feature importance
# -----------------------
plot_importance(model, max_num_features=20)
plt.tight_layout()
plt.show()