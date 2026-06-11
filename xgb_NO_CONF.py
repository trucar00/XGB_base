from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from xgboost import plot_importance
import pandas as pd
import numpy as np
import gc
from pathlib import Path
import pandas as pd
import numpy as np
import json

model_path = Path(f"models/xgb_best_params_no_ra_leak.json")
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

if model_path.exists() == False:
    print("Model not tuned. No best params file exists: ", model_path)
    files = [
        "../../LSTM/three_months/feats_all_w_traps_online/2024_1_3_feats.parquet",
        "../../LSTM/three_months/feats_all_w_traps_online/2024_7_9_feats.parquet",
    ]



    all_mmsis = set()

    for f in files:
        m = pd.read_parquet(
            f,
            columns=["mmsi"],
            engine="pyarrow"
        )["mmsi"].dropna().unique()

        all_mmsis.update(m)

    mmsis = np.array(list(all_mmsis))

    rng = np.random.default_rng(42)
    rng.shuffle(mmsis)

    n = len(mmsis)

    train_mmsi = set(mmsis[: int(0.70 * n)])
    val_mmsi   = set(mmsis[int(0.70 * n): int(0.85 * n)])
    test_mmsi  = set(mmsis[int(0.85 * n):])

    train_parts = []
    val_parts = []
    test_parts = []

    for f in files:

        print("Reading", f)

        tmp = pd.read_parquet(
            f,
            columns=needed_cols,
            engine="pyarrow"
        )

        tmp["sample_weight"] = 1 # give all sample weight = 0, so now unknowns will be treated as conf_no_fishing because they have y_train = 0 such as conf_no_fishing

        tmp = tmp[tmp["sample_weight"] != 0].copy()

        tmp["date_time_utc"] = pd.to_datetime(tmp["date_time_utc"])

        month = tmp["date_time_utc"].dt.month

        tmp["month_sin"] = np.sin(2 * np.pi * month / 12)
        tmp["month_cos"] = np.cos(2 * np.pi * month / 12)

        tmp[BASE_FEATURES] = tmp[BASE_FEATURES].astype(np.float32)
        tmp["month_sin"] = tmp["month_sin"].astype(np.float32)
        tmp["month_cos"] = tmp["month_cos"].astype(np.float32)
        tmp[target] = tmp[target].astype(np.int8)

        train_parts.append(
            tmp[tmp["mmsi"].isin(train_mmsi)].copy()
        )

        val_parts.append(
            tmp[tmp["mmsi"].isin(val_mmsi)].copy()
        )

        test_parts.append(
            tmp[tmp["mmsi"].isin(test_mmsi)].copy()
        )

        del tmp
        gc.collect()

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df   = pd.concat(val_parts, ignore_index=True)
    test_df  = pd.concat(test_parts, ignore_index=True)

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
        "max_depth": [4, 6],
        "min_child_weight": [3],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [300],
        "subsample": [0.8],
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


    best_params = xgb_cv.best_params_
    print("Best params:")
    print(best_params)

    print(f"\nBest CV F1: {xgb_cv.best_score_:.4f}")

    with open(model_path, "w") as f:
        json.dump(best_params, f, indent=2)

#else
print("Best params already exist!")
with open(model_path, "r") as f:
    best_params = json.load(f)
# REFIT
refit_files = [
    "../../LSTM/three_months/feats_all_w_traps_online/2024_1_3_feats.parquet",
    "../../LSTM/three_months/feats_all_w_traps_online/2024_4_6_feats.parquet",
]

all_refit_mmsis = set()

for f in refit_files:
    m = pd.read_parquet(
        f,
        columns=["mmsi"],
        engine="pyarrow"
    )["mmsi"].dropna().unique()

    all_refit_mmsis.update(m)

refit_mmsis = np.array(list(all_refit_mmsis))

rng = np.random.default_rng(42)
rng.shuffle(refit_mmsis)

n = len(refit_mmsis)

train_mmsi_r = set(refit_mmsis[: int(0.70 * n)])
val_mmsi_r   = set(refit_mmsis[int(0.70 * n): int(0.85 * n)])
test_mmsi_r  = set(refit_mmsis[int(0.85 * n):])

train_parts_r = []
val_parts_r = []
test_parts_r = []

for f in refit_files:

    print("Reading", f)

    tmp = pd.read_parquet(
        f,
        columns=needed_cols,
        engine="pyarrow"
    )

    tmp["sample_weight"] = 1
    tmp = tmp[tmp["sample_weight"] != 0].copy()

    tmp["date_time_utc"] = pd.to_datetime(tmp["date_time_utc"])

    month = tmp["date_time_utc"].dt.month

    tmp["month_sin"] = np.sin(2 * np.pi * month / 12)
    tmp["month_cos"] = np.cos(2 * np.pi * month / 12)

    tmp[BASE_FEATURES] = tmp[BASE_FEATURES].astype(np.float32)
    tmp["month_sin"] = tmp["month_sin"].astype(np.float32)
    tmp["month_cos"] = tmp["month_cos"].astype(np.float32)
    tmp[target] = tmp[target].astype(np.int8)

    train_parts_r.append(
        tmp[tmp["mmsi"].isin(train_mmsi_r)].copy()
    )

    val_parts_r.append(
        tmp[tmp["mmsi"].isin(val_mmsi_r)].copy()
    )

    test_parts_r.append(
        tmp[tmp["mmsi"].isin(test_mmsi_r)].copy()
    )

    del tmp
    gc.collect()

train_df_r = pd.concat(train_parts_r, ignore_index=True)
val_df_r   = pd.concat(val_parts_r, ignore_index=True)
test_df_r  = pd.concat(test_parts_r, ignore_index=True)

train_df_r = train_df_r.dropna(subset=[target]).copy()
val_df_r   = val_df_r.dropna(subset=[target]).copy()
test_df_r  = test_df_r.dropna(subset=[target]).copy()

X_train_r = train_df_r[FEATURES]
y_train_r = train_df_r[target].astype(int)
groups_train_r = train_df_r["mmsi"].values

X_val_r = val_df_r[FEATURES]
y_val_r = val_df_r[target].astype(int)

X_test_r = test_df_r[FEATURES]
y_test_r = test_df_r[target].astype(int)

final_xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    **best_params,
)

sample_weight_train_r = compute_sample_weight(
    class_weight="balanced",
    y=y_train_r,
)

final_xgb.fit(
    X_train_r,
    y_train_r,
    sample_weight=sample_weight_train_r,
)

final_xgb.save_model("models/xgb_tuned_2024_1_6_no_ra_leak_NO_CONF.json")

y_pred = final_xgb.predict(X_test_r)
y_prob = final_xgb.predict_proba(X_test_r)[:, 1]

print("\n===== FINAL TEST RESULTS =====")
print(confusion_matrix(y_test_r, y_pred))

print(classification_report(
    y_test_r,
    y_pred,
    target_names=["not fishing", "fishing"],
    digits=4,
))

print(f"Precision: {precision_score(y_test_r, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test_r, y_pred):.4f}")
print(f"F1:        {f1_score(y_test_r, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test_r, y_prob):.4f}")
print(f"PR AUC:    {average_precision_score(y_test_r, y_prob):.4f}")
print(f"Log loss:  {log_loss(y_test_r, y_prob):.4f}")