import os
import json
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import random

print("Loading best params")
with open("xgb_best_params_final.json", "r") as f:
    config = json.load(f)
best_params = config["best_params"]
print(best_params)
 
BASE = "../../LSTM/three_months/feats_new_rule_online"

USE_CONF_LABELS = False
 
# TRAIN: all of 2023
TRAIN_FILES = [
    f"{BASE}/2023_1_3_feats.parquet",     # Q1 2023
    f"{BASE}/2023_4_6_feats.parquet",     # Q2 2023
    f"{BASE}/2023_7_9_feats.parquet",     # Q3 2023
    f"{BASE}/2023_10_12_feats.parquet",   # Q4 2023
]
 
# TEST: the held-out *test* vessels, in 2024
VAL_TEST_FILES = [
    f"{BASE}/2024_1_3_feats.parquet",     # Q1 2024
    f"{BASE}/2024_4_6_feats.parquet",     # Q2 2024
    f"{BASE}/2024_7_9_feats.parquet",     # Q3 2024
    f"{BASE}/2024_10_12_feats.parquet",   # Q4 2024
]
 
BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel",
                 "ra_jerk", "log_dist", "ra_dcog", "log_dt"]
SEASON_FEATURES = ["month_sin", "month_cos"]
FEATURES = BASE_FEATURES + SEASON_FEATURES
 
TARGET = "y_train"
needed_cols = ["mmsi", "date_time_utc", "sample_weight", TARGET] + BASE_FEATURES
 
SEEDS = [0, 1, 2, 3, 4]
THRESHOLD = 0.5

OUTPUT_DIR = "multi_seed_final"
results_csv_path = f"{OUTPUT_DIR}/xgb_train2023_test2024_unseen_no_conf_final.csv"
summary_csv_path = f"{OUTPUT_DIR}/xgb_train2023_test2024_unseen_summary_no_conf_final.csv"

importance_all_path = f"{OUTPUT_DIR}/xgb_feature_importance_all_seeds_final.csv"
importance_summary_path = f"{OUTPUT_DIR}/xgb_feature_importance_summary_final.csv"

def all_mmsis_in(files):
    s = set()
    for f in files:
        mmsis = pd.read_parquet(f, columns=["mmsi"])["mmsi"]
        mmsis = pd.to_numeric(mmsis, errors="coerce").dropna().astype("int64")
        s.update(mmsis.unique())
    return s

def get_global_val_test_mmsis(which, path="../train_val_test_mmsis_FINAL.csv"):
    split_df = pd.read_csv(path)
    split_df["mmsi"] = split_df["mmsi"].astype("int64")
    mmsis = set(split_df.loc[split_df["split"] == which,"mmsi"])
    return mmsis
 
# All vessels in each quarter (no MMSI split -- the split is by TIME).
val_mmsis = get_global_val_test_mmsis(which="validation")
test_mmsis = get_global_val_test_mmsis(which="test")
all_mmsis_in_train = all_mmsis_in(TRAIN_FILES)
train_mmsis = all_mmsis_in_train - val_mmsis - test_mmsis
assert train_mmsis.isdisjoint(val_mmsis), "Train/val MMSIs overlap!"
assert train_mmsis.isdisjoint(test_mmsis), "Train/test MMSIs overlap!"
print(f"Train (all 2023) vessels: {len(train_mmsis)} | Val (2024) vessels: {len(val_mmsis)} | Test (2024) vessels: {len(test_mmsis)}")


def load_feats(files, no_conf, mmsi_keep=None):
    """Read feature parquets, build season features, optionally filter by mmsi.
 
    labeled_only=True keeps only sample_weight == 1 rows, i.e. confident
    fishing / non-fishing. Unknowns (weight 0) are dropped, matching the
    'XGBoost simply ignores unknowns' rule from the thesis.
    """
    parts = []
    for f in files:
        print("Reading", f)
        tmp = pd.read_parquet(f, columns=needed_cols, engine="pyarrow")
 
        if mmsi_keep is not None:
            tmp["mmsi"] = tmp["mmsi"].astype("int64")
            tmp = tmp[tmp["mmsi"].isin(mmsi_keep)]
        if not no_conf:
            tmp = tmp[tmp["sample_weight"] == 1]
 
        tmp = tmp.dropna(subset=[TARGET])
 
        tmp["date_time_utc"] = pd.to_datetime(tmp["date_time_utc"])
        month = tmp["date_time_utc"].dt.month
        tmp["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
        tmp["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)
        tmp[BASE_FEATURES] = tmp[BASE_FEATURES].astype(np.float32)
        tmp[TARGET] = tmp[TARGET].astype(np.int8)
 
        parts.append(tmp[["mmsi", TARGET] + FEATURES].copy())
        del tmp
        gc.collect()
    return pd.concat(parts, ignore_index=True)

print("\nLoading TRAIN (2023)...")
train_df = load_feats(TRAIN_FILES, no_conf=USE_CONF_LABELS, mmsi_keep=train_mmsis)

print("\nLoading TEST (2024, test vessels)...")
random.seed(42)
train_mmsi_list_for_testing = random.sample(sorted(train_mmsis), k=len(train_mmsis) // 2)
test_unseen_df = load_feats(VAL_TEST_FILES, no_conf=USE_CONF_LABELS, mmsi_keep=test_mmsis) # UNSEEN VESSELS TEST, change to train_mmsis if we want SEEN vessels
test_seen_df = load_feats(VAL_TEST_FILES, no_conf=USE_CONF_LABELS, mmsi_keep=train_mmsi_list_for_testing) # SEEN VESSELS TEST, change to train_mmsis if we want SEEN vessels
 
X_train = train_df[FEATURES]
y_train = train_df[TARGET].astype(int)
X_test_unseen  = test_unseen_df[FEATURES]
y_test_unseen  = test_unseen_df[TARGET].astype(int)

X_test_seen  = test_seen_df[FEATURES]
y_test_seen  = test_seen_df[TARGET].astype(int)
 
print(f"\nTrain rows: {len(X_train):,} | pos {int(y_train.sum()):,} "
      f"({y_train.mean():.3%})")
print(f"Test  rows: {len(X_test_unseen):,} | pos {int(y_test_unseen.sum()):,} "
      f"({y_test_unseen.mean():.3%}) | vessels {test_unseen_df['mmsi'].nunique()}")
 
del train_df, test_unseen_df, test_seen_df
gc.collect()

sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)


done_seeds = set()
all_results = []
if os.path.exists(results_csv_path):
    try:
        existing = pd.read_csv(results_csv_path)
        done_seeds = set(existing["seed"].tolist())
        all_results = existing.to_dict("records")
        print(f"Resuming. Already-completed seeds: {sorted(done_seeds)}")
    except Exception as e:
        print(f"Could not read existing results ({e}); starting fresh.")
 
for seed in SEEDS:
    if seed in done_seeds:
        print(f"\n[seed {seed}] Already done. Skipping.")
        continue
 
    print(f"\n========== SEED {seed} ==========")
    np.random.seed(seed)
 
    final_xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **best_params,
    )

    final_xgb.fit(X_train, y_train, sample_weight=sample_weight_train)
    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": final_xgb.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        f"{OUTPUT_DIR}/xgb_feature_importance_seed_{seed}.csv",
        index=False,
    )

    y_pred_unseen = final_xgb.predict(X_test_unseen)
    y_prob_unseen = final_xgb.predict_proba(X_test_unseen)[:, 1]
    y_pred_seen = final_xgb.predict(X_test_seen)
    y_prob_seen = final_xgb.predict_proba(X_test_seen)[:, 1]
 
    precision_unseen = precision_score(y_test_unseen, y_pred_unseen)
    recall_unseen    = recall_score(y_test_unseen, y_pred_unseen)
    f1_unseen        = f1_score(y_test_unseen, y_pred_unseen)
    accuracy_unseen  = accuracy_score(y_test_unseen, y_pred_unseen)
    logloss_unseen   = log_loss(y_test_unseen, y_prob_unseen)
    rocauc_unseen    = roc_auc_score(y_test_unseen, y_prob_unseen)
    prauc_unseen     = average_precision_score(y_test_unseen, y_prob_unseen)

    precision_seen = precision_score(y_test_seen, y_pred_seen)
    recall_seen    = recall_score(y_test_seen, y_pred_seen)
    f1_seen        = f1_score(y_test_seen, y_pred_seen)
    accuracy_seen  = accuracy_score(y_test_seen, y_pred_seen)
    logloss_seen   = log_loss(y_test_seen, y_prob_seen)
    rocauc_seen    = roc_auc_score(y_test_seen, y_prob_seen)
    prauc_seen     = average_precision_score(y_test_seen, y_prob_seen)
 
    print(f"[seed {seed}] TEST UNSEEN 2024 | "
          f"p {precision_unseen:.3f} r {recall_unseen:.3f} f1 {f1_unseen:.3f} "
          f"acc {accuracy_unseen:.3f} logloss {logloss_unseen:.4f} "
          f"rocauc {rocauc_unseen:.3f} prauc {prauc_unseen:.3f}")
    
    print(f"[seed {seed}] TEST SEEN 2024 | "
          f"p {precision_seen:.3f} r {recall_seen:.3f} f1 {f1_seen:.3f} "
          f"acc {accuracy_seen:.3f} logloss {logloss_seen:.4f} "
          f"rocauc {rocauc_seen:.3f} prauc {prauc_seen:.3f}")
 
    all_results.append({
        "seed":      seed,

        "precision_unseen": precision_unseen,
        "recall_unseen":    recall_unseen,
        "f1_unseen":        f1_unseen,
        "accuracy_unseen":  accuracy_unseen,
        "logloss_unseen":   logloss_unseen,
        "rocauc_unseen":    rocauc_unseen,
        "prauc_unseen":     prauc_unseen,
   
        "precision_seen": precision_seen,
        "recall_seen":    recall_seen,
        "f1_seen":        f1_seen,
        "accuracy_seen":  accuracy_seen,
        "logloss_seen":   logloss_seen,
        "rocauc_seen":    rocauc_seen,
        "prauc_seen":     prauc_seen,
    })
 
    # Save incrementally so a crash doesn't lose everything.
    pd.DataFrame(all_results).to_csv(results_csv_path, index=False)
 
 
# ============================================================
# Summary across seeds
# ============================================================
 
df_res = pd.DataFrame(all_results)
print("\n========== SUMMARY ==========")
print(df_res.to_string(index=False))
 
metric_cols = ["f1_unseen", "precision_unseen", "recall_unseen", "accuracy_unseen",
               "logloss_unseen", "rocauc_unseen", "prauc_unseen", 
               "f1_seen", "precision_seen", "recall_seen", "accuracy_seen",
               "logloss_seen", "rocauc_seen", "prauc_seen",]
summary = df_res[metric_cols].agg(["mean", "std"]).T
summary.columns = ["mean", "std"]
print("\nMean / Std across seeds:")
print(summary)
summary.to_csv(summary_csv_path)
print(f"\nPer-seed rows: {results_csv_path}")
print(f"Summary:       {summary_csv_path}")