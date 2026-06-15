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
with open("xgb_best_params.json", "r") as f:
    config = json.load(f)
best_params = config["best_params"]
print(best_params)
 
BASE = "../../LSTM/three_months/feats_new_rule_online"

CONF = False
 
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

results_csv_path = "multi_seed_final/xgb_train2023_test2024.csv"
summary_csv_path = "multi_seed_final/xgb_train2023_test2024_summary.csv"

def all_mmsis_in(files):
    s = set()
    for f in files:
        s.update(pd.read_parquet(f, columns=["mmsi"])["mmsi"].unique())
    return s

def get_global_val_test_mmsis(which, path="../train_val_test_mmsis.csv"):
    split_df = pd.read_csv(path)
    split_df["mmsi"] = split_df["mmsi"].astype("int64")
    return set(split_df.loc[split_df["split"] == which, "mmsi"])
 
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
            tmp = tmp[tmp["mmsi"].isin(mmsi_keep)]
        if no_conf:
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
train_df = load_feats(TRAIN_FILES, no_conf=CONF, mmsi_keep=train_mmsis)

print("\nLoading TEST (2024, test vessels)...")
random.seed(42)
train_mmsi_list = random.sample(sorted(train_mmsis), k=len(train_mmsis) // 2)
test_df = load_feats(VAL_TEST_FILES, no_conf=CONF, mmsi_keep=test_mmsis) # UNSEEN VESSELS TEST, change to train_mmsis if we want SEEN vessels
 
X_train = train_df[FEATURES]
y_train = train_df[TARGET].astype(int)
X_test  = test_df[FEATURES]
y_test  = test_df[TARGET].astype(int)
 
print(f"\nTrain rows: {len(X_train):,} | pos {int(y_train.sum()):,} "
      f"({y_train.mean():.3%})")
print(f"Test  rows: {len(X_test):,} | pos {int(y_test.sum()):,} "
      f"({y_test.mean():.3%}) | vessels {test_df['mmsi'].nunique()}")
 
del train_df, test_df
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
    # final_xgb.save_model(f"models/xgb_train2023_seed{seed}.json")
 
    y_pred = final_xgb.predict(X_test)
    y_prob = final_xgb.predict_proba(X_test)[:, 1]
 
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    accuracy  = accuracy_score(y_test, y_pred)
    logloss   = log_loss(y_test, y_prob)
    rocauc    = roc_auc_score(y_test, y_prob)
    prauc     = average_precision_score(y_test, y_prob)
 
    print(f"[seed {seed}] TEST 2024 | "
          f"p {precision:.3f} r {recall:.3f} f1 {f1:.3f} "
          f"acc {accuracy:.3f} logloss {logloss:.4f} "
          f"rocauc {rocauc:.3f} prauc {prauc:.3f}")
 
    all_results.append({
        "seed":      seed,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
        "logloss":   logloss,
        "rocauc":    rocauc,
        "prauc":     prauc,
    })
 
    # Save incrementally so a crash doesn't lose everything.
    pd.DataFrame(all_results).to_csv(results_csv_path, index=False)
 
 
# ============================================================
# Summary across seeds
# ============================================================
 
df_res = pd.DataFrame(all_results)
print("\n========== SUMMARY ==========")
print(df_res.to_string(index=False))
 
metric_cols = ["f1", "precision", "recall", "accuracy",
               "logloss", "rocauc", "prauc"]
summary = df_res[metric_cols].agg(["mean", "std"]).T
summary.columns = ["mean", "std"]
print("\nMean / Std across seeds:")
print(summary)
summary.to_csv(summary_csv_path)
print(f"\nPer-seed rows: {results_csv_path}")
print(f"Summary:       {summary_csv_path}")