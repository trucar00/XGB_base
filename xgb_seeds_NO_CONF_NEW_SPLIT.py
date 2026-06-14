import os
import json
import gc
from pathlib import Path

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

# ============================================================
# Config
# ============================================================

model_path = Path("models/xgb_best_params_no_ra_leak.json")
# TRAIN: all of 2023
BASE = "../../three_months/feats_new_rule_online"
TRAIN_FILES = [
    f"{BASE}/2023_1_3_feats.parquet",     # Q1 2023
    f"{BASE}/2023_4_6_feats.parquet",     # Q2 2023
    f"{BASE}/2023_7_9_feats.parquet",     # Q3 2023
    f"{BASE}/2023_10_12_feats.parquet",   # Q4 2023
]

# VALIDATION and TEST on 2024. We have MMSIS for validation and MMSIS for testing
VAL_TEST_FILES = [
    f"{BASE}/2024_1_3_feats.parquet",     # Q1 2024
    f"{BASE}/2024_4_6_feats.parquet",     # Q2 2024
    f"{BASE}/2024_7_9_feats.parquet",     # Q3 2024
    f"{BASE}/2024_10_12_feats.parquet",   # Q4 2024
]


BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "log_dist", "ra_dcog", "log_dt"]

SEASON_FEATURES = ["month_sin", "month_cos"]

FEATURES = BASE_FEATURES + SEASON_FEATURES

needed_cols = ["mmsi", "date_time_utc", "sample_weight", target] + BASE_FEATURES

print("Loading best params...")
with open(model_path, "r") as f:
    best_params = json.load(f)

SEEDS = [0, 1, 2, 3, 4]
THRESHOLD = 0.5

results_csv_path = "multi_seed_results/xgb_seed_results_NO_CONF_NEW_SPLIT.csv"

# ============================================================
# MMSI split — FIXED across seeds (rng=42), mirrors LSTM script
# ============================================================

all_refit_mmsis = set()
for f in TRAIN_FILES:
    m = pd.read_parquet(f, columns=["mmsi"], engine="pyarrow")["mmsi"].dropna().unique()
    all_refit_mmsis.update(m)
refit_mmsis = np.array(list(all_refit_mmsis))
split_rng = np.random.default_rng(42)
split_rng.shuffle(refit_mmsis)
n = len(refit_mmsis)
train_mmsi_r = set(refit_mmsis[:int(0.70 * n)])
val_mmsi_r   = set(refit_mmsis[int(0.70 * n):int(0.85 * n)])
test_mmsi_r  = set(refit_mmsis[int(0.85 * n):])

# ============================================================
# Load and split training data ONCE (seed-independent)
# ============================================================

train_parts_r, val_parts_r, test_parts_r = [], [], []
for f in train_files:
    print("Reading", f)
    tmp = pd.read_parquet(f, columns=needed_cols, engine="pyarrow")
    tmp["sample_weight"] = 1
    tmp = tmp[tmp["sample_weight"] == 1].copy()
    tmp["date_time_utc"] = pd.to_datetime(tmp["date_time_utc"])
    month = tmp["date_time_utc"].dt.month
    tmp["month_sin"] = np.sin(2 * np.pi * month / 12)
    tmp["month_cos"] = np.cos(2 * np.pi * month / 12)
    tmp[BASE_FEATURES] = tmp[BASE_FEATURES].astype(np.float32)
    tmp["month_sin"]   = tmp["month_sin"].astype(np.float32)
    tmp["month_cos"]   = tmp["month_cos"].astype(np.float32)
    tmp[target]        = tmp[target].astype(np.int8)
    train_parts_r.append(tmp[tmp["mmsi"].isin(train_mmsi_r)].copy())
    val_parts_r.append(  tmp[tmp["mmsi"].isin(val_mmsi_r)].copy())
    test_parts_r.append( tmp[tmp["mmsi"].isin(test_mmsi_r)].copy())
    del tmp
    gc.collect()

train_df_r = pd.concat(train_parts_r, ignore_index=True).dropna(subset=[target]).copy()
test_df_r  = pd.concat(test_parts_r,  ignore_index=True).dropna(subset=[target]).copy()

X_train_r = train_df_r[FEATURES]
y_train_r = train_df_r[target].astype(int)
X_test_r  = test_df_r[FEATURES]
y_test_r  = test_df_r[target].astype(int)

# Sample weights are deterministic given y_train_r — compute once.
sample_weight_train_r = compute_sample_weight(class_weight="balanced", y=y_train_r)

neg_train = int((y_train_r == 0).sum())
pos_train = int((y_train_r == 1).sum())
pos_weight_value = neg_train / max(pos_train, 1)
print(f"pos_weight (for LSTM-comparable ext_logloss): {pos_weight_value:.4f}")

# ============================================================
# Pre-load 2025 external test set ONCE
# ============================================================

print("Loading 2025 external test set...")
df_2025 = pd.read_parquet(external_test_file, engine="pyarrow")
df_2025["date_time_utc"] = pd.to_datetime(df_2025["date_time_utc"])
_m = df_2025["date_time_utc"].dt.month
df_2025["month_sin"] = np.sin(2 * np.pi * _m / 12)
df_2025["month_cos"] = np.cos(2 * np.pi * _m / 12)

df_2025["sample_weight"] = 1

# ============================================================
# External-prediction + report-based scoring
# ============================================================

def predict_and_score_external(model, df_in, threshold=THRESHOLD, pos_weight_value=1.0):
    df = df_in.copy()
    proba = model.predict_proba(df[FEATURES])[:, 1]
    df["pred_proba"] = proba
    df["pred_fishing"] = (proba >= threshold).astype(int)

    # ----- Pos-weighted BCE on labeled rows (matches LSTM ext_loss) -----
    labeled = df[df["sample_weight"] != 0]
    if len(labeled) > 0:
        y_true = labeled["y_train"].astype(int).to_numpy()
        p = np.clip(labeled["pred_proba"].to_numpy(), 1e-7, 1.0 - 1e-7)
        w = np.where(y_true == 1, pos_weight_value, 1.0)
        per_sample = -(w * (y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
        ext_logloss      = float(per_sample.mean())
        ext_logloss_unw  = float(log_loss(y_true, p))   # plain sklearn, for reference
        ext_n_labeled    = int(len(labeled))
    else:
        ext_logloss = ext_logloss_unw = float("nan")
        ext_n_labeled = 0

    pred_pos_df = df[df["pred_fishing"] == 1]
    tp = int((pred_pos_df["report"] == "fishing").sum())
    fp = int(pred_pos_df["report"].isin(["conf_no_fishing", "unknown"]).sum())

    pred_neg_df = df[df["pred_fishing"] == 0]
    tn = int(pred_neg_df["report"].isin(["conf_no_fishing", "unknown"]).sum())
    fn = int((pred_neg_df["report"] == "fishing").sum())

    precision   = tp / (tp + fp)               if (tp + fp) > 0 else 0.0
    accuracy    = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    recall      = tp / (tp + fn)               if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp)               if (tn + fp) > 0 else 0.0
    f1          = 2 * (precision * recall) / (precision + recall) \
                  if (precision + recall) > 0 else 0.0

    n_pred_fish             = int((df["pred_fishing"] == 1).sum())
    n_pred_no_fish          = int((df["pred_fishing"] == 0).sum())
    n_reported_fish         = int((df["report"] == "fishing").sum())
    n_reported_conf_no_fish = int((df["report"] == "conf_no_fishing").sum())

    unknown_df                  = df[df["report"] == "unknown"]
    n_unknown                   = int(len(unknown_df))
    n_pred_fish_of_unknown      = int((unknown_df["pred_fishing"] == 1).sum())
    n_pred_no_fish_of_unknown   = int((unknown_df["pred_fishing"] == 0).sum())

    return {
        "ext_logloss":        ext_logloss,         # pos-weighted, matches LSTM ext_loss
        "ext_logloss_unw":    ext_logloss_unw,     # plain BCE, for reference
        "ext_n_labeled":      ext_n_labeled,
        "ext_tp":                        tp,
        "ext_fp":                        fp,
        "ext_tn":                        tn,
        "ext_fn":                        fn,
        "ext_accuracy":                  accuracy,
        "ext_recall":                    recall,
        "ext_specificity":               specificity,
        "ext_precision":                 precision,
        "ext_f1":                        f1,
        "ext_n_pred_fish":               n_pred_fish,
        "ext_n_pred_no_fish":            n_pred_no_fish,
        "ext_n_reported_fish":           n_reported_fish,
        "ext_n_reported_no_fish":        n_reported_conf_no_fish,
        "ext_n_unknowns":                n_unknown,
        "ext_n_pred_fish_of_unknown":    n_pred_fish_of_unknown,
        "ext_n_pred_no_fish_of_unknown": n_pred_no_fish_of_unknown,
    }


# ============================================================
# Multi-seed loop with resume support
# ============================================================

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

    final_xgb.fit(X_train_r, y_train_r, sample_weight=sample_weight_train_r)
    #final_xgb.save_model(f"models/xgb_seed{seed}_no_ra_leak_NO_CONF.json")

    # ----- Internal 15% test split -----
    y_pred = final_xgb.predict(X_test_r)
    y_prob = final_xgb.predict_proba(X_test_r)[:, 1]

    int_precision = precision_score(y_test_r, y_pred)
    int_recall    = recall_score(y_test_r, y_pred)
    int_f1        = f1_score(y_test_r, y_pred)
    int_accuracy  = accuracy_score(y_test_r, y_pred)
    int_logloss   = log_loss(y_test_r, y_prob)
    int_rocauc    = roc_auc_score(y_test_r, y_prob)
    int_prauc     = average_precision_score(y_test_r, y_prob)

    print(f"[seed {seed}] INTERNAL TEST | "
          f"p {int_precision:.3f} r {int_recall:.3f} "
          f"f1 {int_f1:.3f} acc {int_accuracy:.3f} "
          f"logloss {int_logloss:.4f}")

    # ----- External 2025 test -----
    ext = predict_and_score_external(final_xgb, df_2025, pos_weight_value=pos_weight_value)
    print(f"[seed {seed}] EXTERNAL 2025 | "
          f"logloss {ext['ext_logloss']:.4f} "
          f"precision {ext['ext_precision']:.3f} "
          f"recall {ext['ext_recall']:.3f} "
          f"specificity {ext['ext_specificity']:.3f} "
          f"f1 {ext['ext_f1']:.3f} "
          f"accuracy {ext['ext_accuracy']:.3f}")

    row = {
        "seed": seed,
        "int_precision": int_precision,
        "int_recall":    int_recall,
        "int_f1":        int_f1,
        "int_accuracy":  int_accuracy,
        "int_logloss":   int_logloss,
        "int_rocauc":    int_rocauc,
        "int_prauc":     int_prauc,
        **ext,
    }
    all_results.append(row)

    # Save incrementally so a crash doesn't lose everything
    pd.DataFrame(all_results).to_csv(results_csv_path, index=False)


# ============================================================
# Summary across seeds
# ============================================================

df_res = pd.DataFrame(all_results)
print("\n========== SUMMARY ==========")
print(df_res.to_string(index=False))

metric_cols = [
    "int_f1", "int_precision", "int_recall", "int_accuracy", "int_logloss",
    "ext_logloss", "ext_f1", "ext_precision", "ext_recall", "ext_accuracy", "ext_specificity",
]
summary = df_res[metric_cols].agg(["mean", "std"]).T
summary.columns = ["mean", "std"]
print("\nMean / Std across seeds:")
print(summary)
summary.to_csv("multi_seed_results/xgb_seed_results_summary_NO_CONF_NO_DIST_NEW_RULE.csv")
print(f"\nPer-seed rows: {results_csv_path}")
print(f"Summary:       multi_seed_results/xgb_seed_results_summary_NO_CONF_NO_DIST_NEW_RULE.csv")