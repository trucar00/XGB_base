import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

KNOWN_REPORTS = ["fishing", "conf_no_fishing"]

def evaluate(df, name):
    # Keep only rows with known true labels
    df_eval = df[df["report"].isin(KNOWN_REPORTS)].copy()

    y_true = (df_eval["report"] == "fishing").astype(int)
    y_pred = df_eval["pred_fishing"].astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"\n===== {name} =====")
    print(f"Evaluation rows: {len(df_eval):,}")
    print(f"Fishing rows: {(y_true == 1).sum():,}")
    print(f"Confident no-fishing rows: {(y_true == 0).sum():,}")

    print(f"\nTP: {tp:,}")
    print(f"FP: {fp:,}")
    print(f"TN: {tn:,}")
    print(f"FN: {fn:,}")

    print(f"\nAccuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1       : {f1_score(y_true, y_pred):.4f}")

    print("\nClassification report")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["confident no fishing", "fishing"],
        digits=4
    ))


df_lstm = pd.read_parquet(
    "predictions/2025_1_3_w_2024_1_3_4_6_model_tuned.parquet",
    engine="pyarrow",
    columns=["pred_fishing", "report"]
)

df_xgb = pd.read_parquet(
    "predictions/predictions_2025.parquet",
    engine="pyarrow",
    columns=["pred_fishing", "report"]
)

evaluate(df_lstm, "LSTM")
evaluate(df_xgb, "XGB")