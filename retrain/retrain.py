# retrain/retrain.py
import pandas as pd
import numpy as np
import pickle
import json
import mlflow
import mlflow.xgboost
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from mlflow.models.signature import infer_signature

BASE = Path(__file__).parent.parent / "models"
DRIFT_RESULT_FILE = Path(__file__).parent.parent / "drift_result.json"
DATA_FILE = BASE / "train_filtered.csv"
SEED = 42

def retrain():
    if not DRIFT_RESULT_FILE.exists():
        print("No drift result found — run monitor.py first")
        return

    with open(DRIFT_RESULT_FILE) as f:
        drift_result = json.load(f)

    drift_detected = drift_result["drift_detected"]

    if not drift_detected:
        print("No drift detected — skipping retrain")
        return

    print("Drift detected — starting retrain...")

    if not DATA_FILE.exists():
        print("train_filtered.csv not found in models/")
        return

    with open(BASE / "feature_metadata.json") as f:
        metadata = json.load(f)

    NUM_COLS = metadata["num_cols"]
    CAT_COLS = metadata["cat_cols"]
    TARGET = "isFraud"

    print("Loading training data...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    available_num = [c for c in NUM_COLS if c in df.columns]
    available_cat = [c for c in CAT_COLS if c in df.columns]

    X = df[available_num + available_cat]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    with open(BASE / "preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    scale_pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())
    print(f"scale_pos_weight: {scale_pos_weight}")

    mlflow.set_experiment("fraud-detection-retrain")

    with mlflow.start_run(run_name="retrain-on-drift"):

        mlflow.log_param("trigger", "drift_detected")
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("seed", SEED)

        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1.5,
            scale_pos_weight=scale_pos_weight,
            random_state=SEED,
            eval_metric="auc",
            early_stopping_rounds=30,
            verbosity=0
        )

        model.fit(
            X_train_processed, y_train,
            eval_set=[(X_val_processed, y_val)],
            verbose=50
        )

        y_pred = model.predict(X_val_processed)
        y_pred_proba = model.predict_proba(X_val_processed)[:, 1]

        metrics = {
            "auc_roc": round(roc_auc_score(y_val, y_pred_proba), 4),
            "f1": round(f1_score(y_val, y_pred), 4),
            "precision": round(precision_score(y_val, y_pred), 4),
            "recall": round(recall_score(y_val, y_pred), 4)
        }
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train_processed, y_pred)
        mlflow.xgboost.log_model(model, "model", signature=signature)

        with open(BASE / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        with open(BASE / "preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)

        metadata["auc_roc"] = metrics["auc_roc"]
        with open(BASE / "feature_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\n=== RETRAIN COMPLETE ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("New model.pkl saved to models/")

if __name__ == "__main__":
    retrain()