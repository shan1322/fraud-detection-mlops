from fastapi import FastAPI
from typing import Dict, Any
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import csv
import os

app = FastAPI(title="Fraud Detection API")

BASE = Path(__file__).parent.parent / "models"
PREDICTIONS_FILE = "incoming_predictions.csv"

with open(BASE / "model.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE / "preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open(BASE / "feature_metadata.json", "r") as f:
    metadata = json.load(f)

NUM_COLS = metadata["num_cols"]
CAT_COLS = metadata["cat_cols"]

@app.get("/")
def root():
    return {
        "service": "Fraud Detection API",
        "model": metadata["best_model"],
        "auc_roc": metadata["auc_roc"],
        "status": "running"
    }

@app.post("/predict")
def predict(transaction: Dict[str, Any]):
    num_data = {col: [transaction.get(col, 0.0)] for col in NUM_COLS}
    cat_data = {col: [transaction.get(col, "missing")] for col in CAT_COLS}
    input_df = pd.DataFrame({**num_data, **cat_data})

    input_processed = preprocessor.transform(input_df)
    proba = float(model.predict_proba(input_processed)[0][1])
    prediction = int(proba >= 0.5)

    log_row = {
        "timestamp": datetime.now().isoformat(),
        "fraud_probability": round(proba, 4),
        "is_fraud_predicted": prediction,
        "risk_level": "HIGH" if proba >= 0.7 else "MEDIUM" if proba >= 0.3 else "LOW",
        "TransactionAmt": transaction.get("TransactionAmt", None),
        "ProductCD": transaction.get("ProductCD", None),
        "card4": transaction.get("card4", None),
    }
    file_exists = os.path.exists(PREDICTIONS_FILE)
    with open(PREDICTIONS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_row)

    return {
        "fraud_probability": round(proba, 4),
        "is_fraud": bool(prediction),
        "risk_level": log_row["risk_level"]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}