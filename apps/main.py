# app/main.py — FastAPI fraud detection service

from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

app = FastAPI(title="Fraud Detection API")

BASE = Path(__file__).parent.parent / "model"

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
    # build dataframe with correct columns
    # missing columns filled with defaults
    num_data = {col: [transaction.get(col, 0.0)] for col in NUM_COLS}
    cat_data = {col: [transaction.get(col, "missing")] for col in CAT_COLS}
    
    input_df = pd.DataFrame({**num_data, **cat_data})
    
    # preprocess + predict
    input_processed = preprocessor.transform(input_df)
    proba = model.predict_proba(input_processed)[0][1]
    prediction = int(proba >= 0.5)
    
    return {
        "fraud_probability": round(float(proba), 4),
        "is_fraud": bool(prediction),
        "risk_level": "HIGH" if proba >= 0.7 else "MEDIUM" if proba >= 0.3 else "LOW"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
