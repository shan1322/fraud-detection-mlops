from fastapi import FastAPI
from typing import Dict, Any
import pickle
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import csv
import os
from github import Github

app = FastAPI(title="Fraud Detection API")

BASE = Path(__file__).parent.parent / "models"
PREDICTIONS_FILE = Path("/tmp/incoming_predictions.csv")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = "shan1322/fraud-detection-mlops"
COMMIT_EVERY = 20

with open(BASE / "model.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE / "preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open(BASE / "feature_metadata.json", "r") as f:
    metadata = json.load(f)

NUM_COLS = metadata["num_cols"]
CAT_COLS = metadata["cat_cols"]

prediction_count = 0

def commit_to_github():
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        content = PREDICTIONS_FILE.read_text()
        try:
            existing = repo.get_contents("incoming_predictions.csv")
            repo.update_file(
                "incoming_predictions.csv",
                f"update predictions {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                content,
                existing.sha
            )
        except:
            repo.create_file(
                "incoming_predictions.csv",
                f"add predictions {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                content
            )
        print(f"Committed predictions to GitHub")
    except Exception as e:
        print(f"GitHub commit failed: {e}")

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
    global prediction_count

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
    }
    for col in NUM_COLS + CAT_COLS:
        log_row[col] = transaction.get(col, None)

    file_exists = PREDICTIONS_FILE.exists()
    with open(str(PREDICTIONS_FILE), "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_row)

    prediction_count += 1

    if prediction_count % COMMIT_EVERY == 0:
        commit_to_github()

    return {
        "fraud_probability": round(proba, 4),
        "is_fraud": bool(prediction),
        "risk_level": log_row["risk_level"]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}