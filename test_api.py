# test_api.py
import pandas as pd
import requests
import os
import sys

# get URL from environment variable or argument
API_URL = os.environ.get("API_URL", "http://localhost:7860")

df = pd.read_csv('models/test_sample.csv')

print(f"Sending transactions to {API_URL}")

for i in range(20):
    row = df.iloc[i].to_dict()
    row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    
    try:
        response = requests.post(f"{API_URL}/predict", json=row, timeout=10)
        print(f"Transaction {i+1}: {response.json()}")
    except Exception as e:
        print(f"Transaction {i+1} failed: {e}")