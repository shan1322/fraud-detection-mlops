import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import ks_2samp, chi2_contingency

BASE = Path(__file__).parent.parent / "models"
PREDICTIONS_FILE = Path(__file__).parent.parent / "incoming_predictions.csv"
DRIFT_RESULT_FILE = Path(__file__).parent.parent / "drift_result.json"

DRIFT_THRESHOLD = 0.05
DRIFT_SHARE = 0.03

def save_result(result):
    with open(DRIFT_RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)

def check_drift():
    if not (BASE / "reference_data.csv").exists():
        print("Reference data missing")
        result = {"drift_detected": False}
        save_result(result)
        return False

    reference = pd.read_csv(BASE / "reference_data.csv")

    with open(BASE / "feature_metadata.json") as f:
        metadata = json.load(f)

    NUM_COLS = metadata.get("num_cols", [])
    CAT_COLS = metadata.get("cat_cols", [])

    reference = reference[[c for c in NUM_COLS + CAT_COLS if c in reference.columns]]

    if not PREDICTIONS_FILE.exists():
        print("No incoming predictions")
        result = {"drift_detected": False}
        save_result(result)
        return False

    incoming = pd.read_csv(PREDICTIONS_FILE)

    if len(incoming) < 20:
        print(f"Only {len(incoming)} rows")
        result = {"drift_detected": False}
        save_result(result)
        return False

    common_num = [c for c in NUM_COLS if c in incoming.columns and c in reference.columns]
    common_cat = [c for c in CAT_COLS if c in incoming.columns and c in reference.columns]

    drifted = 0
    total = 0

    # numeric
    for col in common_num:
        ref_vals = reference[col].dropna()
        inc_vals = incoming[col].dropna()
        if len(ref_vals) < 5 or len(inc_vals) < 5:
            continue
        _, p_value = ks_2samp(ref_vals, inc_vals)
        total += 1
        if p_value < DRIFT_THRESHOLD:
            drifted += 1

    # categorical
    for col in common_cat:
        ref_vals = reference[col].fillna("missing")
        inc_vals = incoming[col].fillna("missing")

        all_cats = list(set(ref_vals.unique()) | set(inc_vals.unique()))
        ref_counts = [ref_vals.value_counts().get(c, 0) for c in all_cats]
        inc_counts = [inc_vals.value_counts().get(c, 0) for c in all_cats]

        try:
            _, p_value, _, _ = chi2_contingency([ref_counts, inc_counts])
            total += 1
            if p_value < DRIFT_THRESHOLD:
                drifted += 1
        except:
            continue

    drift_share = drifted / total if total > 0 else 0
    drift_detected = drift_share >= DRIFT_SHARE

    result = {
        "drift_detected": drift_detected,
        "drift_share": round(drift_share, 4),
        "drifted_columns": drifted,
        "total_columns": total
    }

    print(result)
    save_result(result)

    return drift_detected

if __name__ == "__main__":
    check_drift()