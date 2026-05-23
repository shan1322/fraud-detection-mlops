# monitor/monitor.py
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import ks_2samp, chi2_contingency

BASE = Path(__file__).parent.parent / "models"
PREDICTIONS_FILE = Path(__file__).parent.parent / "incoming_predictions.csv"
DRIFT_RESULT_FILE = Path(__file__).parent.parent / "drift_result.json"

DRIFT_THRESHOLD = 0.05  # p-value threshold
DRIFT_SHARE = 0.2       # 20% of columns drifted = retrain

def check_drift():
    reference = pd.read_csv(BASE / "reference_data.csv")

    with open(BASE / "feature_metadata.json") as f:
        metadata = json.load(f)

    NUM_COLS = metadata["num_cols"]
    CAT_COLS = metadata["cat_cols"]

    reference = reference[[c for c in NUM_COLS + CAT_COLS if c in reference.columns]]

    if not PREDICTIONS_FILE.exists():
        print("No incoming predictions yet — skipping drift check")
        return False

    incoming = pd.read_csv(PREDICTIONS_FILE)

    if len(incoming) < 20:
        print(f"Only {len(incoming)} predictions — need at least 20")
        return False

    common_num = [c for c in NUM_COLS if c in incoming.columns and c in reference.columns]
    common_cat = [c for c in CAT_COLS if c in incoming.columns and c in reference.columns]

    drifted = []
    stable = []

    # numeric — KS test
    for col in common_num:
        ref_vals = reference[col].dropna()
        inc_vals = incoming[col].dropna()
        if len(ref_vals) < 5 or len(inc_vals) < 5:
            continue
        stat, p_value = ks_2samp(ref_vals, inc_vals)
        if p_value < DRIFT_THRESHOLD:
            drifted.append(f"  🚨 {col}: p={round(p_value, 4)}")
        else:
            stable.append(f"  ✅ {col}: p={round(p_value, 4)}")

    # categorical — chi2 test
    for col in common_cat:
        ref_vals = reference[col].fillna("missing")
        inc_vals = incoming[col].fillna("missing")
        all_cats = set(ref_vals.unique()) | set(inc_vals.unique())
        ref_counts = [ref_vals.value_counts().get(c, 0) for c in all_cats]
        inc_counts = [inc_vals.value_counts().get(c, 0) for c in all_cats]
        try:
            stat, p_value, _, _ = chi2_contingency([ref_counts, inc_counts])
            if p_value < DRIFT_THRESHOLD:
                drifted.append(f"  🚨 {col}: p={round(p_value, 4)}")
            else:
                stable.append(f"  ✅ {col}: p={round(p_value, 4)}")
        except:
            continue

    total = len(drifted) + len(stable)
    drift_share = len(drifted) / total if total > 0 else 0
    drift_detected = drift_share >= DRIFT_SHARE

    print(f"\n=== DRIFT REPORT ===")
    print(f"Drifted columns ({len(drifted)}/{total}) — {round(drift_share*100, 1)}%")
    for d in drifted[:20]:
        print(d)
    print(f"\nDrift detected: {drift_detected}")

    result = {
        "drift_detected": drift_detected,
        "drift_share": round(drift_share, 4),
        "drifted_columns": len(drifted),
        "total_columns": total
    }

    with open(DRIFT_RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    return drift_detected

if __name__ == "__main__":
    check_drift()