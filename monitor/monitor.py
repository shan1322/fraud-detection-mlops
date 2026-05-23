# monitor/monitor.py
import pandas as pd
import json
from pathlib import Path

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

BASE = Path(__file__).parent.parent / "models"
PREDICTIONS_FILE = Path(__file__).parent.parent / "incoming_predictions.csv"
DRIFT_RESULT_FILE = Path(__file__).parent.parent / "drift_result.json"

def check_drift():
    reference = pd.read_csv(BASE / "reference_data.csv")

    with open(BASE / "feature_metadata.json") as f:
        metadata = json.load(f)

    NUM_COLS = metadata["num_cols"]
    CAT_COLS = metadata["cat_cols"]
    ALL_COLS = NUM_COLS + CAT_COLS

    reference = reference[[c for c in ALL_COLS if c in reference.columns]]

    if not PREDICTIONS_FILE.exists():
        print("No incoming predictions yet — skipping drift check")
        return False

    incoming = pd.read_csv(PREDICTIONS_FILE)

    if len(incoming) < 5:
        print(f"Only {len(incoming)} predictions — need at least 5")
        return False

    common_cols = [c for c in ALL_COLS if c in incoming.columns and c in reference.columns]

    if len(common_cols) == 0:
        print("No common columns between reference and incoming data")
        return False

    reference_sample = reference[common_cols].sample(
        min(1000, len(reference)), random_state=42
    )
    incoming_sample = incoming[common_cols].sample(
        min(len(incoming), 1000), random_state=42
    )

    report = Report(metrics=[DataDriftPreset(drift_share=0.03)])
    report.run(reference_data=reference_sample, current_data=incoming_sample)

    result = report.as_dict()

    with open(DRIFT_RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    drift_detected = result["metrics"][0]["result"]["dataset_drift"]

    # print per column drift details
    column_results = result["metrics"][1]["result"]["drift_by_columns"]
    print(f"\n=== DRIFT BY COLUMN ===")
    drifted = []
    not_drifted = []
    for col, details in column_results.items():
        drift_score = round(details["drift_score"], 4)
        detected = details["drift_detected"]
        if detected:
            drifted.append(f"  🚨 {col}: {drift_score}")
        else:
            not_drifted.append(f"  ✅ {col}: {drift_score}")

    print(f"\nDrifted columns ({len(drifted)}):")
    for d in drifted:
        print(d)

    print(f"\nStable columns ({len(not_drifted)}):")
    for d in not_drifted:
        print(d)

    print(f"\nTotal drifted: {len(drifted)}/{len(drifted)+len(not_drifted)}")

    if drift_detected:
        print("\n🚨 DATA DRIFT DETECTED")
    else:
        print("\n✅ No significant drift")

    return drift_detected


if __name__ == "__main__":
    check_drift()