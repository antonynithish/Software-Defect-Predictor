import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from app.app import app
from app.predictor import FEATURES


TEST_DATA_PATH = "data/processed/kc1_grouped_test.csv"


def main():
    test_df = pd.read_csv(TEST_DATA_PATH)

    sample = test_df.iloc[0]

    metrics = {
        feature: float(sample[feature])
        for feature in FEATURES
    }

    client = app.test_client()

    # Health check
    response = client.get("/api/health")
    print("\nHEALTH CHECK:")
    print(response.status_code, response.json)

    # Model information
    response = client.get("/api/model-info")
    print("\nMODEL INFORMATION:")
    print(response.status_code)
    print("Features:", response.json["feature_count"])

    # Prediction test
    response = client.post(
        "/api/predict",
        json={"metrics": metrics}
    )

    print("\nPREDICTION RESULT:")
    print("Status:", response.status_code)
    print(response.json)

    if response.status_code != 200:
        raise RuntimeError("Prediction test failed.")

    # History test
    response = client.get("/api/history")

    print("\nPREDICTION HISTORY:")
    print("Status:", response.status_code)
    print(response.json)

    if response.status_code != 200:
        raise RuntimeError("History test failed.")

    print("\nAll backend smoke tests passed!")


if __name__ == "__main__":
    main()
