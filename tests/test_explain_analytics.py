
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from app.app import app
from app.predictor import FEATURES


TEST_DATA_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "kc1_grouped_test.csv"
)


def main():
    client = app.test_client()

    # Load a real KC1 test sample.
    df = pd.read_csv(TEST_DATA_PATH)

    sample = df.iloc[0]

    metrics = {
        feature: float(sample[feature])
        for feature in FEATURES
    }

    # Phase 5: Individual SHAP
    print("\nPHASE 5 - INDIVIDUAL SHAP")

    response = client.post(
        "/api/explain",
        json={"metrics": metrics}
    )

    print("Status:", response.status_code)

    if response.status_code != 200:
        print(response.json)
        raise RuntimeError("Individual SHAP test failed.")

    explanation = response.json

    print(
        "Predicted probability:",
        explanation["predicted_probability"]
    )

    print(
        "Base probability:",
        explanation["base_probability"]
    )

    print("\nTop SHAP contributions:")

    for feature in explanation["top_features"][:5]:
        print(
            feature["feature"],
            feature["shap_value"],
            feature["direction"]
        )

    print("\nPositive contributions:")
    for feature in explanation["positive_contributions"]:
        print(feature["feature"], feature["shap_value"])

    print("\nNegative contributions:")
    for feature in explanation["negative_contributions"]:
        print(feature["feature"], feature["shap_value"])

    # Phase 6: Model analytics
    print("\nPHASE 6 - MODEL ANALYTICS")

    response = client.get("/api/analytics")

    print("Status:", response.status_code)

    if response.status_code != 200:
        print(response.json)
        raise RuntimeError("Analytics API test failed.")

    analytics = response.json

    print(
        "Comparison files:",
        len(analytics["model_comparisons"])
    )

    print(
        "Available evaluation figures:",
        len(analytics["figures"])
    )

    for comparison in analytics["model_comparisons"]:
        print(
            comparison["file"],
            "Records:",
            len(comparison["results"])
        )

    for figure in analytics["figures"]:
        print("Figure:", figure["name"])

    print("\nAll Phase 5 and Phase 6 API tests passed!")


if __name__ == "__main__":
    main()
