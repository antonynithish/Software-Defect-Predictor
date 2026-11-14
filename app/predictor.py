
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "tuned"
    / "kc1_grouped_random_forest_tuned.joblib"
)

FEATURE_NAMES = [
    "loc",
    "v(g)",
    "ev(g)",
    "iv(g)",
    "n",
    "v",
    "l",
    "d",
    "i",
    "e",
    "b",
    "t",
    "lOCode",
    "lOComment",
    "lOBlank",
    "locCodeAndComment",
    "uniq_Op",
    "uniq_Opnd",
    "total_Op",
    "total_Opnd",
    "branchCount",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    if isinstance(artifact, dict):
        pipeline = next(
            (
                artifact[key]
                for key in (
                    "pipeline",
                    "model",
                    "best_model",
                    "best_estimator",
                    "estimator",
                )
                if key in artifact
                and hasattr(artifact[key], "predict_proba")
            ),
            None,
        )

        if pipeline is None:
            raise ValueError(
                "Could not locate the trained pipeline "
                "inside the model artifact."
            )

        features = artifact.get("features", FEATURE_NAMES)
        threshold = float(artifact.get("threshold", 0.5))

    else:
        pipeline = artifact
        features = FEATURE_NAMES
        threshold = 0.5

    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(
            "The loaded model does not support predict_proba."
        )

    features = list(features)

    if set(features) != set(FEATURE_NAMES):
        raise ValueError(
            "The model feature list does not match the "
            "expected KC1 feature schema."
        )

    return pipeline, features, threshold


MODEL, FEATURES, THRESHOLD = load_model()


def get_risk_category(probability):
    # Application display bands, not independently
    # validated defect-risk thresholds.
    if probability < 0.30:
        return "Low"

    if probability < 0.60:
        return "Medium"

    return "High"


def predict_defect(metrics):
    missing = [
        feature for feature in FEATURES
        if feature not in metrics
    ]

    if missing:
        raise ValueError(
            f"Missing features: {', '.join(missing)}"
        )

    values = {}

    for feature in FEATURES:
        try:
            value = float(metrics[feature])
        except (TypeError, ValueError):
            raise ValueError(
                f"{feature} must be a valid number."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"{feature} must be a finite number."
            )

        values[feature] = value

    input_df = pd.DataFrame(
        [[values[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    classifier = (
        MODEL.named_steps["classifier"]
        if hasattr(MODEL, "named_steps")
        and "classifier" in MODEL.named_steps
        else MODEL
    )

    classes = list(classifier.classes_)

    if 1 not in classes:
        raise ValueError(
            "The trained model does not contain "
            "the expected defective class (1)."
        )

    positive_index = classes.index(1)

    probabilities = MODEL.predict_proba(input_df)[0]
    probability = float(probabilities[positive_index])

    is_defective = probability >= THRESHOLD

    return {
        "prediction": (
            "Defective" if is_defective
            else "Non-defective"
        ),
        "defective_probability": round(probability, 4),
        "threshold": round(THRESHOLD, 4),
        "risk_category": get_risk_category(probability),
    }
