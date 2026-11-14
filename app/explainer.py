
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from app.predictor import MODEL, FEATURES, THRESHOLD


ROOT_DIR = Path(__file__).resolve().parents[1]

BACKGROUND_PATH = (
    ROOT_DIR / "data" / "processed" / "kc1_grouped_train.csv"
)

MAX_BACKGROUND_ROWS = 100

_explainer = None
_background = None


def get_model_parts():
    if not hasattr(MODEL, "named_steps"):
        raise TypeError(
            "Individual SHAP currently requires the saved "
            "scikit-learn pipeline."
        )

    classifier = MODEL.named_steps["classifier"]
    imputer = MODEL.named_steps["imputer"]

    classes = list(classifier.classes_)

    if 1 not in classes:
        raise ValueError("Defective class (1) not found.")

    return classifier, imputer, classes.index(1)


def get_background():
    global _background

    if _background is not None:
        return _background

    if not BACKGROUND_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {BACKGROUND_PATH}"
        )

    df = pd.read_csv(BACKGROUND_PATH)

    missing = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Training data is missing features: {missing}"
        )

    X = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace([np.inf, -np.inf], np.nan)

    _, imputer, _ = get_model_parts()

    transformed = imputer.transform(X)

    background = pd.DataFrame(
        transformed,
        columns=FEATURES
    )

    _background = background.sample(
        n=min(MAX_BACKGROUND_ROWS, len(background)),
        random_state=42
    )

    return _background


def get_explainer():
    global _explainer

    if _explainer is not None:
        return _explainer

    classifier, _, _ = get_model_parts()

    background = get_background()

    _explainer = shap.TreeExplainer(
        classifier,
        data=background,
        feature_perturbation="interventional",
        model_output="probability"
    )

    return _explainer


def extract_positive_class_values(shap_values, positive_index):
    if isinstance(shap_values, list):
        values = shap_values[positive_index][0]

    else:
        values = np.asarray(shap_values)

        if values.ndim == 3:
            # Shape: samples x features x classes
            values = values[0, :, positive_index]

        elif values.ndim == 2:
            # Shape: samples x features
            values = values[0]

        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {values.shape}"
            )

    return np.asarray(values, dtype=float).reshape(-1)


def extract_base_value(expected_value, positive_index):
    if isinstance(expected_value, list):
        value = expected_value[positive_index]

    else:
        values = np.asarray(expected_value)

        if values.ndim == 0:
            value = values.item()

        elif values.size > 1:
            value = values.reshape(-1)[positive_index]

        else:
            value = values.reshape(-1)[0]

    return float(value)


def explain_prediction(metrics):
    if not isinstance(metrics, dict):
        raise ValueError("Metrics must be an object.")

    values = {}

    for feature in FEATURES:
        if feature not in metrics:
            raise ValueError(
                f"Missing feature: {feature}"
            )

        try:
            value = float(metrics[feature])

        except (TypeError, ValueError):
            raise ValueError(
                f"{feature} must be numeric."
            )

        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"{feature} must be a finite non-negative number."
            )

        values[feature] = value

    input_df = pd.DataFrame(
        [[values[feature] for feature in FEATURES]],
        columns=FEATURES
    )

    classifier, imputer, positive_index = get_model_parts()

    transformed = imputer.transform(input_df)

    X_explain = pd.DataFrame(
        transformed,
        columns=FEATURES
    )

    explainer = get_explainer()

    try:
        shap_values = explainer.shap_values(
            X_explain,
            check_additivity=False
        )

    except TypeError:
        shap_values = explainer.shap_values(X_explain)

    contributions = extract_positive_class_values(
        shap_values,
        positive_index
    )

    base_value = extract_base_value(
        explainer.expected_value,
        positive_index
    )

    if len(contributions) != len(FEATURES):
        raise ValueError(
            "SHAP feature count does not match model features."
        )

    probability = float(
        MODEL.predict_proba(input_df)[0][positive_index]
    )

    # Check the SHAP decomposition against the model output.
    reconstructed_probability = (
        base_value + float(np.sum(contributions))
    )

    if not np.isclose(
        reconstructed_probability,
        probability,
        atol=0.02
    ):
        raise ValueError(
            "SHAP explanation did not sufficiently match "
            "the model probability. Check SHAP compatibility."
        )

    feature_results = []

    for index, feature in enumerate(FEATURES):
        contribution = float(contributions[index])

        feature_results.append({
            "feature": feature,
            "input_value": values[feature],
            "shap_value": round(contribution, 6),
            "impact_percentage_points": round(
                contribution * 100, 3
            ),
            "direction": (
                "increases" if contribution > 0
                else "decreases" if contribution < 0
                else "neutral"
            ),
            "absolute_impact": round(
                abs(contribution), 6
            )
        })

    feature_results.sort(
        key=lambda item: item["absolute_impact"],
        reverse=True
    )

    positive = sorted(
        [
            item for item in feature_results
            if item["shap_value"] > 0
        ],
        key=lambda item: item["shap_value"],
        reverse=True
    )[:5]

    negative = sorted(
        [
            item for item in feature_results
            if item["shap_value"] < 0
        ],
        key=lambda item: item["shap_value"]
    )[:5]

    return {
        "base_probability": round(base_value, 6),
        "predicted_probability": round(probability, 6),
        "threshold": THRESHOLD,
        "reconstructed_probability": round(
            reconstructed_probability, 6
        ),
        "top_features": feature_results[:10],
        "positive_contributions": positive,
        "negative_contributions": negative,
        "explanation_type": "Individual SHAP",
        "explanation_scale": "Probability"
    }
