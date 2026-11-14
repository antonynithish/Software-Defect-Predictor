
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)


# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "models" / "tuned" /
    "kc1_grouped_random_forest_tuned.joblib"
)

TEST_PATH = (
    ROOT / "data" / "processed" /
    "kc1_grouped_test.csv"
)

OUTPUT_DIR = ROOT / "reports" / "generated"
FIGURE_DIR = OUTPUT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD MODEL AND TEST DATA
# ==================================================

def explain_model():

    print("=" * 65)
    print("SOFTWARE DEFECT PREDICTION - EXPLAINABLE AI")
    print("=" * 65)

    artifact = joblib.load(MODEL_PATH)

    model = artifact["pipeline"]
    features = artifact["features"]
    threshold = artifact["threshold"]

    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df[features]
    y_test = test_df["class"].astype(int)

    print("\nModel:", artifact["model_name"])
    print("Experiment:", artifact["experiment"])
    print("Threshold:", threshold)
    print("Test records:", len(X_test))


    # ==================================================
    # PREDICTIONS
    # ==================================================

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    print("\nFINAL TEST METRICS")

    print("ROC-AUC:",
          round(roc_auc_score(y_test, probabilities), 4))

    print("PR-AUC:",
          round(average_precision_score(
              y_test, probabilities
          ), 4))

    print("Precision:",
          round(precision_score(
              y_test, predictions, zero_division=0
          ), 4))

    print("Recall:",
          round(recall_score(
              y_test, predictions, zero_division=0
          ), 4))

    print("F1:",
          round(f1_score(
              y_test, predictions, zero_division=0
          ), 4))


    # ==================================================
    # CONFUSION MATRIX
    # ==================================================

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-defective", "Defective"]
    )

    display.plot(cmap="Blues", values_format="d")

    plt.title("KC1 - Random Forest Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()


    # ==================================================
    # ROC CURVE
    # ==================================================

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc = roc_auc_score(y_test, probabilities)

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"Random Forest (AUC = {auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random baseline"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve - KC1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "roc_curve.png",
        dpi=300
    )

    plt.close()


    # ==================================================
    # PRECISION-RECALL CURVE
    # ==================================================

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        label=f"Random Forest (AP = {pr_auc:.3f})"
    )

    plt.axhline(
        y=y_test.mean(),
        linestyle="--",
        label="Positive-class prevalence"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve - KC1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "precision_recall_curve.png",
        dpi=300
    )

    plt.close()


    # ==================================================
    # SHAP EXPLAINABILITY
    # ==================================================

    print("\nGenerating SHAP explanations...")

    classifier = model.named_steps["classifier"]
    imputer = model.named_steps["imputer"]

    X_transformed = imputer.transform(X_test)

    X_explain = pd.DataFrame(
        X_transformed,
        columns=features,
        index=X_test.index
    )

    # Limit explanation sample size for efficiency
    X_explain = X_explain.sample(
        n=min(300, len(X_explain)),
        random_state=42
    )

    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(
        X_explain
    )

    # Handle different SHAP output formats
    if isinstance(shap_values, list):

        positive_shap = shap_values[1]

    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:

        positive_shap = shap_values[:, :, 1]

    else:

        positive_shap = shap_values


    # ==================================================
    # SHAP SUMMARY PLOT
    # ==================================================

    plt.figure(figsize=(11, 8))

    shap.summary_plot(
        positive_shap,
        X_explain,
        feature_names=features,
        max_display=15,
        show=False
    )

    plt.title("SHAP Summary - Software Defect Prediction")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ==================================================
    # SHAP FEATURE IMPORTANCE
    # ==================================================

    mean_shap = np.abs(
        positive_shap
    ).mean(axis=0)

    shap_importance = pd.DataFrame({
        "Feature": features,
        "Mean Absolute SHAP": mean_shap
    })

    shap_importance = shap_importance.sort_values(
        "Mean Absolute SHAP",
        ascending=False
    )

    shap_importance.to_csv(
        OUTPUT_DIR / "shap_feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10, 7))

    top_features = shap_importance.head(15).sort_values(
        "Mean Absolute SHAP"
    )

    plt.barh(
        top_features["Feature"],
        top_features["Mean Absolute SHAP"]
    )

    plt.xlabel("Mean Absolute SHAP Value")
    plt.title("Top Software Defect Prediction Features")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "shap_feature_importance.png",
        dpi=300
    )

    plt.close()


    # ==================================================
    # RANDOM FOREST NATIVE FEATURE IMPORTANCE
    # ==================================================

    native_importance = pd.DataFrame({
        "Feature": features,
        "Importance": classifier.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    native_importance.to_csv(
        OUTPUT_DIR / "native_feature_importance.csv",
        index=False
    )


    # ==================================================
    # SAVE PREDICTIONS
    # ==================================================

    prediction_results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": predictions,
        "Defect_Probability": probabilities,
        "Threshold": threshold
    })

    prediction_results.to_csv(
        OUTPUT_DIR / "kc1_test_predictions.csv",
        index=False
    )


    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    print("\nTOP 10 SHAP FEATURES")

    print(
        shap_importance.head(10).to_string(
            index=False
        )
    )

    print("\nGenerated files:")
    print(FIGURE_DIR)

    print("\nSHAP FEATURE IMPORTANCE:")
    print(OUTPUT_DIR / "shap_feature_importance.csv")

    print("\nEXPLAINABLE AI ANALYSIS COMPLETED!")


if __name__ == "__main__":
    explain_model()
