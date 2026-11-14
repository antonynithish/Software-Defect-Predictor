
import pandas as pd
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "generated"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------

def build_models():

    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            ))
        ]),

        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=2,
                random_state=42
            ))
        ])
    }


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy_score(y_test, predictions),

        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),

        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),

        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),

        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),

        "ROC AUC": roc_auc_score(y_test, probabilities),

        "PR AUC": average_precision_score(
            y_test, probabilities
        ),

        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp
    }


# --------------------------------------------------
# TRAINING PIPELINE
# --------------------------------------------------

def train_experiment(experiment_name):

    print("\n" + "=" * 70)
    print("EXPERIMENT:", experiment_name.upper())
    print("=" * 70)

    # Select appropriate dataset files
    if experiment_name == "random":

        train_file = PROCESSED_DIR / "kc1_train.csv"
        test_file = PROCESSED_DIR / "kc1_test.csv"

    elif experiment_name == "grouped":

        train_file = PROCESSED_DIR / "kc1_grouped_train.csv"
        test_file = PROCESSED_DIR / "kc1_grouped_test.csv"

    else:
        raise ValueError("Unknown experiment name")

    # Check dataset files
    if not train_file.exists() or not test_file.exists():

        raise FileNotFoundError(
            f"Missing dataset files for {experiment_name}"
        )

    # Load datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df.drop(columns=["class"])
    y_train = train_df["class"].astype(int)

    X_test = test_df.drop(columns=["class"])
    y_test = test_df["class"].astype(int)

    # Validate datasets
    if list(X_train.columns) != list(X_test.columns):

        raise ValueError(
            "Training and testing features do not match."
        )

    if y_train.nunique() < 2 or y_test.nunique() < 2:

        raise ValueError(
            "Both splits must contain both classes."
        )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("Training defect rate:", round(y_train.mean(), 4))
    print("Testing defect rate:", round(y_test.mean(), 4))

    results = []

    models = build_models()

    # Train each algorithm
    for model_name, model in models.items():

        print("\nTraining:", model_name)

        if model_name == "Gradient Boosting":

            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train
            )

            model.fit(
                X_train,
                y_train,
                classifier__sample_weight=sample_weights
            )

        else:

            model.fit(X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        metrics["Experiment"] = experiment_name
        metrics["Model"] = model_name

        results.append(metrics)

        # Save trained model
        model_filename = (
            f"kc1_{experiment_name}_"
            f"{model_name.lower().replace(' ', '_')}.joblib"
        )

        joblib.dump(
            {
                "pipeline": model,
                "features": list(X_train.columns),
                "model_name": model_name,
                "experiment": experiment_name
            },
            MODEL_DIR / model_filename
        )

        print("F1 Score:", round(metrics["F1 Score"], 4))
        print("Recall:", round(metrics["Recall"], 4))
        print("PR AUC:", round(metrics["PR AUC"], 4))

    return results


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    all_results = []

    # Experiment 1: Random split
    all_results.extend(
        train_experiment("random")
    )

    # Experiment 2: Grouped split
    all_results.extend(
        train_experiment("grouped")
    )

    # Combine results
    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        ["Experiment", "F1 Score"],
        ascending=[True, False]
    )

    # Save comparison report
    results_path = REPORT_DIR / "model_comparison.csv"

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    columns = [
        "Experiment",
        "Model",
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC",
        "PR AUC"
    ]

    print(
        results_df[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nResults saved to:", results_path)

    print("\nALL MODEL EXPERIMENTS COMPLETED!")
