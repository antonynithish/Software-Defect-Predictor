
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.stats import randint, loguniform

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_val_predict
)

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


# ==================================================
# PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "tuned"
REPORT_DIR = ROOT / "reports" / "generated"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# MODEL DEFINITIONS AND SEARCH SPACES
# ==================================================

def build_model_searches():

    searches = {

        "Logistic Regression": {

            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=42
                ))
            ]),

            "params": {
                "classifier__C": loguniform(0.01, 100),
                "classifier__solver": ["lbfgs", "liblinear"],
                "classifier__class_weight": ["balanced"]
            }
        },

        "Decision Tree": {

            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=42
                ))
            ]),

            "params": {
                "classifier__max_depth": [None, 3, 5, 8, 12, 20],
                "classifier__min_samples_split": randint(2, 20),
                "classifier__min_samples_leaf": randint(1, 10),
                "classifier__criterion": ["gini", "entropy"]
            }
        },

        "Random Forest": {

            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1
                ))
            ]),

            "params": {
                "classifier__n_estimators": [150, 250, 400],
                "classifier__max_depth": [None, 8, 15, 25],
                "classifier__min_samples_split": randint(2, 15),
                "classifier__min_samples_leaf": randint(1, 8),
                "classifier__max_features": ["sqrt", 0.5]
            }
        },

        "Gradient Boosting": {

            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", GradientBoostingClassifier(
                    random_state=42
                ))
            ]),

            "params": {
                "classifier__n_estimators": [100, 150, 200, 300],
                "classifier__learning_rate": loguniform(0.01, 0.2),
                "classifier__max_depth": [1, 2, 3, 4],
                "classifier__min_samples_leaf": randint(2, 15),
                "classifier__subsample": [0.7, 0.85, 1.0]
            }
        }
    }

    return searches


# ==================================================
# THRESHOLD OPTIMIZATION
# ==================================================

def find_best_threshold(y_true, probabilities):

    thresholds = np.linspace(0.05, 0.95, 91)

    threshold_results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true, predictions, zero_division=0
        )

        recall = recall_score(
            y_true, predictions, zero_division=0
        )

        f1 = f1_score(
            y_true, predictions, zero_division=0
        )

        threshold_results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    threshold_df = pd.DataFrame(threshold_results)

    # Select highest F1; break ties using higher precision
    best = threshold_df.sort_values(
        ["f1", "precision", "threshold"],
        ascending=[False, False, False]
    ).iloc[0]

    return float(best["threshold"]), threshold_df


# ==================================================
# FINAL TEST EVALUATION
# ==================================================

def evaluate_model(model, X_test, y_test, threshold):

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

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


# ==================================================
# TRAIN AND TUNE EXPERIMENT
# ==================================================

def tune_experiment(experiment_name):

    print("\n" + "=" * 75)
    print("TUNING EXPERIMENT:", experiment_name.upper())
    print("=" * 75)

    # Select dataset
    if experiment_name == "random":

        train_file = PROCESSED_DIR / "kc1_train.csv"
        test_file = PROCESSED_DIR / "kc1_test.csv"

    elif experiment_name == "grouped":

        train_file = PROCESSED_DIR / "kc1_grouped_train.csv"
        test_file = PROCESSED_DIR / "kc1_grouped_test.csv"

    else:
        raise ValueError("Unknown experiment")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df.drop(columns=["class"])
    y_train = train_df["class"].astype(int)

    X_test = test_df.drop(columns=["class"])
    y_test = test_df["class"].astype(int)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Feature columns do not match.")

    # Group identical feature combinations together
    groups = pd.util.hash_pandas_object(
        X_train, index=False
    ).to_numpy()

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    model_searches = build_model_searches()

    experiment_results = []

    for model_name, configuration in model_searches.items():

        print("\n" + "-" * 65)
        print("TUNING MODEL:", model_name)

        search = RandomizedSearchCV(
            estimator=configuration["pipeline"],
            param_distributions=configuration["params"],
            n_iter=8,
            scoring="average_precision",
            cv=cv,
            refit=True,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            error_score="raise"
        )

        # Hyperparameter tuning using training data only
        search.fit(
            X_train,
            y_train,
            groups=groups
        )

        best_model = search.best_estimator_

        print("\nBest cross-validation PR-AUC:",
              round(search.best_score_, 4))

        print("\nBest parameters:")
        print(search.best_params_)

        # Generate out-of-fold probabilities for threshold selection
        oof_probabilities = cross_val_predict(
            best_model,
            X_train,
            y_train,
            groups=groups,
            cv=cv,
            method="predict_proba",
            n_jobs=1
        )[:, 1]

        # Optimize threshold using training predictions
        best_threshold, threshold_df = find_best_threshold(
            y_train,
            oof_probabilities
        )

        print("\nOptimized threshold:", round(best_threshold, 3))

        # Save threshold analysis
        threshold_file = (
            REPORT_DIR /
            f"threshold_{experiment_name}_"
            f"{model_name.lower().replace(' ', '_')}.csv"
        )

        threshold_df.to_csv(
            threshold_file,
            index=False
        )

        # Evaluate on untouched test data
        metrics = evaluate_model(
            best_model,
            X_test,
            y_test,
            best_threshold
        )

        metrics["Experiment"] = experiment_name
        metrics["Model"] = model_name
        metrics["Threshold"] = best_threshold
        metrics["CV PR AUC"] = search.best_score_
        metrics["Best Parameters"] = str(search.best_params_)

        experiment_results.append(metrics)

        # Save optimized model and metadata
        model_filename = (
            f"kc1_{experiment_name}_"
            f"{model_name.lower().replace(' ', '_')}_tuned.joblib"
        )

        joblib.dump(
            {
                "pipeline": best_model,
                "features": list(X_train.columns),
                "model_name": model_name,
                "experiment": experiment_name,
                "threshold": best_threshold,
                "best_params": search.best_params_,
                "cv_pr_auc": search.best_score_
            },
            MODEL_DIR / model_filename
        )

        print("\nTEST RESULTS")
        print("Accuracy:", round(metrics["Accuracy"], 4))
        print("Precision:", round(metrics["Precision"], 4))
        print("Recall:", round(metrics["Recall"], 4))
        print("F1 Score:", round(metrics["F1 Score"], 4))
        print("PR AUC:", round(metrics["PR AUC"], 4))

    return experiment_results


# ==================================================
# MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    all_results = []

    all_results.extend(
        tune_experiment("random")
    )

    all_results.extend(
        tune_experiment("grouped")
    )

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        ["Experiment", "F1 Score"],
        ascending=[True, False]
    )

    results_path = (
        REPORT_DIR / "tuned_model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n" + "=" * 75)
    print("FINAL TUNED MODEL COMPARISON")
    print("=" * 75)

    columns = [
        "Experiment",
        "Model",
        "Threshold",
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

    print("\nALL HYPERPARAMETER TUNING COMPLETED!")
