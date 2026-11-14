
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import arff
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "mendeley" / "KC1.arff"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_kc1():

    print("=" * 60)
    print("KC1 DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Load ARFF dataset
    records, metadata = arff.loadarff(DATASET_PATH)
    df = pd.DataFrame(records)

    # 2. Decode categorical values
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8")
            if isinstance(x, bytes) else x
        )

    print("\nOriginal dataset shape:", df.shape)

    # 3. Convert defect labels
    df["class"] = (
        df["class"].astype(str).str.lower().map({
            "false": 0,
            "true": 1
        })
    )

    if df["class"].isna().any():
        raise ValueError("Unexpected or missing defect labels!")

    # 4. Separate features and target
    X = df.drop(columns=["class"]).apply(
        pd.to_numeric, errors="coerce"
    )

    y = df["class"].astype(int)

    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)

    print("\nFeature count:", X.shape[1])
    print("Target column: class")

    # 5. Dataset quality checks
    print("\nMissing values per feature:")
    print(X.isnull().sum())

    print("\nTotal missing values:", X.isnull().sum().sum())

    print("\nDuplicate complete records:", df.duplicated().sum())

    print("\nDuplicate feature rows:", X.duplicated().sum())

    # Do not automatically delete duplicate feature rows.
    # Different modules can have identical metrics.

    print("\nOriginal class distribution:")
    print(y.value_counts().sort_index())

    # 6. Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 7. Save processed datasets
    train_df = X_train.copy()
    train_df["class"] = y_train

    test_df = X_test.copy()
    test_df["class"] = y_test

    train_df.to_csv(
        OUTPUT_DIR / "kc1_train.csv", index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "kc1_test.csv", index=False
    )

    print("\n========== SPLIT SUMMARY ==========")

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    print("\nSaved files:")
    print(OUTPUT_DIR / "kc1_train.csv")
    print(OUTPUT_DIR / "kc1_test.csv")

    print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    preprocess_kc1()
