
import pandas as pd
from pathlib import Path
from scipy.io import arff

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "mendeley"


def load_arff(file_path):

    records, metadata = arff.loadarff(file_path)

    df = pd.DataFrame(records)

    # Decode byte-string values
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].apply(
            lambda value: value.decode("utf-8")
            if isinstance(value, bytes)
            else value
        )

    return df, metadata


def inspect_datasets():

    print("=" * 65)
    print("NASA SOFTWARE DEFECT DATASET INSPECTION")
    print("=" * 65)

    files = sorted(DATASET_DIR.glob("*.arff"))

    if not files:
        print("No ARFF files found!")
        print("Check the folder:", DATASET_DIR)
        return

    print(f"\nTotal ARFF files found: {len(files)}")

    for file in files:

        print("\n" + "=" * 65)
        print("DATASET:", file.name)
        print("=" * 65)

        try:
            df, metadata = load_arff(file)

            print("\nDataset shape:", df.shape)

            print("\nColumns:")
            print(df.columns.tolist())

            print("\nData types:")
            print(df.dtypes)

            print("\nMissing values:")
            print(df.isnull().sum().sum())

            print("\nLast five columns and their values:")

            for column in df.columns[-5:]:
                print(f"\n{column}")
                print(df[column].value_counts(dropna=False).head(10))

            print("\nFirst five rows:")
            print(df.head())

        except Exception as error:
            print("Error loading dataset:", error)


if __name__ == "__main__":
    inspect_datasets()
