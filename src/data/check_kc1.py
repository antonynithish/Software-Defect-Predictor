
import pandas as pd
from pathlib import Path
from scipy.io import arff

PROJECT_ROOT = Path(__file__).resolve().parents[2]
file_path = PROJECT_ROOT / "data" / "raw" / "mendeley" / "KC1.arff"

records, metadata = arff.loadarff(file_path)
df = pd.DataFrame(records)

# Decode text values
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

print("\n========== KC1 COLUMN NAMES ==========")
for col in df.columns:
    print(col)

print("\n========== LAST FIVE COLUMNS ==========")
for col in df.columns[-5:]:
    print(f"\nColumn: {col}")
    print(df[col].value_counts(dropna=False))

print("\n========== DATA TYPES ==========")
print(df.dtypes)
