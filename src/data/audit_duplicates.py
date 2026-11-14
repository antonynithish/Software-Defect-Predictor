
import pandas as pd
from pathlib import Path
from scipy.io import arff

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "data/raw/mendeley/KC1.arff"

records, metadata = arff.loadarff(FILE)
df = pd.DataFrame(records)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

df["class"] = df["class"].map({"false": 0, "true": 1})

X = df.drop(columns=["class"])
y = df["class"]

# Identify identical feature groups
groups = pd.util.hash_pandas_object(
    X, index=False
)

group_info = pd.DataFrame({
    "group": groups,
    "class": y
})

summary = group_info.groupby("group").agg(
    records=("class", "size"),
    unique_labels=("class", "nunique")
)

print("\n========== DUPLICATE AUDIT ==========")

print("Total records:", len(df))
print("Unique feature groups:", len(summary))

print(
    "Records belonging to repeated feature groups:",
    summary.loc[summary["records"] > 1, "records"].sum()
)

print(
    "Feature groups with conflicting labels:",
    (summary["unique_labels"] > 1).sum()
)

print("\nRepeated feature groups:")
print(summary[summary["records"] > 1].sort_values(
    "records", ascending=False
).head(10))

print("\nConflicting label groups:")
print(summary[summary["unique_labels"] > 1].head(10))

print("\n========== AUDIT COMPLETED ==========")
