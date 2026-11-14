
import pandas as pd
from pathlib import Path
from scipy.io import arff
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]

FILE = ROOT / "data/raw/mendeley/KC1.arff"
OUTPUT = ROOT / "data/processed"

records, metadata = arff.loadarff(FILE)
df = pd.DataFrame(records)

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

df["class"] = df["class"].map({"false": 0, "true": 1})

X = df.drop(columns=["class"])
y = df["class"].astype(int)

# Give identical feature rows the same group ID
groups = pd.util.hash_pandas_object(
    X, index=False
)

# Five-fold grouped stratified split
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Save datasets
train_df = X_train.copy()
train_df["class"] = y_train

test_df = X_test.copy()
test_df["class"] = y_test

OUTPUT.mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUTPUT / "kc1_grouped_train.csv", index=False)
test_df.to_csv(OUTPUT / "kc1_grouped_test.csv", index=False)

# Verify no feature groups overlap
train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

print("\n========== GROUPED SPLIT SUMMARY ==========")

print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nOverlapping feature groups:", len(train_groups & test_groups))

print("\nGrouped split completed successfully!")
