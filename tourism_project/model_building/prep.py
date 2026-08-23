
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ================================
# Data Preparation Script
# ================================

# Load the dataset directly from the repository data folder.
PROJECT_ROOT = Path("tourism_project")
DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset with shape: {df.shape}")

# Perform data cleaning and remove any unnecessary columns.
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])
    print("Removed column: CustomerID")

# Separate features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split the cleaned dataset into training and testing sets, and save them locally.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = X_train.assign(ProdTaken=y_train)
test_df = X_test.assign(ProdTaken=y_test)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_path = OUTPUT_DIR / "train.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

# Make the train and test splits available to the next job as a workflow artifact.
print("\nData preparation complete.")
print(f"Training set saved to: {train_path} (shape: {train_df.shape})")
print(f"Testing set saved to: {test_path} (shape: {test_df.shape})")
