
from pathlib import Path
import pandas as pd

# Project root
PROJECT_ROOT = Path("tourism_project")

# DATA_PATH should also be relative to this PROJECT_ROOT for the script
DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"

# Expected schema based on data dictionary
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups",
    "DurationOfPitch"
]

# Register the dataset with a validation script that checks the expected columns and prints a summary
def validate_dataset(file_path: Path):
    """Validate dataset schema and print summary."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    # Check for missing/extra columns
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    print("=== Validation Report ===")
    print("Missing columns:", missing if missing else "None")
    print("Extra columns:", extra if extra else "None")

    # Dataset summary
    print("\n=== Dataset Summary ===")
    print(f"Path: {file_path}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nColumns:", list(df.columns))

    # Target distribution
    if "ProdTaken" in df.columns:
        print("\nTarget distribution:")
        print(df["ProdTaken"].value_counts(normalize=True).round(2))
    else:
        print("\nTarget column 'ProdTaken' not found.")

    return df

if __name__ == "__main__":
    # Ensure PROJECT_ROOT is defined if the script is run directly
    if 'PROJECT_ROOT' not in locals():
        PROJECT_ROOT = Path(__file__).resolve().parent.parent # Fallback for direct execution
    df = validate_dataset(DATA_PATH)
"""

# Write the content to the file
with open(file_path, "w") as f:
    f.write(file_content.strip())

print(f"File written to: {file_path}")
