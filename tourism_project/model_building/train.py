
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Define project paths for training, testing, and model output
PROJECT_ROOT = Path("tourism_project")
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "splits" / "test.csv"
MODEL_DIR = PROJECT_ROOT / "deployment"
MODEL_PATH = MODEL_DIR / "tourism_model.pkl"

print("Training file:", TRAIN_PATH)
print("Testing file:", TEST_PATH)


# Check that the train and test files exist before proceeding
if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")
if not TEST_PATH.exists():
    raise FileNotFoundError(f"Testing file not found: {TEST_PATH}")


# Load the train and test data from the workflow artifact
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# Define the target column and split features/labels
TARGET = "ProdTaken"
if TARGET not in train_df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


# Remove unnecessary identifier columns if present
if "CustomerID" in X_train.columns:
    X_train = X_train.drop(columns=["CustomerID"])
    X_test = X_test.drop(columns=["CustomerID"])
    print("Removed CustomerID from model features.")


# Identify categorical and numerical columns for preprocessing
categorical_columns = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_columns = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()
print("\nCategorical columns:", categorical_columns)
print("\nNumerical columns:", numerical_columns)


# Build preprocessing pipeline for categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numerical", "passthrough", numerical_columns)
    ]
)


# Define a model and parameters
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
    ]
)


# Define hyperparameter grid for tuning
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}


# Set up MLflow experiment tracking
mlflow.set_experiment("Tourism Package Prediction")

with mlflow.start_run():

    # Tune the model with the defined parameters
    print("\nStarting hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Extract the best model after tuning
    best_model = grid_search.best_estimator_
    print("\nBest parameters:", grid_search.best_params_)

    # Generate predictions and probabilities on the test set
    y_pred = best_model.predict(X_test)
    y_probability = best_model.predict_proba(X_test)[:, 1]

    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probability)

    print("\nModel Evaluation")
    print("================")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC-AUC  :", roc_auc)

    # Log all the tuned parameters
    mlflow.log_params(grid_search.best_params_)

    # Log evaluation metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Save the best model so the pipeline can commit it to the repository
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print("\nModel saved successfully:", MODEL_PATH)

    # Log the saved model file as an artifact in MLflow
    mlflow.log_artifact(str(MODEL_PATH), artifact_path="deployment_model")


print("\n==================================================")
print("MODEL TRAINING COMPLETE")
print("==================================================")
