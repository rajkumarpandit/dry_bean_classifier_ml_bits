"""Load and split the UCI Dry Bean classification dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = APP_DIR / "DryBeanDataset" / "Dry_Bean_Dataset.csv"
DEFAULT_TEST_PATH = APP_DIR / "test_data.csv"
TARGET_COLUMN = "Class"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_and_split_dataset(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the source CSV, validate its schema, and save a stratified test split."""
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset CSV was not found: {dataset_path}")

    data = pd.read_csv(dataset_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain the target column {TARGET_COLUMN!r}.")
    if data.shape[0] < 500 or data.shape[1] - 1 < 12:
        raise ValueError(
            "The assignment requires at least 500 rows and 12 input features; "
            f"found {data.shape[0]} rows and {data.shape[1] - 1} features."
        )
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column {TARGET_COLUMN!r} contains missing values.")

    features = data.drop(columns=[TARGET_COLUMN])
    non_numeric = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"All input features must be numeric; found: {non_numeric}")
    if features.isna().any().any():
        raise ValueError("Input features contain missing values.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Input features contain infinite values.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        data[TARGET_COLUMN].astype(str),
        test_size=test_size,
        random_state=random_state,
        stratify=data[TARGET_COLUMN],
    )

    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_path = Path(test_path)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_data.to_csv(test_path, index=False)
    return X_train, X_test, y_train, y_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_and_split_dataset(
        dataset_path=args.dataset,
        test_path=args.test_output,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Training rows: {len(X_train)}; test rows: {len(X_test)}")
    print(f"Input features: {X_train.shape[1]}; classes: {y_train.nunique()}")
    print(f"Saved upload-ready test data to: {args.test_output.resolve()}")


if __name__ == "__main__":
    main()
