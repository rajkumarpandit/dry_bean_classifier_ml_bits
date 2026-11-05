"""Train all required classifiers and save deployment-ready artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from dataset_gen_and_preprc import (
    DEFAULT_DATASET_PATH,
    DEFAULT_TEST_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    load_and_split_dataset,
)
from ml_common import build_models, calculate_metrics


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"
METADATA_PATH = MODEL_DIR / "metadata.json"


def train_models(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
    model_dir: Path = MODEL_DIR,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> pd.DataFrame:
    """Split the source dataset, train/save each model, and save holdout metrics."""
    X_train, X_test, y_train, y_test = load_and_split_dataset(
        dataset_path=dataset_path,
        test_path=test_path,
        test_size=test_size,
        random_state=random_state,
    )
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    results = []
    model_specs = build_models(random_state=random_state)
    for model_name, (filename, model) in model_specs.items():
        model.fit(X_train, y_train)
        joblib.dump(model, model_dir / filename)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        metrics = calculate_metrics(
            y_test, predictions, probabilities, model.classes_
        )
        results.append({"Model": model_name, **metrics})

    comparison = pd.DataFrame(results)
    comparison.to_csv(model_dir / "model_comparison.csv", index=False)

    metadata = {
        "target_column": TARGET_COLUMN,
        "feature_columns": X_train.columns.tolist(),
        "class_labels": sorted(y_train.unique().tolist()),
        "test_size": test_size,
        "random_state": random_state,
        "metrics_averaging": "Macro average; multiclass AUC uses one-vs-rest.",
        "scikit_learn_version": sklearn.__version__,
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    save_eda_artifacts(X_train, X_test, y_train, y_test, model_dir, random_state)
    return comparison


def save_eda_artifacts(X_train, X_test, y_train, y_test, model_dir, random_state) -> None:
    """Save compact summaries so the app can show EDA without the raw dataset."""
    features = pd.concat([X_train, X_test])
    labels = pd.concat([y_train, y_test]).rename(TARGET_COLUMN)
    full = features.assign(**{TARGET_COLUMN: labels})

    summary = {
        "n_rows": int(len(full)),
        "class_counts": labels.value_counts().to_dict(),
        "statistics": features.describe().T.round(6).to_dict(orient="index"),
        "correlation": features.corr().round(4).to_dict(),
        "feature_min": features.min().to_dict(),
        "feature_max": features.max().to_dict(),
        "class_medians": full.groupby(TARGET_COLUMN).median().to_dict(orient="index"),
    }
    (model_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    sample = full.groupby(TARGET_COLUMN, group_keys=False).sample(
        n=300, replace=False, random_state=random_state
    )
    sample.round(6).to_csv(model_dir / "eda_sample.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    comparison = train_models(
        dataset_path=args.dataset,
        test_path=args.test_output,
        model_dir=args.model_dir,
        random_state=args.random_state,
        test_size=args.test_size,
    )
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved model files and metadata to: {args.model_dir.resolve()}")
    print(f"Saved upload-ready test data to: {args.test_output.resolve()}")


if __name__ == "__main__":
    main()