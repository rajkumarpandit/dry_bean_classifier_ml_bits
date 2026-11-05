"""Interactive Streamlit dashboard for evaluating the saved Dry Bean models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from ml_common import build_models, calculate_metrics


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
METADATA_PATH = MODEL_DIR / "metadata.json"
SAMPLE_TEST_PATH = APP_DIR / "test_data.csv"

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")
st.title("Dry Bean Classifier Lab")
st.write(
    "Compare five classifiers on the UCI Dry Bean dataset. Upload labeled test rows "
    "to evaluate the saved models; uploaded data is never used for training."
)


@st.cache_data
def load_metadata() -> dict:
    with METADATA_PATH.open(encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


model_specs = build_models()
missing_artifacts = [
    str(MODEL_DIR / filename)
    for filename, _ in model_specs.values()
    if not (MODEL_DIR / filename).is_file()
]
if not METADATA_PATH.is_file():
    missing_artifacts.append(str(METADATA_PATH))

if missing_artifacts:
    st.error(
        "Trained model files are missing. Run "
        "`python model_tr_and_eval.py` from this app folder, then restart the app."
    )
    st.code("\n".join(missing_artifacts))
    st.stop()

metadata = load_metadata()
target_column = metadata["target_column"]
feature_columns = metadata["feature_columns"]
class_labels = metadata["class_labels"]

st.sidebar.header("Evaluate a saved model")
model_choice = st.sidebar.selectbox("Classifier", list(model_specs))
uploaded_file = st.sidebar.file_uploader(
    f"Upload labeled test CSV (must include {target_column})",
    type=["csv"],
)

if SAMPLE_TEST_PATH.is_file():
    st.sidebar.download_button(
        "Download sample test CSV",
        data=SAMPLE_TEST_PATH.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )

st.caption(
    f"Dataset: UCI Dry Bean · {len(feature_columns)} numeric features · "
    f"{len(class_labels)} classes · target column: `{target_column}`"
)

comparison_path = MODEL_DIR / "model_comparison.csv"
if comparison_path.is_file():
    st.subheader("Saved-model comparison on the stratified holdout set")
    comparison = pd.read_csv(comparison_path).set_index("Model")
    st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)
    st.caption(
        "Precision, recall, and F1 use macro averaging; multiclass AUC uses "
        "one-vs-rest with macro averaging."
    )

if uploaded_file is None:
    st.info(
        "Upload the generated `test_data.csv` (or another labeled test CSV) to begin.\n\n"
        f"The file must include the `{target_column}` column and at least one row for "
        "each of these classes:\n\n"
        + "\n".join(f"- {label}" for label in class_labels)
    )
    st.stop()

try:
    test_data = pd.read_csv(uploaded_file)
except (UnicodeDecodeError, pd.errors.ParserError) as error:
    st.error(f"Could not read the uploaded file as a valid CSV: {error}")
    st.stop()

if test_data.empty:
    st.error("The uploaded CSV has no data rows.")
    st.stop()

if target_column not in test_data.columns:
    st.error(f"The uploaded CSV must include the target column `{target_column}`.")
    st.stop()

missing_columns = sorted(set(feature_columns) - set(test_data.columns))
extra_columns = sorted(set(test_data.columns) - set(feature_columns) - {target_column})
if missing_columns or extra_columns:
    if missing_columns:
        st.error(f"Missing required feature columns: {', '.join(missing_columns)}")
    if extra_columns:
        st.error(f"Unexpected columns in the CSV: {', '.join(extra_columns)}")
    st.stop()

X_test = test_data.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
invalid_columns = X_test.columns[X_test.isna().any()].tolist()
if invalid_columns:
    st.error(
        "Features must contain numeric, non-missing values. Check: "
        f"{', '.join(invalid_columns)}"
    )
    st.stop()
if not np.isfinite(X_test.to_numpy(dtype=float)).all():
    st.error("Feature values must be finite numbers (no positive or negative infinity).")
    st.stop()

y_test = test_data[target_column].astype(str).str.strip()
if y_test.eq("").any():
    st.error(f"Target column `{target_column}` contains blank labels.")
    st.stop()
unknown_labels = sorted(set(y_test) - set(class_labels))
missing_labels = sorted(set(class_labels) - set(y_test))
if unknown_labels or missing_labels:
    if unknown_labels:
        st.error(f"Unknown target labels: {', '.join(unknown_labels)}")
    if missing_labels:
        st.error(
            "To calculate multiclass AUC, the test CSV must include at least one "
            f"row for every class. Missing: {', '.join(missing_labels)}"
        )
    st.stop()

filename, _ = model_specs[model_choice]
model = load_model(str(MODEL_DIR / filename))
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
classes = model.classes_
metrics = calculate_metrics(y_test, predictions, probabilities, classes)

st.subheader(f"Evaluation results: {model_choice}")
metric_columns = st.columns(6)
for column, (metric_name, value) in zip(metric_columns, metrics.items()):
    column.metric(metric_name, f"{value:.4f}")

st.subheader("Test data preview")
st.dataframe(test_data.head(10), use_container_width=True)

left, right = st.columns(2)
with left:
    st.markdown("#### Confusion matrix")
    matrix = confusion_matrix(y_test, predictions, labels=class_labels)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(class_labels)),
        yticks=range(len(class_labels)),
        xticklabels=class_labels,
        yticklabels=class_labels,
        xlabel="Predicted label",
        ylabel="True label",
    )
    axis.tick_params(axis="x", labelrotation=45)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

with right:
    st.markdown("#### Classification report")
    report = classification_report(
        y_test,
        predictions,
        labels=class_labels,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose()
    st.dataframe(report_frame.style.format("{:.4f}"), use_container_width=True)