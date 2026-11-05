"""Interactive Streamlit dashboard for the saved Dry Bean classification models."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from ml_common import build_models, calculate_metrics


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
METADATA_PATH = MODEL_DIR / "metadata.json"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"
EDA_SUMMARY_PATH = MODEL_DIR / "eda_summary.json"
EDA_SAMPLE_PATH = MODEL_DIR / "eda_sample.csv"
SAMPLE_TEST_PATH = APP_DIR / "test_data.csv"

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="\U0001FAD8", layout="wide")


@st.cache_data
def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as json_file:
        return json.load(json_file)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


PRIMARY_COLOR = "#FF4B4B"
TEXT_COLOR = "#31333F"
MUTED_COLOR = "#808495"
GRID_COLOR = "#E6EAF1"
MODEL_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]

def style_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
            labelColor=TEXT_COLOR,
            titleColor=MUTED_COLOR,
            gridColor=GRID_COLOR,
            domainColor=GRID_COLOR,
            tickColor=GRID_COLOR,
        )
        .configure_legend(
            labelFontSize=12,
            titleFontSize=12,
            labelColor=TEXT_COLOR,
            titleColor=MUTED_COLOR,
            orient="top",
            direction="horizontal",
            symbolType="square",
        )
    )


def show_chart(chart: alt.Chart) -> None:
    st.altair_chart(style_chart(chart), use_container_width=True)

model_specs = build_models()
missing_artifacts = [
    str(MODEL_DIR / filename)
    for filename, _ in model_specs.values()
    if not (MODEL_DIR / filename).is_file()
]
if not METADATA_PATH.is_file():
    missing_artifacts.append(str(METADATA_PATH))

st.title("Dry Bean Classifier Lab")
st.write(
    "Compare five classifiers on the UCI Dry Bean dataset, explore the data, and try "
    "a single-bean prediction. Uploaded data is only used for evaluation, never for training."
)

if missing_artifacts:
    st.error(
        "Trained model files are missing. Run "
        "`python model_tr_and_eval.py` from this app folder, then restart the app."
    )
    st.code("\n".join(missing_artifacts))
    st.stop()

metadata = load_json(str(METADATA_PATH))
target_column = metadata["target_column"]
feature_columns = metadata["feature_columns"]
class_labels = metadata["class_labels"]

st.sidebar.header("Controls")
model_choice = st.sidebar.selectbox(
    "Classifier",
    list(model_specs),
    help="Used for the evaluation tab and the single-bean predictor.",
)
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
st.sidebar.markdown(
    """
    <style>
    .sidebar-credit {
        position: fixed;
        bottom: 0.9rem;
        left: 1.5rem;
        font-size: 0.78rem;
        line-height: 1.5;
        color: #808495;
    }
    </style>
    <div class="sidebar-credit">
        Built by: Raj Kumar Pandit<br>BITS ID: 2026ak05569
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Dataset: UCI Dry Bean \u00b7 {len(feature_columns)} numeric features \u00b7 "
    f"{len(class_labels)} classes \u00b7 target column: `{target_column}`"
)

selected_model_file = MODEL_DIR / model_specs[model_choice][0]


def plot_metric_comparison(comparison: pd.DataFrame) -> None:
    model_names = list(comparison.index)
    metric_names = list(comparison.columns)
    lower = max(0.0, math.floor((comparison.to_numpy().min() - 0.02) * 20) / 20)
    long_form = comparison.reset_index(names="Model").melt(
        id_vars="Model", var_name="Metric", value_name="Score"
    )
    chart = (
        alt.Chart(long_form)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", sort=metric_names, axis=alt.Axis(labelAngle=0, title=None, ticks=False)),
            xOffset=alt.XOffset("Model:N", sort=model_names),
            y=alt.Y(
                "Score:Q",
                scale=alt.Scale(domain=[lower, 1.0], clamp=True),
                axis=alt.Axis(title=f"Score (axis starts at {lower:.2f})", format=".2f"),
            ),
            color=alt.Color(
                "Model:N",
                sort=model_names,
                scale=alt.Scale(domain=model_names, range=MODEL_COLORS),
                legend=alt.Legend(title=None),
            ),
            tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".4f")],
        )
        .properties(height=320)
    )
    show_chart(chart)


def render_evaluation() -> None:
    if COMPARISON_PATH.is_file():
        st.subheader("Saved-model comparison on the stratified holdout set")
        comparison = load_csv(str(COMPARISON_PATH)).set_index("Model")
        st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)
        plot_metric_comparison(comparison)
        st.caption(
            "Precision, recall, and F1 use macro averaging; multiclass AUC uses "
            "one-vs-rest with macro averaging."
        )

    st.subheader("Evaluate on your own test file")
    if uploaded_file is None:
        st.info(
            "Upload the generated `test_data.csv` (or another labeled test CSV) to begin.\n\n"
            f"The file must include the `{target_column}` column and at least one row for "
            "each of these classes:\n\n"
            + "\n".join(f"- {label}" for label in class_labels)
        )
        return

    try:
        test_data = pd.read_csv(uploaded_file)
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        st.error(f"Could not read the uploaded file as a valid CSV: {error}")
        return

    if test_data.empty:
        st.error("The uploaded CSV has no data rows.")
        return

    if target_column not in test_data.columns:
        st.error(f"The uploaded CSV must include the target column `{target_column}`.")
        return

    missing_columns = sorted(set(feature_columns) - set(test_data.columns))
    extra_columns = sorted(set(test_data.columns) - set(feature_columns) - {target_column})
    if missing_columns or extra_columns:
        if missing_columns:
            st.error(f"Missing required feature columns: {', '.join(missing_columns)}")
        if extra_columns:
            st.error(f"Unexpected columns in the CSV: {', '.join(extra_columns)}")
        return

    X_test = test_data.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    invalid_columns = X_test.columns[X_test.isna().any()].tolist()
    if invalid_columns:
        st.error(
            "Features must contain numeric, non-missing values. Check: "
            f"{', '.join(invalid_columns)}"
        )
        return
    if not np.isfinite(X_test.to_numpy(dtype=float)).all():
        st.error("Feature values must be finite numbers (no positive or negative infinity).")
        return

    y_test = test_data[target_column].astype(str).str.strip()
    if y_test.eq("").any():
        st.error(f"Target column `{target_column}` contains blank labels.")
        return
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
        return

    model = load_model(str(selected_model_file))
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    metrics = calculate_metrics(y_test, predictions, probabilities, model.classes_)

    st.markdown(f"#### Results for {model_choice}")
    for column, (metric_name, value) in zip(st.columns(6), metrics.items()):
        column.metric(metric_name, f"{value:.4f}")

    with st.expander("Test data preview"):
        st.dataframe(test_data.head(10), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Confusion matrix")
        matrix = confusion_matrix(y_test, predictions, labels=class_labels)
        cells = pd.DataFrame(
            [
                {"True": true, "Predicted": predicted, "Count": int(matrix[i, j])}
                for i, true in enumerate(class_labels)
                for j, predicted in enumerate(class_labels)
            ]
        )
        base = alt.Chart(cells).encode(
            x=alt.X("Predicted:N", sort=class_labels, axis=alt.Axis(labelAngle=-40, ticks=False)),
            y=alt.Y("True:N", sort=class_labels, axis=alt.Axis(ticks=False)),
        )
        heat = base.mark_rect().encode(
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["True", "Predicted", "Count"],
        )
        threshold = matrix.max() / 2 if matrix.size else 0
        labels = base.mark_text(fontSize=11).encode(
            text="Count:Q",
            color=alt.condition(
                alt.datum.Count > threshold, alt.value("white"), alt.value(TEXT_COLOR)
            ),
        )
        show_chart((heat + labels).properties(height=360))

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
        st.dataframe(
            pd.DataFrame(report).transpose().style.format("{:.4f}"),
            use_container_width=True,
        )


def render_data_explorer() -> None:
    if not EDA_SUMMARY_PATH.is_file() or not EDA_SAMPLE_PATH.is_file():
        st.info("EDA files are missing. Run `python model_tr_and_eval.py` to generate them.")
        return

    summary = load_json(str(EDA_SUMMARY_PATH))
    sample = load_csv(str(EDA_SAMPLE_PATH))

    first, second, third = st.columns(3)
    first.metric("Beans (rows)", f"{summary['n_rows']:,}")
    second.metric("Features", len(feature_columns))
    third.metric("Bean types", len(class_labels))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Class distribution")
        counts = pd.Series(summary["class_counts"]).reindex(class_labels)
        count_frame = counts.rename("Beans").rename_axis("Class").reset_index()
        bars = alt.Chart(count_frame).encode(
            x=alt.X("Class:N", sort=class_labels, axis=alt.Axis(labelAngle=-40, title=None, ticks=False)),
            y=alt.Y("Beans:Q", axis=alt.Axis(title="Number of beans")),
        )
        show_chart(
            (
                bars.mark_bar(color=MODEL_COLORS[0], size=28)
                + bars.mark_text(dy=-8, fontSize=11, color=TEXT_COLOR).encode(text="Beans:Q")
            ).properties(height=300)
        )
        ratio = counts.max() / counts.min()
        st.caption(
            f"The classes are imbalanced ({counts.idxmax()} has {ratio:.1f}x more beans "
            f"than {counts.idxmin()}), which is why macro-averaged metrics and MCC are reported."
        )

    with right:
        st.markdown("#### Feature distribution by class")
        feature = st.selectbox("Feature", feature_columns, key="eda_feature")
        box = (
            alt.Chart(sample[[target_column, feature]])
            .mark_boxplot(
                extent="min-max",
                size=28,
                color=MODEL_COLORS[0],
                median=alt.MarkConfig(color=PRIMARY_COLOR),
            )
            .encode(
                x=alt.X(f"{target_column}:N", sort=class_labels, axis=alt.Axis(labelAngle=-40, title=None, ticks=False)),
                y=alt.Y(f"{feature}:Q", scale=alt.Scale(zero=False), axis=alt.Axis(title=feature)),
            )
            .properties(height=300)
        )
        show_chart(box)
        st.caption("Box plots use a stratified sample of 300 beans per class.")

    st.markdown("#### Feature correlation")
    correlation = pd.DataFrame(summary["correlation"]).loc[feature_columns, feature_columns]
    heatmap_column, pairs_column = st.columns([3, 2])
    with heatmap_column:
        long_corr = correlation.reset_index(names="Feature A").melt(
            id_vars="Feature A", var_name="Feature B", value_name="Correlation"
        )
        heat = (
            alt.Chart(long_corr)
            .mark_rect()
            .encode(
                x=alt.X("Feature B:N", sort=feature_columns, axis=alt.Axis(labelAngle=-55, title=None, ticks=False)),
                y=alt.Y("Feature A:N", sort=feature_columns, axis=alt.Axis(title=None, ticks=False)),
                color=alt.Color(
                    "Correlation:Q",
                    scale=alt.Scale(scheme="redblue", domain=[1, -1]),
                    legend=alt.Legend(orient="right", direction="vertical", title=None, gradientLength=200),
                ),
                tooltip=["Feature A", "Feature B", alt.Tooltip("Correlation:Q", format="+.3f")],
            )
            .properties(height=440)
        )
        show_chart(heat)

    with pairs_column:
        upper = correlation.where(np.triu(np.ones(correlation.shape, dtype=bool), k=1))
        pairs = upper.stack().rename("Correlation").reset_index()
        pairs.columns = ["Feature A", "Feature B", "Correlation"]
        strongest = pairs.reindex(pairs["Correlation"].abs().sort_values(ascending=False).index)
        st.markdown("**Strongest feature pairs**")
        st.dataframe(
            strongest.head(10).style.format({"Correlation": "{:+.3f}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Many size features (Area, Perimeter, axis lengths, ConvexArea, EquivDiameter) "
            "are strongly correlated, so they carry overlapping information."
        )

    with st.expander("Summary statistics"):
        statistics = pd.DataFrame.from_dict(summary["statistics"], orient="index")
        st.dataframe(statistics.loc[feature_columns], use_container_width=True)


def render_predictor() -> None:
    if not EDA_SUMMARY_PATH.is_file():
        st.info("Feature ranges are missing. Run `python model_tr_and_eval.py` to generate them.")
        return

    summary = load_json(str(EDA_SUMMARY_PATH))
    minimums = summary["feature_min"]
    maximums = summary["feature_max"]
    defaults = {name: summary["statistics"][name]["50%"] for name in feature_columns}
    keys = {name: f"predict_{name}" for name in feature_columns}

    for name in feature_columns:
        st.session_state.setdefault(keys[name], float(defaults[name]))

    def apply_preset() -> None:
        preset = st.session_state["predict_preset"]
        if preset in summary["class_medians"]:
            for name in feature_columns:
                st.session_state[keys[name]] = float(summary["class_medians"][preset][name])

    st.write(
        f"Describe a bean with the sliders and see how **{model_choice}** classifies it. "
        "Change the classifier in the sidebar to compare models."
    )
    st.selectbox(
        "Start from the typical bean of a class",
        ["Custom"] + class_labels,
        key="predict_preset",
        on_change=apply_preset,
        help="Fills every slider with that class's median values.",
    )

    columns = st.columns(4)
    for index, name in enumerate(feature_columns):
        low, high = float(minimums[name]), float(maximums[name])
        columns[index % 4].slider(
            name, min_value=low, max_value=high, step=(high - low) / 1000, key=keys[name], format="%.4f"
        )

    row = pd.DataFrame([{name: st.session_state[keys[name]] for name in feature_columns}])
    model = load_model(str(selected_model_file))
    probabilities = pd.Series(model.predict_proba(row)[0], index=model.classes_)
    predicted = probabilities.idxmax()

    st.success(f"Predicted bean type: **{predicted}** ({probabilities[predicted]:.1%} confidence)")
    st.bar_chart(probabilities.sort_values(ascending=False), y_label="Probability")
    st.caption(
        "Bean measurements are strongly correlated, so arbitrary slider combinations can be "
        "physically unrealistic. For meaningful predictions, start from a class preset and "
        "adjust gradually."
    )


evaluate_tab, predictor_tab, explorer_tab = st.tabs(
    ["Evaluate models", "Single-bean predictor", "Data explorer"]
)
with evaluate_tab:
    render_evaluation()
with predictor_tab:
    render_predictor()
with explorer_tab:
    render_data_explorer()
