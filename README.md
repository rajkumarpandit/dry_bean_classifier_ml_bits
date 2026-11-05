# Dry Bean Classifier Lab

An interactive Streamlit application for comparing five classification algorithms on the UCI Dry Bean dataset. The app evaluates uploaded, labeled test rows with saved pipelines; it does not retrain models from uploads.

## Problem statement

Use the same public classification dataset to train and compare Logistic Regression, Decision Tree, K-Nearest Neighbor, Gaussian Naive Bayes, and Support Vector Machine classifiers. Report Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC), and provide an interactive application for evaluating labeled test data.

## Dataset description

The [UCI Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) contains 13,611 records, 16 numeric shape features, and seven bean classes. It has no missing values and exceeds the assignment minimum of 500 instances and 12 features. The target column is `Class`.

The downloaded dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite the UCI Machine Learning Repository dataset page and retain attribution when redistributing it.

## GitHub Repository Link

https://github.com/rajkumarpandit/dry_bean_classifier_ml_bits

## Repository structure

```text
dry_bean_classifier_ml_bits/
│-- app.py                          # Streamlit application
│-- requirements.txt                # Python dependencies
│-- README.md                       # Project documentation
│-- test_data.csv                   # Labeled holdout data for uploading to the app
│-- dataset_gen_and_preprc.py       # Source CSV validation and stratified train/test split
│-- model_tr_and_eval.py            # Model training, evaluation and artifact export
│-- ml_common.py                    # Shared model and metric definitions
│-- LICENSE
│-- model/                          # Saved trained models and preprocessing artifacts
    │-- logistic_regression.joblib
    │-- decision_tree_classifier.joblib
    │-- k_nearest_neighbor_classifier.joblib
    │-- gaussian_naive_bayes.joblib
    │-- support_vector_machine.joblib
    │-- metadata.json               # Feature columns, class labels and target name
    │-- model_comparison.csv        # Holdout metrics for all five models
    │-- eda_summary.json            # Statistics and correlations for the Data explorer
    │-- eda_sample.csv              # Stratified sample used for the box plots
```

- Each `.joblib` file is a complete scikit-learn pipeline, so the feature scaling fitted on the training data is saved together with the classifier.
- The raw source file (`DryBeanDataset/Dry_Bean_Dataset.csv`) is not committed. Download it from the UCI link above only if you want to retrain.

## Running the project

Install the dependencies with `pip install -r requirements.txt`, then start the app. The trained models are included in `model/`, so no training is needed:

```bash
streamlit run app.py
```

To retrain the models (optional), download the dataset from the UCI link above into `DryBeanDataset/Dry_Bean_Dataset.csv` and run:

```bash
python model_tr_and_eval.py
```

Training uses a stratified 80/20 split with random seed 42 and regenerates `test_data.csv` and the `model/` artifacts.

## Using the app

The app has three tabs, and the classifier selected in the sidebar applies to the first two:

- **Evaluate models** — a holdout comparison table and bar chart for all five models, followed by evaluation of an uploaded labeled test CSV with the selected model. The CSV must contain all 16 feature columns and the `Class` target column, with at least one row for each of the seven classes so multiclass AUC can be calculated. The app reports macro-averaged Precision, Recall, and F1, macro one-vs-rest AUC, MCC, an interactive confusion matrix, and a classification report.
- **Single-bean predictor** — set the 16 measurements with sliders (or start from the typical bean of a class) and see the predicted bean type with class probabilities.
- **Data explorer** — class distribution, per-class feature box plots, a correlation heatmap with the strongest feature pairs, and summary statistics.

Charts are interactive (hover for exact values). The sidebar also offers a sample test CSV to download.

## Models and evaluation

The training script saves a CSV comparison table to `model/model_comparison.csv`. The app shows those held-out training results and recomputes all six metrics for the selected saved model on the uploaded test data. Standardization is fitted within each saved model pipeline using training data only.

| Model | Accuracy | AUC Score | Precision (macro) | Recall (macro) | F1 Score (macro) | MCC Score |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9948 | 0.9349 | 0.9314 | 0.9329 | 0.9042 |
| Decision Tree Classifier | 0.8920 | 0.9450 | 0.9075 | 0.9090 | 0.9081 | 0.8696 |
| K-Nearest Neighbor Classifier | 0.9166 | 0.9833 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Gaussian Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Support Vector Machine | 0.9221 | 0.9910 | 0.9359 | 0.9330 | 0.9343 | 0.9059 |

## Live app

https://drybeanclassifier-rajkumarpandit-ml-bits.streamlit.app/
