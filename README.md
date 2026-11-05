# Dry Bean Classifier Lab

An interactive Streamlit application for comparing five classification algorithms on the UCI Dry Bean dataset. The app evaluates uploaded, labeled test rows with saved pipelines; it does not retrain models from uploads.

## Problem statement

Use the same public classification dataset to train and compare Logistic Regression, Decision Tree, K-Nearest Neighbor, Gaussian Naive Bayes, and Support Vector Machine classifiers. Report Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC), and provide an interactive application for evaluating labeled test data.

## Dataset description

The [UCI Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) contains 13,611 records, 16 numeric shape features, and seven bean classes. It has no missing values and exceeds the assignment minimum of 500 instances and 12 features. The target column is `Class`.

The downloaded dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite the UCI Machine Learning Repository dataset page and retain attribution when redistributing it.

## GitHub Repository Link

https://github.com/rajkumarpandit/dry_bean_classifier_ml_bits

## Repository contents

- `app.py` — Streamlit application.
- `dataset_gen_and_preprc.py` — validates the source CSV and creates a stratified train/test split.
- `model_tr_and_eval.py` — trains models, saves artifacts, and writes the comparison table.
- `ml_common.py` — shared model and metric definitions.
- `DryBeanDataset/Dry_Bean_Dataset.csv` — source data (not committed; download it from the UCI link above to retrain).
- `test_data.csv` — generated labeled holdout data for the app upload.
- `model/` — five saved model pipelines, metadata, and holdout comparison table. Generate these files by running the training command below.

## Train and run locally

Use the project virtual environment and run these commands from this folder:

```powershell
& "C:\raj\my_project\machine_learning\venv\Scripts\python.exe" model_tr_and_eval.py
& "C:\raj\my_project\machine_learning\venv\Scripts\python.exe" -m streamlit run app.py
```

By default, training reads `DryBeanDataset/Dry_Bean_Dataset.csv`, uses a stratified 80/20 split with random seed 42, and writes `test_data.csv` and the model artifacts. To use different paths:

```powershell
python model_tr_and_eval.py --dataset "path\to\dataset.csv" --test-output "test_data.csv" --model-dir "model"
```

The uploaded CSV must contain all 16 feature columns and the `Class` target column. It must include at least one row for each of the seven classes so that multiclass AUC can be calculated. The app reports macro-averaged Precision, Recall, and F1, and macro one-vs-rest AUC.

## Models and evaluation

The training script saves a CSV comparison table to `model/model_comparison.csv`. The app shows those held-out training results and recomputes all six metrics for the selected saved model on the uploaded test data. Standardization is fitted within each saved model pipeline using training data only.

| Model | Accuracy | AUC Score | Precision (macro) | Recall (macro) | F1 Score (macro) | MCC Score |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9948 | 0.9349 | 0.9314 | 0.9329 | 0.9042 |
| Decision Tree Classifier | 0.8920 | 0.9450 | 0.9075 | 0.9090 | 0.9081 | 0.8696 |
| K-Nearest Neighbor Classifier | 0.9166 | 0.9833 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Gaussian Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Support Vector Machine | 0.9221 | 0.9910 | 0.9359 | 0.9330 | 0.9343 | 0.9059 |

## Streamlit Community Cloud deployment

1. Run `python model_tr_and_eval.py` locally and check that `test_data.csv` and all five files in `model/` were created.
2. Create a GitHub repository and push this app folder, including `app.py`, `requirements.txt`, `README.md`, `ml_common.py`, both training scripts, the `model/` artifacts, and `test_data.csv`. The source dataset can also be included; it is small and publicly licensed with attribution.
3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with the GitHub account that can access the repository.
4. Choose **Create app** (or **New app**), select the repository and branch, and set the app file to `app.py`.
5. In **Advanced settings**, choose a Python version supported by Streamlit Community Cloud and compatible with the pinned packages in `requirements.txt`.
6. Deploy. Wait for the build to finish, then open the app URL and upload `test_data.csv` to verify the metrics and confusion matrix.
7. Include the repository URL, deployed app URL, and the required BITS Virtual Lab execution screenshot in the assignment submission PDF.

If dependency versions are changed, retrain the saved models with those same versions before redeploying. This avoids incompatibilities between serialized scikit-learn models and the deployment environment.

The assignment requires execution on BITS Virtual Lab. Repeat the required training/evaluation there and capture the requested screenshot for your submission.
