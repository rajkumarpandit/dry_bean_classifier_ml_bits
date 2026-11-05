"""Shared model definitions and multiclass evaluation helpers."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_models(random_state: int = 42) -> dict[str, tuple[str, Pipeline]]:
    """Create the five required classifiers, each with training-only scaling."""
    return {
        "Logistic Regression": (
            "logistic_regression.joblib",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=2000, random_state=random_state)),
                ]
            ),
        ),
        "Decision Tree Classifier": (
            "decision_tree_classifier.joblib",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", DecisionTreeClassifier(random_state=random_state)),
                ]
            ),
        ),
        "K-Nearest Neighbor Classifier": (
            "k_nearest_neighbor_classifier.joblib",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", KNeighborsClassifier(n_neighbors=5)),
                ]
            ),
        ),
        "Gaussian Naive Bayes": (
            "gaussian_naive_bayes.joblib",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", GaussianNB()),
                ]
            ),
        ),
        "Support Vector Machine": (
            "support_vector_machine.joblib",
            Pipeline(
                [
                    (
                        "classifier",
                        CalibratedClassifierCV(
                            estimator=Pipeline(
                                [
                                    ("scaler", StandardScaler()),
                                    ("classifier", SVC(random_state=random_state)),
                                ]
                            ),
                            method="sigmoid",
                            cv=3,
                            ensemble=False,
                        ),
                    ),
                ]
            ),
        ),
    }


def calculate_metrics(y_true, y_pred, probabilities, classes) -> dict[str, float]:
    """Calculate the assignment metrics, using macro averages for multiclass scores."""
    if len(classes) < 2:
        raise ValueError("At least two target classes are required to calculate AUC.")
    if len(classes) == 2:
        auc = roc_auc_score(y_true, probabilities[:, 1])
    else:
        auc = roc_auc_score(
            y_true,
            probabilities,
            labels=list(classes),
            multi_class="ovr",
            average="macro",
        )

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC Score": auc,
        "Precision (macro)": precision_score(
            y_true, y_pred, labels=list(classes), average="macro", zero_division=0
        ),
        "Recall (macro)": recall_score(
            y_true, y_pred, labels=list(classes), average="macro", zero_division=0
        ),
        "F1 Score (macro)": f1_score(
            y_true, y_pred, labels=list(classes), average="macro", zero_division=0
        ),
        "MCC Score": matthews_corrcoef(y_true, y_pred),
    }
