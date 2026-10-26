"""CSV validation and evaluation shared by training and the local app."""

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

ROOT = Path(__file__).resolve().parent
TARGET = "is_phishing"
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
MAX_ROWS = 20_000
MAX_CSV_BYTES = 5 * 1024 * 1024
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}


def load_artifacts(model_dir=None):
    """Load trusted local pipelines for one dataset; never load uploaded pickles."""
    folder = Path(model_dir).expanduser().resolve() if model_dir is not None else ROOT / "model"
    metadata = json.loads((folder / "metadata.json").read_text())
    models = {}
    for name, filename in MODEL_FILES.items():
        with (folder / filename).open("rb") as model_file:
            models[name] = pickle.load(model_file)
    classes = list(next(iter(models.values())).classes_)
    metadata.setdefault("classes", [value.item() if isinstance(value, np.generic) else value for value in classes])
    metadata.setdefault("positive_class", metadata["classes"][-1] if len(classes) == 2 else None)
    metadata.setdefault("metric_average", "binary" if len(classes) == 2 else "weighted")
    return metadata, models


def validate_data(frame, metadata, max_rows=MAX_ROWS):
    """Check a labeled CSV against its fitted dataset's feature/target schema.

    Categorical features stay strings; numeric features are converted to numbers.
    This performs validation only: all fitted preprocessing lives in the model.
    """
    if frame.empty:
        raise ValueError("The CSV must contain at least one data row.")
    if max_rows is not None and len(frame) > max_rows:
        raise ValueError(f"Upload test data only, with at most {max_rows:,} rows.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")

    features = metadata["feature_names"]
    target_column = metadata.get("target_column", TARGET)
    expected = features + [target_column]
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        messages = []
        if missing:
            messages.append("Missing columns: " + ", ".join(missing))
        if extra:
            messages.append("Unexpected columns: " + ", ".join(map(str, extra)))
        raise ValueError(". ".join(messages) + ". Use test_data.csv as a template.")

    data = frame[expected].copy()
    if data.isna().any().any():
        raise ValueError("Missing values, NaN and infinity are not allowed.")
    feature_types = metadata.get("feature_types", {name: "numeric" for name in features})
    for name in features:
        if feature_types.get(name, "numeric") == "categorical":
            data[name] = data[name].astype(str)
        else:
            try:
                data[name] = pd.to_numeric(data[name], errors="raise")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Feature {name} must contain numeric values.") from exc
            if not np.isfinite(data[name].to_numpy(dtype=float)).all():
                raise ValueError("Missing values, NaN and infinity are not allowed.")
        allowed = metadata.get("feature_values", {}).get(name)
        if allowed is not None and not data[name].isin(allowed).all():
            raise ValueError(f"Invalid values in {name}. Allowed codes: {allowed}.")

    classes = metadata.get("classes")
    legacy = target_column == TARGET and "feature_values" in metadata and classes in (None, [0, 1])
    if legacy or (classes is not None and all(isinstance(value, (int, float)) for value in classes)):
        try:
            data[target_column] = pd.to_numeric(data[target_column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError("Target labels must be numeric for this dataset.") from exc
        if not np.isfinite(data[target_column].to_numpy(dtype=float)).all():
            raise ValueError("Missing values, NaN and infinity are not allowed.")
    elif classes is not None and all(isinstance(value, str) for value in classes):
        data[target_column] = data[target_column].astype(str)
    allowed_classes = [0, 1] if legacy else classes
    if allowed_classes is not None and not data[target_column].isin(allowed_classes).all():
        if legacy:
            raise ValueError("is_phishing must be 0 (legitimate) or 1 (phishing).")
        raise ValueError(f"Unknown target labels in {target_column}. Expected classes: {allowed_classes}.")
    if legacy:
        return data[features].astype(np.int8), data[target_column].to_numpy(dtype=np.int8)
    return data[features], data[target_column].to_numpy()


def evaluate_model(model, features, target, metadata=None):
    """Evaluate a fitted binary or multiclass model without fitting any data.

    Binary precision/recall/F1 use the selected positive class. Multiclass
    precision/recall/F1 and one-vs-rest AUC are support-weighted. AUC requires
    all fitted classes in the test data and continuous decision/probability
    scores, never thresholded predictions.
    """
    metadata = metadata or {}
    predictions = model.predict(features)
    classes = list(model.classes_)
    if len(classes) < 2:
        raise ValueError("Saved classifiers must contain at least two classes.")
    target = np.asarray(target)
    if not pd.Series(target).isin(classes).all():
        raise ValueError(f"Unknown target labels. Expected classes: {classes}.")
    binary = len(classes) == 2
    positive = metadata.get("positive_class", classes[-1]) if binary else None
    if binary and positive not in classes:
        raise ValueError("The positive class is not one of the fitted model's classes.")
    class_labels = metadata.get("target_labels", {})
    class_names = [str(class_labels.get(str(value), str(value))) for value in classes]
    scores = None
    auc = None
    auc_reason = None
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features))
        scores = probabilities[:, classes.index(positive)] if binary else probabilities
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features))
        if binary and positive != classes[-1]:
            scores = -scores
    if len(pd.unique(target)) != len(classes):
        auc_reason = "AUC requires test rows from every fitted class."
    elif scores is None:
        auc_reason = "This model does not expose probability or decision scores."
    elif binary:
        auc = float(roc_auc_score(target == positive, scores))
    elif hasattr(model, "predict_proba"):
        auc = float(roc_auc_score(target, scores, labels=classes, multi_class="ovr", average="weighted"))
    else:
        auc_reason = "Multiclass AUC requires fitted class probabilities. Retrain this profile with probability-enabled SVM."

    metric_options = {"average": "binary", "pos_label": positive} if binary else {"average": "weighted"}

    metrics = {
        "Accuracy": float(accuracy_score(target, predictions)),
        "AUC": auc,
        "Precision": float(precision_score(target, predictions, zero_division=0, **metric_options)),
        "Recall": float(recall_score(target, predictions, zero_division=0, **metric_options)),
        "F1": float(f1_score(target, predictions, zero_division=0, **metric_options)),
        "MCC": float(matthews_corrcoef(target, predictions)),
    }
    return {
        "metrics": metrics,
        "predictions": np.asarray(predictions),
        "scores": np.asarray(scores) if scores is not None else None,
        "classes": classes,
        "class_names": class_names,
        "auc_reason": auc_reason,
        "confusion_matrix": confusion_matrix(target, predictions, labels=classes),
        "classification_report": classification_report(
            target, predictions, labels=classes,
            target_names=class_names, output_dict=True, zero_division=0,
        ),
    }


def evaluate_models(models, features, target, metadata=None):
    details = {name: evaluate_model(model, features, target, metadata) for name, model in models.items()}
    table = pd.DataFrame([
        {"Model": name, **result["metrics"]} for name, result in details.items()
    ])
    return table[["Model", *METRICS]], details
