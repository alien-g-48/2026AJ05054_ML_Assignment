"""Train the five assignment classifiers and save everything needed by app.py.

Run from the project folder: python train.py
"""

import argparse
import csv
import hashlib
import json
import pickle
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import type_of_target

from dataset_source import load_phishing_dataset
from utils import MAX_CSV_BYTES, MAX_ROWS, ROOT, TARGET, MODEL_FILES, validate_data, evaluate_models

SEED = 2026


def split_dataset(features, target, test_size=0.2, custom=False):
    """Keep identical feature vectors together to prevent duplicate leakage."""
    # Preserve the original dataset's exact partition. For new datasets, hash
    # cells rather than joining text so embedded commas cannot merge groups.
    feature_keys = pd.util.hash_pandas_object(features, index=False) if custom else features.astype(str).agg(",".join, axis=1)
    groups = pd.factorize(feature_keys, sort=True)[0]
    n_splits = max(2, round(1 / test_size)) if custom else 5
    if custom:
        group_counts = pd.DataFrame({"class": target, "group": groups}).groupby("class")["group"].nunique()
        if group_counts.min() < n_splits:
            raise ValueError(f"Each class needs at least {n_splits} distinct feature groups for this test size. Remove duplicate rows or increase --test-size.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    if custom:
        candidates = [
            (train_index, test_index)
            for train_index, test_index in splitter.split(features, target, groups)
            if len(np.unique(target[train_index])) == len(np.unique(target))
            and len(np.unique(target[test_index])) == len(np.unique(target))
        ]
        if not candidates:
            raise ValueError("A grouped split containing every class in both partitions could not be created.")
        train_index, test_index = min(candidates, key=lambda indices: abs(len(indices[1]) / len(features) - test_size))
    else:
        train_index, test_index = next(splitter.split(features, target, groups))
    assert not set(groups[train_index]) & set(groups[test_index])
    return train_index, test_index, groups


def create_models(feature_types=None, multiclass=False):
    # Each saved pipeline contains its fitted preprocessing and classifier.
    # All parameters are fixed baselines, not chosen using the test scores.
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", "passthrough"),
            ("classifier", DecisionTreeClassifier(max_depth=12, min_samples_leaf=3, random_state=SEED)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=15, weights="distance", algorithm="ball_tree", n_jobs=1)),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", "passthrough"),
            ("classifier", GaussianNB(var_smoothing=1e-8)),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=3.0, kernel="rbf", gamma="scale", probability=multiclass, random_state=SEED)),
        ]),
    }
    if feature_types is not None:
        numeric = [name for name, kind in feature_types.items() if kind == "numeric"]
        categorical = [name for name, kind in feature_types.items() if kind == "categorical"]
        for name, model in models.items():
            # Each pipeline owns its encoder/scaler. They are fitted only when
            # model.fit(x_train, y_train) is called, never on test data.
            transformers = []
            if numeric:
                scale = StandardScaler() if name in ("Logistic Regression", "KNN", "Support Vector Machine") else "passthrough"
                transformers.append(("numeric", scale, numeric))
            if categorical:
                transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
            models[name] = Pipeline([
                ("preprocessor", ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)),
                ("classifier", model.named_steps["classifier"]),
            ])
    return models


def count_classes(target):
    return {str(label): int(count) for label, count in pd.Series(target).value_counts().sort_index().items()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train five classifiers locally and save a reusable dataset profile.")
    parser.add_argument("--data", type=Path, help="Full labeled training CSV for a new classification dataset.")
    parser.add_argument("--target", help="Target column in the training CSV (required with --data).")
    parser.add_argument("--name", help="Display name for the dataset.")
    parser.add_argument("--positive-class", help="Binary class treated as positive, as written in the target column.")
    parser.add_argument("--output", "--output-dir", type=Path, help="Profile folder; default: model_profiles/<dataset-name> for custom data.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Approximate grouped holdout fraction, between 0.1 and 0.5 (default: 0.2).")
    args = parser.parse_args(argv)
    if args.data is not None and not args.target:
        parser.error("--target is required when --data is supplied.")
    if args.data is None and any([args.target, args.name, args.positive_class]):
        parser.error("--target, --name and --positive-class require --data.")
    if not 0.1 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.1 and 0.5.")
    if args.data is None and args.test_size != 0.2:
        parser.error("The bundled assignment's split is fixed. Use --data for a custom split.")
    return args


def prepare_custom_dataset(frame, args):
    """Record a tabular classification schema without learning preprocessing."""
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    if args.target not in frame.columns:
        raise ValueError(f"Target column {args.target!r} is missing from the CSV.")
    if len(frame) < 500 or len(frame.columns) - 1 < 12:
        raise ValueError("The assignment requires at least 500 rows and 12 input features, excluding the target.")
    if frame.isna().any().any():
        raise ValueError("Missing values are not supported. Clean missing features and target labels before training.")
    if type_of_target(frame[args.target]) not in ("binary", "multiclass"):
        raise ValueError("The target must be a binary or multiclass classification label, not a continuous value.")
    classes = np.unique(frame[args.target]).tolist()
    if len(classes) < 2:
        raise ValueError("The target must contain at least two classes.")
    if pd.Series(frame[args.target]).value_counts().min() < 5:
        raise ValueError("Each target class needs at least five rows to create a stratified holdout.")
    positive = classes[-1] if len(classes) == 2 else None
    if args.positive_class is not None:
        if len(classes) != 2:
            raise ValueError("--positive-class applies only to binary classification datasets.")
        matches = [value for value in classes if str(value) == args.positive_class]
        if not matches:
            raise ValueError(f"--positive-class must be one of {classes}.")
        positive = matches[0]
    features = frame.columns.drop(args.target).tolist()
    feature_types = {name: "numeric" if pd.api.types.is_numeric_dtype(frame[name]) else "categorical" for name in features}
    metadata = {
        "target_column": args.target,
        "target_labels": {str(value): str(value) for value in classes},
        "classes": classes,
        "positive_class": positive,
        "metric_average": "binary" if len(classes) == 2 else "weighted",
        "feature_names": features,
        "feature_types": feature_types,
    }
    # Use exactly the same conversion rules later applied to uploaded CSVs.
    features, target = validate_data(frame, metadata, max_rows=None)
    return features, target, metadata


def main(argv=None):
    args = parse_args(argv)
    custom = args.data is not None
    source = {}
    data_path = args.data.expanduser().resolve() if custom else None
    if custom:
        # pandas otherwise renames duplicate CSV headers before validation.
        with data_path.open(newline="", encoding="utf-8-sig") as source_file:
            header = next(csv.reader(source_file), [])
        if len(header) != len(set(header)):
            raise ValueError("Duplicate column names are not allowed.")
        if not header or any(not name.strip() for name in header):
            raise ValueError("Every CSV column must have a non-empty name.")
        frame = pd.read_csv(data_path)
    else:
        print("Downloading UCI Phishing Websites directly into memory...")
        frame, source = load_phishing_dataset()
    if custom:
        features, target, schema = prepare_custom_dataset(frame, args)
        dataset_name = args.name or data_path.stem.replace("_", " ").strip()
        slug = re.sub(r"[^a-z0-9]+", "-", dataset_name.lower()).strip("-") or "custom-dataset"
        output = args.output or ROOT / "model_profiles" / slug
    else:
        features, target = validate_data(frame, source)
        dataset_name = source["name"]
        output = args.output or ROOT / "model"
        schema = {
            "target_column": TARGET,
            "target_labels": {"0": "Legitimate", "1": "Phishing"},
            "classes": [0, 1], "positive_class": 1, "metric_average": "binary",
            "feature_names": source["feature_names"],
            "feature_values": source["feature_values"],
            "feature_types": {name: "numeric" for name in source["feature_names"]},
        }
    output = output.expanduser().resolve()
    if custom and output == (ROOT / "model").resolve():
        raise ValueError("Choose a different --output folder to preserve the bundled assignment models.")
    if custom and data_path == output / "test_data.csv":
        raise ValueError("Choose a different --output folder so the held-out test CSV cannot overwrite your full training CSV.")

    # 2. Reserve one common held-out partition for all five classifiers.
    train_index, test_index, groups = split_dataset(features, target, args.test_size, custom=custom)
    x_train, x_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = target[train_index], target[test_index]
    # Reject an unusable export before fitting or replacing any saved models.
    # The exact same held-out CSV must be downloadable and uploadable in the app.
    if len(test_index) > MAX_ROWS:
        raise ValueError(f"The held-out test set exceeds {MAX_ROWS:,} rows. Use a smaller --test-size or a smaller representative dataset for this local app.")
    test_csv_bytes = frame.iloc[test_index].to_csv(index=False).encode("utf-8")
    if len(test_csv_bytes) > MAX_CSV_BYTES:
        raise ValueError("The held-out test CSV exceeds 5 MB. Use a smaller --test-size or reduce the dataset size before training.")

    # 3. Fit only on training rows, then save complete pipelines.
    output.mkdir(parents=True, exist_ok=True)
    models = create_models(schema["feature_types"] if custom else None, multiclass=len(schema["classes"]) > 2)
    for name, model in models.items():
        model.fit(x_train, y_train)
        with (output / MODEL_FILES[name]).open("wb") as model_file:
            pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Trained and saved: {name}")

    # 4. Evaluate every model on the same test data and export that labeled CSV.
    table, details = evaluate_models(models, x_test, y_test, schema)
    table.to_csv(output / "metrics.csv", index=False, float_format="%.10f")
    (output / "test_data.csv").write_bytes(test_csv_bytes)
    if not custom and output == (ROOT / "model").resolve():
        (ROOT / "test_data.csv").write_bytes(test_csv_bytes)
    conflicts = pd.DataFrame({"group": groups, "target": target}).groupby("group")["target"].nunique()
    metadata = {
        "dataset": {
            "name": dataset_name, "url": source.get("source_url", ""),
            "rows": len(frame), "feature_count": len(features.columns),
            "license": source.get("license", "Not recorded; check your dataset source"), "doi": source.get("doi", ""),
            "csv_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest() if custom else source["csv_sha256"],
            **({"download_url": source["download_url"], "archive_sha256": source["archive_sha256"], "arff_sha256": source["arff_sha256"]} if not custom else {}),
        },
        **schema,
        "test_data_file": "test_data.csv",
        "split": {
            "method": (f"Closest held-out fold of shuffled StratifiedGroupKFold(n_splits={max(2, round(1 / args.test_size))}); identical feature vectors form a group." if custom else "First held-out fold of shuffled StratifiedGroupKFold(n_splits=5); identical feature vectors form a group."),
            "requested_test_fraction": args.test_size,
            "actual_test_fraction": len(test_index) / len(frame),
            "seed": SEED, "train_rows": len(train_index), "test_rows": len(test_index),
            "train_class_counts": count_classes(y_train), "test_class_counts": count_classes(y_test),
            "duplicate_feature_rows": len(frame) - len(np.unique(groups)),
            "conflicting_feature_groups": int((conflicts > 1).sum()),
            "feature_group_overlap": 0,
        },
        "training": {
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "serialization": "Python pickle", "pickle_protocol": pickle.HIGHEST_PROTOCOL,
            "hyperparameters": {name: model.named_steps["classifier"].get_params() for name, model in models.items()},
            "settings": "Fixed baselines; training-only scaling; no test-set tuning.",
        },
        "evaluation": {
            name: {"metrics": result["metrics"],
                   "confusion_matrix": result["confusion_matrix"].tolist(),
                   "classification_report": result["classification_report"]}
            for name, result in details.items()
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"\nTraining rows: {len(train_index):,}; test rows: {len(test_index):,}")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved dataset profile: {output}")


if __name__ == "__main__":
    main()
