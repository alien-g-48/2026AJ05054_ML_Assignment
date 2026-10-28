"""Exercise locally trained dataset profiles, independent of the phishing example.

These synthetic datasets are test fixtures, not assignment dataset submissions.
All training artifacts are created in a temporary folder and removed afterward.
"""

from base64 import b64decode
import hashlib
from html.parser import HTMLParser
from io import BytesIO
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from streamlit.testing.v1 import AppTest
from matplotlib.figure import Figure

from downloads import sample_download_url
from test_downloads import assert_sample_download, sample_download_href
from train import main as train_main, prepare_custom_dataset
from utils import METRICS, MODEL_FILES, ROOT, evaluate_model, evaluate_models, load_artifacts, validate_data


class EmbeddedMatrixParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "img" and attributes.get("alt", "").endswith(" confusion matrix"):
            self.images.append(attributes)


def assert_no_comparison_chart(test, app):
    """The removed chart and its metric/replay controls must stay absent."""
    test.assertNotIn("comparison_metric", [item.key for item in app.selectbox])
    test.assertNotIn("replay_comparison", [item.key for item in app.button])
    test.assertFalse(any(item.value == "Animated metric comparison" for item in app.subheader))
    test.assertFalse(any('class="metric-comparison-chart"' in item.value for item in app.markdown))


class MulticlassProbabilityFixture:
    """Known class probabilities with some incorrect maximum-score predictions."""

    classes_ = np.array([101, 202, 303])
    probabilities = np.array([
        [0.70, 0.20, 0.10],
        [0.15, 0.65, 0.20],
        [0.10, 0.25, 0.65],
        [0.35, 0.55, 0.10],
        [0.05, 0.60, 0.35],
        [0.60, 0.10, 0.30],
        [0.50, 0.30, 0.20],
        [0.20, 0.10, 0.70],
    ])

    def predict(self, features):
        return self.classes_[self.probabilities[:len(features)].argmax(axis=1)]

    def predict_proba(self, features):
        return self.probabilities[:len(features)]


def make_fixture(multiclass=False):
    features, labels = make_classification(
        n_samples=600, n_features=10, n_informative=6, n_redundant=2,
        n_classes=3 if multiclass else 2, n_clusters_per_class=1,
        weights=[0.45, 0.35, 0.20] if multiclass else [0.65, 0.35],
        random_state=43 if multiclass else 31,
    )
    frame = pd.DataFrame(features, columns=[f"measurement_{index}" for index in range(10)])
    frame["protocol"] = np.array(["001", "002", "abc"])[np.arange(len(frame)) % 3]
    frame["region"] = np.where(np.arange(len(frame)) % 2, "north", "south")
    frame["event_status"] = np.array([101, 202, 303])[labels] if multiclass else np.array(["normal", "attack"])[labels]
    # Include genuine duplicate feature vectors to exercise the grouped split.
    return pd.concat([frame, frame.iloc[:60]], ignore_index=True)


class GenericDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = TemporaryDirectory(prefix="classification-profile-tests-")
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.folder = Path(cls.workspace.name)
        cls.profiles = {}
        for mode in ("binary", "multiclass"):
            source = make_fixture(multiclass=mode == "multiclass")
            data_path = cls.folder / f"{mode}.csv"
            source.to_csv(data_path, index=False)
            profile = cls.folder / mode
            command = [
                sys.executable, str(ROOT / "train.py"), "--data", str(data_path),
                "--target", "event_status", "--name", f"Synthetic {mode} test fixture",
                "--output", str(profile),
            ]
            if mode == "binary":
                command.extend(["--positive-class", "attack"])
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
            if result.returncode:
                raise AssertionError(f"Custom {mode} training failed:\n{result.stdout}\n{result.stderr}")
            metadata, models = load_artifacts(profile)
            test_data = pd.read_csv(profile / "test_data.csv")
            features, target = validate_data(test_data, metadata)
            comparison, details = evaluate_models(models, features, target, metadata)
            cls.profiles[mode] = {
                "path": profile, "source": pd.read_csv(data_path), "source_path": data_path,
                "metadata": metadata, "models": models, "test_data": test_data,
                "features": features, "target": target, "comparison": comparison, "details": details,
            }

    def test_cli_creates_complete_profiles_with_unmodified_holdout_and_no_group_leakage(self):
        for mode, profile in self.profiles.items():
            with self.subTest(dataset=mode):
                metadata = profile["metadata"]
                source = profile["source"]
                test_data = profile["test_data"]
                self.assertEqual(metadata["target_column"], "event_status")
                self.assertEqual(metadata["dataset"]["feature_count"], 12)
                self.assertEqual(metadata["dataset"]["rows"], 660)
                self.assertEqual(metadata["feature_types"]["protocol"], "categorical")
                self.assertEqual(metadata["feature_types"]["measurement_0"], "numeric")
                self.assertEqual(metadata["dataset"]["csv_sha256"], hashlib.sha256(profile["source_path"].read_bytes()).hexdigest())
                self.assertTrue(all((profile["path"] / filename).is_file() for filename in MODEL_FILES.values()))
                stored = pd.read_csv(profile["path"] / "metrics.csv")
                pd.testing.assert_frame_equal(profile["comparison"], stored, check_exact=False, atol=1e-9, rtol=1e-9)
                self.assertEqual(metadata["split"]["train_rows"] + len(test_data), len(source))
                feature_names = metadata["feature_names"]
                # Reading/writing floating point values can change a last bit, so
                # identify source rows by exact round-trip CSV representation.
                serialized_source = source[feature_names].round(9).astype(str).agg("|".join, axis=1)
                serialized_test = set(test_data[feature_names].round(9).astype(str).agg("|".join, axis=1))
                test_mask = serialized_source.isin(serialized_test)
                self.assertEqual(int(test_mask.sum()), len(test_data))
                self.assertFalse(set(serialized_source[~test_mask]) & serialized_test)
                self.assertEqual(metadata["split"]["feature_group_overlap"], 0)
                training_features = source.loc[~test_mask, feature_names]
                for name, model in profile["models"].items():
                    self.assertIsInstance(model, Pipeline)
                    np.testing.assert_array_equal(model.classes_, metadata["classes"])
                    preprocessing = model.named_steps["preprocessor"]
                    scaler = preprocessing.named_transformers_["numeric"]
                    if isinstance(scaler, StandardScaler):
                        self.assertEqual(int(scaler.n_samples_seen_), len(training_features))
                        np.testing.assert_allclose(scaler.mean_, training_features.iloc[:, :10].mean().to_numpy(), atol=1e-12)
                    encoder = preprocessing.named_transformers_["categorical"]
                    for feature, categories in zip(["protocol", "region"], encoder.categories_):
                        self.assertEqual(set(categories), set(training_features[feature].astype(str)))

    def test_binary_string_labels_and_first_sorted_positive_class_use_correct_scores(self):
        profile = self.profiles["binary"]
        self.assertEqual(profile["metadata"]["positive_class"], "attack")
        for name, model in profile["models"].items():
            with self.subTest(model=name):
                result = profile["details"][name]
                classes = list(model.classes_)
                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(profile["features"])[:, classes.index("attack")]
                else:
                    scores = -model.decision_function(profile["features"])
                self.assertAlmostEqual(result["metrics"]["AUC"], roc_auc_score(profile["target"] == "attack", scores))
                for metric, function in (("Precision", precision_score), ("Recall", recall_score), ("F1", f1_score)):
                    self.assertAlmostEqual(result["metrics"][metric], function(profile["target"], result["predictions"], pos_label="attack", zero_division=0))
                self.assertEqual(result["confusion_matrix"].shape, (2, 2))
                self.assertEqual(result["class_names"], ["attack", "normal"])

    def test_multiclass_metrics_are_support_weighted_and_use_probabilities(self):
        fixture = MulticlassProbabilityFixture()
        features = np.zeros((8, 2))
        target = np.array([101, 202, 303, 101, 202, 303, 101, 303])
        result = evaluate_model(fixture, features, target)
        predictions = fixture.predict(features)
        expected = {
            "Accuracy": accuracy_score(target, predictions),
            "AUC": roc_auc_score(target, fixture.probabilities, multi_class="ovr", average="weighted"),
            "Precision": precision_score(target, predictions, average="weighted", zero_division=0),
            "Recall": recall_score(target, predictions, average="weighted", zero_division=0),
            "F1": f1_score(target, predictions, average="weighted", zero_division=0),
            "MCC": matthews_corrcoef(target, predictions),
        }
        self.assertEqual(list(result["metrics"]), METRICS)
        for metric, value in expected.items():
            self.assertAlmostEqual(result["metrics"][metric], value)
        self.assertEqual(result["confusion_matrix"].shape, (3, 3))
        self.assertEqual(int(result["confusion_matrix"].sum()), len(target))
        profile = self.profiles["multiclass"]
        for name, detail in profile["details"].items():
            with self.subTest(model=name):
                self.assertTrue(all(np.isfinite(value) for value in detail["metrics"].values()))
                probabilities = profile["models"][name].predict_proba(profile["features"])
                self.assertAlmostEqual(detail["metrics"]["AUC"], roc_auc_score(profile["target"], probabilities, multi_class="ovr", average="weighted"))

    def test_missing_fitted_class_reports_unavailable_auc_with_full_matrix(self):
        for mode, profile in self.profiles.items():
            target_class = profile["metadata"]["classes"][0]
            frame = profile["test_data"].loc[profile["test_data"]["event_status"] == target_class].head(6)
            features, target = validate_data(frame, profile["metadata"])
            comparison, details = evaluate_models(profile["models"], features, target, profile["metadata"])
            self.assertTrue(comparison["AUC"].isna().all())
            class_count = len(profile["metadata"]["classes"])
            for name, result in details.items():
                with self.subTest(dataset=mode, model=name):
                    self.assertIsNone(result["metrics"]["AUC"])
                    self.assertTrue(result["auc_reason"])
                    self.assertEqual(result["confusion_matrix"].shape, (class_count, class_count))
                    self.assertEqual(int(result["confusion_matrix"].sum()), len(frame))
                    self.assertTrue(all(np.isfinite(value) for metric, value in result["metrics"].items() if metric != "AUC"))

    def test_generic_csv_validation_supports_reordering_and_unseen_categories(self):
        for mode, profile in self.profiles.items():
            with self.subTest(dataset=mode):
                metadata = profile["metadata"]
                original = profile["test_data"].head(8)
                reordered = original.iloc[:, ::-1].astype(str)
                features, target = validate_data(reordered, metadata)
                np.testing.assert_allclose(features.iloc[:, :10], original[metadata["feature_names"]].iloc[:, :10], atol=1e-12)
                np.testing.assert_array_equal(target, original["event_status"].to_numpy())
                unseen = original.copy()
                unseen["protocol"] = "unseen-in-training"
                features, target = validate_data(unseen, metadata)
                with patch.object(Pipeline, "fit", side_effect=AssertionError("Evaluation must not train")):
                    comparison, details = evaluate_models(profile["models"], features, target, metadata)
                self.assertEqual(len(comparison), 5)
                self.assertTrue(all(len(result["predictions"]) == len(unseen) for result in details.values()))

    def test_generic_csv_errors_identify_schema_and_label_problems(self):
        profile = self.profiles["binary"]
        base = profile["test_data"].head(6).copy()
        bad_cases = [
            base.drop(columns="event_status"),
            base.drop(columns="measurement_0"),
            base.assign(unexpected=1),
            pd.concat([base, base[["region"]]], axis=1),
            base.iloc[:0],
        ]
        for column, value in (("event_status", "unknown-class"), ("event_status", None), ("protocol", None), ("measurement_0", np.nan), ("measurement_0", np.inf), ("measurement_0", "invalid-number")):
            malformed = base.astype(object)
            malformed.loc[malformed.index[0], column] = value
            bad_cases.append(malformed)
        for index, malformed in enumerate(bad_cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                validate_data(malformed, profile["metadata"])
        with self.assertRaisesRegex(ValueError, "Unknown target labels"):
            evaluate_model(next(iter(profile["models"].values())), profile["features"].head(2), np.array(["unknown-class", "attack"]), profile["metadata"])

    def test_custom_training_rejects_invalid_assignment_datasets_before_fitting(self):
        base = self.profiles["binary"]["source"].copy()
        continuous = base.copy()
        continuous["event_status"] = np.linspace(0.01, 0.99, len(base))
        missing = base.copy()
        missing.loc[0, "protocol"] = None
        invalid_datasets = [
            base.head(499),
            base.drop(columns="measurement_0"),
            base.drop(columns="event_status"),
            base.assign(event_status="one-class"),
            continuous,
            missing,
        ]
        args = SimpleNamespace(target="event_status", positive_class=None)
        for index, malformed in enumerate(invalid_datasets):
            with self.subTest(case=index), \
                    patch.object(Pipeline, "fit", side_effect=AssertionError("Invalid datasets must not train")), \
                    self.assertRaises(ValueError):
                prepare_custom_dataset(malformed, args)

    def test_oversized_holdout_is_rejected_before_training_or_creating_outputs(self):
        source_path = self.profiles["binary"]["source_path"]
        original_source = source_path.read_bytes()
        limits = [
            ("rows", {"MAX_ROWS": 1}, "held-out test set exceeds"),
            ("bytes", {"MAX_CSV_BYTES": 1}, "held-out test CSV exceeds"),
        ]
        for label, settings, message in limits:
            output = self.folder / f"rejected-{label}-profile"
            with self.subTest(limit=label), \
                    patch.multiple("train", **settings), \
                    patch.object(Pipeline, "fit", side_effect=AssertionError("An unusable holdout must be rejected before training")) as fit, \
                    self.assertRaisesRegex(ValueError, message):
                train_main([
                    "--data", str(source_path), "--target", "event_status",
                    "--name", "Oversized holdout fixture", "--output", str(output),
                ])
            fit.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(source_path.read_bytes(), original_source)

    def test_local_app_renders_custom_profile_and_download_without_training(self):
        profile = self.profiles["multiclass"]
        upload_state = {"profile": None, "bytes": None}

        def current_upload(*args, **kwargs):
            # Streamlit scopes the upload widget to the selected model profile.
            # A file provided for one dataset must not populate another one.
            if kwargs["key"] != f"test_upload_{upload_state['profile']}":
                return None
            return SimpleNamespace(name="selected_test.csv", getvalue=lambda: upload_state["bytes"])

        def assert_waiting_for_upload(app):
            self.assertEqual(list(app.exception), [])
            self.assertEqual(list(app.error), [])
            self.assertEqual(len(app.dataframe), 0)
            self.assertEqual(len(app.metric), 0)
            self.assertNotIn("model_selection", [item.key for item in app.selectbox])
            self.assertNotIn("comparison_metric", [item.key for item in app.selectbox])
            self.assertFalse(any('class="metric-comparison-chart"' in item.value for item in app.markdown))
            self.assertFalse(any(" confusion matrix" in item.value and "<img" in item.value for item in app.markdown))

        with TemporaryDirectory(prefix="custom-app-downloads-") as static_root, \
                patch.dict(os.environ, {"ML_MODEL_DIR": str(profile["path"])}), \
                patch.object(Pipeline, "fit", side_effect=AssertionError("The app must never train models")), \
                patch("streamlit.file_uploader", side_effect=current_upload), \
                patch("downloads.sample_download_url", side_effect=lambda data: sample_download_url(data, static_root=static_root)):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            assert_waiting_for_upload(app)
            self.assertEqual(app.title[0].value, "ML Assignment 2 · 2026AJ05054")
            self.assertTrue(any(profile["metadata"]["dataset"]["name"] in item.value for item in app.markdown))
            downloaded_bytes = assert_sample_download(self, app, static_root, (profile["path"] / "test_data.csv").read_bytes())
            custom_download_href = sample_download_href(app)
            downloaded_frame = pd.read_csv(BytesIO(downloaded_bytes))
            pd.testing.assert_frame_equal(downloaded_frame, profile["test_data"])
            upload_state.update(profile=profile["path"].resolve(), bytes=downloaded_bytes)
            app.run()
            self.assertEqual(list(app.exception), [])
            pd.testing.assert_frame_equal(app.dataframe[0].value, profile["comparison"])
            assert_no_comparison_chart(self, app)
            self.assertTrue(any("3 target classes give a 3 × 3 matrix" in item.value for item in app.caption))
            self.assertEqual(app.selectbox(key="model_selection").options, list(MODEL_FILES))
            for name in MODEL_FILES:
                app.selectbox(key="model_selection").select(name).run()
                self.assertEqual(list(app.exception), [])
                self.assertEqual([metric.label for metric in app.metric], METRICS)
                self.assertEqual([metric.value for metric in app.metric], [f"{profile['details'][name]['metrics'][metric]:.4f}" for metric in METRICS])
                report = app.dataframe[1].value
                self.assertTrue(all(str(value) in " ".join(map(str, report.index)) for value in [101, 202, 303]))
            # Switching datasets must switch downloads and wait for that
            # dataset's upload, rather than displaying the earlier results.
            default_metadata, default_models = load_artifacts()
            default_frame = pd.read_csv(ROOT / "test_data.csv")
            default_features, default_target = validate_data(default_frame, default_metadata)
            default_comparison, _ = evaluate_models(default_models, default_features, default_target, default_metadata)
            app.selectbox(key="dataset_profile").select((ROOT / "model").resolve()).run()
            assert_waiting_for_upload(app)
            assert_sample_download(self, app, static_root, (ROOT / "test_data.csv").read_bytes())
            self.assertNotEqual(sample_download_href(app), custom_download_href)
            upload_state.update(profile=(ROOT / "model").resolve(), bytes=(ROOT / "test_data.csv").read_bytes())
            app.run()
            self.assertEqual(list(app.exception), [])
            pd.testing.assert_frame_equal(app.dataframe[0].value, default_comparison)
            assert_no_comparison_chart(self, app)
            app.selectbox(key="dataset_profile").select(profile["path"].resolve()).run()
            assert_waiting_for_upload(app)
            assert_sample_download(self, app, static_root, (profile["path"] / "test_data.csv").read_bytes())
            self.assertEqual(sample_download_href(app), custom_download_href)
            upload_state.update(profile=profile["path"].resolve(), bytes=downloaded_bytes)
            app.run()
            self.assertEqual(list(app.exception), [])
            pd.testing.assert_frame_equal(app.dataframe[0].value, profile["comparison"])
            assert_no_comparison_chart(self, app)

    def test_app_evaluates_uploaded_generic_csv_and_handles_invalid_uploads(self):
        profile = self.profiles["binary"]
        metadata = profile["metadata"]
        one_class = profile["test_data"].loc[profile["test_data"]["event_status"] == "attack"].head(8)
        features, target = validate_data(one_class, metadata)
        expected, _ = evaluate_models(profile["models"], features, target, metadata)
        uploads = [
            ("one-class.csv", one_class, None),
            ("missing-column.csv", one_class.drop(columns="measurement_0"), "Missing columns"),
            ("unknown-target.csv", one_class.assign(event_status="unknown-label"), "Unknown target labels"),
        ]
        with TemporaryDirectory(prefix="uploaded-app-downloads-") as static_root, \
                patch.dict(os.environ, {"ML_MODEL_DIR": str(profile["path"])}), \
                patch.object(Pipeline, "fit", side_effect=AssertionError("Uploaded test data must never train")), \
                patch("downloads.sample_download_url", side_effect=lambda data: sample_download_url(data, static_root=static_root)):
            for filename, frame, error_message in uploads:
                payload = frame.to_csv(index=False).encode()
                upload = SimpleNamespace(name=filename, getvalue=lambda payload=payload: payload)
                with self.subTest(upload=filename), patch("streamlit.file_uploader", return_value=upload):
                    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
                    self.assertEqual(list(app.exception), [])
                    assert_sample_download(self, app, static_root, (profile["path"] / "test_data.csv").read_bytes())
                    if error_message:
                        self.assertTrue(any(error_message in item.value for item in app.error))
                        self.assertEqual(len(app.metric), 0)
                    else:
                        self.assertEqual(list(app.error), [])
                        pd.testing.assert_frame_equal(app.dataframe[0].value, expected)
                        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "AUC"), "N/A")
                        self.assertTrue(any("AUC" in item.value for item in app.warning))
                        assert_no_comparison_chart(self, app)
            self.assertEqual(len(list(Path(static_root).rglob("test_data.csv"))), 1)

    def test_new_csv_refreshes_every_model_matrix_and_preview_in_the_same_session(self):
        profile = self.profiles["binary"]
        first_frame = profile["test_data"].head(24).copy()
        second_frame = profile["test_data"].tail(36).copy()
        # Invert the fixture labels deliberately so stale results would be
        # obvious even for classifiers that predict both subsets accurately.
        second_frame["event_status"] = second_frame["event_status"].map({"attack": "normal", "normal": "attack"})
        upload_state = {"bytes": first_frame.to_csv(index=False).encode()}
        callbacks = {}
        figures = []
        original_savefig = Figure.savefig

        def save_figure(figure, *args, **kwargs):
            figures.append(figure)
            return original_savefig(figure, *args, **kwargs)

        def current_upload(*args, **kwargs):
            callbacks["on_change"] = kwargs.get("on_change")
            return SimpleNamespace(name="current_test.csv", getvalue=lambda: upload_state["bytes"])

        with TemporaryDirectory(prefix="refresh-app-downloads-") as static_root, \
                patch.dict(os.environ, {"ML_MODEL_DIR": str(profile["path"])}), \
                patch.object(Pipeline, "fit", side_effect=AssertionError("Changing test data must never train models")), \
                patch("streamlit.file_uploader", side_effect=current_upload), \
                patch.object(Figure, "savefig", autospec=True, side_effect=save_figure), \
                patch("downloads.sample_download_url", side_effect=lambda data: sample_download_url(data, static_root=static_root)):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            previous_table = None
            for position, frame in enumerate((first_frame, second_frame)):
                if position:
                    upload_state["bytes"] = frame.to_csv(index=False).encode()
                    self.assertTrue(callable(callbacks["on_change"]))
                    # Exercise the actual callback and transfer its state to the
                    # same AppTest session, as Streamlit does on a file change.
                    callback_state = {"upload_revision": 0}
                    with patch("streamlit.session_state", callback_state):
                        callbacks["on_change"]()
                    self.assertEqual(callback_state["upload_revision"], 1)
                    app.session_state["upload_revision"] = callback_state["upload_revision"]
                    app.run()
                self.assertEqual(list(app.exception), [])
                self.assertEqual(list(app.error), [])
                self.assertTrue(any(f"{len(frame)} test rows" in message.value for message in app.success))
                features, target = validate_data(frame, profile["metadata"])
                expected_table, details = evaluate_models(profile["models"], features, target, profile["metadata"])
                pd.testing.assert_frame_equal(app.dataframe[0].value, expected_table)
                if previous_table is not None:
                    self.assertFalse(expected_table.equals(previous_table))
                previous_table = expected_table
                flow = next(item.value for item in app.markdown if 'class="dataset-flow"' in item.value)
                for label, count in (
                    ("Full source dataset", profile["metadata"]["dataset"]["rows"]),
                    ("Used for training", profile["metadata"]["split"]["train_rows"]),
                    ("Reserved for testing", profile["metadata"]["split"]["test_rows"]),
                ):
                    self.assertIn(f"<span>{label}</span><strong>{count:,} rows</strong>", flow)
                summary = next(item.value for item in app.markdown if 'class="dataset-summary"' in item.value)
                current_count = re.search(r'Rows being evaluated now</div>\s*<div class="summary-value">([^<]+)', summary)
                self.assertIsNotNone(current_count)
                self.assertEqual(int(current_count.group(1).replace(",", "")), len(frame))
                assert_no_comparison_chart(self, app)
                pd.testing.assert_frame_equal(app.dataframe[0].value, expected_table)
                for name in MODEL_FILES:
                    app.selectbox(key="model_selection").select(name).run()
                    self.assertEqual(list(app.exception), [])
                    result = details[name]
                    self.assertEqual([metric.value for metric in app.metric], [f"{result['metrics'][metric]:.4f}" for metric in METRICS])
                    figure = figures[-1]
                    matrix_axis = figure.axes[0]
                    np.testing.assert_array_equal(matrix_axis.images[0].get_array(), result["confusion_matrix"])
                    self.assertIn(name, matrix_axis.get_title())
                    self.assertGreaterEqual(figure.get_size_inches()[0], 10)
                    self.assertGreaterEqual(figure.get_size_inches()[1], 7)
                    self.assertEqual(len(matrix_axis.texts), len(result["classes"]) ** 2)
                    expected_annotations = [
                        f"{value:,}\n{(value / row.sum() if row.sum() else 0):.1%}"
                        for row in result["confusion_matrix"] for value in row
                    ]
                    self.assertEqual([text.get_text() for text in matrix_axis.texts], expected_annotations)
                    embedded = EmbeddedMatrixParser()
                    for element in app.markdown:
                        embedded.feed(element.value)
                    self.assertEqual(len(embedded.images), 1)
                    image = embedded.images[0]
                    self.assertEqual(image["alt"], f"{name} confusion matrix")
                    self.assertTrue(image["src"].startswith("data:image/png;base64,"))
                    png_bytes = b64decode(image["src"].split(",", 1)[1], validate=True)
                    self.assertEqual(png_bytes[:8], b"\x89PNG\r\n\x1a\n")
                    self.assertGreater(len(png_bytes), 1_000)
                    self.assertTrue(any(f"Selected model: {name}" in caption.value for caption in app.caption))
                    preview = frame.head(20).reset_index(drop=True).copy()
                    preview["predicted_event_status"] = result["predictions"][:len(preview)]
                    pd.testing.assert_frame_equal(app.dataframe[2].value.reset_index(drop=True), preview)

    def test_upload_preserves_leading_zeros_in_categorical_features_and_string_labels(self):
        source = self.profiles["multiclass"]["source"].copy()
        source["event_status"] = source["event_status"].map({101: "001", 202: "002", 303: "attack"})
        data_path = self.folder / "string_label_training.csv"
        source.to_csv(data_path, index=False)
        profile = self.folder / "string-label-profile"
        trained = subprocess.run([
            sys.executable, str(ROOT / "train.py"), "--data", str(data_path),
            "--target", "event_status", "--name", "Leading-zero string labels",
            "--output", str(profile),
        ], cwd=ROOT, capture_output=True, text=True, timeout=90)
        self.assertEqual(trained.returncode, 0, f"{trained.stdout}\n{trained.stderr}")
        metadata, models = load_artifacts(profile)
        heldout = pd.read_csv(profile / "test_data.csv", dtype={"protocol": str, "event_status": str})
        numeric_looking = heldout.loc[
            heldout["protocol"].isin(["001", "002"])
            & heldout["event_status"].isin(["001", "002"])
        ].head(24)
        self.assertEqual(set(numeric_looking["event_status"]), {"001", "002"})
        self.assertEqual(set(numeric_looking["protocol"]), {"001", "002"})
        features, target = validate_data(numeric_looking, metadata)
        expected, _ = evaluate_models(models, features, target, metadata)
        payload = numeric_looking.to_csv(index=False).encode()
        upload = SimpleNamespace(name="leading-zero-test.csv", getvalue=lambda: payload)
        with TemporaryDirectory(prefix="leading-zero-app-downloads-") as static_root, \
                patch.dict(os.environ, {"ML_MODEL_DIR": str(profile)}), \
                patch.object(Pipeline, "fit", side_effect=AssertionError("Uploaded test data must never train")), \
                patch("streamlit.file_uploader", return_value=upload), \
                patch("downloads.sample_download_url", side_effect=lambda data: sample_download_url(data, static_root=static_root)):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(list(app.error), [])
            pd.testing.assert_frame_equal(app.dataframe[0].value, expected)
            preview = app.dataframe[-1].value
            self.assertEqual(preview["protocol"].tolist(), numeric_looking.head(20)["protocol"].tolist())
            self.assertEqual(preview["event_status"].tolist(), numeric_looking.head(20)["event_status"].tolist())
            assert_sample_download(self, app, static_root, (profile / "test_data.csv").read_bytes())
            self.assertEqual(len(list(Path(static_root).rglob("test_data.csv"))), 1)


if __name__ == "__main__":
    unittest.main()
