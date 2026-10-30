"""Check the assignment's data split, saved models, metrics, and local app.

Run from the project folder: python -m unittest discover -s tests -v
"""

import pickle
import re
from html.parser import HTMLParser
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from streamlit.testing.v1 import AppTest

from dataset_source import SOURCE_METADATA
from downloads import sample_download_url
from test_downloads import assert_sample_download
from train import split_dataset
from utils import (
    MAX_ROWS, METRICS, MODEL_FILES, ROOT, TARGET,
    evaluate_model, evaluate_models, load_artifacts, validate_data,
)


class ProbabilityFixture:
    """Perfect ranking with deliberately imperfect thresholded predictions."""

    classes_ = np.array([0, 1])

    def predict(self, features):
        return np.array([0, 0, 1, 1])

    def predict_proba(self, features):
        positive = np.array([0.1, 0.8, 0.2, 0.9])
        return np.column_stack([1 - positive, positive])


class DecisionScoreFixture:
    classes_ = np.array([0, 1])

    def predict(self, features):
        return np.array([0, 0, 1, 1])

    def decision_function(self, features):
        return np.array([-2.0, 1.0, -1.0, 2.0])


class SelectionBannerParser(HTMLParser):
    def __init__(self, class_name="model-selection"):
        super().__init__()
        self.class_name = class_name
        self.selections = []
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attributes = dict(attrs)
        if self.class_name in attributes.get("class", "").split():
            self.selections.append(attributes)


class AssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata, cls.models = load_artifacts()
        cls.test_data = pd.read_csv(ROOT / "test_data.csv")
        cls.test_features, cls.test_target = validate_data(cls.test_data, cls.metadata)
        cls.table, cls.details = evaluate_models(cls.models, cls.test_features, cls.test_target, cls.metadata)

    def test_dataset_meets_requirements_and_source_is_recorded(self):
        dataset = self.metadata["dataset"]
        self.assertGreaterEqual(dataset["rows"], 500)
        self.assertGreaterEqual(dataset["feature_count"], 12)
        self.assertEqual((dataset["rows"], dataset["feature_count"]), (11_055, 30))
        self.assertEqual(set(self.test_target), {0, 1})
        self.assertTrue(dataset["url"].startswith("https://archive.ics.uci.edu/"))
        self.assertEqual(SOURCE_METADATA["source_url"], dataset["url"])
        self.assertEqual(SOURCE_METADATA["feature_names"], self.metadata["feature_names"])
        self.assertEqual(SOURCE_METADATA["feature_values"], self.metadata["feature_values"])
        self.assertRegex(dataset["csv_sha256"], r"^[0-9a-f]{64}$")

    def test_one_reproducible_test_partition_without_duplicate_leakage(self):
        # Reproduce the splitting algorithm with a small fixture, without
        # downloading or retaining the full UCI source dataset in the project.
        rng = np.random.default_rng(42)
        features = pd.DataFrame({
            name: rng.choice(self.metadata["feature_values"][name], 600)
            for name in self.metadata["feature_names"]
        })
        target = np.tile([0, 1], 300)
        features = pd.concat([features, features.iloc[:60]], ignore_index=True)
        target = np.concatenate([target, target[:60]])
        train_index, test_index, groups = split_dataset(features, target)
        repeated_train, repeated_test, _ = split_dataset(features, target)
        np.testing.assert_array_equal(train_index, repeated_train)
        np.testing.assert_array_equal(test_index, repeated_test)
        self.assertFalse(set(train_index) & set(test_index))
        self.assertEqual(len(train_index) + len(test_index), len(features))
        self.assertFalse(set(groups[train_index]) & set(groups[test_index]))
        self.assertEqual(self.metadata["split"]["train_rows"], 8_841)
        self.assertEqual(self.metadata["split"]["test_rows"], len(self.test_data))
        self.assertEqual(len(self.test_data), 2_214)
        self.assertEqual(self.metadata["split"]["train_rows"] + len(self.test_data), self.metadata["dataset"]["rows"])
        self.assertEqual(self.metadata["split"]["feature_group_overlap"], 0)

    def test_five_saved_pipelines_contain_training_only_preprocessing(self):
        self.assertEqual(list(self.models), list(MODEL_FILES))
        expected_types = [
            "LogisticRegression", "DecisionTreeClassifier", "KNeighborsClassifier",
            "GaussianNB", "SVC",
        ]
        training_rows = self.metadata["split"]["train_rows"]
        for (name, model), expected_type in zip(self.models.items(), expected_types):
            with self.subTest(model=name):
                self.assertIsInstance(model, Pipeline)
                classifier = model.named_steps["classifier"]
                self.assertEqual(type(classifier).__name__, expected_type)
                self.assertEqual(classifier.n_features_in_, 30)
                np.testing.assert_array_equal(model.classes_, [0, 1])
                filename = ROOT / "model" / MODEL_FILES[name]
                self.assertEqual(filename.suffix, ".pkl")
                with filename.open("rb") as stream:
                    independently_loaded = pickle.load(stream)
                np.testing.assert_array_equal(independently_loaded.predict(self.test_features.head(8)), model.predict(self.test_features.head(8)))
                scaler = model.named_steps["scaler"]
                if isinstance(scaler, StandardScaler):
                    self.assertEqual(int(scaler.n_samples_seen_), training_rows)
                    self.assertEqual(scaler.mean_.shape, (30,))
                    self.assertTrue(np.isfinite(scaler.mean_).all())
                else:
                    self.assertEqual(scaler, "passthrough")
        self.assertEqual(len(self.models["KNN"].named_steps["classifier"]._y), training_rows)
        self.assertEqual(
            self.models["Decision Tree"].named_steps["classifier"].tree_.n_node_samples[0],
            training_rows,
        )
        self.assertEqual(
            self.models["Naive Bayes"].named_steps["classifier"].class_count_.sum(),
            training_rows,
        )

    def test_all_six_metrics_match_saved_test_results(self):
        stored = pd.read_csv(ROOT / "model" / "metrics.csv")
        self.assertEqual(list(self.table.columns), ["Model", *METRICS])
        pd.testing.assert_frame_equal(self.table, stored, check_exact=False, atol=1e-9, rtol=1e-9)
        for name, result in self.details.items():
            with self.subTest(model=name):
                self.assertEqual(list(result["metrics"]), METRICS)
                for metric, value in result["metrics"].items():
                    self.assertTrue(np.isfinite(value))
                    self.assertGreaterEqual(value, -1 if metric == "MCC" else 0)
                    self.assertLessEqual(value, 1)
                    self.assertAlmostEqual(value, self.metadata["evaluation"][name]["metrics"][metric])
                self.assertEqual(result["confusion_matrix"].shape, (2, 2))
                self.assertEqual(int(result["confusion_matrix"].sum()), len(self.test_data))
                self.assertEqual(len(result["predictions"]), len(self.test_data))

    def test_auc_uses_positive_continuous_scores(self):
        for fixture in (ProbabilityFixture(), DecisionScoreFixture()):
            with self.subTest(score_method=type(fixture).__name__):
                result = evaluate_model(fixture, np.zeros((4, 1)), np.array([0, 1, 0, 1]))
                self.assertEqual(result["metrics"], {
                    "Accuracy": 0.5, "AUC": 1.0, "Precision": 0.5,
                    "Recall": 0.5, "F1": 0.5, "MCC": 0.0,
                })
                np.testing.assert_array_equal(result["confusion_matrix"], [[1, 1], [1, 1]])

    def test_naive_bayes_recall_matches_manual_phishing_counts_and_report(self):
        """Perfect phishing recall can coexist with many legitimate false alarms."""
        result = evaluate_model(
            self.models["Naive Bayes"], self.test_features,
            self.test_target, self.metadata,
        )
        predictions = result["predictions"]
        actual_phishing = self.test_target == 1
        predicted_phishing = predictions == 1
        true_positive = int(np.count_nonzero(actual_phishing & predicted_phishing))
        false_negative = int(np.count_nonzero(actual_phishing & ~predicted_phishing))
        false_positive = int(np.count_nonzero(~actual_phishing & predicted_phishing))
        true_negative = int(np.count_nonzero(~actual_phishing & ~predicted_phishing))
        self.assertEqual(self.metadata.get("positive_class", result["classes"][-1]), 1)
        np.testing.assert_array_equal(
            result["confusion_matrix"],
            [[true_negative, false_positive], [false_negative, true_positive]],
        )
        self.assertEqual((true_negative, false_positive, false_negative, true_positive), (408, 825, 0, 981))
        manual_recall = true_positive / (true_positive + false_negative)
        manual_precision = true_positive / (true_positive + false_positive)
        self.assertEqual(result["metrics"]["Recall"], manual_recall)
        self.assertEqual(result["metrics"]["Recall"], 1.0)
        self.assertAlmostEqual(result["metrics"]["Precision"], manual_precision)
        self.assertLess(manual_precision, .55)
        self.assertEqual(result["classification_report"]["Phishing"]["recall"], manual_recall)
        self.assertEqual(result["classification_report"]["Phishing"]["support"], int(actual_phishing.sum()))
        self.assertAlmostEqual(result["classification_report"]["Legitimate"]["recall"], true_negative / (true_negative + false_positive))

    def test_reordered_numeric_csv_columns_are_supported(self):
        reordered = self.test_data.iloc[:10, ::-1].astype(str)
        features, target = validate_data(reordered, self.metadata)
        pd.testing.assert_frame_equal(features, self.test_features.iloc[:10])
        np.testing.assert_array_equal(target, self.test_target[:10])

    def test_invalid_test_csvs_are_rejected_with_useful_messages(self):
        base = self.test_data.iloc[:4].copy()
        first_feature = self.metadata["feature_names"][0]
        bad_cases = {
            "empty": (base.iloc[:0], "at least one"),
            "missing target": (base.drop(columns=TARGET), "Missing columns"),
            "missing feature": (base.drop(columns=first_feature), "Missing columns"),
            "unexpected column": (base.assign(extra=1), "Unexpected columns"),
            "duplicate column": (pd.concat([base, base[[first_feature]]], axis=1), "Duplicate"),
            "too many rows": (pd.concat([base.iloc[[0]]] * (MAX_ROWS + 1), ignore_index=True), "at most"),
        }
        for label, value, column, message in [
            ("text", "text", first_feature, "numeric"),
            ("missing value", np.nan, first_feature, "Missing values"),
            ("infinite value", np.inf, first_feature, "infinity"),
            ("invalid feature code", 42, first_feature, "Invalid values"),
            ("invalid target", 2, TARGET, "is_phishing must"),
        ]:
            malformed = base.astype(object)
            malformed.loc[malformed.index[0], column] = value
            bad_cases[label] = (malformed, message)
        for label, (frame, message) in bad_cases.items():
            with self.subTest(case=label), self.assertRaisesRegex(ValueError, message):
                validate_data(frame, self.metadata)

    def test_one_class_test_data_has_unavailable_auc_and_valid_other_metrics(self):
        for label in (0, 1):
            one_class = self.test_data[self.test_data[TARGET] == label].head(5)
            features, target = validate_data(one_class, self.metadata)
            comparison, details = evaluate_models(self.models, features, target)
            self.assertTrue(comparison["AUC"].isna().all())
            for name, result in details.items():
                with self.subTest(label=label, model=name):
                    self.assertIsNone(result["metrics"]["AUC"])
                    self.assertEqual(result["confusion_matrix"].shape, (2, 2))
                    self.assertEqual(int(result["confusion_matrix"].sum()), 5)
                    self.assertTrue(all(
                        np.isfinite(value) for metric, value in result["metrics"].items()
                        if metric != "AUC"
                    ))

    def assert_no_evaluation_results(self, app):
        """No old scores, model controls or prediction outputs may remain."""
        self.assert_no_footer(app)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.dataframe), 0)
        result_labels = {"Choose a metric to compare", "Select a model to inspect"}
        self.assertFalse(result_labels & {item.label for item in app.selectbox})
        self.assertNotIn("model_selection", [item.key for item in app.sidebar.selectbox])
        self.assertFalse(any('class="sidebar-selected-model"' in item.value for item in app.sidebar.markdown))
        result_titles = {
            "Comparison of all five models", "Animated metric comparison",
            "Confusion matrix", "Classification report",
        }
        self.assertFalse(result_titles & {item.value for item in app.subheader})
        self.assertFalse(any(
            'class="metric-comparison-chart"' in item.value
            or 'class="confusion-figure"' in item.value
            or 'class="dataset-summary"' in item.value
            for item in app.markdown
        ))

    def assert_assignment_identity_and_no_chart(self, app):
        self.assertEqual(app.title[0].value, "BITS: ML Assignment 2")
        rendered_text = " ".join(item.value for item in app.markdown) + " ".join(item.value for item in app.caption)
        self.assertIn("2026AJ05054", rendered_text)
        self.assertIn('class="hero-student"', rendered_text)
        self.assert_no_footer(app)
        self.assertIn(self.metadata["dataset"]["name"], rendered_text)
        self.assertNotIn("Animated metric comparison", rendered_text)
        self.assertNotIn('class="metric-comparison-chart"', rendered_text)
        self.assertNotIn("comparison_metric", [item.key for item in app.selectbox])
        self.assertNotIn("replay_comparison", [item.key for item in app.button])
        self.assertFalse(any(item.value == "Animated metric comparison" for item in app.subheader))

    def assert_no_footer(self, app):
        markup = " ".join(item.value for item in app.markdown)
        self.assertNotIn('class="page-footer"', markup)
        self.assertNotIn('class="footer-status"', markup)
        self.assertNotIn("Evaluation · Saved models · No retraining", markup)

    def assert_no_selected_model_results(self, app):
        """A valid upload alone must not display a model's evaluation outputs."""
        self.assert_no_footer(app)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.dataframe), 0)
        self.assertFalse(any(
            item.value in {"Confusion matrix", "Classification report"}
            or item.value.endswith(": evaluation metrics")
            for item in app.subheader
        ))
        self.assertFalse(any('class="confusion-figure"' in item.value for item in app.markdown))
        self.assert_sidebar_model_picker(app)
        self.assertTrue(any(
            'class="model-waiting"' in item.value and "sidebar" in item.value.lower()
            for item in app.main.markdown
        ))
        self.assert_no_comparison_table(app)

    def assert_sidebar_model_picker(self, app, selected=None):
        """One sidebar control owns the selection; results belong to the main page."""
        side_pickers = [item for item in app.sidebar.selectbox if item.key == "model_selection"]
        self.assertEqual(len(side_pickers), 1)
        self.assertEqual(side_pickers[0].options, list(MODEL_FILES))
        self.assertEqual(side_pickers[0].value, selected)
        self.assertNotIn("model_selection", [item.key for item in app.main.selectbox])
        self.assertEqual(sum(item.key == "model_selection" for item in app.selectbox), 1)
        self.assertEqual(len(app.sidebar.metric), 0)
        self.assertEqual(len(app.sidebar.dataframe), 0)
        indicators = [
            item.value for item in app.sidebar.markdown
            if 'class="sidebar-selected-model"' in item.value
        ]
        if selected is None:
            self.assertEqual(indicators, [])
        else:
            self.assertEqual(len(indicators), 1)
            parser = SelectionBannerParser("sidebar-selected-model")
            parser.feed(indicators[0])
            self.assertEqual(len(parser.selections), 1)
            self.assertEqual(parser.selections[0]["data-model"], selected)
            self.assertIn(selected, indicators[0])
            self.assertNotIn("Results ready", indicators[0])

    def assert_no_comparison_table(self, app):
        self.assertNotIn("Comparison of all five models", [item.value for item in app.subheader])
        for table in app.dataframe:
            self.assertNotEqual(list(table.value.columns), ["Model", *METRICS])

    def assert_selected_model_results(self, app, name, result, frame):
        """Exactly one set of metric cards belongs to the chosen classifier."""
        self.assert_no_footer(app)
        self.assert_sidebar_model_picker(app, name)
        definitions = {
            "Logistic Regression": "estimates class probabilities",
            "Decision Tree": "learned if-then rules",
            "KNN": "15 neighbours",
            "Naive Bayes": "independent within each class",
            "Support Vector Machine": "RBF kernel",
        }
        markup = " ".join(item.value for item in app.markdown)
        self.assertNotIn("Results ready", markup)
        self.assertNotIn('class="model-ready"', markup)
        selected_banner = next(item.value for item in app.main.markdown if 'class="model-selection"' in item.value)
        self.assertIn('class="model-definition"', selected_banner)
        self.assertIn("Basic definition", selected_banner)
        self.assertIn(definitions[name], selected_banner)
        self.assertEqual([item.label for item in app.metric], METRICS)
        self.assertEqual([item.label for item in app.main.metric], METRICS)
        self.assertEqual(len({item.label for item in app.metric}), 6)
        self.assertEqual(
            [item.value for item in app.metric],
            [f"{result['metrics'][metric]:.4f}" for metric in METRICS],
        )
        self.assertEqual(
            [item.value for item in app.subheader if item.value.endswith(": evaluation metrics")],
            [f"{name}: evaluation metrics"],
        )
        self.assertEqual(sum('class="confusion-figure"' in item.value for item in app.markdown), 1)
        self.assertEqual(len(app.dataframe), 2)
        report = pd.DataFrame({
            label: values for label, values in result["classification_report"].items()
            if isinstance(values, dict)
        }).T
        report.index.name = "Class"
        pd.testing.assert_frame_equal(app.dataframe[0].value, report)
        preview = frame.head(20).copy()
        preview[f"predicted_{TARGET}"] = result["predictions"][:len(preview)]
        pd.testing.assert_frame_equal(app.dataframe[1].value.reset_index(drop=True), preview.reset_index(drop=True))
        self.assert_no_comparison_table(app)

    def assert_model_selection_animation(self, app, name):
        banners = [item.value for item in app.main.markdown if 'class="model-selection"' in item.value]
        self.assertEqual(len(banners), 1)
        parser = SelectionBannerParser()
        parser.feed(banners[0])
        self.assertEqual(len(parser.selections), 1)
        attributes = parser.selections[0]
        self.assertEqual(attributes["data-model"], name)
        self.assertEqual(attributes["role"], "status")
        self.assertEqual(attributes["aria-live"], "polite")
        self.assertIn(name, banners[0])
        self.assertNotIn("script", parser.tags)
        self.assertNotRegex(banners[0], r"\bon(?:click|load|error)\s*=")
        animation_css = " ".join(item.value for item in app.markdown if "@keyframes model-choice-" in item.value)
        tokens = re.findall(r"@keyframes\s+((?:model|metric|result)-choice-[a-f0-9]+)", animation_css)
        self.assertEqual(len(tokens), 3)
        self.assertEqual({token.split("-choice-")[0] for token in tokens}, {"model", "metric", "result"})
        sidebar_tokens = re.findall(r"@keyframes\s+(sidebar-choice-[a-f0-9]+)", animation_css)
        self.assertEqual(len(sidebar_tokens), 1)
        self.assertEqual(sidebar_tokens[0].split("-choice-")[-1], tokens[0].split("-choice-")[-1])
        normalized = re.sub(r"\s+", "", animation_css)
        self.assertIn("@media(prefers-reduced-motion:reduce)", normalized)
        self.assertIn("animation:none!important", normalized)
        self.assertIn(".sidebar-selected-model", normalized)
        return tuple(tokens + sidebar_tokens)

    def test_initial_app_waits_for_upload_and_keeps_dataset_details_and_download(self):
        with TemporaryDirectory(prefix="waiting-app-downloads-") as static_root, \
                patch.object(Pipeline, "fit", side_effect=AssertionError("The app must never train models")), \
                patch("utils.evaluate_models", side_effect=AssertionError("No upload means no evaluation")) as evaluate, \
                patch("downloads.sample_download_url", side_effect=lambda data: sample_download_url(data, static_root=static_root)):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertEqual(list(app.exception), [])
            self.assert_assignment_identity_and_no_chart(app)
            self.assert_no_evaluation_results(app)
            evaluate.assert_not_called()
            flow = next(item.value for item in app.markdown if 'class="dataset-flow"' in item.value)
            for count in ("11,055 rows", "8,841 rows", "2,214 rows"):
                self.assertIn(count, flow)
            self.assertTrue(any("Upload test CSV" == item.label for item in app.get("file_uploader")))
            self.assertTrue(any("upload" in item.value.lower() for item in app.info))
            assert_sample_download(self, app, static_root, (ROOT / "test_data.csv").read_bytes())

    def test_uploaded_sample_shows_metrics_once_only_after_selection_without_training(self):
        upload = SimpleNamespace(name="uploaded-test.csv", getvalue=lambda: (ROOT / "test_data.csv").read_bytes())
        with TemporaryDirectory(prefix="default-app-downloads-") as static_root, \
                patch.object(Pipeline, "fit", side_effect=AssertionError("The app must never train models")), \
                patch("streamlit.file_uploader", return_value=upload), \
                patch("downloads.sample_download_url", side_effect=lambda data: sample_download_url(data, static_root=static_root)):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertEqual(list(app.exception), [])
            self.assert_assignment_identity_and_no_chart(app)
            self.assertEqual(app.selectbox(key="model_selection").options, list(MODEL_FILES))
            self.assertIsNone(app.selectbox(key="model_selection").value)
            self.assert_no_selected_model_results(app)
            flow = next(item.value for item in app.markdown if 'class="dataset-flow"' in item.value)
            self.assertIn("11,055 rows", flow)
            self.assertIn("8,841 rows", flow)
            self.assertIn("2,214 rows", flow)
            self.assertFalse(any("deploy" in button.label.lower() for button in app.button))
            assert_sample_download(self, app, static_root, (ROOT / "test_data.csv").read_bytes())
            for name in MODEL_FILES:
                app.selectbox(key="model_selection").select(name).run()
                self.assertEqual(list(app.exception), [])
                self.assert_assignment_identity_and_no_chart(app)
                self.assert_selected_model_results(app, name, self.details[name], self.test_data)
                self.assertTrue(any("2 target classes give a 2 × 2 matrix" in item.value for item in app.caption))
                self.assertTrue(any("Confusion matrix" == title.value for title in app.subheader))
            app.selectbox(key="model_selection").select(None).run()
            self.assertEqual(list(app.exception), [])
            self.assertIsNone(app.selectbox(key="model_selection").value)
            self.assert_no_selected_model_results(app)
            app.selectbox(key="model_selection").select("Naive Bayes").run()
            self.assertEqual(list(app.exception), [])
            self.assert_selected_model_results(app, "Naive Bayes", self.details["Naive Bayes"], self.test_data)

    def test_model_selection_callback_replays_accessible_animation_without_duplicate_results(self):
        upload = SimpleNamespace(name="animation-test.csv", getvalue=lambda: (ROOT / "test_data.csv").read_bytes())
        with TemporaryDirectory(prefix="selection-animation-downloads-") as static_root, \
                patch.object(Pipeline, "fit", side_effect=AssertionError("Selection must never train models")) as fit, \
                patch("streamlit.file_uploader", return_value=upload), \
                patch("downloads.sample_download_url", side_effect=lambda data: sample_download_url(data, static_root=static_root)):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertEqual(list(app.exception), [])
            self.assert_assignment_identity_and_no_chart(app)
            self.assert_no_selected_model_results(app)
            self.assertTrue(any('class="model-waiting"' in item.value for item in app.markdown))

            app.selectbox(key="model_selection").select("Logistic Regression").run()
            self.assertEqual(list(app.exception), [])
            first_revision = app.session_state["model_selection_revision"]
            first_animation = self.assert_model_selection_animation(app, "Logistic Regression")
            self.assert_selected_model_results(app, "Logistic Regression", self.details["Logistic Regression"], self.test_data)
            app.run()
            self.assertEqual(app.session_state["model_selection_revision"], first_revision)
            self.assertEqual(self.assert_model_selection_animation(app, "Logistic Regression"), first_animation)

            app.selectbox(key="model_selection").select("Support Vector Machine").run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(app.session_state["model_selection_revision"], first_revision + 1)
            second_animation = self.assert_model_selection_animation(app, "Support Vector Machine")
            self.assertTrue(all(first != second for first, second in zip(first_animation, second_animation)))
            self.assert_selected_model_results(app, "Support Vector Machine", self.details["Support Vector Machine"], self.test_data)

            app.selectbox(key="model_selection").select(None).run()
            self.assertEqual(list(app.exception), [])
            self.assert_no_selected_model_results(app)
            self.assertFalse(any('class="model-selection"' in item.value for item in app.markdown))
            app.selectbox(key="model_selection").select("Logistic Regression").run()
            self.assertEqual(list(app.exception), [])
            self.assertNotEqual(self.assert_model_selection_animation(app, "Logistic Regression"), first_animation)
            self.assert_selected_model_results(app, "Logistic Regression", self.details["Logistic Regression"], self.test_data)
            fit.assert_not_called()

    def test_upload_validation_and_removal_hide_previous_results_in_the_same_session(self):
        first = self.test_data.head(40).copy()
        second = self.test_data.tail(55).copy()
        second[TARGET] = 1 - second[TARGET]
        state = {"upload": None}
        callbacks = {}

        def current_upload(*args, **kwargs):
            callbacks["on_change"] = kwargs.get("on_change")
            return state["upload"]

        def replace_upload(app, frame, filename):
            if frame is None:
                state["upload"] = None
            else:
                payload = frame.to_csv(index=False).encode()
                state["upload"] = SimpleNamespace(name=filename, getvalue=lambda: payload)
            self.assertTrue(callable(callbacks["on_change"]))
            callback_state = {"upload_revision": app.session_state["upload_revision"] if "upload_revision" in app.session_state else 0}
            previous_revision = callback_state["upload_revision"]
            with patch("streamlit.session_state", callback_state):
                callbacks["on_change"]()
            self.assertEqual(callback_state["upload_revision"], previous_revision + 1)
            app.session_state["upload_revision"] = callback_state["upload_revision"]
            app.run()
            self.assertEqual(list(app.exception), [])

        with TemporaryDirectory(prefix="upload-cycle-downloads-") as static_root, \
                patch.object(Pipeline, "fit", side_effect=AssertionError("Upload changes must never train models")) as fit, \
                patch("streamlit.file_uploader", side_effect=current_upload), \
                patch("downloads.sample_download_url", side_effect=lambda data: sample_download_url(data, static_root=static_root)):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertEqual(list(app.exception), [])
            self.assert_no_evaluation_results(app)

            replace_upload(app, first, "first.csv")
            first_features, first_target = validate_data(first, self.metadata)
            first_scores, first_details = evaluate_models(self.models, first_features, first_target, self.metadata)
            self.assertIsNone(app.selectbox(key="model_selection").value)
            self.assert_no_selected_model_results(app)
            app.selectbox(key="model_selection").select("Logistic Regression").run()
            self.assertEqual(list(app.exception), [])
            self.assert_selected_model_results(app, "Logistic Regression", first_details["Logistic Regression"], first)
            self.assertTrue(any("first.csv" in item.value for item in app.success))
            self.assertTrue(any('class="confusion-figure"' in item.value for item in app.markdown))

            replace_upload(app, second, "replacement.csv")
            second_features, second_target = validate_data(second, self.metadata)
            second_scores, second_details = evaluate_models(self.models, second_features, second_target, self.metadata)
            self.assertFalse(first_scores.equals(second_scores))
            self.assertEqual(app.selectbox(key="model_selection").value, "Logistic Regression")
            self.assert_selected_model_results(app, "Logistic Regression", second_details["Logistic Regression"], second)

            replace_upload(app, first.drop(columns=TARGET), "invalid.csv")
            self.assertTrue(any("Missing columns" in item.value for item in app.error))
            self.assert_no_evaluation_results(app)
            self.assertEqual(list(app.success), [])
            assert_sample_download(self, app, static_root, (ROOT / "test_data.csv").read_bytes())

            replace_upload(app, None, "")
            self.assert_no_evaluation_results(app)
            self.assertEqual(list(app.error), [])
            self.assertTrue(any("upload" in item.value.lower() for item in app.info))

            replace_upload(app, second, "second.csv")
            self.assertIsNone(app.selectbox(key="model_selection").value)
            self.assert_no_selected_model_results(app)
            self.assertTrue(any("second.csv" in item.value for item in app.success))
            self.assertEqual(list(app.error), [])
            app.selectbox(key="model_selection").select("Support Vector Machine").run()
            self.assertEqual(list(app.exception), [])
            selected = app.selectbox(key="model_selection").value
            expected = second_details[selected]
            self.assert_selected_model_results(app, selected, expected, second)
            fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
