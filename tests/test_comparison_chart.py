"""Check chart scores, signed bar positions, accessibility and safe labels."""

from html.parser import HTMLParser
import math
import re
import unittest

import numpy as np
import pandas as pd

from comparison_chart import build_metric_chart_html
from utils import METRICS


class MetricChartParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.figure = None
        self.rows = []
        self.tags = []
        self.current = None
        self.score_row = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = attributes.get("class", "").split()
        self.tags.append(tag)
        if tag == "figure" and "metric-comparison-chart" in classes:
            self.figure = attributes
        if tag == "div" and "metric-chart-row" in classes:
            self.current = {"attributes": attributes, "fill": None, "score": "", "zero_markers": 0}
            self.rows.append(self.current)
        if self.current is not None and "metric-bar-fill" in classes:
            self.current["fill"] = attributes
        if self.current is not None and "metric-chart-score" in classes:
            self.score_row = self.current
        if self.current is not None and "metric-chart-zero" in classes:
            self.current["zero_markers"] += 1

    def handle_data(self, data):
        if self.score_row is not None:
            self.score_row["score"] += data

    def handle_endtag(self, tag):
        if tag == "span":
            self.score_row = None


def parse_metric_chart(markup):
    parser = MetricChartParser()
    parser.feed(markup)
    return parser


def chart_from_app(app):
    charts = [element.value for element in app.markdown if 'class="metric-comparison-chart"' in element.value]
    if len(charts) != 1:
        raise AssertionError(f"Expected one comparison chart, found {len(charts)}")
    return charts[0]


def assert_metric_chart(test, markup, comparison, metric):
    """Compare rendered labels and CSS geometry with the evaluated score table."""
    parsed = parse_metric_chart(markup)
    test.assertIsNotNone(parsed.figure)
    test.assertEqual(parsed.figure["data-metric"], metric)
    minimum, maximum = (-1, 1) if metric == "MCC" else (0, 1)
    test.assertEqual(float(parsed.figure["data-domain-min"]), minimum)
    test.assertEqual(float(parsed.figure["data-domain-max"]), maximum)
    test.assertEqual(len(parsed.rows), len(comparison))
    available = [float(value) for value in comparison[metric] if value is not None and math.isfinite(float(value))]
    best = max(available) if available else None
    for rendered, (name, value) in zip(parsed.rows, comparison[["Model", metric]].itertuples(index=False, name=None)):
        attributes = rendered["attributes"]
        test.assertEqual(attributes["data-model"], str(name))
        test.assertEqual(attributes["role"], "listitem")
        test.assertEqual(rendered["zero_markers"], 1 if metric == "MCC" else 0)
        if value is None or not math.isfinite(float(value)):
            test.assertEqual(attributes["data-available"], "false")
            test.assertEqual(attributes["data-value"], "")
            test.assertEqual(rendered["score"], "N/A")
            test.assertIsNone(rendered["fill"])
            test.assertNotIn("is-best", attributes["class"].split())
            continue
        value = float(value)
        test.assertEqual(attributes["data-available"], "true")
        test.assertAlmostEqual(float(attributes["data-value"]), value)
        test.assertEqual(rendered["score"], f"{value:.4f}")
        test.assertEqual("is-best" in attributes["class"].split(), math.isclose(value, best, rel_tol=1e-10, abs_tol=1e-12))
        fill = rendered["fill"]
        test.assertIsNotNone(fill)
        zero_position = (0 - minimum) / (maximum - minimum) * 100
        score_position = (value - minimum) / (maximum - minimum) * 100
        expected_left = min(zero_position, score_position)
        expected_width = abs(score_position - zero_position)
        declarations = dict(item.split(":", 1) for item in fill["style"].split(";") if ":" in item)
        for name, expected in (("left", expected_left), ("width", expected_width)):
            test.assertAlmostEqual(float(fill[f"data-{name}"]), expected, places=7)
            test.assertAlmostEqual(float(declarations[name].removesuffix("%")), expected, places=7)
        test.assertEqual("is-negative" in fill["class"].split(), value < 0)
    return parsed


class ComparisonChartTests(unittest.TestCase):
    @staticmethod
    def scores():
        return pd.DataFrame({
            "Model": ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"],
            "Accuracy": [.82, .90, .88, .76, .90],
            "AUC": [.86, .92, .89, .80, .94],
            "Precision": [.84, .91, .87, .79, .93],
            "Recall": [.80, .89, .90, .72, .92],
            "F1": [.82, .90, .88, .75, .92],
            "MCC": [-.60, -.20, .0, .65, .65],
        })

    def test_all_six_metrics_render_the_exact_scores_and_correct_fixed_axis(self):
        comparison = self.scores()
        for metric in METRICS:
            with self.subTest(metric=metric):
                assert_metric_chart(self, build_metric_chart_html(comparison, metric), comparison, metric)

    def test_unavailable_auc_has_no_bar_and_is_never_presented_as_zero(self):
        comparison = self.scores().assign(AUC=[None, np.nan, None, np.nan, None])
        markup = build_metric_chart_html(comparison, "AUC")
        parsed = assert_metric_chart(self, markup, comparison, "AUC")
        self.assertTrue(all(row["score"] == "N/A" and row["fill"] is None for row in parsed.rows))
        self.assertIn("not a zero score", markup)

    def test_negative_mcc_extends_left_of_zero_and_equal_winners_are_highlighted(self):
        comparison = self.scores()
        parsed = assert_metric_chart(self, build_metric_chart_html(comparison, "MCC"), comparison, "MCC")
        first_fill = parsed.rows[0]["fill"]
        self.assertAlmostEqual(float(first_fill["data-left"]), 20)
        self.assertAlmostEqual(float(first_fill["data-width"]), 30)
        winners = [row["attributes"]["data-model"] for row in parsed.rows if "is-best" in row["attributes"]["class"].split()]
        self.assertEqual(winners, ["Naive Bayes", "Support Vector Machine"])
        self.assertEqual(parsed.rows[2]["score"], "0.0000")

    def test_model_names_and_replay_keys_cannot_create_executable_html(self):
        comparison = self.scores()
        malicious = '<img src=x onerror="alert(1)"> & <script>alert(2)</script>'
        comparison.loc[0, "Model"] = malicious
        markup = build_metric_chart_html(comparison, "Accuracy", animation_key='</style><script>alert(3)</script>')
        parsed = assert_metric_chart(self, markup, comparison, "Accuracy")
        self.assertEqual(parsed.rows[0]["attributes"]["data-model"], malicious)
        self.assertNotIn("script", parsed.tags)
        self.assertNotIn("img", parsed.tags)
        self.assertNotIn("onerror", parsed.rows[0]["attributes"])
        self.assertIn("&lt;img", markup)

    def test_replay_restarts_animation_and_reduced_motion_preserves_visible_values(self):
        comparison = self.scores()
        first = build_metric_chart_html(comparison, "Accuracy", animation_key="first")
        same = build_metric_chart_html(comparison, "Accuracy", animation_key="first")
        replay = build_metric_chart_html(comparison, "Accuracy", animation_key="replay")
        self.assertEqual(first, same)
        self.assertNotEqual(parse_metric_chart(first).figure["id"], parse_metric_chart(replay).figure["id"])
        normalized = re.sub(r"\s+", "", first)
        self.assertIn("@media(prefers-reduced-motion:reduce)", normalized)
        self.assertIn("animation:none!important", normalized)
        self.assertIn("transform:scaleX(1)!important", normalized)
        assert_metric_chart(self, replay, comparison, "Accuracy")

    def test_invalid_tables_metrics_and_out_of_range_scores_have_clear_errors(self):
        comparison = self.scores()
        cases = [
            (comparison, "unsupported"),
            (comparison.drop(columns="Model"), "Accuracy"),
            (comparison.drop(columns="Accuracy"), "Accuracy"),
            (comparison.iloc[:0], "Accuracy"),
            (comparison.assign(Accuracy=1.1), "Accuracy"),
            (comparison.assign(MCC=-1.1), "MCC"),
        ]
        for table, metric in cases:
            with self.subTest(metric=metric, rows=len(table)), self.assertRaises(ValueError):
                build_metric_chart_html(table, metric)


if __name__ == "__main__":
    unittest.main()
