"""Local Streamlit app for evaluating the five saved classifiers."""

from io import BytesIO
from base64 import b64encode
from html import escape
from pathlib import Path
import json
import hashlib
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import streamlit as st

from utils import MAX_CSV_BYTES, MAX_ROWS, METRICS, ROOT, evaluate_models, load_artifacts, validate_data
from downloads import sample_download_url
from comparison_chart import build_metric_chart_html


MAX_TEST_ROWS = MAX_ROWS

st.set_page_config(
    page_title="UCI Phishing Websites | ML Assignment 2",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown(
    f"<style>{(ROOT / 'styles.css').read_text()}</style>",
    unsafe_allow_html=True,
)


def discover_profiles():
    """Find trusted, locally trained datasets, never user-uploaded model files."""
    configured = os.environ.get("ML_MODEL_DIR")
    folders = [ROOT / "model", *sorted((ROOT / "model_profiles").glob("*"))]
    if configured:
        folders.insert(0, Path(configured).expanduser())
    profiles = {}
    for folder in folders:
        folder = folder.resolve()
        if folder in profiles or not (folder / "metadata.json").is_file():
            continue
        try:
            info = json.loads((folder / "metadata.json").read_text())
            profiles[folder] = info["dataset"]["name"]
        except (OSError, ValueError, KeyError):
            continue
    return profiles


def artifact_version(model_dir):
    """Invalidate caches when any model or preprocessing artifact changes."""
    return tuple(
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(Path(model_dir).glob("*"))
        if path.is_file()
    )


@st.cache_resource(show_spinner=False)
def cached_artifacts(version, model_dir):
    return load_artifacts(model_dir)


@st.cache_data(show_spinner=False, max_entries=4)
def evaluate_csv(csv_bytes, version, model_dir, upload_revision=0):
    """Validate a test file and use saved models; this never fits a model."""
    if len(csv_bytes) > MAX_CSV_BYTES:
        raise ValueError("Choose a test CSV smaller than 5 MB.")
    # Pandas otherwise renames duplicate headers, hiding a malformed schema.
    header = pd.read_csv(BytesIO(csv_bytes), header=None, nrows=1, dtype=str, keep_default_na=False)
    if header.iloc[0].duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    metadata, models = cached_artifacts(version, model_dir)
    # A numeric-looking subset of text categories (for example "001") must
    # retain the exact spelling seen by the fitted encoder and target labels.
    csv_types = {
        name: str for name, kind in metadata.get("feature_types", {}).items()
        if kind == "categorical"
    }
    if all(isinstance(label, str) for label in metadata["classes"]):
        csv_types[metadata["target_column"]] = str
    frame = pd.read_csv(BytesIO(csv_bytes), dtype=csv_types)
    if frame.empty:
        raise ValueError("The test CSV must contain at least one data row.")
    if len(frame) > MAX_TEST_ROWS:
        raise ValueError("Upload no more than 20,000 test rows at a time.")
    features, target = validate_data(frame, metadata)
    comparison, details = evaluate_models(models, features, target, metadata)
    return frame, comparison, details


def refresh_test_results():
    """The uploader triggers a full Streamlit rerun after this callback.

    Give each upload a new evaluation revision, even when its filename and
    contents match an earlier upload. Models remain fitted and read-only.
    """
    st.session_state["upload_revision"] = st.session_state.get("upload_revision", 0) + 1


def show_confusion_matrix(matrix, labels, model_name):
    count = len(labels)
    size = min(16, max(7, count * .65))
    figure, axis = plt.subplots(figsize=(size + 4, size), dpi=140)
    figure.patch.set_facecolor("white")
    palette = LinearSegmentedColormap.from_list("classification_matrix", ["#eaf5f0", "#177c68"])
    heatmap = axis.imshow(matrix, cmap=palette, vmin=0, vmax=max(int(matrix.max()), 1))
    axis.set_title(f"{model_name} — Confusion matrix", fontsize=18, color="#172b45", fontweight="bold", pad=22)
    axis.set_xticks(range(count), labels=labels, rotation=35 if count > 3 else 0, ha="right" if count > 3 else "center")
    axis.set_yticks(range(count), labels=labels)
    axis.set_xlabel("Predicted label", color="#52677c", labelpad=16, fontsize=13)
    axis.set_ylabel("Actual label", color="#52677c", labelpad=16, fontsize=13)
    axis.tick_params(length=0, pad=12, labelsize=13 if count <= 3 else 9, colors="#52677c")
    axis.set_xticks([value - .5 for value in range(count + 1)], minor=True)
    axis.set_yticks([value - .5 for value in range(count + 1)], minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    threshold = matrix.max() * 0.75
    for row in range(count):
        for column in range(count):
            row_total = matrix[row].sum()
            percentage = matrix[row, column] / row_total if row_total else 0
            annotation = f"{matrix[row, column]:,}"
            if count <= 4:
                annotation += f"\n{percentage:.1%}"
            axis.text(
                column,
                row,
                annotation,
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#173e3a",
                fontsize=23 if count == 2 else max(6, 19 - count),
                fontweight="bold",
            )
    colorbar = figure.colorbar(heatmap, ax=axis, fraction=.045, pad=.06, shrink=.85)
    colorbar.set_label("Number of test records", color="#52677c", fontsize=11, labelpad=12)
    colorbar.ax.tick_params(labelsize=10, colors="#52677c")
    colorbar.outline.set_visible(False)
    figure.tight_layout()
    # Embed the plot so a rerun cannot leave an expired media URL in a browser.
    # Reserving its natural aspect ratio also keeps the page stable as it loads.
    plot_bytes = BytesIO()
    figure.savefig(plot_bytes, format="png", dpi=140)
    plot_base64 = b64encode(plot_bytes.getvalue()).decode("ascii")
    st.markdown(
        '<figure class="confusion-figure">'
        f'<img src="data:image/png;base64,{plot_base64}" '
        f'alt="{escape(model_name)} confusion matrix" '
        f'width="{int(figure.get_figwidth() * 140)}" height="{int(figure.get_figheight() * 140)}">'
        '</figure>',
        unsafe_allow_html=True,
    )
    plt.close(figure)


def main():
    profiles = discover_profiles()
    if not profiles:
        st.error("No saved dataset models were found.")
        st.info("Train a dataset locally with python train.py, then refresh this page.")
        st.stop()
    if len(profiles) > 1:
        model_dir = st.sidebar.selectbox(
            "Trained dataset", list(profiles), format_func=lambda path: profiles[path],
            key="dataset_profile",
        )
    else:
        model_dir = next(iter(profiles))
    version = artifact_version(model_dir)
    try:
        metadata, models = cached_artifacts(version, str(model_dir))
    except (OSError, ValueError, KeyError) as error:
        st.error(f"The saved model files could not be loaded: {error}")
        st.info("Run `python train.py` from the project folder, then restart the app.")
        st.stop()

    dataset = metadata["dataset"]
    with st.container(key="hero"):
        st.markdown(
            '<div class="hero-eyebrow">MACHINE LEARNING / CLASSIFICATION</div>',
            unsafe_allow_html=True,
        )
        st.title(dataset["name"])
        st.write("Compare five classifiers, explore prediction errors, and evaluate saved models on labeled test data.")
        classification_type = "Binary classification" if len(metadata["classes"]) == 2 else "Multiclass classification"
        st.markdown(
            '<div class="hero-facts"><span>5 classification models</span>'
            f'<span>6 evaluation metrics</span><span>{classification_type}</span></div>',
            unsafe_allow_html=True,
        )
    sample_path = model_dir / "test_data.csv"
    if not sample_path.exists() and model_dir == (ROOT / "model").resolve():
        sample_path = ROOT / "test_data.csv"
    try:
        sample_bytes = sample_path.read_bytes()
    except OSError as error:
        st.error(f"The bundled test_data.csv could not be read: {error}")
        st.stop()

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><div class="brand-symbol" aria-hidden="true">'
            '<svg viewBox="0 0 24 26" fill="none"><path d="M12 2 21 6v7c0 5-6 9-9 11-3-2-9-6-9-11V6l9-4Z" '
            'stroke="currentColor" stroke-width="1.5"/><path d="m8 13 3 3 5-6" stroke="currentColor" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg></div>'
            '<div><div class="brand-name">ML Assignment 2</div>'
            '<div class="brand-subtitle">Classifier Comparison</div></div></div>'
            '<div class="sidebar-step">01 / EVALUATION DATA</div>',
            unsafe_allow_html=True,
        )
        st.header("Test data")
        st.write(
            "Download a matching test sample or upload labeled test data for the selected trained dataset."
        )
        # Embed the exact CSV bytes so saving does not depend on a transient
        # Streamlit media URL. Only the saved sample is made downloadable.
        st.markdown(
            f'<a class="sample-download" href="data:text/csv;base64,{b64encode(sample_bytes).decode("ascii")}" '
            'download="test_data.csv" target="_self">'
            '<span aria-hidden="true">↓</span> Download test_data.csv</a>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<a class="sample-preview" href="{sample_download_url(sample_bytes)}" '
            'target="_blank" rel="noopener noreferrer">Open CSV in a new tab</a>',
            unsafe_allow_html=True,
        )
        st.caption("If an embedded preview blocks saving, open the local app in your regular browser. The same sample is also in the project folder.")
        st.write("")
        uploaded_file = st.file_uploader(
            "Upload test CSV", type=["csv"], key=f"test_upload_{model_dir}",
            help=f"Include {dataset['feature_count']} input features and the {metadata['target_column']} target column.",
            on_change=refresh_test_results,
        )
        st.caption("CSV only · Up to 5 MB · 20,000 rows · Results refresh automatically")
        st.divider()
        st.markdown(
            '<div class="sidebar-note"><strong>A fair comparison.</strong><br>'
            'All five models evaluate the same rows. Uploads are used for evaluation, with no retraining.</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Current dataset: {dataset['name']}")
        st.caption(f"Target column: {metadata['target_column']}")
        with st.expander("Required CSV format"):
            st.write("These feature columns and the target are required. Column order can differ.")
            st.code(",".join(metadata["feature_names"] + [metadata["target_column"]]), language=None)
            st.write("Classes: " + ", ".join(map(str, metadata["classes"])))
            st.caption("For a different dataset, first train its models locally using the instructions in README.md.")

    if uploaded_file is None:
        csv_bytes = sample_bytes
        data_name = "bundled test_data.csv"
    else:
        csv_bytes = uploaded_file.getvalue()
        data_name = uploaded_file.name

    try:
        with st.spinner("Evaluating saved models on the test data..."):
            frame, comparison, details = evaluate_csv(
                csv_bytes, version, str(model_dir), st.session_state.get("upload_revision", 0),
            )
    except (ValueError, KeyError, UnicodeDecodeError, pd.errors.ParserError) as error:
        st.error(f"The test CSV could not be evaluated: {error}")
        st.info("Download the test sample for the selected dataset to check the format. To use unrelated features or a different target, first train a separate dataset profile using train.py.")
        st.stop()

    if uploaded_file is not None:
        st.success(f"Results refreshed from {data_name}: {len(frame):,} test rows evaluated with all five saved models.")

    split = metadata.get("split", {})
    if "train_rows" in split and "test_rows" in split:
        st.markdown(
            '<div class="dataset-flow" aria-label="Training and test split">'
            f'<div><span>Full source dataset</span><strong>{dataset["rows"]:,} rows</strong></div>'
            '<span class="flow-arrow" aria-hidden="true">→</span>'
            f'<div><span>Used for training</span><strong>{split["train_rows"]:,} rows</strong></div>'
            '<span class="flow-plus" aria-hidden="true">+</span>'
            f'<div><span>Reserved for testing</span><strong>{split["test_rows"]:,} rows</strong></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption("The models learned from the training rows. On startup, this app evaluates the reserved test rows. Uploading a CSV replaces the evaluation data below.")

    # Uploaded filenames are escaped before entering the static HTML summary.
    source_label = "Bundled held-out test set" if uploaded_file is None else "Uploaded test set"
    st.markdown(
        '<div class="dataset-summary">'
        f'<div class="summary-item"><div class="summary-label">{source_label}</div>'
        f'<div class="summary-value summary-file" title="{escape(data_name)}">{escape(data_name)}</div>'
        '<div class="summary-detail">Current evaluation source</div></div>'
        f'<div class="summary-item"><div class="summary-label">Rows being evaluated now</div><div class="summary-value">{len(frame):,}</div>'
        '<div class="summary-detail">Current test data only</div></div>'
        f'<div class="summary-item"><div class="summary-label">Input features</div><div class="summary-value">{dataset["feature_count"]}</div>'
        f'<div class="summary-detail">{len(metadata["classes"])} target classes</div></div></div>',
        unsafe_allow_html=True,
    )
    dataset_text = f"[{dataset['name']}]({dataset['url']})" if dataset.get("url") else dataset["name"]
    st.caption(f"Training dataset: {dataset_text} · Target: {metadata['target_column']} · {len(metadata['classes'])} classes")
    with st.expander("How the dataset is used"):
        st.write("The training command loads the full source dataset, creates a separate training and test partition, and fits all five classifiers using only the training rows. The fitted preprocessing and classifiers are saved in .pkl files.")
        st.write("Starting the app loads those saved models and evaluates the default held-out test CSV. Uploading a labeled test CSV updates the comparison table, chart, metrics, confusion matrix, report and predictions using that file. The models are not retrained.")
        st.write("Removing the upload restores the default held-out test results. Evaluating the entire original dataset would include training rows and would not provide an independent test of performance.")
    auc_reasons = {result["auc_reason"] for result in details.values() if result.get("auc_reason")}
    if auc_reasons:
        st.warning("AUC is unavailable: " + " ".join(sorted(auc_reasons)))

    with st.container(key="comparison"):
        st.markdown('<div class="section-eyebrow">01 / THE BIG PICTURE</div>', unsafe_allow_html=True)
        st.subheader("Comparison of all five models")
        st.caption("One test set. Five perspectives. The highest value in each metric is highlighted.")
        st.dataframe(
            comparison.style.highlight_max(subset=METRICS, color="#e0f2e9"),
            hide_index=True,
            width="stretch",
            column_config={
                metric: st.column_config.NumberColumn(metric, format="%.4f")
                for metric in METRICS
            },
        )

    with st.container(key="metric_chart_panel"):
        st.markdown('<div class="section-eyebrow">02 / VISUAL COMPARISON</div>', unsafe_allow_html=True)
        st.subheader("Animated metric comparison")
        st.caption(f"All five models on {data_name} · {len(frame):,} test rows")
        metric_column, replay_column = st.columns([3, 1], vertical_alignment="bottom")
        with metric_column:
            chart_metric = st.selectbox("Choose a metric to compare", METRICS, key="comparison_metric")
        with replay_column:
            if st.button("Replay chart animation", key="replay_comparison", width="stretch"):
                st.session_state["chart_replay_revision"] = st.session_state.get("chart_replay_revision", 0) + 1
        animation_key = ":".join([
            str(model_dir), hashlib.sha256(csv_bytes).hexdigest(),
            str(st.session_state.get("upload_revision", 0)),
            str(st.session_state.get("chart_replay_revision", 0)),
        ])
        st.markdown(
            build_metric_chart_html(comparison, chart_metric, animation_key=animation_key),
            unsafe_allow_html=True,
        )

    with st.container(key="model_detail"):
        st.markdown('<div class="section-eyebrow">03 / LOOK CLOSER</div>', unsafe_allow_html=True)
        selected_model = st.selectbox("Select a model to inspect", list(models), key="model_selection")
        result = details[selected_model]
        st.subheader(f"{selected_model}: evaluation metrics")
        if len(result["classes"]) == 2:
            positive = metadata["positive_class"]
            positive_name = metadata.get("target_labels", {}).get(str(positive), str(positive))
            st.caption(f"Precision, recall and F1 use {positive_name} ({positive}) as the positive class. AUC uses continuous scores.")
        else:
            st.caption("Precision, recall and F1 are support-weighted across classes. AUC is weighted one-vs-rest ROC-AUC.")
        metric_help = {
            "Accuracy": "Fraction of all records classified correctly.",
            "AUC": "ROC area under the curve from continuous scores; weighted one-vs-rest for multiclass data.",
            "Precision": "Correct positive predictions divided by positive predictions; support-weighted for multiclass data.",
            "Recall": "Detected positives divided by actual positives; support-weighted for multiclass data.",
            "F1": "Harmonic mean of precision and recall; support-weighted for multiclass data.",
            "MCC": "Matthews correlation coefficient: agreement using all four confusion-matrix cells.",
        }
        for first in (0, 3):
            columns = st.columns(3)
            for column, metric in zip(columns, METRICS[first : first + 3]):
                value = result["metrics"][metric]
                with column, st.container(key=f"metric_{metric.lower()}"):
                    st.metric(
                        metric,
                        "N/A" if value is None or pd.isna(value) else f"{value:.4f}",
                        help=metric_help[metric],
                    )

    with st.container(key="matrix_panel"):
        st.markdown('<div class="section-eyebrow">04 / UNDERSTAND THE ERRORS</div>', unsafe_allow_html=True)
        st.subheader("Confusion matrix")
        st.caption(f"Selected model: {selected_model} · Evaluation data: {data_name}")
        class_count = len(result["classes"])
        st.caption(f"{class_count} target classes give a {class_count} × {class_count} matrix. Input features do not determine the matrix size.")
        st.caption("Rows are actual classes and columns are predicted classes. The diagonal contains correct predictions.")
        show_confusion_matrix(result["confusion_matrix"], result["class_names"], selected_model)
        correct = int(result["confusion_matrix"].trace())
        st.caption(f"All {len(frame):,} test records are shown: {correct:,} correct predictions and {len(frame) - correct:,} errors. Cell percentages are calculated within each actual class.")
    with st.container(key="report_panel"):
        st.markdown('<div class="section-eyebrow">05 / CLASS BY CLASS</div>', unsafe_allow_html=True)
        st.subheader("Classification report")
        st.caption("A detailed view of performance for every class in this dataset.")
        report = {
            label: values
            for label, values in result["classification_report"].items()
            if isinstance(values, dict)
        }
        report_table = pd.DataFrame(report).T
        report_table.index.name = "Class"
        st.dataframe(
            report_table.style.format({
                "precision": "{:.4f}", "recall": "{:.4f}",
                "f1-score": "{:.4f}", "support": "{:.0f}",
            }), width="stretch",
        )
        st.caption("Support is the number of actual records. Undefined precision, recall or F1 is shown as 0.")

    with st.expander("Preview current test data and predictions"):
        preview = frame.head(20).copy()
        preview[f"predicted_{metadata['target_column']}"] = result["predictions"][: len(preview)]
        st.dataframe(preview, hide_index=True, width="stretch")
        st.caption("Showing the first 20 rows. The target column contains the actual label.")

    st.markdown(
        '<div class="page-footer"><span>Machine Learning · Assignment 2</span>'
        '<span class="footer-status">Evaluation · Saved models · No retraining</span></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
