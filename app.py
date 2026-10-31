"""Local Streamlit app for evaluating the five saved classifiers."""

from io import BytesIO
from base64 import b64encode
from html import escape
from pathlib import Path
import hashlib
import json
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import streamlit as st

from utils import MAX_CSV_BYTES, MAX_ROWS, METRICS, ROOT, evaluate_models, load_artifacts, validate_data
from downloads import sample_download_url


MAX_TEST_ROWS = MAX_ROWS
ASSIGNMENT_TITLE = "BITS: ML Assignment 2"
STUDENT_ID = "2026AJ05054"
APP_TITLE = f"{ASSIGNMENT_TITLE} · {STUDENT_ID}"
MODEL_PRESENTATION = {
    "Logistic Regression": (
        "LR",
        "A classification algorithm that estimates class probabilities from a weighted combination of input features and predicts the most likely class.",
        "#087f75",
    ),
    "Decision Tree": (
        "DT",
        "A classification algorithm that follows a sequence of learned if-then rules about the features to decide which class a record belongs to.",
        "#4f62b6",
    ),
    "KNN": (
        "KNN",
        "K-Nearest Neighbors predicts a record's class using similar training examples nearby. This implementation uses 15 neighbours and gives closer examples more influence.",
        "#2871a4",
    ),
    "Naive Bayes": (
        "NB",
        "A probabilistic classifier based on Bayes' theorem that assumes features are independent within each class. The Gaussian version used here models each feature within a class using a normal, bell-shaped distribution.",
        "#9d6c12",
    ),
    "Support Vector Machine": (
        "SVM",
        "A classifier that learns a boundary separating classes while seeking a wide margin between them. The RBF kernel used here allows that boundary to be nonlinear.",
        "#7853b3",
    ),
}

st.set_page_config(
    page_title=APP_TITLE,
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


def refresh_model_selection():
    """Replay the selection transition when the native dropdown changes."""
    st.session_state["model_selection_revision"] = st.session_state.get("model_selection_revision", 0) + 1


def sidebar_model_picker(panel, models, ready=False, invalid=False):
    """Keep the model control beside the upload without bypassing CSV checks."""
    with panel:
        st.markdown('<div class="sidebar-step">02 / CLASSIFIER</div>', unsafe_allow_html=True)
        st.header("Choose a model", anchor=False)
        if not ready:
            message = "Fix the uploaded CSV to unlock the models." if invalid else "Upload a valid test CSV to unlock the five models."
            st.markdown(
                '<div class="sidebar-model-lock"><span aria-hidden="true">↑</span>'
                f'<p>{message}</p></div>',
                unsafe_allow_html=True,
            )
            return None
        selected_model = st.selectbox(
            "Select a model to inspect", list(models), index=None,
            placeholder="Choose a model...", key="model_selection",
            help="View one classifier at a time. Switch models to compare results on the same uploaded test data.",
            on_change=refresh_model_selection,
        )
        if selected_model is None:
            st.markdown(
                '<p class="sidebar-model-hint">Choose a classifier to display its definition and results.</p>',
                unsafe_allow_html=True,
            )
        else:
            symbol, _, accent = MODEL_PRESENTATION[selected_model]
            st.markdown(
                f'<div class="sidebar-selected-model" data-model="{escape(selected_model, quote=True)}" '
                f'style="--model-accent:{accent}">'
                f'<span class="sidebar-model-symbol" aria-hidden="true">{symbol}</span>'
                '<div class="sidebar-model-copy"><span class="sidebar-selected-label">Viewing</span>'
                f'<strong>{escape(selected_model)}</strong></div></div>',
                unsafe_allow_html=True,
            )
        return selected_model


def show_selected_model(model_name):
    """Show selection feedback; animation never changes scores or predictions."""
    symbol, description, accent = MODEL_PRESENTATION[model_name]
    revision = ":".join([
        model_name,
        str(st.session_state.get("model_selection_revision", 0)),
        str(st.session_state.get("upload_revision", 0)),
    ])
    # Safe CSS identifiers change on selection, so transitions replay even
    # when Streamlit reuses existing metric elements during a rerun.
    token = hashlib.sha256(revision.encode()).hexdigest()[:12]
    st.markdown(
        f"""<style>
@keyframes model-choice-{token} {{
  from {{opacity:.3;transform:translateY(8px) scale(.985);}}
  to {{opacity:1;transform:none;}}
}}
@keyframes metric-choice-{token} {{
  from {{opacity:.25;transform:translateY(10px);}}
  to {{opacity:1;transform:none;}}
}}
@keyframes result-choice-{token} {{
  from {{opacity:.4;transform:translateY(8px);}}
  to {{opacity:1;transform:none;}}
}}
@keyframes sidebar-choice-{token} {{
  from {{opacity:.35;transform:translateX(-7px);}}
  to {{opacity:1;transform:none;}}
}}
.st-key-model_detail {{--model-accent:{accent};}}
.model-selection {{animation:model-choice-{token} .42s ease-out both;}}
.sidebar-selected-model {{animation:sidebar-choice-{token} .4s ease-out both;}}
[class*="st-key-metric_"] [data-testid="stMetric"] {{
  animation-name:metric-choice-{token};animation-duration:.48s;animation-fill-mode:both;
}}
.st-key-matrix_panel,.st-key-report_panel {{
  animation-name:result-choice-{token};animation-duration:.5s;animation-fill-mode:both;
}}
@media (prefers-reduced-motion:reduce) {{
  .model-selection,.sidebar-selected-model,[class*="st-key-metric_"] [data-testid="stMetric"],
  .st-key-matrix_panel,.st-key-report_panel {{animation:none!important;transform:none!important;}}
}}
</style>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="model-selection" data-model="{escape(model_name, quote=True)}" '
        f'role="status" aria-live="polite" aria-atomic="true" style="--model-accent:{accent}">'
        f'<span class="model-symbol" aria-hidden="true">{symbol}</span>'
        '<div class="model-copy"><span class="model-kicker">Selected classifier</span>'
        f'<strong>{escape(model_name)}</strong>'
        '<p class="model-definition"><span class="model-definition-label">Basic definition</span>'
        f'{escape(description)}</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def show_confusion_matrix(matrix, labels, model_name):
    count = len(labels)
    size = min(16, max(7, count * .65))
    figure_size = (8.6, 6.4) if count <= 3 else (size + 4, size)
    figure, axis = plt.subplots(figsize=figure_size, dpi=140)
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
    descriptions = [
        f"Actual {label}: " + "; ".join(
            f"predicted {predicted_label} {int(matrix[row, column]):,}"
            for column, predicted_label in enumerate(labels)
        )
        for row, label in enumerate(labels)
    ]
    alternative = f"{model_name} confusion matrix. " + ". ".join(descriptions) + "."
    st.markdown(
        f'<div class="matrix-viewport" role="region" aria-label="{escape(model_name, quote=True)} confusion matrix chart" tabindex="0">'
        '<figure class="confusion-figure">'
        f'<img src="data:image/png;base64,{plot_base64}" '
        f'alt="{escape(alternative, quote=True)}" '
        f'width="{int(figure.get_figwidth() * 140)}" height="{int(figure.get_figheight() * 140)}">'
        '</figure></div>',
        unsafe_allow_html=True,
    )
    plt.close(figure)


def show_matrix_values(matrix, labels):
    """Expose the same plotted counts as a keyboard-accessible HTML table."""
    headers = ''.join(f'<th scope="col">{escape(str(label))}</th>' for label in labels)
    rows = ''.join(
        '<tr>' + f'<th scope="row">{escape(str(label))}</th>'
        + ''.join(f'<td>{int(value):,}</td>' for value in matrix[index]) + '</tr>'
        for index, label in enumerate(labels)
    )
    st.markdown(
        '<div class="matrix-table-wrapper" role="region" aria-label="Confusion matrix counts" tabindex="0">'
        '<table class="matrix-table"><caption>Rows are actual classes. Columns are predicted classes.</caption>'
        '<thead><tr><th scope="col">Actual / Predicted</th>' + headers + '</tr></thead>'
        '<tbody>' + rows + '</tbody></table></div>',
        unsafe_allow_html=True,
    )


def main():
    profiles = discover_profiles()
    if not profiles:
        st.error("No saved dataset models were found.")
        st.info("Train a dataset locally with python train.py, then refresh this page.")
        st.stop()
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><div class="brand-symbol" aria-hidden="true">'
            '<svg viewBox="0 0 24 26" fill="none"><path d="M12 2 21 6v7c0 5-6 9-9 11-3-2-9-6-9-11V6l9-4Z" '
            'stroke="currentColor" stroke-width="1.5"/><path d="m8 13 3 3 5-6" stroke="currentColor" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg></div>'
            f'<div><div class="brand-name">{ASSIGNMENT_TITLE}</div>'
            f'<div class="brand-subtitle">Student ID {STUDENT_ID}</div></div></div>',
            unsafe_allow_html=True,
        )
        if len(profiles) > 1:
            model_dir = st.selectbox(
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
            '<div class="hero-eyebrow"><span>MACHINE LEARNING / CLASSIFICATION</span>'
            f'<span class="hero-student">Student ID · {STUDENT_ID}</span></div>',
            unsafe_allow_html=True,
        )
        st.title(ASSIGNMENT_TITLE, anchor=False)
        st.markdown(
            f'<p class="hero-dataset">Dataset: {escape(dataset["name"])}</p>',
            unsafe_allow_html=True,
        )
        st.write("Upload your test data, choose a classifier, and explore its predictions.")
        classification_type = "Binary classification" if len(metadata["classes"]) == 2 else "Multiclass classification"
        st.markdown(
            '<div class="hero-facts"><span>5 classification models</span>'
            f'<span>6 evaluation metrics</span><span>{dataset["feature_count"]} input features</span>'
            f'<span>{classification_type}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="mobile-controls-hint"><span aria-hidden="true">↖</span> '
            'Use the top-left arrow to open test data and model controls.</p>',
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
        with st.container(key="sidebar_upload_panel"):
            st.markdown('<div class="sidebar-step">01 / TEST DATA</div>', unsafe_allow_html=True)
            st.header("Test data", anchor=False)
            # Only the saved sample is made downloadable; uploads stay private.
            st.markdown(
                f'<a class="sample-download" href="data:text/csv;base64,{b64encode(sample_bytes).decode("ascii")}" '
                'download="test_data.csv" target="_self">'
                '<span aria-hidden="true">↓</span><span>Download test CSV</span></a>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<a class="sample-preview" href="{sample_download_url(sample_bytes)}" '
                'target="_blank" rel="noopener noreferrer">Open CSV in a new tab</a>',
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Upload test CSV", type=["csv"], key=f"test_upload_{model_dir}",
                help=f"Include {dataset['feature_count']} input features and the {metadata['target_column']} target column.",
                on_change=refresh_test_results,
            )
            st.caption("CSV · Up to 5 MB · 20,000 rows")
        # Reserve the position here; populate it once CSV validation is known.
        model_picker_panel = st.container(key="sidebar_model_panel")
        with st.expander("Required CSV format"):
            st.caption(f"Current dataset: {dataset['name']}")
            st.caption(f"Target column: {metadata['target_column']}")
            st.write("These feature columns and the target are required. Column order can differ.")
            st.code("\n".join(metadata["feature_names"] + [metadata["target_column"]]), language=None)
            st.write("Classes: " + ", ".join(map(str, metadata["classes"])))
            st.caption("If an embedded preview blocks saving, open the app in your regular browser. The sample is also in the project folder.")
            st.caption("For a different dataset, first train its models locally with train.py.")

    with st.expander("How the dataset is used"):
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
        dataset_text = f"[{dataset['name']}]({dataset['url']})" if dataset.get("url") else dataset["name"]
        st.caption(f"Training dataset: {dataset_text} · {dataset['feature_count']} input features · Target: {metadata['target_column']} · {len(metadata['classes'])} classes")
        st.write("The training command loads the full source dataset, creates a separate training and test partition, and fits all five classifiers using only the training rows. The fitted preprocessing and classifiers are saved in .pkl files.")
        st.write("The app loads those saved models and waits for a labeled test CSV. After uploading a valid file, choose a model to display its metrics, confusion matrix, report and predictions using only that file. Switch models to inspect each classifier on the same test data. The models are not retrained.")
        st.write("Removing the uploaded file hides the results until another valid test CSV is uploaded. Evaluating the entire original dataset would include training rows and would not provide an independent test of performance.")

    if uploaded_file is None:
        sidebar_model_picker(model_picker_panel, models)
        with st.container(key="upload_prompt"):
            st.markdown('<div class="section-eyebrow">READY TO EVALUATE</div>', unsafe_allow_html=True)
            st.subheader("Upload your test data to begin", anchor=False)
            st.write("Use **Test data** in the sidebar to upload the sample CSV or another matching labeled file.")
            st.info("After upload, choose a classifier in the sidebar to view its results.")
        return

    csv_bytes = uploaded_file.getvalue()
    data_name = uploaded_file.name
    try:
        with st.spinner("Evaluating saved models on your uploaded test data..."):
            frame, _comparison, details = evaluate_csv(
                csv_bytes, version, str(model_dir), st.session_state.get("upload_revision", 0),
            )
    except (ValueError, KeyError, UnicodeDecodeError, pd.errors.ParserError) as error:
        sidebar_model_picker(model_picker_panel, models, invalid=True)
        st.error(f"The test CSV could not be evaluated: {error}")
        st.info("Upload a valid labeled test CSV to display results. Download the test sample for the selected dataset to check the required format.")
        return

    selected_model = sidebar_model_picker(model_picker_panel, models, ready=True)
    if selected_model is None:
        st.success(f"Test data loaded from {data_name}: {len(frame):,} test rows. Choose a classifier in the sidebar.")
    else:
        st.success(f"{len(frame):,} test rows evaluated with {selected_model}.")

    # Uploaded filenames are escaped before entering the static HTML summary.
    st.markdown(
        '<div class="dataset-summary">'
        '<div class="summary-item"><div class="summary-label">Uploaded test set</div>'
        f'<div class="summary-value summary-file" title="{escape(data_name)}">{escape(data_name)}</div>'
        '<div class="summary-detail">Current evaluation source</div></div>'
        f'<div class="summary-item"><div class="summary-label">Rows being evaluated now</div><div class="summary-value">{len(frame):,}</div>'
        '<div class="summary-detail">Current test data only</div></div>'
        f'<div class="summary-item"><div class="summary-label">Input features</div><div class="summary-value">{dataset["feature_count"]}</div>'
        f'<div class="summary-detail">{len(metadata["classes"])} target classes</div></div></div>',
        unsafe_allow_html=True,
    )
    with st.container(key="model_detail"):
        st.markdown('<div class="section-eyebrow">01 / MODEL RESULTS</div>', unsafe_allow_html=True)
        if selected_model is None:
            st.markdown(
                '<div class="model-waiting"><span class="model-symbol" aria-hidden="true">↗</span>'
                '<div class="model-copy"><strong>Choose a model in the left sidebar</strong>'
                '<p>Open the sidebar and select a classifier. Its six metrics, confusion matrix and classification report will appear here.</p>'
                '</div></div>',
                unsafe_allow_html=True,
            )
        else:
            result = details[selected_model]
            show_selected_model(selected_model)
            st.subheader(f"{selected_model}: evaluation metrics", anchor=False)
            if result.get("auc_reason"):
                st.warning("AUC is unavailable: " + result["auc_reason"])
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
                "MCC": "Matthews correlation coefficient: overall agreement from -1 to 1. Higher values indicate better classification.",
            }
            with st.container(key="metrics_grid"):
                for metric in METRICS:
                    value = result["metrics"][metric]
                    with st.container(key=f"metric_{metric.lower()}"):
                        st.metric(
                            metric,
                            "N/A" if value is None or pd.isna(value) else f"{value:.4f}",
                            help=metric_help[metric],
                        )

    if selected_model is None:
        return

    with st.container(key="matrix_panel"):
        st.markdown('<div class="section-eyebrow">02 / UNDERSTAND THE ERRORS</div>', unsafe_allow_html=True)
        st.subheader("Confusion matrix", anchor=False)
        st.caption(f"Selected model: {selected_model} · Evaluation data: {data_name}")
        class_count = len(result["classes"])
        st.caption(f"{class_count} × {class_count} matrix · Rows: actual class · Columns: predicted class")
        show_confusion_matrix(result["confusion_matrix"], result["class_names"], selected_model)
        st.markdown('<p class="matrix-scroll-hint">Scroll across the chart to view every label, or open the values below.</p>', unsafe_allow_html=True)
        correct = int(result["confusion_matrix"].trace())
        st.caption(f"All {len(frame):,} test records are shown: {correct:,} correct predictions and {len(frame) - correct:,} errors. Percentages are calculated within each actual class.")
        with st.expander("Confusion matrix values"):
            show_matrix_values(result["confusion_matrix"], result["class_names"])
    with st.container(key="report_panel"):
        st.markdown('<div class="section-eyebrow">03 / CLASS BY CLASS</div>', unsafe_allow_html=True)
        st.subheader("Classification report", anchor=False)
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

if __name__ == "__main__":
    main()
