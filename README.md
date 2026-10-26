# Machine Learning Assignment 2: UCI Phishing Websites

Compare five classification models using your chosen dataset in a local Streamlit application. The project includes trained models and held-out test data for UCI Phishing Websites, and supports other tabular binary or multiclass datasets through saved dataset profiles. Each profile contains five trained models, their fitted preprocessing, dataset metadata, evaluation results and labeled test data.

Train a new dataset with `train.py` before evaluating it in the app. Models trained on one set of features cannot evaluate an unrelated dataset: choose the matching saved profile, then upload its test CSV. The app evaluates saved models without retraining on uploaded data. Everything remains local, and the app has no deployment controls.

## 1. Problem statement

Implement Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine on one classification dataset. Compare all five models on the same held-out rows using Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC). The assignment requires a public Kaggle or UCI dataset with at least 12 input features and 500 records.

The bundled assignment example predicts phishing versus legitimate websites from pre-extracted indicators. Its positive class is phishing. A false negative is a phishing website classified as legitimate; a false positive is a legitimate website incorrectly flagged as phishing. This example evaluates labeled feature records rather than scanning raw URLs.

## 2. Dataset description

### Bundled example: UCI Phishing Websites

The dataset description, measured comparison and observations below document the supplied example. They do not describe results for a new dataset; document that dataset and regenerate its results when preparing your own submission.

| Item | Details |
| --- | --- |
| Dataset | UCI Phishing Websites, dataset ID 327 |
| Official source | https://archive.ics.uci.edu/dataset/327/phishing+websites |
| Training download | [Official UCI archive](https://archive.ics.uci.edu/static/public/327/phishing+websites.zip), loaded directly into memory |
| DOI | https://doi.org/10.24432/C51W2X |
| Instances | 11,055 |
| Input features | 30, excluding the target column |
| Classification type | Binary: legitimate or phishing |
| Full-dataset class counts | 6,157 legitimate; 4,898 phishing |
| Missing values | None in the supplied data |
| Project target | `is_phishing`: 0 = legitimate, 1 = phishing |
| Original target mapping | UCI `Result=-1` becomes 1; UCI `Result=+1` becomes 0 |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |

The dataset meets the minimum requirements: 30 features is at least 12, and 11,055 instances is at least 500. It is a historical cybersecurity dataset for phishing detection, rather than a chronological incident-response log. UCI describes the original collections as PhishTank, MillerSmiles and Google search operators.

Features cover URL structure, links and forms, browser/page behavior, domain information and reputation indicators. Examples include `URL_Length`, `having_At_Symbol`, `SSLfinal_State`, `URL_of_Anchor`, `SFH`, `DNSRecord` and `web_traffic`. Original feature names and integer category codes are preserved. The complete feature list and feature-specific allowed values are in `model/metadata.json`. Feature codes of -1 are valid values, not missing data.

### Preprocessing and common train/test split

- Running `python train.py` fetches the original UCI dataset directly from its public archive URL, parses it in memory and remaps its target to `is_phishing`. Internet access is required for this training command. The full training dataset is not stored in a `data/` folder.
- The generated `test_data.csv` contains all 30 original input features and the remapped `is_phishing` target. Do not remap these labels again. Saved models and the held-out CSV can be evaluated locally without fetching the full dataset.
- All 11,055 records are retained. There are 5,270 repeated feature rows beyond the first occurrence and 64 feature patterns with conflicting labels.
- Identical feature vectors are grouped together before splitting, even when their labels differ. This prevents the same feature pattern appearing in both training and test data.
- A shuffled `StratifiedGroupKFold` with five folds and seed 2026 supplies one held-out fold. This is one evaluation split, not five-fold cross-validation of model performance.
- Training contains 8,841 records. Test data contains 2,214 records: 1,233 legitimate and 981 phishing. The split is approximately 80/20 because whole groups are kept intact. Train/test feature-group overlap is zero.
- Logistic Regression, KNN and SVM use `StandardScaler` fitted on training data only. The tree and Gaussian Naive Bayes use the original encoded inputs. Each scaler is saved inside its model pipeline.
- All five models use the same rows, features and labels. Their settings are fixed baselines; the test set is not used to choose hyperparameters.

## 3. GitHub Repository Link

Pending. The project is currently local only; repository creation and publishing are outside the current work.

## 4. Models used

These fixed baseline settings apply to the bundled example. Custom dataset profiles use the same five classifiers, with numeric scaling and categorical encoding saved inside each pipeline.

| ML Model Name | Implementation |
| --- | --- |
| Logistic Regression | StandardScaler + LogisticRegression: C=1.0, max_iter=2000 |
| Decision Tree | DecisionTreeClassifier: max_depth=12, min_samples_leaf=3, random_state=2026 |
| KNN | StandardScaler + KNeighborsClassifier: 15 neighbors, distance weights, Euclidean distance, ball_tree, n_jobs=1 |
| Naive Bayes | GaussianNB: var_smoothing=1e-8 |
| Support Vector Machine | StandardScaler + SVC: RBF kernel, C=3.0, gamma=scale |

### Comparison of evaluation metrics

The following measured results belong to the bundled phishing example and use the same 2,214 held-out test records. Values are rounded to four decimal places; full-precision results are saved in `model/metrics.csv`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9273 | 0.9751 | 0.9476 | 0.8848 | 0.9151 | 0.8532 |
| Decision Tree | 0.9345 | 0.9718 | 0.9372 | 0.9134 | 0.9251 | 0.8672 |
| KNN | 0.9250 | 0.9723 | 0.9232 | 0.9062 | 0.9146 | 0.8479 |
| Naive Bayes | 0.6274 | 0.9684 | 0.5432 | 1.0000 | 0.7040 | 0.4240 |
| Support Vector Machine | 0.9508 | 0.9887 | 0.9580 | 0.9297 | 0.9436 | 0.9003 |

Precision, recall and F1 are binary metrics with phishing as class 1. AUC is ROC-AUC from class-1 probabilities for Logistic Regression, Decision Tree, KNN and GaussianNB; SVM uses its continuous decision function. AUC is not calculated from predicted class labels. An SVM decision score is not a probability.

Accuracy measures the fraction of all correct predictions. Precision measures the fraction of phishing alerts that are correct. Recall measures the fraction of actual phishing records detected. F1 is the harmonic mean of precision and recall. MCC uses all four confusion-matrix cells and ranges from -1 to 1. AUC measures ranking across thresholds, so a high AUC need not imply good predictions at the default classification threshold.

### Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Accuracy is 92.73% and phishing precision is 94.76%. It raises 48 false alarms and misses 113 phishing records. Its linear boundary provides a useful baseline, but recall is lower than the tree, KNN and SVM on this split. |
| Decision Tree | Accuracy is 93.45% and F1 is 0.9251. It misses 85 phishing records and raises 60 false alarms. It can capture nonlinear indicator interactions; its default-threshold accuracy is better than Logistic Regression despite a lower AUC. |
| KNN | Accuracy is 92.50% and F1 is 0.9146. It detects 889 phishing records, misses 92 and raises 74 false alarms. It is competitive, although Euclidean distance over encoded categories is an approximation. |
| Naive Bayes | Recall is 1.0000, but it incorrectly flags 825 legitimate records. Precision falls to 0.5432, accuracy to 62.74% and MCC to 0.4240. Correlated categorical indicators make its independent Gaussian assumptions questionable, which may help explain the behavior. Its AUC of 0.9684 shows that strong ranking can coexist with poor predictions at the default boundary. |
| Support Vector Machine | It gives the highest accuracy (95.08%), AUC (0.9887), precision (0.9580), F1 (0.9436) and MCC (0.9003) of these five baselines. It detects 912 of 981 phishing records, misses 69 and raises 40 false alarms. Its RBF kernel represents nonlinear relationships between features. |
| Overall, Winner for your dataset? | Support Vector Machine offers the best balanced performance on this held-out split, supported by F1 and MCC. GaussianNB has higher recall but many more false alarms. This is a conclusion about the measured split, not every future dataset. |

For SVM, the confusion matrix is `[[1193, 40], [69, 912]]`. Rows are actual labels and columns are predicted labels, both ordered `[legitimate, phishing]`. All models' confusion matrices and classification reports are saved in `model/metadata.json`.

### Understanding the confusion matrix

The matrix size comes from the number of target classes, not the number of input features. This dataset has 30 input features but only two target classes, **Legitimate** and **Phishing**, so its confusion matrix correctly has two rows and two columns. A dataset with seven target classes produces a 7 × 7 matrix. Changing the displayed size does not change the number of classes.

Read each row as the actual class and each column as the predicted class. For the SVM example above, 1,193 legitimate and 912 phishing records are classified correctly on the diagonal; 40 legitimate records are flagged as phishing and 69 phishing records are missed.

The dataset is historical. Its feature codes and reputation indicators may not describe current websites well. Grouping prevents identical-input leakage, but there are no reliable domain/time identifiers for proving domain-independent or future-traffic performance. Scaling does not turn categorical codes into continuous measurements. This single split provides point estimates rather than uncertainty intervals.

## 5. Run locally

### On this Mac: the environment and trained models are ready

Open Terminal and run these lines in order:

```bash
cd "/Users/gangamaheshwar/Downloads/ML Assignment 2"
source .venv/bin/activate
python -m streamlit run app.py
```

Open the **Local URL** printed in Terminal, usually `http://localhost:8501`. Keep the Terminal running while using the app. Press **Control + C** in that Terminal to stop it.

The page heading shows the selected dataset name. The page starts with that profile's held-out test dataset. Use **Select a model to inspect** to view each classifier. The comparison table shows all five models on the current CSV, and the selected model's confusion matrix appears in a large, full-width panel. If multiple profiles exist, use the **Trained dataset** dropdown to switch between them.

Upload the matching `test_data.csv` to demonstrate the upload feature. Streamlit automatically reruns when a CSV is uploaded; after validation, the app refreshes every result using the uploaded rows: the comparison table, comparison chart, selected model's six metrics, confusion matrix and classification report. A success message identifies the accepted upload. Removing the upload restores the profile's saved held-out data. No separate refresh button or training step is required for test uploads.

### What happens to the dataset?

1. **Training loads the full source.** For the bundled example, `python train.py` fetches all **11,055 UCI records** directly from the source URL into memory.
2. **One split is shared by every model.** The script reserves **8,841 training rows** and **2,214 held-out test rows**. It keeps identical feature vectors together to avoid overlap between the partitions.
3. **Training fits and saves five pipelines.** Each model learns only from the training rows. Its `.pkl` file contains the fitted classifier and any required preprocessing. The reserved rows are saved as `test_data.csv`.
4. **Opening the app evaluates saved models.** Running `python -m streamlit run app.py` loads those `.pkl` files and evaluates the **2,214 held-out rows**. It does not download the full dataset or train again.
5. **An upload changes the evaluation data.** Uploading a compatible labeled test CSV replaces the current evaluation rows. All displayed scores and charts are recalculated from that CSV, while the saved models remain fitted exactly as before. Removing the upload returns to the saved holdout.

These row counts describe the supplied UCI example. Another dataset profile or an uploaded CSV can have different counts; the **Rows being evaluated now** summary shows the current count. Open **How the dataset is used** for the same training and evaluation explanation inside the app.

### Compare models visually

Below the metrics comparison table, **Animated metric comparison** displays a horizontal bar chart for all five classifiers. Use **Choose a metric to compare** to select Accuracy, AUC, Precision, Recall, F1 or MCC. Model labels remain readable without truncation and scores use four decimal places. Every model tied for the best score is highlighted in teal with a **Best** label; unavailable scores are labeled **N/A** rather than being shown as zero.

The chart uses a fixed 0–1 axis for Accuracy, AUC, Precision, Recall and F1, and a fixed −1–1 axis for MCC. Negative MCC scores extend left from the central zero line. These fixed scales make comparisons consistent when the evaluation data changes. Uploading a CSV or choosing another metric updates the chart. Use **Replay chart animation** to view the bar animation again. The chart respects the system's reduced-motion preference and adds no new package dependency.

**Download test_data.csv** contains the selected profile's exact held-out CSV bytes. Each profile has its own sample; switching datasets updates the link. Downloading does not retrain a model or reset your evaluation. **Open CSV in a new tab** provides a separate preview. Only saved samples are made available; uploaded test files are not published.

Some embedded browser previews block file saving. If clicking the download does not save a CSV, open `http://localhost:8501` in your regular browser. You can also use the existing `test_data.csv` in the project folder (or the selected profile folder) directly. The automated tests verify download contents, filename and profile switching; the local HTTP endpoint also returns the exact sample bytes.

### Use your own classification dataset

1. Choose a public UCI or Kaggle classification dataset with at least 500 rows and 12 input features, excluding the target. Save one labeled CSV locally, with one target column and no dataframe index column. Record its source and license for your assignment.
2. Clean missing values and infinity before training. Numeric and categorical input columns are supported; the target must contain at least two discrete classes. Each class needs at least five rows and enough distinct feature groups for the grouped split (five groups per class at the default 20% test size).
3. With the virtual environment activated, train all five models on your full labeled CSV:

```bash
python train.py --data "/path/to/your_dataset.csv" --target "label" --name "My Dataset" --output model_profiles/my-dataset
```

Replace the CSV path and `label` with your actual file and target column. The command reserves a common held-out test partition, fits preprocessing on training rows only and saves the profile to `model_profiles/my-dataset/`. Numeric features are standardized for Logistic Regression, KNN and SVM. Categorical features are one-hot encoded, and the fitted encoder handles categories not seen during training.

4. Start the app with the three Mac commands above, or refresh it if it is already running. Select **My Dataset** from **Trained dataset**. The app discovers profiles in `model_profiles/` alongside the bundled `model/` profile. Upload only labeled test rows with the same schema; the generated `model_profiles/my-dataset/test_data.csv` is ready to use.

By default, binary Precision, Recall and F1 use the last sorted target label as the positive class. Set it explicitly when necessary, for example by adding `--positive-class "yes"`. For multiclass problems, Precision, Recall and F1 are support-weighted, and AUC uses support-weighted one-versus-rest probabilities. Accuracy and MCC use all classes. Class names and the confusion matrix come from the selected profile.

The optional `--test-size 0.2` accepts values from 0.1 to 0.5. The fraction is approximate because identical feature vectors stay together in a grouped split. All classes must appear in both partitions; very small classes or too few unique groups need a different dataset or test fraction. These are baseline models: large datasets or many categorical values can increase memory use and SVM training time.

The generated holdout must fit the app's 20,000-row and 5 MB limits. Training checks both limits before fitting or replacing models. If the holdout is too large, use a smaller test fraction or a smaller representative dataset.

To load a trusted local profile stored outside `model_profiles/`, start the app with its path:

```bash
ML_MODEL_DIR="/path/to/saved-profile" python -m streamlit run app.py
```

### First setup on another computer

Use Python 3.12. In a Terminal inside the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate` instead of the `source` command. The `.venv` directory is specific to its computer; create it again after copying the project to another machine. The saved `.pkl` models and `test_data.csv` are ready for evaluation. To reproduce their training, run `python train.py` with internet access before starting the app.

### Retrain the bundled example or verify the project

```bash
python train.py
python -m unittest discover -s tests -v
```

With no arguments, `train.py` downloads the UCI phishing dataset into memory and overwrites its `model/` artifacts and root `test_data.csv`. `dataset_source.py` defines `load_phishing_dataset()` and `SOURCE_METADATA`, including the official URL, citation, license, original feature schema and target mapping. Custom datasets require `--data` and `--target` and use their own profile folders. If you change a dataset or training settings, regenerate the assignment's dataset description, metric table and observations from that profile's saved results. The app itself never fits a classifier on uploaded data.

All five classifiers are saved as genuine Python pickle files (`.pkl`), including their fitted preprocessing. Pickle support comes from Python's standard library. The app loads only trusted local model profiles, not uploaded model files.

## 6. CSV upload format

Use a UTF-8 CSV with exactly the selected profile's input feature columns and target column. The target labels must be among those used to train that profile. Column order may differ; the app reorders features correctly. Each profile's `test_data.csv` provides the required template.

For the bundled example, this means the 30 feature columns listed in `model/metadata.json` and target `is_phishing`: 0 = legitimate and 1 = phishing. Its original integer category codes remain required. New profiles use their own feature types and target labels, so their CSVs do not need phishing feature names.

Missing or extra columns, duplicate headers, invalid fixed category codes, nonnumeric values in numeric columns, NaN/infinity, empty files and files larger than 5 MB or 20,000 rows are rejected. Do not include an extra dataframe index column. Categorical text columns are supported for profiles trained with those columns.

AUC is unavailable when an upload omits any fitted class. The remaining metrics and a confusion matrix covering all fitted classes are still displayed. Undefined Precision, Recall and F1 use zero; MCC is zero when its denominator is zero. Upload unseen test data for a meaningful evaluation; arbitrary uploads are not certified as independent of training.

## 7. Project structure

```text
ML Assignment 2/
  app.py                 Local Streamlit evaluation app
  styles.css             Responsive styling, hover effects and reduced-motion-aware animations
  train.py               Training CLI for the example or a custom classification CSV
  dataset_source.py      UCI URL retrieval, in-memory parsing and source attribution
  utils.py               CSV validation, artifact loading and six-metric evaluation
  downloads.py           Direct downloads of the selected profile's test sample
  comparison_chart.py    Animated six-metric comparison of the five classifiers
  requirements.txt       Pinned dependencies
  README.md              Dataset, results, observations and run instructions
  test_data.csv          Bundled example's labeled held-out test data
  static/                Locally served sample downloads, generated automatically
  model/
    logistic_regression.pkl
    decision_tree.pkl
    knn.pkl
    naive_bayes.pkl
    support_vector_machine.pkl
    metrics.csv          Five-model comparison
    metadata.json        Schema, split, parameters, versions and evaluation reports
  model_profiles/        Created when training custom datasets
    my-dataset/          Five .pkl pipelines, metadata.json, metrics.csv and test_data.csv
  tests/
    test_project.py      Checks for artifacts, split isolation, metrics and app behavior
  .streamlit/
    config.toml          Local app settings; developer deployment controls hidden
```

## 8. BITS Virtual Lab evidence and attribution

The assignment still requires execution on **BITS Virtual Lab** and one genuine screenshot of that execution. A local run does not fulfill that evidence requirement. Run the same training and app commands there, then capture the actual lab screen. The screenshot, final PDF and repository link remain pending.

Dataset attribution: Mohammad, R. & McCluskey, L. (2012). *Phishing Websites* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X. License: https://creativecommons.org/licenses/by/4.0/. Changes: conversion of the original ARFF data to CSV, remapping the target, and creating a grouped training/test partition. Original feature names and codes are retained.
