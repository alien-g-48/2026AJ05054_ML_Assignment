# BITS: ML Assignment 2 · 2026AJ05054: UCI Phishing Websites

## 1. Problem statement

Classify websites as legitimate or phishing using their extracted website features. Implement Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes and Support Vector Machine on the same dataset. Compare their performance using Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC).

## 2. Dataset description

| Item | Details |
| --- | --- |
| Dataset | UCI Phishing Websites |
| Public repository | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites) |
| Dataset URL | [Official UCI dataset archive](https://archive.ics.uci.edu/static/public/327/phishing+websites.zip) |
| Instances | 11,055 |
| Input features | 30, excluding the target column |
| Feature description | Encoded indicators of URL structure, website links and forms, page behavior, domain information and reputation |
| Classification type | Binary classification |
| Target column | `is_phishing`: 0 = legitimate, 1 = phishing |
| Class distribution | 6,157 legitimate; 4,898 phishing |
| Missing values | None |
| Training and test split | 8,841 training records; 2,214 held-out test records |
| License | [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) |

The dataset satisfies the assignment's minimum requirements of 12 input features and 500 instances. Training loads the dataset directly from the UCI URL and maps the original target `Result=-1` to phishing (1), and `Result=1` to legitimate (0).

Identical feature vectors are kept in the same partition using a grouped, stratified split with random seed 2026. All five models use the same training and test rows. Logistic Regression, KNN and SVM use standardization fitted only on the training data; Decision Tree and Gaussian Naive Bayes use the original encoded features.

Dataset attribution: Mohammad, R. & McCluskey, L. (2012). *Phishing Websites* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C51W2X](https://doi.org/10.24432/C51W2X).

## 3. GitHub Repository Link

[2026AJ05054_ML_Assignment](https://github.com/alien-g-48/2026AJ05054_ML_Assignment)

Live Streamlit app: [https://2026aj05054mlassignment-phishing.streamlit.app/](https://2026aj05054mlassignment-phishing.streamlit.app/)

## 4. Models used

### Comparison of evaluation metrics

Results below are calculated on the same 2,214 held-out test records and rounded to four decimal places.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9273 | 0.9751 | 0.9476 | 0.8848 | 0.9151 | 0.8532 |
| Decision Tree | 0.9345 | 0.9718 | 0.9372 | 0.9134 | 0.9251 | 0.8672 |
| KNN | 0.9250 | 0.9723 | 0.9232 | 0.9062 | 0.9146 | 0.8479 |
| Naive Bayes | 0.6274 | 0.9684 | 0.5432 | 1.0000 | 0.7040 | 0.4240 |
| Support Vector Machine | 0.9508 | 0.9887 | 0.9580 | 0.9297 | 0.9436 | 0.9003 |

Phishing (1) is the positive class for Precision, Recall and F1. AUC uses continuous class-1 probabilities for Logistic Regression, Decision Tree, KNN and Naive Bayes, and decision scores for SVM.

### Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Provides a useful linear baseline with 92.73% accuracy and 0.9476 precision. Its recall of 0.8848 is lower than the other models, with 113 phishing records missed. |
| Decision Tree | Achieves 93.45% accuracy and 0.9251 F1, improving on Logistic Regression. It misses 85 phishing records and incorrectly flags 60 legitimate records. |
| KNN | Achieves 92.50% accuracy and 0.9146 F1. It is competitive with Logistic Regression, but makes 166 total errors on this test set. |
| Naive Bayes | Detects every phishing record, giving 1.0000 recall, but incorrectly flags 825 legitimate records. Its low precision of 0.5432 leads to the lowest accuracy and MCC. Its high AUC does not translate into strong predictions at the default classification threshold. |
| Support Vector Machine | Achieves the highest accuracy (95.08%), AUC (0.9887), Precision (0.9580), F1 (0.9436) and MCC (0.9003). It makes 109 errors: 40 false alarms and 69 missed phishing records. |
| Overall, Winner for your dataset? | **Support Vector Machine** provides the best balanced performance on this held-out test set. Naive Bayes has higher recall, but SVM combines strong phishing detection with substantially fewer false alarms. |

## 5. Overall observations

- **SVM provides the best overall performance on this test set.** It achieves 95.08% accuracy, 0.9436 F1 and 0.9003 MCC, with 40 false alarms and 69 missed phishing records. It leads five of the six evaluation metrics; Naive Bayes has the highest recall.
- **Decision Tree is the next strongest model by accuracy, F1 and MCC.** Logistic Regression and KNN also provide competitive baselines, with accuracies of 92.73% and 92.50%, respectively.
- **Phishing detection involves a trade-off between missed attacks and false alarms.** Naive Bayes detects all 981 phishing records, but also flags 825 legitimate records as phishing. Its perfect recall therefore comes with low precision (0.5432).
- **Model selection should consider several metrics together.** Naive Bayes achieves a high AUC of 0.9684 despite only 62.74% accuracy at its default classification threshold. Precision, recall, F1, MCC and the confusion matrix help explain behavior that accuracy or AUC alone can hide.
- **The conclusions apply to the evaluated dataset and split.** All models use the same grouped holdout, identical feature vectors stay within one partition, and scaling is fitted only on training data. These choices support a consistent comparison; further validation on additional unseen data would help assess how well the results generalize.
