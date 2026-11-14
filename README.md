# Automated Software Bug and Defect Predictor

An interpretable machine learning application that predicts whether a software module is likely to contain defects using software metrics. It combines machine learning classification, model evaluation, SHAP explainability and a Flask-based web interface.

## 1. Project Overview

Software defects can increase maintenance costs, affect reliability and delay software releases. Identifying potentially defective modules early can help developers prioritize software testing and quality assurance.

This project uses software metrics to estimate the likelihood that a software module belongs to the defective class.

The application provides:
- Software defect prediction using a trained Random Forest classifier.
- Defect probability and classification threshold.
- Individual prediction explanations using SHAP.
- Comparison of four machine learning algorithms.
- Model evaluation visualizations.
- Prediction history using SQLite.
- A web-based interface built with Flask.

**Project type:** Machine Learning / Software Engineering  
**Dataset:** NASA KC1  
**Application:** Python Flask  
**Primary model:** Tuned Random Forest with grouped evaluation

## 2. Objectives

1. Prepare and analyse software defect datasets.
2. Compare multiple machine learning classification algorithms.
3. Evaluate models using both ordinary and grouped data splits.
4. Tune model hyperparameters using cross-validation.
5. Select a classification threshold using out-of-fold predictions.
6. Explain individual predictions using SHAP.
7. Develop a web application for interactive defect prediction.

## 3. Dataset

The project uses the KC1 software defect dataset.

**Dataset source:** [Mendeley Data – Software Defect Prediction Dataset](https://data.mendeley.com/datasets/923xvkk5mm/1)

The KC1 dataset contains software module metrics and a binary defect label.

- Total records: 2,109
- Software metric features: 21
- Target column: `class`
- Non-defective class: 0 (`false`)
- Defective class: 1 (`true`)

The features describe software size, complexity, operators, operands and related code characteristics.

Examples include:
- `loc` – Lines of code
- `v(g)` – Cyclomatic complexity
- `ev(g)` – Essential complexity
- `total_Op` – Total operators
- `total_Opnd` – Total operands
- `branchCount` – Branch count

## 4. Methodology

### 4.1 Data preprocessing

The original ARFF dataset is loaded and converted into a structured tabular format.

The preprocessing procedure includes:
- Decoding categorical values.
- Converting defect labels into binary values.
- Separating features and target.
- Converting features into numeric values.
- Replacing infinite values with missing values.
- Checking missing values and duplicate records.

Duplicate feature rows are not automatically removed because different software modules can have identical metric values.

### 4.2 Data splitting

Two evaluation strategies are implemented.

**Ordinary stratified split**

- 80% training data.
- 20% testing data.
- Stratification preserves class proportions.

**Grouped stratified split**

Identical feature combinations are assigned the same group identifier. StratifiedGroupKFold is used to create a split that prevents identical feature groups from appearing in both training and testing data.

The grouped split is particularly important because duplicate feature vectors can otherwise make evaluation overly optimistic.

### 4.3 Machine learning algorithms

Four classification algorithms are compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

The training pipeline includes median imputation. Standard scaling is applied to Logistic Regression.

Class weighting or sample weighting is used where configured to account for class imbalance.

### 4.4 Hyperparameter tuning

RandomizedSearchCV is used to explore model hyperparameters.

The tuning configuration includes:
- Three-fold StratifiedGroupKFold cross-validation.
- Eight randomly sampled configurations per model.
- Average Precision (PR-AUC) as the model selection metric.
- Fixed random states for reproducibility.

### 4.5 Classification threshold selection

The default classification threshold of 0.50 is not assumed to be optimal.

Out-of-fold training probabilities are used to evaluate thresholds from 0.05 to 0.95. The threshold with the highest F1 score is selected. Higher precision is used as the tie-breaker.

The selected threshold for the deployed grouped Random Forest model is **0.47**.

### 4.6 Model explainability

SHAP (SHapley Additive exPlanations) is used to explain individual predictions.

The explanation identifies which software metrics contribute positively or negatively to the model's predicted defective-class probability.

Global SHAP feature importance is also used to understand which features have greater average influence on model predictions.

SHAP explanations describe model behaviour; they do not establish causal relationships.

## 5. Model Evaluation

The models are evaluated using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall correct predictions |
| Balanced Accuracy | Average performance across both classes |
| Precision | Proportion of predicted defects that are actual defects |
| Recall | Proportion of actual defects identified |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Ranking performance across classification thresholds |
| PR-AUC | Precision-recall performance, useful for imbalanced data |

### Selected grouped Random Forest results

| Metric | Result |
|---|---:|
| Classification threshold | 0.47 |
| Accuracy | 0.8152 |
| Balanced Accuracy | 0.7398 |
| Precision | 0.4316 |
| Recall | 0.6308 |
| F1 Score | 0.5125 |
| ROC-AUC | 0.8207 |
| PR-AUC | 0.4547 |

These are the observed results for the selected grouped Random Forest experiment on its held-out test split. They should not be interpreted as a guarantee of performance on other datasets or software projects.

## 6. Application Features

### Defect Prediction
Users enter software metrics and receive a predicted class, defect probability and classification threshold.

### Individual SHAP Explanation
Displays the baseline probability and feature-level contributions for an individual prediction.

### Prediction History
Stores previous predictions and their associated metrics in a local SQLite database.

### Model Insights
Displays global SHAP feature importance.

### Model Performance
Displays model comparison results and evaluation figures, including:
- Confusion matrix
- ROC curve
- Precision-recall curve

## 7. Technology Stack

| Area | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy, SciPy |
| Machine learning | Scikit-learn |
| Explainable AI | SHAP |
| Model storage | Joblib |
| Backend | Flask |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Visualization | Matplotlib |

## 8. Project Structure

```text
Software-Defect-Predictor/
│
├── app/
│   ├── app.py
│   ├── predictor.py
│   ├── explainer.py
│   ├── analytics.py
│   ├── database.py
│   ├── templates/
│   └── static/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── tuned/
│
├── src/
│   ├── data/
│   └── models/
│
├── tests/
│   ├── run_all_tests.py
│   ├── smoke_test_api.py
│   └── test_explain_analytics.py
│
├── reports/
│   └── generated/
│
├── docs/
├── notebooks/
├── database/
├── requirements.txt
├── .gitignore
└── README.md
```

## 9. Installation and Execution

### Requirements

- Python 3.10 or later, subject to dependency compatibility.
- pip
- Git (for cloning the repository)

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Software-Defect-Predictor
```

### Step 2: Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 4: Start the application

```powershell
python -m app.app
```

Open the following address in a browser:

```text
http://127.0.0.1:5000
```

### Step 5: Run tests

```powershell
python tests/run_all_tests.py
```

The application requires the committed trained model, grouped training CSV and generated analytics files. These should be present in the repository at the documented paths.

## 10. Limitations

- The model is trained using the KC1 dataset and may not generalize to other software projects.
- Software metrics alone cannot capture every factor that contributes to defects.
- The grouped test set is relatively small.
- Threshold selection and model selection can introduce optimism into reported performance.
- SHAP values explain model predictions, not the underlying causes of software defects.
- The application is an experimental decision-support prototype, not a replacement for software testing or code review.

## 11. Future Work

- Validate the approach on additional software defect datasets.
- Evaluate generalization across projects and datasets.
- Investigate more robust validation strategies, including nested cross-validation.
- Explore additional software metrics and feature selection techniques.
- Study calibration of predicted probabilities.
- Improve the usability of the prediction interface.

## 12. Disclaimer

This project is developed for academic and educational purposes. Predictions should be treated as estimates to support software quality analysis, not as definitive evidence that a software module contains defects.
