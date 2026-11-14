# Model Methodology and Experimental Results

## 1. Introduction

This document describes the machine learning methodology used in the Automated Software Bug and Defect Predictor project.

The objective is to develop and evaluate classification models that estimate whether a software module is defective based on its software metrics.

The methodology includes dataset preprocessing, duplicate analysis, data splitting, model training, hyperparameter tuning, classification threshold selection, model evaluation and explainability.

## 2. Dataset Description

### 2.1 Dataset Source

The project uses the KC1 software defect dataset obtained from Mendeley Data.

Dataset source:

https://data.mendeley.com/datasets/923xvkk5mm/1

### 2.2 Dataset Characteristics

| Property | Description |
|---|---|
| Dataset | KC1 |
| Total records | 2,109 |
| Number of features | 21 |
| Target variable | class |
| Classification type | Binary classification |
| Non-defective records | 1,783 |
| Defective records | 326 |

The dataset contains software metrics describing the characteristics of individual software modules.

### 2.3 Input Features

The 21 input features are:

1. loc
2. v(g)
3. ev(g)
4. iv(g)
5. n
6. v
7. l
8. d
9. i
10. e
11. b
12. t
13. lOCode
14. lOComment
15. lOBlank
16. locCodeAndComment
17. uniq_Op
18. uniq_Opnd
19. total_Op
20. total_Opnd
21. branchCount

The target variable is `class`:

- 0 – Non-defective
- 1 – Defective

## 3. Data Preprocessing

The original dataset is provided in ARFF format.

The preprocessing pipeline performs the following operations:

1. Load the ARFF dataset using SciPy.
2. Convert the dataset into a Pandas DataFrame.
3. Decode categorical byte values.
4. Convert the target labels into binary numerical values.
5. Separate input features and target labels.
6. Convert feature columns into numeric values.
7. Replace infinite feature values with missing values.
8. Inspect missing values and duplicate records.

### 3.1 Duplicate Analysis

Duplicate feature combinations were investigated before model development.

The analysis identified repeated feature vectors, including some feature combinations associated with conflicting defect labels.

These records were not automatically deleted.

The reason is that different software modules can have identical metric values while having different defect labels. Removing them without investigation could discard relevant information.

This observation motivated the use of grouped data splitting.

## 4. Data Splitting Strategies

Two different splitting strategies were implemented to examine model performance.

### 4.1 Ordinary Stratified Split

The first experiment uses an 80:20 stratified train-test split.

Configuration:

- Training proportion: 80%
- Testing proportion: 20%
- Random state: 42
- Stratification: Enabled

Stratification maintains approximately similar class proportions in the training and testing datasets.

### 4.2 Grouped Stratified Split

The second experiment uses StratifiedGroupKFold with five folds.

Identical feature combinations are assigned the same group identifier using Pandas feature-row hashing.

The grouped split ensures that identical feature groups are not distributed across both training and testing datasets.

The selected split produced:

| Property | Training | Testing |
|---|---:|---:|
| Total records | 1,687 | 422 |
| Non-defective | 1,426 | 357 |
| Defective | 261 | 65 |

The overlap between training and testing feature groups was verified to be zero.

### 4.3 Why Grouped Evaluation Matters

When identical feature combinations appear in both training and testing datasets, a model may benefit from seeing repeated patterns during training.

This can make evaluation results overly optimistic.

Grouped splitting addresses this particular issue by keeping identical feature combinations together.

Therefore, grouped evaluation provides an additional and more cautious assessment of model performance for this dataset.

## 5. Machine Learning Algorithms

Four supervised machine learning algorithms were implemented and compared.

### 5.1 Logistic Regression

Logistic Regression is a linear classification algorithm that estimates the probability of a binary outcome.

It is included as a baseline model because it provides a relatively simple classification approach.

Standard scaling and median imputation are used in its preprocessing pipeline.

### 5.2 Decision Tree

A Decision Tree learns classification rules through a sequence of feature-based decisions.

It can capture nonlinear relationships and provides an interpretable tree-based structure.

Median imputation is included in its pipeline.

### 5.3 Random Forest

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

The grouped baseline experiment uses 300 trees and balanced subsampling of class weights.

It was selected as the deployed model after the tuning and evaluation experiments.

### 5.4 Gradient Boosting

Gradient Boosting builds an ensemble of decision trees sequentially.

Each new tree attempts to improve the performance of the existing ensemble.

Sample weighting is used in the baseline training implementation to account for class imbalance.

## 6. Model Evaluation Metrics

Since the dataset contains substantially more non-defective than defective records, accuracy alone is insufficient.

The following metrics are used:

### 6.1 Accuracy

Measures the proportion of all predictions that are correct.

### 6.2 Balanced Accuracy

Calculates the average recall across the two classes.

It is useful when class distributions are imbalanced.

### 6.3 Precision

Measures the proportion of predicted defective modules that are actually defective.

### 6.4 Recall

Measures the proportion of actual defective modules correctly identified by the model.

### 6.5 F1 Score

The harmonic mean of precision and recall.

It provides a balance between identifying defects and avoiding false defect predictions.

### 6.6 ROC-AUC

Measures the model's ability to rank defective modules above non-defective modules across classification thresholds.

### 6.7 PR-AUC

Average Precision summarizes the precision-recall relationship across thresholds.

It is particularly useful when evaluating models on imbalanced classification datasets.

## 7. Hyperparameter Tuning

RandomizedSearchCV was used to explore different hyperparameter configurations for the four classification algorithms.

The tuning configuration includes:

| Parameter | Configuration |
|---|---|
| Search method | RandomizedSearchCV |
| Configurations per model | 8 |
| Cross-validation | StratifiedGroupKFold |
| Number of folds | 3 |
| Selection metric | Average Precision |
| Random state | 42 |

Hyperparameter tuning is performed using training data.

The best estimator is selected based on cross-validation average precision.

### 7.1 Hyperparameter Search

The search spaces include model-specific parameters such as:

- Logistic Regression: regularization strength and solver.
- Decision Tree: maximum depth, minimum samples split and minimum samples per leaf.
- Random Forest: number of estimators, maximum depth and minimum samples per leaf.
- Gradient Boosting: number of estimators, learning rate, tree depth and subsampling.

The selected parameters are saved with the trained model artifact.

## 8. Classification Threshold Optimization

A model's predicted probability does not automatically determine the final classification. A threshold is required.

The default threshold of 0.50 may not provide the best balance between precision and recall.

This project evaluates thresholds between 0.05 and 0.95 using out-of-fold training probabilities.

For each threshold, precision, recall and F1 score are calculated.

The threshold with the highest F1 score is selected. If multiple thresholds have the same F1 score, higher precision is preferred.

The selected threshold for the deployed grouped Random Forest model is:

**0.47**

This threshold is stored with the trained model and used during application predictions.

## 9. Experimental Results

### 9.1 Grouped Tuned Model Comparison

The following results were obtained from the grouped evaluation experiment.

| Model | Threshold | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.47 | 0.8152 | 0.7398 | 0.4316 | 0.6308 | 0.5125 | 0.8207 | 0.4547 |
| Logistic Regression | 0.66 | 0.8412 | 0.7174 | 0.4861 | 0.5385 | 0.5109 | 0.8373 | 0.4808 |
| Gradient Boosting | 0.18 | 0.7204 | 0.7466 | 0.3290 | 0.7846 | 0.4636 | 0.8366 | 0.4682 |
| Decision Tree | 0.64 | 0.7796 | 0.6936 | 0.3627 | 0.5692 | 0.4431 | 0.7198 | 0.3256 |

### 9.2 Result Interpretation

The grouped Logistic Regression model achieved the highest accuracy, precision, ROC-AUC and PR-AUC among these four reported experiments.

The grouped Random Forest achieved the highest F1 score, with a value of 0.5125, and recall of 0.6308.

Gradient Boosting achieved the highest recall of 0.7846, but its precision was lower.

The Decision Tree produced the lowest ROC-AUC and PR-AUC among the four grouped tuned models.

These results demonstrate that model selection depends on the evaluation metric and the intended balance between detecting defects and avoiding false positives.

### 9.3 Final Model Selection

The tuned grouped Random Forest was selected for application deployment.

Its F1 score was slightly higher than that of grouped Logistic Regression, while providing a reasonable balance between precision and recall.

The selected model artifact is:

`models/tuned/kc1_grouped_random_forest_tuned.joblib`

It contains the trained preprocessing and classification pipeline, feature information, selected threshold and model metadata.

The application loads this artifact to generate predictions.

## 10. Model Explainability Using SHAP

SHAP (SHapley Additive exPlanations) is used to interpret the predictions generated by the Random Forest model.

The explanation process uses the trained tree-based classifier and a background sample from the grouped training dataset.

### 10.1 Global Feature Importance

Mean absolute SHAP values are calculated to identify features with greater average influence on model predictions.

The top ten features are:

| Feature | Mean Absolute SHAP |
|---|---:|
| lOCode | 0.072413 |
| total_Opnd | 0.037718 |
| v | 0.033380 |
| loc | 0.032309 |
| i | 0.031636 |
| total_Op | 0.020282 |
| n | 0.020201 |
| t | 0.019100 |
| e | 0.019058 |
| b | 0.018428 |

These values describe the average magnitude of feature contributions.

They do not indicate whether a feature increases or decreases the probability of defects, and they do not establish causal relationships.

### 10.2 Individual Prediction Explanation

For an individual prediction, SHAP calculates the contribution of each input feature.

The explanation includes:

- Baseline probability.
- Predicted defective-class probability.
- Positive contributions.
- Negative contributions.

The baseline and feature contributions are checked against the model's predicted probability.

This provides users with additional insight into why the model generated a particular prediction.

## 11. Implementation

The trained model is integrated into a Flask application.

The user enters software metrics through the web interface.

The backend:

1. Validates the input.
2. Loads the trained model pipeline.
3. Generates defect probability.
4. Applies the saved threshold.
5. Stores the prediction in SQLite.
6. Generates an individual SHAP explanation when requested.

The application also provides model comparisons, evaluation plots and prediction history.

## 12. Research Limitations

The following limitations should be considered when interpreting the results.

### 12.1 Dataset Generalization

The model is trained using the KC1 dataset. Its performance may differ when applied to other software projects, programming languages or development environments.

### 12.2 Class Imbalance

The dataset contains substantially more non-defective modules than defective modules. Although class weighting and multiple evaluation metrics are used, the model still produces false positives and false negatives.

### 12.3 Validation Strategy

The experiments use grouped train-test splitting and grouped cross-validation to reduce the effect of duplicate feature combinations.

However, the hyperparameter search, threshold selection and final model comparisons are not a fully nested validation procedure.

Comparing multiple models using the same test set can introduce model-selection bias. Therefore, the reported test results should be treated as experimental estimates rather than definitive evidence of generalization performance.

### 12.4 Dataset-Specific Evaluation

Grouped splitting prevents identical feature combinations from crossing the training and testing boundary. It does not establish generalization to completely independent software projects.

### 12.5 Explainability Limitations

SHAP describes the behaviour of the trained model. It does not prove that a particular software metric causes a defect.

## 13. Future Research Directions

Potential future investigations include:

1. Cross-project defect prediction using independent datasets.
2. Nested cross-validation for more robust model selection.
3. Probability calibration.
4. Feature selection and feature engineering.
5. Investigation of additional software metrics.
6. Evaluation on more recent software defect datasets.
7. Comparison of explainability methods.

## 14. Conclusion

This project demonstrates a machine learning workflow for software defect prediction using software metrics.

Four classification algorithms were implemented and compared. Grouped evaluation was incorporated to reduce the influence of duplicate feature combinations across training and testing data.

Hyperparameter tuning, threshold optimization and SHAP explainability were integrated into the workflow.

The selected grouped Random Forest model was deployed in a Flask application that provides interactive predictions, individual explanations, model analytics and prediction history.

The project provides a foundation for further investigation into interpretable software defect prediction and more robust cross-project evaluation.