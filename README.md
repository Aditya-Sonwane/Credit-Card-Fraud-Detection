# Credit Card Fraud Detection

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions using SMOTE and XGBoost, built to handle severe class imbalance (~0.17% fraud rate) across 284,807 real-world transactions.

## Overview

Fraud detection is an extreme class-imbalance problem — fraudulent transactions are a tiny fraction of total volume, so naive models can score high on accuracy while catching almost no fraud. This pipeline is built to avoid that trap:

- Train/test split happens **before** any preprocessing, preventing data leakage
- `RobustScaler` handles the `Amount` feature, which contains extreme outliers
- `SMOTE` oversampling is applied **only** to the training set — the test set keeps its real-world imbalance
- `XGBClassifier` is tuned for imbalanced tabular data
- Evaluation reports ROC-AUC, PR-AUC (Average Precision), and a full classification report — not just accuracy
- A singleton `FraudInferenceService` serves predictions from memory for low-latency scoring

## Results

Evaluated on an untouched, raw-imbalance test set of 56,962 transactions (98 fraud cases, 0.17%):

| Metric | Score |
|---|---|
| ROC-AUC | 0.9815 |
| PR-AUC (Average Precision) | 0.8779 |
| Fraud Precision | 0.72 |
| Fraud Recall | 0.87 |
| Fraud F1-score | 0.79 |

**Confusion Matrix**

|  | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 56,831 | 33 |
| **Actual Fraud** | 13 | 85 |

Out of 98 real fraud cases in the test set, the model correctly caught 85 (87% recall) while producing only 33 false alarms out of 56,864 legitimate transactions.

**Inference latency**: ~7 ms per transaction after a one-time ~17 ms model load.

## Architecture

DataPreprocessingPipeline  → stratified split, RobustScaler, SMOTE
ModelTrainer                → XGBoost training + evaluation
FraudInferenceService       → singleton, in-memory, low-latency scoring

**DataPreprocessingPipeline** — Splits raw data into train/test first, fits `RobustScaler` only on the training partition, then applies `SMOTE` exclusively to the training set. The test set is transformed but never resampled, preserving the true class imbalance for honest evaluation.

**ModelTrainer** — Builds and trains an `XGBClassifier` with hyperparameters suited to imbalanced fraud data (`max_depth=6`, `learning_rate=0.05`, `scale_pos_weight`, `eval_metric='logloss'`). Evaluates on the untouched test set with ROC-AUC, PR-AUC, a classification report, and a confusion matrix.

**FraudInferenceService** — A singleton service that loads the trained model and preprocessing artifacts into memory exactly once, then serves `predict_live_transaction()` calls with no repeated disk I/O.

## Dataset

Uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle (ULB Machine Learning Group) — 284,807 transactions by European cardholders in September 2013, with 492 fraud cases (0.172%).

> The dataset is not included in this repository due to size and licensing. Download `creditcard.csv` from Kaggle and place it in the `data/` folder before retraining. See [`data/README.md`](data/README.md) for details.

## Pretrained Model Included

Trained model artifacts (`models/xgb_fraud_model.joblib`, `models/preprocessing_artifacts.joblib`) are included in this repository, so you can run inference immediately without needing the dataset or retraining. The dataset is only required if you want to retrain the model from scratch.

## Installation

```powershell
git clone https://github.com/Aditya-Sonwane/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### To retrain the model
1. Download `creditcard.csv` from Kaggle and place it in the `data/` folder.
2. Run the pipeline:

```powershell
python src/fraud_detection_pipeline.py
```

This will:
- Load and preprocess the data
- Train the XGBoost model
- Print evaluation metrics (ROC-AUC, PR-AUC, classification report)
- Save `xgb_fraud_model.joblib` and `preprocessing_artifacts.joblib` to `models/`
- Run a mock real-time prediction through `FraudInferenceService`

### To run inference only
Since pretrained artifacts are already included in `models/`, you can skip straight to using `FraudInferenceService` in your own script without needing the dataset:

```python
from fraud_detection_pipeline import FraudInferenceService

service = FraudInferenceService()
result = service.predict_live_transaction(transaction_features)
```

## Tech Stack

- Python 3.10+
- pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost
- joblib

## Project Structure

Credit-Card-Fraud-Detection/
│   .gitignore
│   README.md
│   requirements.txt
│
├───data
│       README.md
│
├───models
│       preprocessing_artifacts.joblib
│       xgb_fraud_model.joblib
│
└───src
fraud_detection_pipeline.py