import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Project-relative paths 
BASE_DIR = Path(__file__).resolve().parent.parent   
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV_PATH = DATA_DIR / "creditcard.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "xgb_fraud_model.joblib"
DEFAULT_ARTIFACTS_PATH = MODELS_DIR / "preprocessing_artifacts.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("FraudDetectionSystem")


class DataPreprocessingPipeline:
    """Leakage-free split -> scale -> SMOTE pipeline for fraud data."""

    def __init__(
        self,
        target_col: str = "Class",
        amount_col: str = "Amount",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.target_col = target_col
        self.amount_col = amount_col
        self.test_size = test_size
        self.random_state = random_state
        self.scaler: RobustScaler = RobustScaler()
        self._feature_columns: Optional[list] = None
        self._is_fitted = False

    def load_data(self, filepath: Path) -> pd.DataFrame:
        logger.info(f"Loading raw transaction data from: {filepath}")
        df = pd.read_csv(filepath)
        fraud_ratio = df[self.target_col].mean() * 100
        logger.info(
            f"Loaded {len(df):,} transactions | "
            f"Fraud ratio: {fraud_ratio:.4f}% "
            f"({df[self.target_col].sum():,} fraud cases)"
        )
        return df

    def split_scale_resample(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._feature_columns = [c for c in df.columns if c != self.target_col]
        X = df[self._feature_columns].copy()
        y = df[self.target_col].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        logger.info(
            f"Train set: {len(X_train):,} rows | fraud={y_train.sum():,} "
            f"({y_train.mean()*100:.4f}%)"
        )
        logger.info(
            f"Test set: {len(X_test):,} rows | fraud={y_test.sum():,} "
            f"({y_test.mean()*100:.4f}%)"
        )

        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[[self.amount_col]] = self.scaler.fit_transform(X_train[[self.amount_col]])
        X_test[[self.amount_col]] = self.scaler.transform(X_test[[self.amount_col]])
        self._is_fitted = True

        smote = SMOTE(random_state=self.random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logger.info(
            f"Post-SMOTE training set: {len(X_train_res):,} rows | "
            f"class balance = {pd.Series(y_train_res).value_counts().to_dict()}"
        )

        return (
            X_train_res.values,
            y_train_res.values,
            X_test.values,
            y_test.values,
        )

    def transform_single_transaction(self, transaction_df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Scaler has not been fitted.")
        transaction_df = transaction_df.copy()
        transaction_df[[self.amount_col]] = self.scaler.transform(
            transaction_df[[self.amount_col]]
        )
        return transaction_df[self._feature_columns].values

    def save_artifacts(self, path: Path = DEFAULT_ARTIFACTS_PATH) -> None:
        joblib.dump(
            {"scaler": self.scaler, "feature_columns": self._feature_columns},
            path,
        )
        logger.info(f"Preprocessing artifacts saved to: {path}")


@dataclass
class EvaluationReport:
    roc_auc: float
    pr_auc: float
    classification_report_str: str
    confusion_matrix: np.ndarray


class ModelTrainer:
    """XGBoost training and evaluation for imbalanced fraud detection."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: Optional[XGBClassifier] = None

    def build_model(self, scale_pos_weight: float = 1.0) -> XGBClassifier:
        self.model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1.0,
            min_child_weight=3,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=-1,
        )
        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> XGBClassifier:
        if self.model is None:
            self.build_model()

        logger.info("Training XGBClassifier on SMOTE-balanced training data...")
        start = time.time()

        eval_set = [(X_val, y_val)] if X_val is not None else None
        fit_kwargs: Dict[str, Any] = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False

        self.model.fit(X_train, y_train, **fit_kwargs)
        elapsed = time.time() - start
        logger.info(f"Training complete in {elapsed:.2f}s.")
        return self.model

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> EvaluationReport:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        report_str = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
        cm = confusion_matrix(y_test, y_pred)

        logger.info("=" * 65)
        logger.info("EVALUATION ON UNTOUCHED, RAW-IMBALANCE TEST SET")
        logger.info("=" * 65)
        logger.info(f"ROC-AUC Score           : {roc_auc:.4f}")
        logger.info(f"PR-AUC (Avg. Precision) : {pr_auc:.4f}")
        logger.info("Classification Report:\n" + report_str)
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info("=" * 65)

        return EvaluationReport(
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            classification_report_str=report_str,
            confusion_matrix=cm,
        )

    def save_model(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        if self.model is None:
            raise RuntimeError("No trained model to save.")
        joblib.dump(self.model, path)
        logger.info(f"Trained model saved to: {path}")


class FraudInferenceService:
    """Singleton, in-memory, low-latency fraud scoring service."""

    _instance: Optional["FraudInferenceService"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        artifacts_path: Path = DEFAULT_ARTIFACTS_PATH,
        decision_threshold: float = 0.5,
        model_obj: Optional[XGBClassifier] = None,
        scaler_obj: Optional[RobustScaler] = None,
        feature_columns: Optional[list] = None,
    ):
        if FraudInferenceService._initialized:
            return

        logger.info("Initializing FraudInferenceService...")
        start = time.time()

        if model_obj is not None:
            self.model = model_obj
        else:
            self.model = joblib.load(model_path)

        if scaler_obj is not None and feature_columns is not None:
            self.scaler = scaler_obj
            self.feature_columns = feature_columns
        else:
            artifacts = joblib.load(artifacts_path)
            self.scaler = artifacts["scaler"]
            self.feature_columns = artifacts["feature_columns"]

        self.decision_threshold = decision_threshold
        self._amount_col = "Amount"

        elapsed = time.time() - start
        logger.info(f"Model & artifacts loaded into memory in {elapsed*1000:.2f} ms.")

        FraudInferenceService._initialized = True

    def predict_live_transaction(
        self, transaction_features: Dict[str, float]
    ) -> Dict[str, Any]:
        start = time.perf_counter()

        row = pd.DataFrame([transaction_features], columns=self.feature_columns)
        row[[self._amount_col]] = self.scaler.transform(row[[self._amount_col]])

        proba = float(self.model.predict_proba(row.values)[:, 1][0])
        is_fraud = proba >= self.decision_threshold

        if proba >= 0.9:
            risk_tier = "CRITICAL"
        elif proba >= 0.5:
            risk_tier = "HIGH"
        elif proba >= 0.1:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "is_fraud": bool(is_fraud),
            "fraud_probability": round(proba, 6),
            "risk_tier": risk_tier,
            "latency_ms": round(latency_ms, 3),
        }

    @classmethod
    def reset_singleton(cls) -> None:
        cls._instance = None
        cls._initialized = False


def main(csv_path: Path = DEFAULT_CSV_PATH) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            f"Download creditcard.csv from Kaggle and place it in the 'data/' folder."
        )

    preprocessor = DataPreprocessingPipeline(
        target_col="Class", amount_col="Amount", test_size=0.2, random_state=42
    )
    raw_df = preprocessor.load_data(csv_path)
    X_train_res, y_train_res, X_test, y_test = preprocessor.split_scale_resample(raw_df)
    preprocessor.save_artifacts(DEFAULT_ARTIFACTS_PATH)

    trainer = ModelTrainer(random_state=42)
    trainer.build_model(scale_pos_weight=1.0)
    trainer.train(X_train_res, y_train_res)

    report = trainer.evaluate(X_test, y_test)
    trainer.save_model(DEFAULT_MODEL_PATH)

    print("\n" + "=" * 65)
    print(f"FINAL RESULTS  |  ROC-AUC: {report.roc_auc:.4f}  |  PR-AUC: {report.pr_auc:.4f}")
    print("=" * 65 + "\n")

    inference_service = FraudInferenceService(
        model_path=DEFAULT_MODEL_PATH,
        artifacts_path=DEFAULT_ARTIFACTS_PATH,
        decision_threshold=0.5,
    )

    same_instance = FraudInferenceService()
    assert inference_service is same_instance, "Singleton pattern violated!"
    logger.info("Singleton verified: repeated instantiation reuses the in-memory instance.")

    sample_transaction = raw_df.drop(columns=["Class"]).iloc[0].to_dict()
    result = inference_service.predict_live_transaction(sample_transaction)

    print("Mock real-time prediction on a sample transaction:")
    print(f"  -> Is Fraud?          : {result['is_fraud']}")
    print(f"  -> Fraud Probability  : {result['fraud_probability']}")
    print(f"  -> Risk Tier          : {result['risk_tier']}")
    print(f"  -> Inference Latency  : {result['latency_ms']} ms")


if __name__ == "__main__":
    main()