"""
Model training script for tabular regression.
Applies feature engineering discovered during EDA, evaluates via 5-Fold Cross-Validation,
and serializes the trained model and metadata for inference.
"""

import argparse
import json
import logging
import os
import sys
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train regression model pipeline.")
    parser.add_argument(
        "--train_path",
        type=str,
        default="train.csv",
        help="Path to training dataset CSV (default: train.csv)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save trained model artifacts (default: models)",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="target",
        help="Name of target column (default: target)",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augments dataset with quadratic interaction for feature '6'
    identified during exploratory data analysis.
    """
    df_out = df.copy()
    df_out["6_sq"] = df_out["6"] ** 2
    return df_out


def load_data(file_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(file_path):
        logger.error(f"Training file not found: {file_path}")
        sys.exit(1)

    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' missing from dataset.")
        sys.exit(1)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    random_state: int,
) -> Tuple[Ridge, float]:
    X_engineered = engineer_features(X)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_predictions = np.zeros(len(y))

    logger.info(f"Running {n_splits}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_engineered, y), start=1):
        X_train, y_train = X_engineered.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X_engineered.iloc[val_idx], y.iloc[val_idx]

        model = Ridge(alpha=1e-5)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        oof_predictions[val_idx] = val_pred
        fold_rmse = root_mean_squared_error(y_val, val_pred)
        logger.info(f"Fold {fold}/{n_splits} - RMSE: {fold_rmse:.2e}")

    overall_rmse = root_mean_squared_error(y, oof_predictions)
    logger.info(f"Overall Out-of-Fold RMSE: {overall_rmse:.2e}")

    logger.info("Fitting final production estimator on entire training dataset...")
    final_model = Ridge(alpha=1e-5)
    final_model.fit(X_engineered, y)

    return final_model, overall_rmse


def save_artifacts(model: Ridge, feature_names: list, cv_rmse: float, model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    bundle = {
        "model": model,
        "feature_names": feature_names,
        "cv_rmse": cv_rmse,
    }
    joblib.dump(bundle, model_path)
    logger.info(f"Model bundle saved to {model_path}")

    metadata = {
        "model_type": "Ridge_with_Quadratic_Feature",
        "n_features_in": len(feature_names),
        "cross_val_rmse": float(cv_rmse),
    }
    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved to {meta_path}")


def main() -> None:
    args = parse_args()
    X, y = load_data(args.train_path, args.target_col)
    model, cv_rmse = train_and_evaluate(X, y, args.n_splits, args.random_state)
    feature_names = list(engineer_features(X).columns)
    save_artifacts(model, feature_names, cv_rmse, args.model_dir)
    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
