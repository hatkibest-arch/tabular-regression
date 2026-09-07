"""
Model inference script for tabular regression.
Loads serialized model bundle, applies required transformations,
and outputs predictions to CSV.
"""

import argparse
import logging
import os
import sys

import joblib
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions for test data.")
    parser.add_argument(
        "--test_path",
        type=str,
        default="hidden_test.csv",
        help="Path to hidden test CSV file (default: hidden_test.csv)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/model.joblib",
        help="Path to saved model bundle (default: models/model.joblib)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="predictions.csv",
        help="Output path for predictions CSV (default: predictions.csv)",
    )
    return parser.parse_args()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_out["6_sq"] = df_out["6"] ** 2
    return df_out


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}. Run train.py first.")
        sys.exit(1)

    if not os.path.exists(args.test_path):
        logger.error(f"Test dataset not found: {args.test_path}")
        sys.exit(1)

    logger.info(f"Loading model from {args.model_path}...")
    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    expected_features = bundle["feature_names"]

    logger.info(f"Loading test features from {args.test_path}...")
    test_df = pd.read_csv(args.test_path)

    test_eng = engineer_features(test_df)

    missing_cols = set(expected_features) - set(test_eng.columns)
    if missing_cols:
        logger.error(f"Missing required columns in test data: {missing_cols}")
        sys.exit(1)

    test_features = test_eng[expected_features]

    logger.info(f"Generating predictions for {len(test_features)} rows...")
    preds = model.predict(test_features)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    result_df = pd.DataFrame({"prediction": preds})
    result_df.to_csv(args.output_path, index=False)

    logger.info(f"Predictions successfully written to {args.output_path}")
    logger.info(f"Sample preview:\n{result_df.head()}")


if __name__ == "__main__":
    main()
