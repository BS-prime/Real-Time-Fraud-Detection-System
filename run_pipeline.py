"""Run the default fraud detection training pipeline."""

from fraud_detection.training_pipeline import run_training_pipeline


if __name__ == "__main__":
    run_training_pipeline(algo_name="xgboost")
