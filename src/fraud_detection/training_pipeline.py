"""End-to-end training pipeline."""

import logging
from datetime import datetime
from pathlib import Path

from fraud_detection.data_ingestion import generate_transactions_data
from fraud_detection.feature_engineering import feature_engineer
from fraud_detection.model_evaluation import model_evaluator
from fraud_detection.model_training import model_trainer
from fraud_detection.threshold_optimization import threshold_optimizer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Orchestrate the fraud detection training workflow.
    """

    def __init__(self) -> None:
        self.steps = [
            "Generating transaction data",
            "Engineering features",
            "Training model",
            "Optimizing decision threshold",
            "Evaluating model",
        ]

    def run(
        self,
        n_tx: int = 10_000,
        n_users: int = 500,
        seed: int = 42,
        algo_name: str = "xgboost",
    ) -> dict[str, object]:
        logger.info("FRAUD DETECTION TRAINING PIPELINE STARTED")
        start_time = datetime.now()

        logger.info("[1/5] %s", self.steps[0])
        generate_transactions_data(n_tx=n_tx, n_users=n_users, seed=seed)
        simulated_file = f"simulated_transactions_seed_{seed}.csv"

        logger.info("[2/5] %s", self.steps[1])
        feature_engineer(simulated_file)
        features_file = f"fraud_features_seed_{seed}.csv"

        logger.info("[3/5] %s", self.steps[2])
        X_test, y_test = model_trainer(csv_name=features_file, algo_name=algo_name)
        model_name = f"{algo_name}_seed_{seed}.json"

        logger.info("[4/5] %s", self.steps[3])
        y_prob, y_pred = threshold_optimizer(X_test, y_test, model_name=model_name)

        logger.info("[5/5] %s", self.steps[4])
        model_evaluator(model_name, X_test, y_test, y_prob, y_pred)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("PIPELINE COMPLETED SUCCESSFULLY in %.2f seconds", duration)

        return {
            "status": "SUCCESS",
            "model_name": model_name,
            "seed": seed,
            "duration_seconds": duration,
            "completed_at": end_time.isoformat(),
        }


def run_training_pipeline(
    n_tx: int = 10_000,
    n_users: int = 500,
    seed: int = 42,
    algo_name: str = "xgboost",
) -> dict[str, object]:

    return TrainingPipeline().run(
        n_tx=n_tx,
        n_users=n_users,
        seed=seed,
        algo_name=algo_name,
    )
