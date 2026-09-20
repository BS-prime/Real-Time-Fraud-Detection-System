"""
End-to-end orchestration for the fraud detection training workflow.
"""

import logging
from datetime import UTC, datetime

from fraud_detection.data_ingestion import simulate_transactions_data
from fraud_detection.feature_engineering import batch_feature_engineer
from fraud_detection.model_evaluation import run_model_evaluation
from fraud_detection.model_training import ModelTrainer
from fraud_detection.threshold_optimization import run_threshold_optimization

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Run the same stage sequence used by DVC from a Python entrypoint.
    """

    def run(
        self,
        n_tx: int = 1_000_000,
        n_users: int = 5_000,
        seed: int = 42,
        algo_name: str | None = None,
    ) -> dict[str, object]:
        logger.info("Fraud detection training pipeline started")
        start_time = datetime.now(tz=UTC)

        simulated_path = simulate_transactions_data(
            n_tx=n_tx,
            n_users=n_users,
            seed=seed,
        )
        train_path, _ = batch_feature_engineer(simulated_path)

        trainer = ModelTrainer(train_file_path=train_path)
        if algo_name:
            trained_models = trainer.train_all(
                stop_on_error=True,
                feature_path=train_path,
                algo_names=[algo_name],
            )
        else:
            trained_models = trainer.train_all(stop_on_error=False, feature_path=train_path)

        threshold_summary = run_threshold_optimization()
        evaluation_summary = run_model_evaluation()

        end_time = datetime.now(tz=UTC)
        duration = (end_time - start_time).total_seconds()
        logger.info("Pipeline completed successfully in %.2f seconds", duration)

        return {
            "status": "SUCCESS",
            "seed": seed,
            "trained_models": trained_models,
            "threshold_summary": threshold_summary,
            "evaluation_summary": evaluation_summary,
            "duration_seconds": duration,
            "completed_at": end_time.isoformat(),
        }


def run_training_pipeline(
    n_tx: int = 10_000,
    n_users: int = 500,
    seed: int = 42,
    algo_name: str | None = None,
) -> dict[str, object]:
    """
    Execute the Python training pipeline.
    """

    return TrainingPipeline().run(
        n_tx=n_tx,
        n_users=n_users,
        seed=seed,
        algo_name=algo_name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training_pipeline()
