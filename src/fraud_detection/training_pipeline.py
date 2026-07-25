"""End-to-end training pipeline."""

from datetime import datetime

from fraud_detection.data_ingestion import generate_transactions_data
from fraud_detection.feature_engineering import feature_engineer
from fraud_detection.model_evaluation import model_evaluator
from fraud_detection.model_training import model_trainer
from fraud_detection.threshold_optimization import threshold_optimizer


def run_training_pipeline(
    n_tx: int = 10_000,
    n_users: int = 500,
    seed: int = 42,
    algo_name: str = "xgboost",
) -> dict:
    """Run the project pipeline from data generation to evaluation."""
    print("\n=== FRAUD DETECTION TRAINING PIPELINE STARTED ===\n")
    start_time = datetime.now()

    print("[1/5] Generating transaction data...")
    generate_transactions_data(n_tx=n_tx, n_users=n_users, seed=seed)
    simulated_file = f"simulated_transactions_seed_{seed}.csv"

    print("[2/5] Engineering features...")
    feature_engineer(simulated_file)
    features_file = f"fraud_features_seed_{seed}.csv"

    print("[3/5] Training model...")
    X_test, y_test = model_trainer(csv_name=features_file, algo_name=algo_name)
    model_name = f"{algo_name}_seed_{seed}.json"

    print("[4/5] Optimizing decision threshold...")
    y_prob, y_pred = threshold_optimizer(X_test, y_test, model_name=model_name)

    print("[5/5] Evaluating model...")
    model_evaluator(model_name, X_test, y_test, y_prob, y_pred)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print(f"Duration: {duration:.2f} seconds\n")

    return {
        "status": "SUCCESS",
        "model_name": model_name,
        "seed": seed,
        "duration_seconds": duration,
        "completed_at": end_time.isoformat(),
    }
