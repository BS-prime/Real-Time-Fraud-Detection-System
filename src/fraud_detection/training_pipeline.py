# ==================================================================================================================
# --- Import Libraries ---
# ==================================================================================================================
from datetime import datetime

from fraud_detection import (
    generate_transactions_data,
    feature_engineer,
    model_trainer,
    threshold_optimizer,
    model_evaluator,
)

# ==================================================================================================================
# --- Defining the actual function ---
# ==================================================================================================================

def run_training_pipeline(
    n_tx: int | None = 10_000,
    n_users: int | None = 500,
    seed: int | None = 42,
    algo_name: str | None = None,
) -> dict:
    """
    Docstring for run_training_pipeline

    :param n_tx: Number of trsactions for the dataset
    :type n_tx: int | None
    :param n_users: Number of users for the dataset
    :type n_users: int | None
    :param seed: Reproducible seed for dataset
    :type seed: int | None
    :param algo_name: Select the algo Example: "XGBoost" or "RandomForest".
    :type algo_name: str | None
    :return: Description
    :rtype: dict
    """

    """
    Orchestrates the full fraud detection training pipeline.

    Stages:
    1. Data generation
    2. Feature engineering
    3. Model training
    4. Threshold optimization
    5. Model evaluation

    Returns metadata about the pipeline run.
    """

    print("\n=== FRAUD DETECTION TRAINING PIPELINE STARTED ===\n")

    start_time = datetime.now()

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data
    # ------------------------------------------------------------------

    print("[1/5] Generating transaction data...")

    generate_transactions_data(n_tx=n_tx, n_users=n_users, seed=seed)

    simulated_file = f"simulated_transactions_seed_{seed}.csv"

    # ------------------------------------------------------------------
    # Step 2: Feature engineering
    # ------------------------------------------------------------------

    print("[2/5] Engineering features...")

    feature_engineer(simulated_file)

    features_file = f"fraud_features_seed_{seed}.csv"

    # ------------------------------------------------------------------
    # Step 3: Train model
    # ------------------------------------------------------------------

    print("[3/5] Training model...")

    X_test, y_test = model_trainer(csv_name=features_file, algo_name=algo_name)

    # ------------------------------------------------------------------
    # Step 4: Threshold optimization
    # ------------------------------------------------------------------

    print("[4/5] Optimizing decision threshold...")

    y_prob, y_pred = threshold_optimizer(X_test, y_test, algo_name=algo_name)

    # ------------------------------------------------------------------
    # Step 5: Model evaluation
    # ------------------------------------------------------------------

    print("[5/5] Evaluating model...")

    model_evaluator(algo_name, X_test, y_test, y_prob, y_pred)

    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds()

    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print(f"Duration: {duration:.2f} seconds\n")

    return {
        "status": "SUCCESS",
        "algo_name": algo_name,
        "seed": seed,
        "duration_seconds": duration,
        "completed_at": end_time.isoformat(),
    }
