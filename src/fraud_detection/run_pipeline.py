from fraud_detection import run_training_pipeline


if __name__ == "__main__":
    result = run_training_pipeline(seed=42, algo_name="xgboost")

    print(result)