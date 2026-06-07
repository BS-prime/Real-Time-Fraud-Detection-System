# Fraud Guard 2026: Real-Time Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Model](https://img.shields.io/badge/Model-XGBoost-orange)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED)
![Testing](https://img.shields.io/badge/Tests-pytest-brightgreen)

Fraud Guard 2026 is a deployable machine learning decision system for real-time transaction fraud detection. It combines behavioral profiling, geospatial anomaly detection, cost-sensitive thresholding, explainable decision rules, and a FastAPI inference layer into one end-to-end project.

The goal of this project is not just to train a model. The goal is to show how a fraud detection model can be turned into software that a risk team could actually use: reliable inputs, engineered signals, optimized business decisions, API responses, model artifacts, monitoring reports, and a containerized runtime.

## What Makes This Project Stand Out

Most ML portfolio projects stop at notebook accuracy. This project goes further by treating fraud detection as a production decision problem.

- It turns raw transaction behavior into model-ready fraud signals.
- It detects unrealistic travel behavior using geospatial velocity.
- It handles rare-event classification with cost-aware model training and thresholding.
- It serves predictions through a real API, not just a notebook cell.
- It returns business actions such as `ALLOW`, `CHALLENGE`, and `BLOCK`.
- It includes drift monitoring so the model can be evaluated after deployment.

For a recruiter or hiring manager, this project signals practical ML engineering ability: modeling, backend API design, MLOps thinking, deployment readiness, and business-aware decision logic.

## Snapshot

| Signal | Evidence in this project |
| --- | --- |
| End-to-end ML delivery | Synthetic data generation, feature engineering, training, thresholding, evaluation, API serving |
| Production mindset | Dockerfile, persisted artifacts, API validation, deterministic startup checks |
| Fraud domain understanding | Spend anomalies, impossible travel, transaction frequency, authentication signals |
| Business awareness | Threshold optimization based on false-positive and false-negative costs |
| Backend skills | FastAPI service with Pydantic request validation and structured JSON responses |
| Monitoring awareness | Evidently drift report generation for post-training model checks |
| Testing discipline | pytest checks for model probability output and risk behavior |

## Business Problem

Fraud teams need decisions that are fast, explainable, and aligned with financial risk. A default classifier threshold is rarely enough because the cost of missing fraud is very different from the cost of challenging a legitimate customer.

This system is built around that reality. It scores a transaction, interprets the risk, and returns an action that can be used by a payment workflow or fraud review queue.

## Core Capabilities

### Real-Time Fraud Scoring API

The FastAPI service accepts a transaction request and returns a complete fraud decision:

- Fraud probability from the trained model
- Risk band: `LOW`, `MEDIUM`, `HIGH`, or `VERY_HIGH`
- Recommended action: `ALLOW`, `CHALLENGE`, or `BLOCK`
- Plain-English decision reasons
- Fallback indicators when user history is missing
- Model version metadata

### Geospatial Impossible-Travel Detection

Fraud Guard calculates the distance between a user's current and previous transaction locations using the Haversine formula. It then derives an implied travel velocity in km/h.

This creates a strong fraud signal for account takeover patterns where the same user appears to transact from locations that are physically unrealistic within the available time window.

### Behavioral Feature Engineering

The model is trained on features that reflect how fraud analysts often think about suspicious behavior:

- `amount_ratio`: current transaction amount compared with the user's average spend
- `tx_count_24h`: recent transaction frequency
- `dist_from_last_tx_km`: distance from the previous transaction
- `travel_velocity_kmph`: implied travel speed between transactions
- Authentication method signals
- Merchant category signals

### Cost-Sensitive Decision Thresholding

The project does not rely on a generic `0.5` probability cutoff. It searches for an operating threshold that minimizes business cost using false-positive and false-negative penalties.

Current XGBoost threshold metadata is stored in:

```text
artifacts/model_thresholds/optimal_threshold_xgboost_seed_42.json
```

### Explainable Decision Layer

The API response is designed for humans as well as systems. Instead of returning only a probability, it explains why a transaction looks risky.

Example explanations include:

- Transaction amount is significantly higher than normal behavior.
- Transaction location implies unrealistic travel speed.
- Transaction frequency is unusually high within 24 hours.

This makes the project stronger as a portfolio piece because it shows the bridge between model output and operational decision-making.

## System Architecture

```text
Synthetic Transaction Data
        |
        v
Feature Engineering
        |
        v
Model Training
        |
        v
Business Threshold Optimization
        |
        v
Model Evaluation and Drift Monitoring
        |
        v
FastAPI Inference Service
        |
        v
Fraud Score + Risk Band + Recommended Action + Explanation
```

## Tech Stack

- Python 3.12
- FastAPI
- Pydantic v2
- Uvicorn
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- SHAP
- Evidently
- Matplotlib
- Docker
- pytest

## Project Structure

```text
.
|-- artifacts/
|   |-- models/                  # Trained model artifacts
|   `-- model_thresholds/         # Optimized decision thresholds
|-- configs/
|   `-- config.yaml               # Model and evaluation configuration
|-- data/
|   |-- simulated/                # Generated transaction data
|   `-- features/                 # Model-ready engineered features
|-- notebooks/                    # Exploration and development notebooks
|-- reports/                      # Drift and evaluation outputs
|-- src/
|   |-- api/
|   |   `-- main.py               # FastAPI inference app
|   `-- fraud_detection/
|       |-- data_ingestion.py
|       |-- feature_engineering.py
|       |-- model_training.py
|       |-- threshold_optimization.py
|       |-- model_evaluation.py
|       |-- drift_monitoring.py
|       `-- training_pipeline.py
|-- tests/
|   `-- test_fraud_model.py
|-- Dockerfile
|-- pyproject.toml
`-- run_pipeline.py
```

## Run with Docker

Build the image locally:

```bash
docker build -t fraud-detection-system .
```

Run the API:

```bash
docker run -p 8000:8000 fraud-detection-system
```

Open the interactive API docs:

```text
http://localhost:8000/docs
```

The container starts the service with:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Run Locally

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project:

```bash
pip install -e ".[dev]"
```

Start the API:

```bash
uvicorn src.api.main:app --reload
```

Health check:

```text
http://localhost:8000/
```

## Example Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "USER_123",
    "amount": 4200.0,
    "lat": 40.7128,
    "lon": -74.0060,
    "auth_method": "Password",
    "category": "travel",
    "time_delta_min": 5,
    "tx_count_24h": 24
  }'
```

Example response:

```json
{
  "model_version": "XGBoost_v:1.0",
  "fraud_probability": 0.9718,
  "risk_band": "VERY_HIGH",
  "recommended_action": "BLOCK",
  "decision_reasons": [
    "Transaction amount significantly higher than user's normal spending",
    "Transaction location implies unrealistic travel speed",
    "Unusually high number of transactions in past 24 hours"
  ],
  "fallbacks_used": []
}
```

## Training Pipeline

Run the full pipeline:

```bash
python run_pipeline.py
```

Pipeline stages:

1. Generate synthetic transactions.
2. Engineer fraud-focused features.
3. Train the configured model.
4. Optimize the decision threshold.
5. Evaluate the model and generate reports.

Model configuration lives in:

```text
configs/config.yaml
```

## Testing

```bash
pytest -q
```

The tests validate that:

- Model predictions stay within a valid probability range.
- Fraud risk does not drop unexpectedly when transaction amount increases sharply.

## Monitoring

The project includes an Evidently-based monitoring workflow for:

- Data quality checks
- Data drift detection
- Target drift detection
- Prediction score comparison between reference and current datasets

A generated drift report is included in:

```text
reports/drift_report_20260201_071922.html
```

## Production Improvements I Would Add Next

To move this closer to a production fraud platform, I would add:

- A real online feature store or database instead of mock user history.
- Structured logging for every prediction and decision.
- Request tracing for auditability.
- CI checks for tests, linting, Docker builds, and API smoke tests.
- Model and threshold version tracking in an experiment registry.
- Authentication and rate limiting for the API.
- Automated alerts when drift crosses a retraining threshold.

## Portfolio Takeaway

Fraud Guard 2026 shows the full path from ML experimentation to deployable decision software. It demonstrates the ability to build models, engineer fraud signals, package inference logic, expose a production-style API, monitor model behavior, and communicate results in a way that business and engineering teams can both understand.
