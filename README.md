# 🛡️ Fraud Guard 2026

### Real-Time Fraud Detection Pipeline with Geospatial Intelligence and Behavioral Profiling

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![API](https://img.shields.io/badge/API-FastAPI-009688)
![Model](https://img.shields.io/badge/model-XGBoost-orange)
![Container](https://img.shields.io/badge/container-Docker-2496ED)

------------------------------------------------------------------------

## Overview

Fraud Guard 2026 is a production-style, real-time fraud detection system
designed to detect anomalous financial transactions using geospatial
physics, behavioral baselines, and cost-sensitive machine learning.

The system exposes a low-latency FastAPI inference service (\<15ms)
capable of transforming raw transactions into explainable, actionable
fraud decisions.

This project implements an end-to-end machine learning pipeline 
consisting of data generation, feature engineering, model training, 
cost-aware threshold optimization, and deployment via a real-time FastAPI 
inference service.

This project focuses on deployment realism, decision clarity, and model
transparency---not just model accuracy.

------------------------------------------------------------------------

## Run Instantly Using Docker (Recommended)

Pull the prebuilt image from Docker Hub:

``` bash
docker pull bsprime777/fraud-guard-v:1.1.0
```

Run the container:

``` bash
docker run -p 8000:8000 bsprime777/fraud-guard-v:1.1.0
```

Access API documentation:

    http://localhost:8000/docs

Health check:

    http://localhost:8000/

This runs the exact production inference environment with no setup
required.

------------------------------------------------------------------------
## Running the pipeline (Jupyter Notebook)

``` bash
from fraud_detection import (
    generate_transactions_data,
    feature_engineer,
    model_trainer,
    threshold_optimizer,
    model_evaluator,
)

seed = 42
model_name = "xgboost_seed_42.json"

generate_transactions_data(seed=seed)
feature_engineer(f"simulated_transactions_seed_{seed}.csv")

X_test, y_test = model_trainer(f"fraud_features_seed_{seed}.csv")

y_prob, y_pred = threshold_optimizer(
    X_test,
    y_test,
    model_name=model_name
)

model_evaluator(
    model_name,
    X_test,
    y_test,
    y_prob,
    y_pred
)
```
------------------------------------------------------------------------

## Core Capabilities

### 📊 Core Feature: Geospatial Velocity
To identify fraud, the system calculates the physical distance between consecutive transactions using:

$$\text{dist} = 2r \arcsin\left(\sqrt{\sin^2\left(\frac{\phi_2-\phi_1}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\lambda_2-\lambda_1}{2}\right)}\right)$$

If the velocity ($\text{dist} / \text{time}$) exceeds a commercial flight threshold (e.g., 900 km/h), the risk score is automatically elevated.

------------------------------------------------------------------------

### Behavioral Anomaly Detection

Evaluates transactions relative to behavioral baselines:

-   amount_ratio
-   tx_count_24h
-   travel_velocity_kmph
-   authentication method
-   category anomalies

------------------------------------------------------------------------

### Cost-Sensitive Machine Learning

Optimized for minimizing financial loss using:

-   XGBoost with imbalance weighting
-   Explicit business-cost threshold optimization

------------------------------------------------------------------------

### High-Performance Inference

Uses native XGBoost Booster interface:

-   Direct C++ execution
-   Minimal latency
-   JSON-based model loading
-   No sklearn runtime overhead

------------------------------------------------------------------------

### Decision Layer

Separates prediction from business action.

Example response:

``` json
{
  "model_version": "fraud_xgb_v1.0",
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

------------------------------------------------------------------------

### Explainability (Offline Validation)

SHAP was used during evaluation to validate learned feature importance.

Confirmed top fraud signals:

-   amount_ratio
-   travel_velocity_kmph
-   tx_count_24h

Plots available in:

    reports/shap/

------------------------------------------------------------------------

### Drift Monitoring

Monitors:

-   Feature distribution drift
-   Data quality degradation
-   Concept drift

Using Evidently AI.

------------------------------------------------------------------------

## System Architecture

    Transaction Input
          │
          ▼
    Feature Engineering
          │
          ▼
    XGBoost Booster (C++ inference)
          │
          ▼
    Decision Layer
          │
          ├── Risk Band
          ├── Recommended Action
          ├── Decision Reasons
          └── Fallback Transparency
          │
          ▼
    FastAPI Response (<15 ms)

------------------------------------------------------------------------

## Tech Stack

Modeling: - XGBoost - Scikit-Learn

API: - FastAPI - Pydantic v2 - Uvicorn

Explainability (offline): - SHAP

Monitoring: - Evidently AI

Containerization: - Docker

------------------------------------------------------------------------

## Local Development

Run locally:

``` bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

------------------------------------------------------------------------

## Testing

Run:

``` bash
pytest -q
```

------------------------------------------------------------------------

## Project Focus

Demonstrates:

-   Real-time ML inference deployment
-   Cost-aware fraud decision systems
-   Explainable ML validation
-   Production-grade API design
-   Containerized deployment

------------------------------------------------------------------------