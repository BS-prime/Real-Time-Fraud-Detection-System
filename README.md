# 🛡️ Fraud Guard 2026

## Real-Time Fraud Detection Pipeline with Geospatial Intelligence & Behavioral Profiling

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![API](https://img.shields.io/badge/API-FastAPI-009688)
![Model](https://img.shields.io/badge/model-XGBoost-orange)
![Container](https://img.shields.io/badge/container-Docker-2496ED)

------------------------------------------------------------------------

## 📌 Overview

Fraud Guard 2026 is a production-oriented real-time fraud detection
system designed to identify anomalous financial transactions using
geospatial physics, behavioral profiling, and cost-sensitive machine
learning.

This project demonstrates the full lifecycle of a deployable ML system,
including:

-   Deterministic training pipeline orchestration
-   Behavioral and geospatial feature engineering
-   Cost-aware threshold optimization
-   Low-latency FastAPI inference deployment
-   Containerized runtime via Docker
-   Explainable model validation using SHAP

The system exposes a FastAPI inference service capable of sub-15ms
response time, converting raw transactions into explainable fraud
decisions.

------------------------------------------------------------------------

## 🚀 Key Engineering Solutions

### 🌍 Geospatial Velocity Detection ("Impossible Traveler")

Implemented a physics-based anomaly signal using the Haversine formula
to compute travel velocity between consecutive transactions.

Detects fraud patterns such as credential theft, account takeover, and
location spoofing.

------------------------------------------------------------------------

### ⚖️ Cost-Sensitive Fraud Optimization

Handled extreme class imbalance (\~98% legitimate transactions) using:

-   XGBoost scale_pos_weight
-   Explicit decision threshold optimization

Optimizes fraud detection based on financial loss trade-offs.

------------------------------------------------------------------------

### ⚡ High-Performance Native Inference Architecture

Uses XGBoost's native Booster interface for:

-   Direct execution in optimized C++ runtime
-   Lower latency and memory overhead
-   Deterministic JSON model loading

Typical inference latency: \< 15 ms

------------------------------------------------------------------------

### 🧠 Behavioral Anomaly Feature Engineering

Engineered deviation-based behavioral features:

-   amount_ratio
-   travel_velocity_kmph
-   tx_count_24h

------------------------------------------------------------------------

### 🔍 Explainable Model Validation

SHAP was used during evaluation to validate learned feature importance.

Confirmed top fraud signals:

-   amount_ratio
-   travel_velocity_kmph
-   tx_count_24h

Plots available in:

    reports/shap/

------------------------------------------------------------------------

### 📦 Containerized Deployment

Docker enables reproducible, environment-independent deployment.

------------------------------------------------------------------------

## 🐳 Run Instantly Using Docker (Recommended)

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

## 🔁 Training Pipeline

```bash
python run_pipeline.py
```

Pipeline stages:

1.  Data generation\
2.  Feature engineering\
3.  Model training\
4.  Threshold optimization\
5.  Model evaluation

------------------------------------------------------------------------

## 🧠 Decision Layer

Example response:

Separates prediction from business action.

Example response:

``` json
{
  "model_version": "xgboost_v:1.0",
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

## 🧰 Tech Stack

Machine Learning: - XGBoost - Scikit-Learn

API: - FastAPI - Pydantic - Uvicorn

Explainability(Offline): - SHAP

Monitoring: - Evidently AI

Deployment: - Docker

------------------------------------------------------------------------

## 🎯 Project Focus

Demonstrates:

-   Real-time ML inference deployment
-   Cost-aware fraud decision systems
-   Explainable ML validation
-   Production-grade API design
-   Containerized deployment

------------------------------------------------------------------------
