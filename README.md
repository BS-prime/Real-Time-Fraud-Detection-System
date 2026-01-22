# 🛡️ Real-Time Fraud Guard 2026
**An End-to-End MLOps Pipeline for Geospatial & Behavioral Anomaly Detection**

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Framework](https://img.shields.io/badge/Framework-FastAPI-009688)

## 📌 Project Overview
In 2026, static fraud rules are obsolete. This project implements a high-performance Machine Learning service designed to detect fraudulent transactions using **Geospatial Velocity** and **Behavioral Profiling**. It moves beyond "Black Box" AI by integrating **SHAP explainability** and **real-time drift monitoring**.

### 🚀 Key Engineering Solutions
* **The Impossible Traveler Logic:** Developed a custom feature to detect transactions that exceed physical travel limits using the Haversine formula.
* **Cost-Sensitive Learning:** Leveraged XGBoost's `scale_pos_weight` to address a 98% class imbalance, prioritizing **Recall (100%)** over simple Accuracy.
* **Explainable AI (XAI):** Integrated SHAP into the inference pipeline to provide "Reason Codes" for every blocked transaction, satisfying financial regulatory requirements.
* **Business-Cost-Aware-Threshold:** The decision threshold is optimized to minimize financial loss, explicitly prioritizing the reduction of costly false positives.
* **Production Architecture:** Containerized via Docker with a FastAPI gateway capable of **<15ms inference latency**.

---

## 🏗️ Tech Stack
* **Modeling:** XGBoost, Scikit-Learn
* **Explainability:** SHAP
* **API Layer:** FastAPI, Pydantic V2, Uvicorn
* **Containerization:** Docker (Slim-image optimization)
* **Observability:** Evidently AI (Data & Concept Drift)

---

## 📊 Core Feature: Geospatial Velocity
To identify fraud, the system calculates the physical distance between consecutive transactions using:

$$\text{dist} = 2r \arcsin\left(\sqrt{\sin^2\left(\frac{\phi_2-\phi_1}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\lambda_2-\lambda_1}{2}\right)}\right)$$

If the velocity ($\text{dist} / \text{time}$) exceeds a commercial flight threshold (e.g., 900 km/h), the risk score is automatically elevated.

---

## 🛠️ Installation & Deployment
### 1. Build the Docker Container
```bash
docker build -t fraud-guard-2026 .