from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import math

# ==============================================================================
# App
# ==============================================================================

app = FastAPI(title="Fraud Guard 2026")



# ==============================================================================
# Model Loading (Explicit, Safe)
# ==============================================================================

MODEL_PATH = Path("model/fraud_detection_model.json")

if not MODEL_PATH.exists():
    raise RuntimeError("Model file not found")

model = xgb.Booster()
model.load_model(MODEL_PATH)

MODEL_VERSION = "fraud_xgb_v1.0"



# ==============================================================================
# Constants
# ==============================================================================

FEATURE_COLUMNS = [
    "amount",
    "amount_ratio",
    "coord_delta",
    "travel_velocity",
    "auth_method_Biometric",
]

GLOBAL_AVG_SPEND = 60.0
DEFAULT_TIME_DELTA_MIN = 6.0  # fallback only



# ==============================================================================
# Request Schema
# ==============================================================================

class Transaction(BaseModel):
    user_id: str
    amount: float = Field(gt=0)
    lat: float
    lon: float
    auth_method: Literal["Biometric", "OTP", "Password"]
    time_delta_min: float | None = Field(
        default=None, gt=0, description="Minutes since last transaction"
    )



# ==============================================================================
# Mock Feature Store
# ==============================================================================

user_history = {
    "USER_123": {
        "avg_spend": 45.0,
        "last_lat": 34.05,
        "last_lon": -118.24
    }
}



# ==============================================================================
# Helpers
# ==============================================================================

def coord_delta(lat1, lon1, lat2, lon2) -> float:
    
    R = 6371.0  # Earth radius in km

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def fraud_decision(prob: float) -> str:
    if prob > 0.91:
        return "Block"
    if prob > 0.6:
        return "Review"
    return "Allow"



# ==============================================================================
# Prediction Endpoint
# ==============================================================================

@app.post("/predict")
async def predict_fraud(tx: Transaction):
    history = user_history.get(
        tx.user_id,
        {
            "avg_spend": GLOBAL_AVG_SPEND,
            "last_lat": tx.lat,
            "last_lon": tx.lon
        }
    )

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    amount_ratio = tx.amount / (history["avg_spend"] + 1e-6)

    delta = coord_delta(
        tx.lat,
        tx.lon,
        history["last_lat"],
        history["last_lon"]
    )

    time_delta = tx.time_delta_min or DEFAULT_TIME_DELTA_MIN
    travel_velocity = delta / time_delta

    feature_vector = pd.DataFrame(
        [[
            tx.amount,
            amount_ratio,
            delta,
            travel_velocity,
            1 if tx.auth_method == "Biometric" else 0
        ]],
        columns=FEATURE_COLUMNS
    )

    # -----------------------------
    # Inference
    # -----------------------------
    try:
        dmatrix = xgb.DMatrix(feature_vector)
        probability = float(model.predict(dmatrix)[0])
    except Exception:
        raise HTTPException(status_code=500, detail="Model inference failed")

    decision = fraud_decision(probability)

    return {
        "model_version": MODEL_VERSION,
        "decision": decision,
        "fraud_probability": round(probability, 4),
        "risk_level": "High" if decision != "Allow" else "Low"
    }



# ==============================================================================
# Local Run
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Use the following  url on your browser after running the python file 
# http://localhost:8000/docs
