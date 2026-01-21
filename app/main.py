from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import xgboost as xgb
import numpy as np
from pathlib import Path

# ==============================================================================
# Model loading (fail fast)
# ==============================================================================

MODEL_PATH = Path("model/fraud_detection_model.json")

if not MODEL_PATH.exists():
    raise RuntimeError("Model file not found")

model = xgb.Booster()
model.load_model(str(MODEL_PATH))

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
DEFAULT_TIME_DELTA_MIN = 6.0

# ==============================================================================
# App
# ==============================================================================

app = FastAPI(title="Fraud Guard 2026")

# ==============================================================================
# Request schema
# ==============================================================================

class Transaction(BaseModel):
    user_id: str
    amount: float = Field(gt=0)
    lat: float
    lon: float
    auth_method: Literal["Biometric", "OTP", "Password"]
    time_delta_min: float | None = Field(default=None, gt=0)

# ==============================================================================
# Mock feature store
# ==============================================================================

user_history = {
    "USER_123": {
        "avg_spend": 45.0,
        "last_lat": 34.05,
        "last_lon": -118.24,
    }
}

# ==============================================================================
# Helpers
# ==============================================================================

def coord_delta(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * (2 * np.arcsin(np.sqrt(a)))

def fraud_decision(prob: float) -> str:
    if prob > 0.91:
        return "Block"
    if prob > 0.60:
        return "Review"
    return "Allow"

# ==============================================================================
# Prediction endpoint
# ==============================================================================

@app.post("/predict")
async def predict_fraud(tx: Transaction):

    # --- fetch history ---
    history = user_history.get(
        tx.user_id,
        {
            "avg_spend": GLOBAL_AVG_SPEND,
            "last_lat": tx.lat,
            "last_lon": tx.lon,
        },
    )

    # --- feature engineering ---
    amount_ratio = tx.amount / (history["avg_spend"] + 1e-6)

    delta = coord_delta(
        tx.lat,
        tx.lon,
        history["last_lat"],
        history["last_lon"],
    )

    time_delta = tx.time_delta_min or DEFAULT_TIME_DELTA_MIN
    travel_velocity = delta / time_delta

    features = np.array(
        [[
            tx.amount,
            amount_ratio,
            delta,
            travel_velocity,
            1.0 if tx.auth_method == "Biometric" else 0.0,
        ]],
        dtype=np.float32,
    )

    if features.shape[1] != len(FEATURE_COLUMNS):
        raise HTTPException(status_code=500, detail="Feature vector mismatch")

    # --- inference ---
    try:
        dmatrix = xgb.DMatrix(features, feature_names=FEATURE_COLUMNS)
        probability = float(model.predict(dmatrix)[0])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    decision = fraud_decision(probability)

    return {
        "model_version": MODEL_VERSION,
        "decision": decision,
        "fraud_probability": round(probability, 4),
        "risk_level": "High" if decision != "Allow" else "Low",
    }

# ==============================================================================
# Local run
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
