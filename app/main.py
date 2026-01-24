from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import xgboost as xgb
import numpy as np
from pathlib import Path
from datetime import datetime

# ==============================================================================
# Model loading (fail fast)
# ==============================================================================

MODEL_PATH = Path("artifacts/model/fraud_detection_model.json")

if not MODEL_PATH.exists():
    raise RuntimeError("Model file not found")

model = xgb.Booster()
model.load_model(str(MODEL_PATH))

MODEL_VERSION = "fraud_xgb_v1.0"

# ==============================================================================
# Constants
# ==============================================================================

FEATURE_COLUMNS = [
    'amount',  
    'lat', 
    'lon',
    'hour', 
    'day_of_week', 
    'tx_count_24h', 
    'avg_spend_user', 
    'amount_ratio', 
    'dist_from_last_tx_km', 
    'travel_velocity_kmph', 
    'auth_method_PIN', 
    'auth_method_Password',
    'category_food', 
    'category_grocery', 
    'category_tech', 
    'category_travel',
    'category_utilities'
]

GLOBAL_AVG_SPEND = 60.0
DEFAULT_TIME_DELTA_MIN = 6.0

# ==============================================================================
# --- 1. initializing fastapi instance ---
# ==============================================================================

app = FastAPI(title="Fraud Guard 2026")



# ==============================================================================
# --- 2. Request schema ---
# ==============================================================================

class Transaction(BaseModel):
    user_id: str
    amount: float = Field(gt=0)
    lat: float | None  = Field(le= 90.00, ge=-90.0)
    lon: float | None  = Field(le= 180.00, ge=-180.0)
    auth_method: Literal["Biometric", "OTP", "Password"]
    category: Literal["food", "grocery", "tech", "travel", "utilities", "entertainment"]
    time_delta_min: float | None = Field(default=None, gt=0)
    tx_count_24h: int | None = Field(default=3, ge=0)


# ==============================================================================
# Mock feature store
# ==============================================================================

# since 
user_history = {
    "USER_123": {
        "avg_spend": 45.0,
        "last_lat": 34.05,
        "last_lon": -118.24,
    }
}



# ==============================================================================
# helper functions
# ==============================================================================

def coord_delta(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0  # in km of course
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * (2 * np.arcsin(np.sqrt(a)))


# using probability by the model we make a decision
def fraud_decision(prob: float) -> str:
    if prob > 0.91: # remember the business cost aware threshold
        return "Block"
    if prob > 0.60:
        return "Review"
    return "Allow"


# for current day of the week
day_of_week = datetime.today().weekday()
# current hour
hour = datetime.today().hour

# ==============================================================================
# Prediction endpoint
# ==============================================================================
@app.get("/")
def health_check():
    return {"status": "Fraud Guard 2026 is up and running."}

@app.post("/predict")
async def predict_fraud(tx: Transaction):


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- fetch history ---
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    history = user_history.get(
        tx.user_id,
        {
            "avg_spend": GLOBAL_AVG_SPEND,
            "last_lat": tx.lat,
            "last_lon": tx.lon,
        },
    )

    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- real time feature engineering ---
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ratio between current amount and historical average
    amount_ratio = tx.amount / (history["avg_spend"] + 1e-6)

    # distance from last transaction
    dist_from_last_tx_km = coord_delta(
        tx.lat,
        tx.lon,
        history["last_lat"],
        history["last_lon"],
    )

    # travel velocity in km/h
    travel_velocity_kmph = dist_from_last_tx_km / (tx.time_delta_min/60.0)
    
    # handle auth method one-hot encoding
    auth_method_PIN = 1 if tx.auth_method == "PIN" else 0
    auth_method_Password = 1 if tx.auth_method == "Password" else 0
    
    # handle category one-hot encoding
    category_food = 1 if tx.category == "food" else 0
    category_grocery = 1 if tx.category == "grocery" else 0
    category_tech = 1 if tx.category == "tech" else 0
    category_travel = 1 if tx.category == "travel" else 0
    category_utilities = 1 if tx.category == "utilities" else 0

        
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- deterministic fallback ---
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    time_delta = tx.time_delta_min or DEFAULT_TIME_DELTA_MIN
    
    features = np.array(
        [[
            tx.amount,
            tx.lat,
            tx.lon,
            hour,
            day_of_week,
            tx.tx_count_24h,
            history["avg_spend"],
            amount_ratio,
            dist_from_last_tx_km,
            travel_velocity_kmph,
            auth_method_PIN,
            auth_method_Password,
            category_food,
            category_grocery,
            category_tech,
            category_travel,
            category_utilities
        ]],
        dtype=np.float32,
    )

    if features.shape[1] != len(FEATURE_COLUMNS):
        raise HTTPException(
            status_code=500, 
            detail="Feature vector mismatch"
        )


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- inference ---
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    try:
        dmatrix = xgb.DMatrix(
            features, 
            feature_names=FEATURE_COLUMNS
        )
        probability = float(
            model.predict(dmatrix)[0]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Inference error: {e}"
        )

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


# Use this link to test the API once running: http://127.0.0.1:8000/docs