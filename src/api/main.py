# ==============================================================================
# --- Import the libraries ---
# ==============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import json
import xgboost as xgb
import numpy as np
from pathlib import Path
from datetime import datetime



# ==============================================================================
# --- Model loading (fail fast) ---
# ==============================================================================

MODEL_PATH = Path("artifacts/models")

MODEL_FILE = MODEL_PATH / "xgboost_seed_42.json"

if not MODEL_FILE.exists():
    raise RuntimeError("Model file not found")

model = xgb.Booster()
model.load_model(str(MODEL_FILE))

MODEL_VERSION = "XGBoost_v:1.0"



# ==============================================================================
# --- Load optimal threshold ---
# ==============================================================================

THRESHOLD_PATH = Path(
    "artifacts/model_thresholds/optimal_threshold_xgboost_seed_42.json"
)
if not (THRESHOLD_PATH).exists():
    raise RuntimeError("Threshold file not found")

with open(THRESHOLD_PATH) as f:
    THRESHOLD = json.load(f)["best_threshold"]



# ==============================================================================
# --- Constants ---
# ==============================================================================

# Define all the features for further use
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

# default values for missing user history
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
    auth_method: Literal["Biometric", "PIN", "Password"]
    category: Literal["food", "grocery", "tech", "travel", "utilities", "entertainment"]
    time_delta_min: float | None = Field(default=None, gt=0)
    tx_count_24h: int | None = Field(default=3, ge=0)



# ==============================================================================
# Mock feature store
# ==============================================================================

# since we don't have a real feature store, we'll use mock user historical data
# In a real-world scenario, this would be fetched from a database or feature store
user_history = {
    "USER_123": {
        "avg_spend": 45.0,
        "last_lat": 34.05,
        "last_lon": -118.24,
    }
}



# ==============================================================================
# --- helper functions ---
# ==============================================================================

def coord_delta(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0  # in km of course
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * (2 * np.arcsin(np.sqrt(a)))


# for current day of the week
day_of_week = datetime.today().weekday()

# current hour
hour = datetime.today().hour



# ==============================================================================
# --- Decision Logic Layer (Business Logic) ---
# ==============================================================================

def get_risk_band(probability: float) -> str:
    """Maps probability to human-readable risk level."""
    if probability >= 0.85:
        return "VERY_HIGH"
    elif probability >= 0.65:
        return "HIGH"
    elif probability >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


def decide_action(probability: float, threshold: float) -> str:
    """Determines business action based on probability."""
    if probability >= threshold:
        return "BLOCK"
    elif probability >= 0.50:
        return "CHALLENGE"
    else:
        return "ALLOW"


def get_decision_reasons(
    amount_ratio: float,
    travel_velocity_kmph: float,
    tx_count_24h: int
) -> list[str]:
    """Provides human-readable explanation signals."""
    
    reasons = []

    if amount_ratio >= 3:
        reasons.append("Transaction amount significantly higher than user's normal spending")

    if travel_velocity_kmph >= 800:
        reasons.append("Transaction location implies unrealistic travel speed")

    if tx_count_24h >= 20:
        reasons.append("Unusually high number of transactions in past 24 hours")

    if amount_ratio >= 10 or travel_velocity_kmph >= 900: # commercial flight threshold (e.g., 900 km/h)
        reasons.append("Extreme deviation from user spending behavior")

    return reasons[:3]  # limit to top 3


def get_fallbacks_used(
    tx,
    history
) -> list[str]:
    """Reports fallback defaults used during feature engineering."""
    
    fallbacks = []

    if tx.time_delta_min is None:
        fallbacks.append("DEFAULT_TIME_DELTA_USED")

    if history["avg_spend"] == GLOBAL_AVG_SPEND:
        fallbacks.append("GLOBAL_AVG_SPEND_USED")

    if history["last_lat"] == tx.lat and history["last_lon"] == tx.lon:
        fallbacks.append("NO_LOCATION_HISTORY")

    return fallbacks



# ==============================================================================
# --- 3. API Endpoints ---
# ==============================================================================

@app.get("/")
def health_check():
    return {"status": "Fraud Guard 2026 is up and running."}


@app.post("/predict")
async def predict_fraud(tx: Transaction):


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- fetch history ---
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Usually in real cases history get fetch from a real database
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
    
    # travel velocity in km/h
    travel_velocity_kmph = dist_from_last_tx_km / (time_delta / 60.0)
    
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- prepare feature vector ---
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- Decision Layer ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    risk = get_risk_band(probability)

    action = decide_action(
        probability,
        THRESHOLD
    )

    reasons = get_decision_reasons(
        amount_ratio,
        travel_velocity_kmph,
        tx.tx_count_24h
    )

    fallbacks_used = get_fallbacks_used(
        tx,
        history
    )


    # ==============================================================================
    # --- Final Response ---
    # ==============================================================================

    return {
        "model_version": MODEL_VERSION,

        # model output
        "fraud_probability": round(probability, 4),

        # business interpretation
        "risk_band": risk,
        "recommended_action": action,

        # explainability
        "decision_reasons": reasons,

        # transparency
        "fallbacks_used": fallbacks_used
    }



# ==============================================================================
# Local run
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# Use this link to test the API once running: http://localhost:8000/docs