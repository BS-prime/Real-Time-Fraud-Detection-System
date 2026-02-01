# ==============================================================================================
# --- Import libraries ---
# ==============================================================================================

import xgboost as xgb
import numpy as np
from pathlib import Path

# Define model path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "artifacts"
    / "model"
    / "v1.0_xgb_fraud_detection_model.json"
)

import pytest

@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Trained model not found"
)



# ==============================================================================================
# --- Model Output Tests ---
# ==============================================================================================

def test_model_output_range():

    """
    Ensure the model always predicts a probability between 0 and 1.
    """

    
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    
    # Create mock data (1 row, matching training features)
    mock_data = np.random.rand(1, 17).astype(np.float32)
 
    dmatrix = xgb.DMatrix(mock_data)
    
    prediction = model.predict(dmatrix)[0]
    assert 0 <= prediction <= 1, "Model output is not a valid probability!"



# ==============================================================================================
# --- Heuristic Tests ---
# ==============================================================================================

def test_high_amount_risk():
    
    """
    Heuristic Test:
    As transaction amount increases significantly,
    predicted fraud risk should not decrease sharply.
    """
    
    
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))


    # Base transaction (low amount) 
    base = np.array(
    [[
        20.0,        # tx.amount, amount is low, to make it a normal transaction
        12.9716,      # tx.lat
        77.5946,      # tx.lon
        14,           # hour (2 PM)
        2,            # day_of_week (Wednesday)
        3,            # tx.tx_count_24h
        350.0,        # history["avg_spend"]
        0.057,         # amount_ratio (20 / 350)
        1.2,          # dist_from_last_tx_km
        5.0,          # travel_velocity_kmph
        1.0,          # auth_method_PIN
        0.0,          # auth_method_Password
        1.0,          # category_food
        0.0,          # category_grocery
        0.0,          # category_tech
        0.0,          # category_travel
        0.0           # category_utilities
    ]],
    dtype=np.float32
)

    # High-amount transaction 
    high = base.copy()

    high[0, 0] = 10_000  # Update the amount feature (index 0)
    high[0, 7] = high[0, 0] / high[0, 6]  # Update amount_ratio accordingly

    # Create DMatrix for both
    d_low = xgb.DMatrix(base)
    d_high = xgb.DMatrix(high)

    # Make the predictions
    low_risk = model.predict(d_low)[0]
    high_risk = model.predict(d_high)[0]

    # Assert that risk does not drop significantly with higher amount
    assert high_risk >= low_risk * 0.7, (
        f"Risk dropped unexpectedly when amount increased: "
        f"low={low_risk}, high={high_risk}"
    )
