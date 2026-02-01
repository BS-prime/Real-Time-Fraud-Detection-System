# ==============================================================================================
# --- Import libraries ---
# ==============================================================================================

import xgboost as xgb
import numpy as np



# ==============================================================================================
# --- Model Output Tests ---
# ==============================================================================================

def test_model_output_range():

    """
    Ensure the model always predicts a probability between 0 and 1.
    """

    
    model = xgb.Booster()
    model.load_model("artifacts/models/artifacts/model/v1.0_xgb_fraud_detection_model.json")
    
    # Create mock data (1 row, matching training features)
    mock_data = np.random.rand(1, 5) 
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
    model.load_model("artifacts/models/artifacts/model/v1.0_xgb_fraud_detection_model.json")

    amount_idx = 0  # 👈 MUST match training feature order

    # Base transaction (low amount)
    base = np.array([[10, 0.5, 0.2, 0.1, 0.3]])

    # High-amount transaction (everything same except amount)
    high = base.copy()
    high[0, amount_idx] = 10_000

    d_low = xgb.DMatrix(base)
    d_high = xgb.DMatrix(high)

    low_risk = model.predict(d_low)[0]
    high_risk = model.predict(d_high)[0]

    assert high_risk >= low_risk * 0.7, (
        f"Risk dropped unexpectedly when amount increased: "
        f"low={low_risk}, high={high_risk}"
    )
