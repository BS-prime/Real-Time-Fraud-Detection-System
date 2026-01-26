# ================================================================================
# --- 1. Import Libraries ---
# ================================================================================

from pathlib import Path
import json
import numpy as np
from sklearn.metrics import confusion_matrix

# Define the path to save the threshold
THRESHOLD_PATH = Path.cwd().parent / "model" / "threshold.json"



# ================================================================================
# --- 2. Define the function ---
# ================================================================================

def find_optimal_threshold(y_true, y_prob, cost_fp=1, cost_fn=10):
    
    
    
    # Search thresholds from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    lowest_cost = float("inf")

    
    
    # Evaluate each threshold
    for t in thresholds:
        
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = cost_fp * fp + cost_fn * fn

        if total_cost < lowest_cost:
            lowest_cost = total_cost
            best_threshold = t

    return best_threshold, lowest_cost



# Example usage after model training
best_t, cost = find_optimal_threshold(
    y_val,
    model.predict_proba(X_val)[:, 1],
    cost_fp=1,
    cost_fn=10
)



with open(THRESHOLD_PATH, "w") as f:
    json.dump({"threshold": best_t}, f)



print(f"Saved optimal threshold: {best_t:.2f}")
