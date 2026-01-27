# ===================================================================================
# --- 1. Import the libraries ---
# ===================================================================================

from pathlib import Path
import json
import numpy as np
from sklearn.metrics import confusion_matrix



# ===================================================================================
# --- 2. Threshold Optimization Functions ---
# ===================================================================================

def compute_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> float:
    
    # Compute business cost based on confusion matrix.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (cost_fp * fp) + (cost_fn * fn)



# ===================================================================================
# --- 3. Find Optimal Threshold ---
# ===================================================================================

# Find the optimal probability threshold to minimize business cost.
def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
    thresholds: np.ndarray | None = None,
) -> dict:
    
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    min_cost = float("inf")

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cost = compute_cost(y_true, y_pred, cost_fp, cost_fn)

        if cost < min_cost:
            min_cost = cost
            best_threshold = float(t)

    return {
        "best_threshold": best_threshold,
        "min_cost": float(min_cost),
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
    }
# Put the result in a variable to use later in the save_threshold function in the
# parameter threshold_info.


# ===================================================================================
# --- 4. Save Threshold Metadata ---
# ===================================================================================

def save_threshold(
    threshold_info: dict,
    path: Path,
) -> None:
    
    # Save threshold information as JSON.
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(threshold_info, f, indent=2)
