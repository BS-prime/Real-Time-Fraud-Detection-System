# ====================================================================================================
# --- Import the libraries ---
# ====================================================================================================

from pathlib import Path
import json
import numpy as np
from sklearn.metrics import confusion_matrix



# ====================================================================================================
# --- Core Threshold Optimization Function ---
# ====================================================================================================

def threshold_optimizer(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_path: str,
        cost_fp: float | None = 1.0,
        cost_fn: float | None = 10.0,  
) -> json:
    
    '''
    Function to find the optimal threshold that minimizes 
    business cost and save it in a JSON file.
    '''


    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- 1. Defining Paths ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Locate the project root directory
    ROOT_DIR = Path(__file__).resolve().parents[2]

    # Define the input and output paths
    OUTPUT_DIR = ROOT_DIR / Path("artifacts/model_thresholds")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract model name from the model path for naming the JSON file.
    model_name = Path(model_path).stem
        
    # Setting up the path to save the json file
    path = OUTPUT_DIR / f"optimal_threshold_{model_name}.json"
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- 2. Threshold Optimization Functions ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def compute_cost(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cost_fp: float,
        cost_fn: float,
    ) -> float:
        
        """
        Function to compute business cost based on confusion matrix.
        """
        
        # Compute business cost based on confusion matrix.
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return (cost_fp * fp) + (cost_fn * fn)



    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- 3. Find Optimal Threshold ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Find the optimal probability threshold to minimize business cost.
    def find_optimal_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        cost_fp: float | None = 1.0,
        cost_fn: float | None = 10.0
    ) -> dict:
        
        """
        Find the optimal threshold that minimizes business cost.
        """
        
        
        thresholds = np.linspace(0.01, 0.99, 99)

        y_true = y_true.to_numpy()

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
    
    # Call the find_optimal_threshold function to get the optimal threshold information.
    threshold_info = find_optimal_threshold(
        y_true=y_true,
        y_prob=y_pred,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
    )


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- 4. Save Threshold Metadata ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def save_threshold(
        threshold_info = threshold_info,
        path = path
    ) -> json:
        
        '''
        Function to save new threshold information in a JSON file.
        '''
        

        # Save the threshold information as a JSON file.
        with open(path, "w") as f:
            json.dump(threshold_info, f, indent=2)

    # Call the save_threshold function to save the optimal threshold information.
    save_threshold(threshold_info, OUTPUT_DIR)

    return threshold_info , path 
