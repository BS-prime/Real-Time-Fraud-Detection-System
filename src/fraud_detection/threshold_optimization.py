# ====================================================================================================
# --- Import the libraries ---
# ====================================================================================================

from pathlib import Path

import json
import joblib

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.metrics import confusion_matrix



# ====================================================================================================
# --- Core Threshold Optimization Function ---
# ====================================================================================================

def threshold_optimizer(
            X_test: pd.DataFrame,
            y_test: pd.Series,
            model_name: str,
            cost_fp: float | None = 1.0,
            cost_fn: float | None = 10.0,
            save: bool = True  
    ) -> tuple:

    '''
    Docstring for threshold_optimizer
    
    :param X_test: Put the X_test data that you want to use for threshold optimization, 
    it should be in the same format as the data used for model training.
    :type X_test: list
    :param y_test: Put the y_test data that you want to use for threshold optimization,
    it should be in the same format as the data used for model training.
    :type y_test: list
    :param model_name: The name of the model that you want to optimize the threshold for, 
    it should be in the format 'model_id_seed_XX.json' where model_id is the name of the
    model and XX is the seed number used in data generation. 
    The model should be located in artifacts/model folder.
    :type model_name: str
    :param cost_fp: Just put the cost of false positive that you want to use for 
    threshold optimization, it should be a positive number. The default value is 1.0.
    :type cost_fp: float | None
    :param cost_fn: Just put the cost of false negative that you want to use for 
    threshold optimization, it should be a positive number. The default value is 10.0.
    :type cost_fn: float | None
    :param save: Set this to True if you want to save the optimal threshold information 
    in a JSON file, the JSON file will be saved in artifacts/model_thresholds folder. 
    The default value is True
    :type save: bool
    :return: The function returns a tuple containing the predicted probabilities(y_prob) 
    and the final predicted(y_pred_final) labels based on the optimal threshold.
    :rtype: tuple
    '''


    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- 0. Defining Directories ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Locate the project root directory
    ROOT_DIR = Path(__file__).resolve().parents[2]

    # Locate the model 
    MODEL_PATH = ROOT_DIR / "artifacts" / "models" / model_name

    # Define threshold output path
    OUTPUT_DIR = ROOT_DIR / Path("artifacts/model_thresholds")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract model name from the model path for naming the JSON file.
    model_name = Path(MODEL_PATH).stem
        
    # Setting up the path to save the json file
    path = OUTPUT_DIR / f"optimal_threshold_{model_name}.json"
    
    
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- 2. Load model ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Load the model according to the model type
    try:
        model = joblib.load(MODEL_PATH)
    except:
        model = xgb.Booster()
        model.load_model(str(MODEL_PATH))
    
    # Get predicted probabilities for the positive class (fraud) based on the model type.
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]  # Get probability of positive class 
    else:
        y_prob = model.predict(xgb.DMatrix(X_test))  # For XGBoost, use DMatrix for prediction
    

    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- 2. Business cost calculation ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def compute_cost(
            y_test = y_test,
            y_pred = None,
            cost_fp = cost_fp,
            cost_fn = cost_fn,
        ) -> float:
        
        """
        Function to compute business cost based on confusion matrix.
        """
        


        # Compute business cost based on confusion matrix.
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        return (cost_fp * fp) + (cost_fn * fn)



    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- 3. Find Optimal Threshold ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Find the optimal probability threshold to minimize business cost.
    def find_optimal_threshold(
            y_test = y_test,
            y_prob = y_prob,
            cost_fp = cost_fp,
            cost_fn = cost_fn,
        ) -> dict:
        
        """
        Find the optimal threshold that minimizes business cost.
        """
        
        # Define a range of numbers to evaluate threshold
        thresholds = np.linspace(0.01, 0.99, 99)

        # Convert the list to numpy array
        y_test = y_test.to_numpy()
        
        # Initialize variables to track the best threshold and minimum cost.
        best_threshold = 0.5
        min_cost = float("inf")

        # Run a loop to find the best threshold that minimizes the business cost.
        for t in thresholds:
            
            y_pred = (y_prob >= t).astype(int)
            
            cost = compute_cost(
                y_test=y_test, 
                y_pred=y_pred, 
                cost_fp=cost_fp, 
                cost_fn=cost_fn
            )

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
        y_test=y_test,
        y_prob=y_prob,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
    )


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- 4. Save Threshold Metadata ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    if save:
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
        save_threshold(threshold_info, path)


    
    
    best_threshold = threshold_info['best_threshold']
    
    print("=" * 70)
    
    # Print the path to the json file
    print(f"Optimal threshold information saved to: {OUTPUT_DIR}")
    
    print()
    
    # Print the threshold for the particular model
    print(f"The threshold for the model: {model_name} is: {best_threshold}")
    
    print("=" * 70)
    
    y_pred_final = (y_prob >= best_threshold).astype(int)

    return  y_prob, y_pred_final
