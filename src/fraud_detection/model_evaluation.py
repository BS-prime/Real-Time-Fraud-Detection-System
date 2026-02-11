# ======================================================================================================================================================
# --- Import Libraries ---
# ======================================================================================================================================================

from pathlib import Path
import joblib

import pandas as pd
import numpy as np

import shap
import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.metrics import (
    classification_report,
    average_precision_score, 
    precision_recall_curve
)



# ======================================================================================================================================================
# --- The Model Evaluation Function ---
# ======================================================================================================================================================

def model_evaluator(
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_prob: np.ndarray | pd.Series,
        y_pred_final: np.ndarray | pd.Series
    ):
    
    '''
    Docstring for model_evaluator
    
    :param model_name: The name of the model that you want to evaluate, it should be in the format 
    'model_id_seed_XX.json' where model_id is the name of the model and XX is the seed number 
    used in data generation. The model should be located in artifacts/model folder.
    
    :type model_name: str
    :type X_test: pd.DataFrame
    :type y_test: pd.Series
    :type y_pred_final: np.ndarray | pd.Series
    '''
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- Path Declaration ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ROOT_DIR = Path(__file__).resolve().parents[2]
    
    # Model path
    MODEL_PATH = ROOT_DIR / "artifacts" / "models" / model_name

    # Output path for evaluation results
    EVALUATION_OUTPUT_DIR = ROOT_DIR / "reports" / "model_evaluation"
    EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # SHAP output path
    SHAP_OUTPUT_DIR = ROOT_DIR / "reports" / "shap"
    SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- 1. AUC-PR score and Classification Report ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    print(f"AUC-PR for {model_name}: {average_precision_score(y_test, y_prob):.4f}\n")
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred_final))

    
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- 2. Precision-Recall Curve ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    average_precision = average_precision_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
    
    plt.savefig(EVALUATION_OUTPUT_DIR / f"precision_recall_curve_{model_name}.png")
    
    plt.close()



    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- 3. SHAP Summary Plot ---
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Load the model according to the model type
    try:
        model = joblib.load(MODEL_PATH)
    except:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
    
    # Calculate SHAP values for the test set
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Put shap_values as a 2d array for ploting purposes 
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    elif getattr(shap_values, "ndim", 2) == 3:
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_values_to_plot = shap_values


    plt.figure(figsize=(6,14))
    
    shap.summary_plot(shap_values_to_plot, X_test, plot_type="bar")
    
    plt.savefig(SHAP_OUTPUT_DIR / f"shap_summary_{model_name}.png")
    
    plt.close() 


        
