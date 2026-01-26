# ================================================================================
# --- 1. Importing the Libraries ---
# ================================================================================

import shap
import numpy as np



# ================================================================================
# --- 2. Defining the class ---
# ================================================================================

class ShapExplainer:
    
    
    # Initialize with model and feature names
    def __init__(self, model, feature_names):
        
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names


    # Explain the prediction for a single sample
    def explain(self, X, top_k=3):
        
        shap_values = self.explainer.shap_values(X)
        shap_row = shap_values[0]

        contributions = list(zip(self.feature_names, shap_row))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        return [
            {"feature": f, "contribution": float(v)}
            for f, v in contributions[:top_k]
        ]
