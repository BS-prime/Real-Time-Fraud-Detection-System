# ==========================================================================
# --- Import Libraries ---
# ==========================================================================

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier


# ==========================================================================
# --- Define the actual function ---
# ==========================================================================
def model_trainer(
        data_path: str | None = 'artifacts/data/simulated_transactions_seed_42.csv'        
):
    
    '''
    Docstring for model_trainer
    
    :param data_path: Just give the path from the root directory
    :type data_path: str | None
    '''


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- Path 

# 1. Load Config & Extract Seed
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_path = 'artifacts/data/simulated_transactions_seed_42.csv'
seed_val = int(re.search(r'_seed_(\d+)', data_path).group(1))

# 2. Map String names to actual Scikit-Learn Classes
algo_map = {
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "SVC": SVC
}

# 3. Data Setup (Placeholder for your loading logic)
X_train, X_test, y_train, y_test = 

# 4. Loop through Config and Train
for model_id, settings in config['models'].items():
    print(f"Running GridSearch for: {model_id}")
    
    # Initialize the specific class
    base_model = algo_map[settings['type']](random_state=seed_val)
    
    # Setup GridSearch
    grid = GridSearchCV(base_model, settings['params'], cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    
    # Save the Best Version
    best_model = grid.best_estimator_
    save_path = Path(f"artifacts/models/{model_id}_seed_{seed_val}.pkl")
    joblib.dump(best_model, save_path)
    
    print(f"Best Score: {grid.best_score_:.4f} | Saved to: {save_path}")
