# ========================================================================
# --- Import Libraries ---
# ========================================================================

from pathlib import Path
import pandas as pd
import numpy as np



# ========================================================================
# --- Load Data ---
# ========================================================================

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a DataFrame.
    """
    df = pd.read_csv(file_path)
    return df



# ========================================================================
