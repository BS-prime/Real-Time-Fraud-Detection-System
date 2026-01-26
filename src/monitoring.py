# ===================================================================================
# --- Import the libraries ---
# ===================================================================================

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset, 
    TargetDriftPreset, 
    DataQualityPreset
)



def generate_monitoring_report(reference_path, current_path, output_path="reports/drift_report.html"):
    


    # ===================================================================================
    # --- 1. Load your baseline (training) and live data ---
    # ===================================================================================
    
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)



    # ===================================================================================
    # --- 2. Setup the Evidently Report ---
    # ===================================================================================
    
    # We check for: Data Drift (Inputs), Target Drift (Fraud rates), and Data Quality
    report = Report(metrics=[
        DataDriftPreset(), 
        TargetDriftPreset(),
        DataQualityPreset()
    ])



    # ===================================================================================
    # --- 3. Run the analysis ---
    # ===================================================================================
    
    report.run(reference_data=reference, current_data=current)
    


    # ===================================================================================
    # --- 4. Save as a professional dashboard ---
    # ===================================================================================
    
    report.save_html(output_path)
    print(f"✅ Monitoring Report generated: {output_path}")

# Example usage
if __name__ == "__main__":
    # Example usage
    generate_monitoring_report(
        "data/processed/train_baseline.csv", 
        "data/processed/production_jan_2026.csv"
    )