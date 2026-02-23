import os
import glob
import pandas as pd
from src.models.analysis import StatisticalAnalyzer
from src.models.classifier import FaultClassifier

def run_analysis():
    print("Starting Post-Training Analysis...")
    
    # 1. Load Data
    processed_files = glob.glob('data/processed/*.csv')
    dfs = []
    for f in processed_files:
        dfs.append(pd.read_csv(f))
        
    if not dfs:
        print("No processed data found. Run main.py first.")
        return
        
    master_df = pd.concat(dfs, ignore_index=True)
    
    # 2. Statistical Analysis
    analyzer = StatisticalAnalyzer()
    analyzer.perform_feature_significance(master_df)
    
    # 3. We need a trained model for SHAP
    # Since we train it on the fly in main.py, let's retrain quickly on a subset
    print("Retraining a quick XGBoost model for SHAP Explainability...")
    
    ignore_cols = ['timestamp', 'scenario', 'run_id', 'fault_type', 'fault_severity', 
                   'time_to_next_fault', 'fault_within_5s', 'fault_within_30s', 'event_id']
    # Keep only numeric columns for features to avoid SHAP string to float conversion errors
    numeric_df = master_df.select_dtypes(include=['number'])
    feature_cols = [c for c in numeric_df.columns if c not in ignore_cols]
    
    # Needs to have faults
    train_df = master_df.dropna(subset=['fault_type']) 
    X_train = train_df[feature_cols]
    y_train = train_df['fault_type']
    
    if len(y_train) < 50 or len(y_train.unique()) < 2:
        print("Not enough fault data for SHAP analysis. Skipping.")
        return
        
    classifier = FaultClassifier()
    classifier.fit(X_train, y_train)
    
    analyzer.plot_shap_explanations(classifier, X_train, feature_cols)
    print("Analysis complete. Check 'Analysis_Results' folder.")

if __name__ == '__main__':
    run_analysis()
