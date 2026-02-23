import os
import glob
import pandas as pd
import numpy as np

# Import our custom modules
from src.data_generation.generator import CombatScenarioGenerator
from src.features.engineering import FeatureEngineer
from src.models.classifier import FaultClassifier

def run_pipeline():
    print("="*50)
    print("DRDO PREDICTIVE MAINTENANCE PIPIELINE STARTED")
    print("="*50)
    
    # ---------------------------------------------------------
    # PART 1: Data Generation
    # ---------------------------------------------------------
    raw_dir = 'data/raw'
    if not os.path.exists(raw_dir) or len(glob.glob(f'{raw_dir}/*.csv')) == 0:
        print("\n[1] GENERATING PHYSICAL SCENARIO DATA...")
        generator = CombatScenarioGenerator(output_dir=raw_dir)
        # Baseline Data
        for i in range(4):
            generator.generate_scenario(120, 'baseline', run_id=i+1)
        # Combat Scenarios
        scenarios = ['arctic_cold', 'desert_heat', 'artillery_firing', 'rough_terrain', 'weapons_active', 'emp_simulation']
        for val, scenario in enumerate(scenarios):
            generator.generate_scenario(30, scenario, run_id=val+1)
    else:
        print("\n[1] RAW DATA FOUND. SKIPPING GENERATION...")
            
    # ---------------------------------------------------------
    # PART 2: Feature Engineering
    # ---------------------------------------------------------
    processed_dir = 'data/processed'
    if not os.path.exists(processed_dir) or len(glob.glob(f'{processed_dir}/*.csv')) == 0:
        print("\n[2] ENGINEERING FEATURES (FFT, DERIVATIVES, LAGS)...")
        engineer = FeatureEngineer()
        
        raw_files = glob.glob(f'{raw_dir}/*_run*.csv')
        raw_files = [f for f in raw_files if 'faults' not in f]
        
        for file in raw_files:
            df = pd.read_csv(file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            featured_df = engineer.engineer_features(df)
            
            os.makedirs(processed_dir, exist_ok=True)
            filename = os.path.basename(file)
            featured_df.to_csv(os.path.join(processed_dir, filename), index=False)
            print(f"Processed and saved {filename}")
    else:
        print("\n[2] PROCESSED DATA FOUND. SKIPPING ENGINEERING...")
        
    # ---------------------------------------------------------
    # PART 3: ML Modeling - XGBoost Classification
    # ---------------------------------------------------------
    print("\n[3] TRAINING XGBOOST + SMOTE CLASSIFIER...")
    processed_files = glob.glob(f'{processed_dir}/*.csv')
    
    # Load all processed data
    dfs = []
    for f in processed_files:
        dfs.append(pd.read_csv(f))
    master_df = pd.concat(dfs, ignore_index=True)
    
    # We will classify 'fault_type' using the engineered features
    # Drop columns that are not predictors
    ignore_cols = ['timestamp', 'scenario', 'run_id', 'fault_type', 'fault_severity', 
                   'time_to_next_fault', 'fault_within_5s', 'fault_within_30s', 'event_id']
    
    feature_cols = [c for c in master_df.columns if c not in ignore_cols]
    
    # Simple temporal split: use 80% for train, 20% for test (ordered by time)
    master_df = master_df.sort_values(by=['scenario', 'run_id', 'timestamp']).reset_index(drop=True)
    train_size = int(len(master_df) * 0.8)
    
    X_train = master_df.loc[:train_size, feature_cols]
    y_train = master_df.loc[:train_size, 'fault_type']
    
    X_test = master_df.loc[train_size:, feature_cols]
    y_test = master_df.loc[train_size:, 'fault_type']
    
    # Only train if we have faults in the training set
    if len(y_train.unique()) > 1:
        classifier = FaultClassifier()
        classifier.fit(X_train, y_train)
        
        # Simple Evaluation
        from sklearn.metrics import classification_report
        y_pred = classifier.predict(X_test)
        print("\n--- XGBoost Classification Report on Test Set ---")
        print(classification_report(y_test, y_pred, zero_division=0))
    else:
        print("Not enough fault classes in training split to evaluate classifier.")
        
    print("\n======================================================")
    print("PIPELINE COMPLETED")
    print("======================================================")

if __name__ == '__main__':
    run_pipeline()
