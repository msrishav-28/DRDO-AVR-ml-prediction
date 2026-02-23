import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from scipy.stats import kruskal, wilcoxon
import os

class StatisticalAnalyzer:
    def __init__(self, output_dir='Analysis_Results'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def perform_feature_significance(self, df):
        print("Running Kruskal-Wallis tests across scenarios...")
        # Compare base features across different physical scenarios to prove simulation fidelity
        scenarios = df['scenario'].unique()
        features = ['voltage', 'current', 'power', 'fft_power_5_10hz']
        
        results = []
        for feature in features:
            groups = [df[df['scenario'] == s][feature].dropna().values for s in scenarios]
            
            # Subsample to keep tests fast and avoid massive memory overhead
            groups = [g[np.random.choice(len(g), min(len(g), 5000), replace=False)] 
                      if len(g) > 5000 else g for g in groups]
                      
            stat, p_val = kruskal(*groups)
            results.append({'Feature': feature, 'H-Statistic': stat, 'p-value': p_val})
            
        res_df = pd.DataFrame(results)
        res_df.to_csv(os.path.join(self.output_dir, 'kruskal_significance.csv'), index=False)
        print(res_df)
        return res_df

    def plot_shap_explanations(self, model, X_test, feature_names):
        print("Generating SHAP summary plot...")
        # Ensure we don't pass the massive scikit-learn pipeline to SHAP
        # Extraction from CalibratedClassifierCV
        # calibrated_classifiers_ contains the fitted classifiers for each CV fold
        # We take the first fold's pipeline and extract the xgboost step
        fitted_pipeline = model.calibrated_model.calibrated_classifiers_[0].estimator
        xgb_model = fitted_pipeline.named_steps['xgb']
        
        # Explain the native model
        explainer = shap.TreeExplainer(xgb_model)
        
        # Use a manageable sample for SHAP
        sample_idx = np.random.choice(X_test.shape[0], min(X_test.shape[0], 500), replace=False)
        X_sample = X_test.iloc[sample_idx]
        
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure(figsize=(10, 8))
        # shap_values represents a list for multi-class. We plot absolute mean importance
        shap.summary_plot(shap_values, X_sample, plot_type="bar", feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'shap_summary_auto.png'), dpi=300)
        plt.close()

if __name__ == '__main__':
    print("Analyzer Module Loaded.")
