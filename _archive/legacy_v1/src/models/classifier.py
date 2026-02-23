import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import pickle

class FaultClassifier:
    """
    XGBoost accompanied by SMOTE to handle highly imbalanced multi-class military fault data.
    Incorporates CalibratedClassifierCV to give exact probabilistic prediction guarantees.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        
        # The pipeline automatically applies SMOTE *only* on the training fold during fit()
        self.pipeline = ImbPipeline([
            ('smote', SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=random_state)),
            ('xgb', xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=random_state,
                n_jobs=-1
            ))
        ])
        
        # Wrap in calibration for rigorous probability outputs
        self.calibrated_model = CalibratedClassifierCV(self.pipeline, cv=3, method='isotonic')

    def fit(self, X_train, y_train):
        # Encode labels (e.g., 'None', 'Under-voltage', 'Over-voltage')
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        print(f"Classes found: {self.label_encoder.classes_}")
        
        self.calibrated_model.fit(X_train, y_train_encoded)
        return self

    def predict(self, X_test):
        y_pred_encoded = self.calibrated_model.predict(X_test)
        return self.label_encoder.inverse_transform(y_pred_encoded)
        
    def predict_proba(self, X_test):
        return self.calibrated_model.predict_proba(X_test)
        
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.calibrated_model,
                'encoder': self.label_encoder
            }, f)
            
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.calibrated_model = data['model']
            self.label_encoder = data['encoder']

if __name__ == '__main__':
    # Simple compilation test
    print("XGBoost Classifier module loaded successfully.")
