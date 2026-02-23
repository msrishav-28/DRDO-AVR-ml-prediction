import warnings
import pandas as pd
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

class RulForecaster:
    """
    Wrapper around TemporalFusionTransformer for Remaining Useful Life (RUL) prediction.
    """
    def __init__(self, max_encoder_length=120, max_prediction_length=50):
        # By default (at 10Hz):
        # 120 steps = 12 seconds of history lookback
        # 50 steps = 5 seconds of future forecasting
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.model = None
        self.training_dataset = None

    def prepare_dataset(self, df):
        print("Preparing TFT Dataset...")
        
        # TFT requires an integer time index
        df = df.copy()
        
        # Ensure correct types
        df['scenario'] = df['scenario'].astype(str)
        df['run_id'] = df['run_id'].astype(str)
        
        # Create group ID
        df['group_id'] = df['scenario'] + "_" + df['run_id']
        
        # Create continuous time index per group
        df['time_idx'] = df.groupby('group_id').cumcount()
        
        # We need to filter out rows where RUL is missing (NaN)
        # TFT cannot train on NaN targets
        df = df.dropna(subset=['time_to_next_fault'])
        
        if len(df) == 0:
            raise ValueError("No valid sequences found after dropping NaNs.")
            
        self.training_dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="time_to_next_fault",
            group_ids=["group_id"],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["scenario"],
            time_varying_known_reals=["time_idx", "load_percent"], # Things we know in future
            time_varying_unknown_reals=[
                "voltage", "current", "temperature", 
                "v_ripple", "dV_dt", "power"
            ], # Sensor readings
            target_normalizer=GroupNormalizer(
                groups=["scenario"], transformation="softplus"
            ), 
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        return self.training_dataset

    def build_model(self, learning_rate=0.03, hidden_size=64, attention_head_size=4, dropout=0.1):
        if self.training_dataset is None:
            raise ValueError("Call prepare_dataset before building the model.")
            
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_size // 2,
            output_size=7, # 7 quantiles (e.g., 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98)
            loss=QuantileLoss(),
            log_interval=10, 
            reduce_on_plateau_patience=4
        )
        return self.model

if __name__ == '__main__':
    print("TFT module loaded successfully.")
