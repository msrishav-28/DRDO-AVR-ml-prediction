import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

class FeatureEngineer:
    def __init__(self, fs=10):
        self.fs = fs # 10Hz sampling frequency
        self.dt = 1.0 / fs
        
    def engineer_features(self, df):
        print("Engineering features...")
        # Ensure we don't mix time series across different runs or scenarios
        df = df.sort_values(by=['scenario', 'run_id', 'timestamp']).reset_index(drop=True)
        
        # We will apply transformations group by group
        engineered_dfs = []
        for (scenario, run_id), group in df.groupby(['scenario', 'run_id']):
            processed_group = self._process_group(group.copy())
            engineered_dfs.append(processed_group)
            
        return pd.concat(engineered_dfs, ignore_index=True)
        
    def _process_group(self, group):
        # 1. Rolling & Lag Features (Time Domain)
        for lag in [1, 5, 10]:
            group[f'voltage_lag_{lag}'] = group['voltage'].shift(lag)
            
        for window in [5, 10, 20]:
            group[f'voltage_rolling_mean_{window}'] = group['voltage'].rolling(window=window).mean()
            group[f'voltage_rolling_std_{window}'] = group['voltage'].rolling(window=window).std()
            group[f'current_rolling_mean_{window}'] = group['current'].rolling(window=window).mean()
            group[f'current_rolling_std_{window}'] = group['current'].rolling(window=window).std()

        # 2. Derivative Features (Rate of Change)
        group['dV_dt'] = group['voltage'].diff() / self.dt
        group['d2V_dt2'] = group['dV_dt'].diff() / self.dt
        group['dI_dt'] = group['current'].diff() / self.dt

        # 3. Cross-Variable Combinations
        group['power'] = group['voltage'] * group['current']
        group['thermal_load'] = group['power'] * group['temperature']
        
        # Prevent division by zero
        safe_current = group['current'].replace(0, 1e-6)
        safe_dI_dt = group['dI_dt'].replace(0, 1e-6)
        
        group['V_I_ratio'] = group['voltage'] / safe_current
        group['impedance_proxy'] = group['dV_dt'] / safe_dI_dt
        
        # 4. FFT Frequency Domain Features (Rolling Windows for localized spectra)
        # We use a 20-step (2-second) sliding window to compute local FFTs
        fft_window_size = 20
        
        # Initialize columns
        group['fft_dominant_freq'] = 0.0
        group['fft_power_2_5hz'] = 0.0
        group['fft_power_5_10hz'] = 0.0
        group['spectral_entropy'] = 0.0
        
        # Only compute if sequence is long enough
        if len(group) >= fft_window_size:
            # We can vectorize rolling apply or use a fast loop
            v_values = group['voltage'].values
            
            dom_freqs = np.zeros(len(v_values))
            power_2_5 = np.zeros(len(v_values))
            power_5_10 = np.zeros(len(v_values))
            entropy = np.zeros(len(v_values))
            
            for i in range(fft_window_size, len(v_values)):
                window = v_values[i-fft_window_size:i]
                spectrum = np.abs(rfft(window))
                freqs = rfftfreq(len(window), self.dt)
                
                # Dominant frequency (ignore DC component at index 0)
                dom_freqs[i] = freqs[np.argmax(spectrum[1:]) + 1] if len(spectrum) > 1 else 0
                
                # Band powers
                power_2_5[i] = np.sum(spectrum[(freqs >= 2) & (freqs <= 5)])
                power_5_10[i] = np.sum(spectrum[(freqs >= 5) & (freqs <= 10)])
                
                # Spectral Entropy
                prob_spect = spectrum / (np.sum(spectrum) + 1e-10)
                
                # mask 0s to prevent nan in log
                prob_spect_safe = prob_spect[prob_spect > 0]
                entropy[i] = -np.sum(prob_spect_safe * np.log(prob_spect_safe + 1e-10))
                
            group['fft_dominant_freq'] = dom_freqs
            group['fft_power_2_5hz'] = power_2_5
            group['fft_power_5_10hz'] = power_5_10
            group['spectral_entropy'] = entropy
            
        # Clean up NaNs from shifting and rolling
        # We use bfill then ffill
        group.bfill(inplace=True)
        group.ffill(inplace=True)
        group.fillna(0, inplace=True)
            
        return group

if __name__ == '__main__':
    import glob
    import os
    
    # Simple test logic
    raw_files = glob.glob('data/raw/*_run*.csv')
    raw_files = [f for f in raw_files if 'faults' not in f] # exclude fault logs
    
    if len(raw_files) == 0:
        print("No raw files found. Run data_generation first.")
        exit()
        
    print(f"Found {len(raw_files)} raw files to process.")
    
    engineer = FeatureEngineer()
    
    for file in raw_files:
        print(f"Processing {file}...")
        df = pd.read_csv(file)
        
        # Convert timestamp strings back to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        featured_df = engineer.engineer_features(df)
        
        # Save to processed directory
        filename = os.path.basename(file)
        os.makedirs('data/processed', exist_ok=True)
        featured_df.to_csv(os.path.join('data/processed', filename), index=False)
        print(f"Saved processed features to data/processed/{filename}")
