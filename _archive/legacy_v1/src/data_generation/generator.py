import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def generate_ou_voltage(n_steps, dt=0.1, mu=28.0, theta=2.5, sigma=0.15, min_v=16.0, max_v=48.0, seed=None):
    """
    Generates baseline voltage using the Ornstein-Uhlenbeck process.
    Models a PI-regulated 28VDC system bounded by MIL-STD-1275E hard limits.
    """
    rng = np.random.default_rng(seed)
    V = np.zeros(n_steps)
    V[0] = mu
    for t in range(1, n_steps):
        dW = rng.normal(0, np.sqrt(dt))
        # Mean reversion + noise
        V[t] = V[t-1] + theta * (mu - V[t-1]) * dt + sigma * dW
        
    return np.clip(V, min_v, max_v)

class CombatScenarioGenerator:
    def __init__(self, output_dir='data/raw', freq_hz=10):
        self.output_dir = output_dir
        self.freq_hz = freq_hz
        self.dt = 1.0 / freq_hz
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_scenario(self, duration_minutes, scenario='baseline', run_id=1, seed=None):
        print(f"Generating {scenario} data (Run {run_id})...")
        rng = np.random.default_rng(seed)
        
        n_steps = int(duration_minutes * 60 * self.freq_hz)
        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=i*self.dt) for i in range(n_steps)]
        
        # 1. Base Physical Quantities
        # Base voltage uses OU Process (MIL-STD-1275E nominal 28VDC)
        voltage = generate_ou_voltage(n_steps, dt=self.dt, mu=28.0, theta=2.5, sigma=0.15, seed=seed)
        
        # Nominal temperature (ambient 25C + ~40C operating delta = 65C)
        temperature = np.full(n_steps, 65.0) + rng.normal(0, 0.5, n_steps)
        
        # Base physical load tied to current (nominal 45A)
        load_percent = np.full(n_steps, 0.45) + rng.normal(0, 0.05, n_steps)
        current = load_percent * 100.0 # scale back to amps
        
        # Scenario Flags
        transient_flag = np.zeros(n_steps, dtype=int)
        event_id = np.zeros(n_steps, dtype=int)
        event_counter = 1
        
        # Fault injection probabilities and tracking
        fault_log = []
        
        # 2. Add Scenario-Specific Physics & Perturbations
        voltage, current, temperature, load_percent, transient_flag, event_id, event_counter, fault_log = self._apply_scenario_physics(
            scenario, n_steps, voltage, current, temperature, load_percent, transient_flag, event_id, event_counter, fault_log, timestamps, rng
        )
        
        # 3. Compile DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'load_percent': load_percent,
            'v_ripple': np.zeros(n_steps), # Placeholder for derived ripple
            'transient_flag': transient_flag,
            'event_id': event_id,
            'scenario': scenario,
            'run_id': run_id
        })
        
        # 4. Generate Target Labels
        df, fault_df = self._generate_labels(df, fault_log)
        
        # 5. Save Output
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(os.path.join(self.output_dir, f'{scenario}_run{run_id}_{timestamp_str}.csv'), index=False)
        if not fault_df.empty:
            fault_df.to_csv(os.path.join(self.output_dir, f'{scenario}_run{run_id}_faults_{timestamp_str}.csv'), index=False)
            
        return df, fault_df

    def _apply_scenario_physics(self, scenario, n_steps, voltage, current, temperature, load_percent, transient_flag, event_id, event_counter, fault_log, timestamps, rng):
        # Default physical parameters
        E_a = 0.7 # Activation energy (eV)
        k_B = 8.617e-5 # Boltzmann constant (eV/K)
        T_ref = 298.0 # Reference temp (K)
        
        if scenario == 'desert_heat':
            # Arrhenius thermal derating
            T_desert = 333.0 # +60C ambient
            temperature += (T_desert - 298.0) # Shift ambient up
            # Resistance increases, causing voltage sag under load
            R_ref = 0.02
            # R_hot(T) = R_ref * exp[ E_a/k_B * (1/T_ref - 1/T) ]
            temp_k = temperature + 273.15
            R_hot = R_ref * np.exp((E_a / k_B) * ((1.0 / T_ref) - (1.0 / temp_k)))
            voltage -= (current * (R_hot - R_ref))
            
            self._inject_faults(n_steps, 0.0015, timestamps, fault_log, voltage, rng)

        elif scenario == 'arctic_cold':
            # Arrhenius thermal derating (negative shift)
            T_arctic = 233.0 # -40C ambient
            temperature += (T_arctic - 298.0)
            
            # Cold-crank transient at t=0 per MIL-STD-1275E (drop to 16V for 0.5s)
            crank_steps = int(0.5 * self.freq_hz)
            voltage[:crank_steps] = 16.0
            transient_flag[:crank_steps] = 1
            event_id[:crank_steps] = event_counter
            event_counter += 1
            
            self._inject_faults(n_steps, 0.001, timestamps, fault_log, voltage, rng)

        elif scenario == 'artillery_firing':
            # Impulsive load steps: I_surge(t) = I_peak * e^(-t/tau)
            I_peak = 300.0
            tau = 0.05 # 50ms
            R_int = 0.025
            L = 0.001 # 1mH
            
            # Fire events: Poisson distributed, mean rate 0.5/min
            mean_rate_per_sec = 0.5 / 60.0
            prob_per_step = mean_rate_per_sec * self.dt
            
            for i in range(n_steps):
                if rng.random() < prob_per_step and i + int(0.2*self.freq_hz) < n_steps:
                    # Fire event
                    surge_len = int(0.2 * self.freq_hz) # last 200ms
                    t_surge = np.arange(0, surge_len) * self.dt
                    I_surge = I_peak * np.exp(-t_surge / tau)
                    
                    # dI/dt
                    dI_dt = - (I_peak / tau) * np.exp(-t_surge / tau)
                    
                    # V_bus(t) = V_OC - I_surge(t)*R_int - L * dI_surge/dt
                    V_sag = I_surge * R_int + L * dI_dt
                    
                    voltage[i:i+surge_len] -= V_sag
                    current[i:i+surge_len] += I_surge
                    load_percent[i:i+surge_len] = current[i:i+surge_len] / 100.0
                    
                    transient_flag[i:i+surge_len] = 1
                    event_id[i:i+surge_len] = event_counter
                    event_counter += 1
                    
            self._inject_faults(n_steps, 0.003, timestamps, fault_log, voltage, rng)

        elif scenario == 'rough_terrain':
            # Vibration-induced contact resistance oscillation
            # R_contact(t) = R_0(1 + A*sin(2*pi*f_vib*t + phi))
            R_0 = 0.015
            A = 0.15
            # Broad band vibration between 2-15Hz
            t_array = np.arange(n_steps) * self.dt
            f_vib = rng.uniform(2, 15, size=n_steps)
            
            # Compute phase integral
            phase = np.cumsum(2 * np.pi * f_vib * self.dt)
            R_contact = R_0 * (1 + A * np.sin(phase))
            
            voltage -= (current * R_contact)
            
            self._inject_faults(n_steps, 0.002, timestamps, fault_log, voltage, rng)

        elif scenario == 'weapons_active':
            # MIL-STD-1275E Section 5.3 switching transients
            # 5 successive switch events with 1s interval
            
            # Pick a random start time for the sequence
            start_idx = rng.integers(int(self.freq_hz), n_steps - int(6 * self.freq_hz))
            
            L = 0.002 # 2mH
            
            for event in range(5):
                idx = start_idx + int(event * 1.0 * self.freq_hz) # 1s intervals
                if idx + 5 >= n_steps: continue
                
                # Load jumps from 10% to 85% in 50ms (0.05s) -> 1 timestep approx
                # Actually delta I = 75A in 5ms -> dI/dt = 75 / 0.005 = 15000 A/s
                dI_dt = 15000 
                V_spike = L * dI_dt # 30V spike
                
                # Apply spike
                voltage[idx] += V_spike
                current[idx:] += 75.0 # Load stays high
                load_percent[idx:] += 0.75
                
                transient_flag[idx:idx+2] = 1
                event_id[idx:idx+2] = event_counter
                event_counter += 1
                
            self._inject_faults(n_steps, 0.004, timestamps, fault_log, voltage, rng)

        elif scenario == 'emp_simulation':
            # MIL-STD-461G RS105 double exponential pulse
            # V_EMP(t) = V_0 * E_0 * (e^(-alpha*t) - e^(-beta*t))
            alpha = 4e6
            beta = 4.76e8
            V_0 = 50000.0
            E_0 = 0.001
            
            # One major EMP pulse in the middle
            idx = n_steps // 2
            
            # High resolution subsampling for the pulse (since it occurs in microseconds)
            # At 10Hz, the pulse is a single delta spike.
            voltage[idx] += (V_0 * E_0) # Add a 50V spike instantly
            transient_flag[idx] = 1
            event_id[idx] = event_counter
            event_counter += 1
            
            # Add +2dB wideband noise floor permanently after event
            # 2dB voltage multiplier = 10^(2/20) = 1.258
            # Just increase the noise amplitude of the OU process manually
            noise_increase = rng.normal(0, 0.15 * 1.258, n_steps - idx)
            voltage[idx:] += noise_increase
            
            self._inject_faults(n_steps, 0.005, timestamps, fault_log, voltage, rng)
            
        else: # baseline
            self._inject_faults(n_steps, 0.0005, timestamps, fault_log, voltage, rng)

        # Final MIL-STD hard clipping
        voltage = np.clip(voltage, 16.0, 48.0)
            
        return voltage, current, temperature, load_percent, transient_flag, event_id, event_counter, fault_log
        
    def _inject_faults(self, n_steps, prob, timestamps, fault_log, voltage, rng):
        for i in range(10, n_steps - 10):
            if rng.random() < prob:
                if rng.random() < 0.5:
                    fault_type = 'Under-voltage'
                    severity = rng.integers(1, 4)
                    voltage[i:i+3] -= rng.uniform(5, 12) # Sag
                else:
                    fault_type = 'Over-voltage'
                    severity = rng.integers(1, 4)
                    voltage[i:i+3] += rng.uniform(5, 12) # Swell
                    
                fault_log.append({
                    'timestamp': timestamps[i],
                    'fault_type': fault_type,
                    'severity': severity
                })

    def _generate_labels(self, df, fault_log):
        # Default labels
        df['time_to_next_fault'] = np.nan # RUL in seconds
        df['fault_within_5s'] = 0
        df['fault_within_30s'] = 0
        df['fault_type'] = 'None'
        df['fault_severity'] = 0
        
        fault_df = pd.DataFrame(fault_log)
        
        if fault_df.empty:
            # If no faults, RUL is just max time remaining (censored)
            df['time_to_next_fault'] = (df['timestamp'].max() - df['timestamp']).dt.total_seconds()
            return df, fault_df
            
        fault_times = pd.to_datetime(fault_df['timestamp']).sort_values().values
        
        # Calculate RUL mapping
        # Iterates backwards or uses searchsorted to find the *next* fault
        current_times = df['timestamp'].values
        next_fault_idx = np.searchsorted(fault_times, current_times)
        
        # for indices where a next fault exists
        valid_mask = next_fault_idx < len(fault_times)
        
        # Time to next fault in seconds
        rul_deltas = (fault_times[next_fault_idx[valid_mask]] - current_times[valid_mask]).astype('timedelta64[ms]').astype(float) / 1000.0
        df.loc[valid_mask, 'time_to_next_fault'] = rul_deltas
        
        # For data points after the *last* fault, mark as censored (time remaining in run)
        df.loc[~valid_mask, 'time_to_next_fault'] = (df['timestamp'].max() - df['timestamp'][~valid_mask]).dt.total_seconds()
        
        # Classification Targets
        df.loc[df['time_to_next_fault'] <= 5.0, 'fault_within_5s'] = 1
        df.loc[df['time_to_next_fault'] <= 30.0, 'fault_within_30s'] = 1
        
        # Apply fault types exactly AT the fault timestamp
        for _, row in fault_df.iterrows():
            mask = df['timestamp'] == row['timestamp']
            df.loc[mask, 'fault_type'] = row['fault_type']
            df.loc[mask, 'fault_severity'] = row['severity']
            
        return df, fault_df

if __name__ == '__main__':
    generator = CombatScenarioGenerator()
    
    # 1. Baseline Data (4 runs of 120 mins)
    for i in range(4):
        generator.generate_scenario(120, 'baseline', run_id=i+1)
        
    # 2. Combat Scenarios (30 mins each)
    scenarios = ['arctic_cold', 'desert_heat', 'artillery_firing', 'rough_terrain', 'weapons_active', 'emp_simulation']
    for val, scenario in enumerate(scenarios):
        generator.generate_scenario(30, scenario, run_id=val+1)

